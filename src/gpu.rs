//! CUDA engine for qwen3moe. Same numerics as the CPU reference (f32 activations and KV
//! cache, weights dequantized on the fly), with the dense weights resident in VRAM and the
//! experts in host RAM, cached in VRAM slots by `ExpertCache` (ubatches up to 32 tokens; larger
//! ones read the cached experts and stream the rest through staging slots, as llama.cpp).
//!
//! Kernel seam: kernels are loaded by name from an embedded fatbin built from
//! kernels/eomoe.cu (build.rs); an all-Rust module exporting the same names can replace it.

use crate::expert_cache::ExpertCache;
use crate::gguf::{GgmlType, Gguf, TensorInfo};
use crate::quant::dequant_row;
use crate::qwen3moe::{route, Engine, Weights};
use cudarc::driver::{result, sys::CUdeviceptr, CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PinnedHostSlice, PushKernelArg};
use rayon::prelude::*;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/eomoe.fatbin"));

type Ptr = CUdeviceptr;

/// Larger ubatches touch most experts of a layer: caching them would evict the decode set.
const CACHE_MAX_BATCH: usize = 32;
/// Staging slots for the misses of uncached (large) ubatches, used in waves.
const STAGE: usize = 32;
/// VRAM left free after sizing the expert cache to "as much as fits".
const VRAM_RESERVE: usize = 192 << 20;

/// Layers whose experts have the same layout share one bank of slots and one cache.
struct Group {
    cache: ExpertCache,
    bank: Ptr,
    /// gate | up | down bytes of one expert.
    slot_bytes: usize,
}

/// Every expert in page-locked RAM in slot layout (gate | up | down), so a miss is one async
/// DMA at full PCIe speed (pageable copies are ~30% slower and block the host).
/// ponytail: pins all expert bytes (~16 GiB for Qwen3-30B); a RAM-tight host gets the
/// pageable fallback, add a config switch / partial pinning if that needs tuning.
struct HostExperts {
    buf: PinnedHostSlice<u8>,
    /// [n_layer] offset of expert 0 of the layer.
    base: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    /// Bytes uploaded into cache slots.
    pub bytes: u64,
}

macro_rules! launch {
    ($s:expr, $f:expr, $grid:expr, $block:expr, [$($a:ident),*]) => {{
        let cfg = LaunchConfig { grid_dim: $grid, block_dim: ($block, 1, 1), shared_mem_bytes: 0 };
        let mut b = $s.stream.launch_builder(&$f);
        $(b.arg(&$a);)*
        unsafe { b.launch(cfg) }.expect("kernel launch");
    }};
}

struct Kernels {
    mm_f32: CudaFunction,
    mm_q4_k: CudaFunction,
    mm_q6_k: CudaFunction,
    mm_q4_k_ids: CudaFunction,
    mm_q6_k_ids: CudaFunction,
    rmsnorm: CudaFunction,
    qk_norm_rope: CudaFunction,
    attention: CudaFunction,
    add_inplace: CudaFunction,
    gather_rows: CudaFunction,
    swiglu: CudaFunction,
    moe_combine: CudaFunction,
}

/// A quantized matrix in VRAM: n_rows rows of k elements.
#[derive(Clone, Copy)]
struct Mat {
    p: Ptr,
    ty: GgmlType,
    row_bytes: usize,
    k: usize,
    n_rows: usize,
}

struct DLayer {
    attn_norm: Ptr,
    wq: Mat,
    wk: Mat,
    wv: Mat,
    wo: Mat,
    q_norm: Ptr,
    k_norm: Ptr,
    ffn_norm: Ptr,
    router: Mat,
}

#[derive(Clone, Copy)]
struct Scratch {
    x: Ptr,
    h: Ptr,
    q: Ptr,
    attn: Ptr,
    o: Ptr,
    rl: Ptr,
    /// Gathered rows: inputs, [gate | up] outputs, swiglu activations, down outputs.
    xg: Ptr,
    gu: Ptr,
    act: Ptr,
    db: Ptr,
    logits: Ptr,
    /// Per layer routing, see `forward_chunk`: [gidx | trow | tw | wp].
    meta: Ptr,
}

pub struct GpuModel {
    w: Weights,
    stream: Arc<CudaStream>,
    k: Kernels,
    layers: Vec<DLayer>,
    out_norm: Ptr,
    output: Mat,
    /// Per layer: K then V, each n_ctx x (n_head_kv * head_dim) f32.
    kv: Ptr,
    groups: Vec<Group>,
    group_of: Vec<usize>,
    /// STAGE slots of the largest expert, for uncached ubatches.
    stage: Ptr,
    stage_bytes: usize,
    stats: CacheStats,
    host: Option<HostExperts>,
    /// Page-locked staging for the per-layer routing upload.
    meta_host: PinnedHostSlice<u8>,
    s: Scratch,
    n_ctx: usize,
    max_batch: usize,
    n_past: usize,
    /// Owns every device allocation the raw pointers above point into.
    _bufs: Vec<CudaSlice<u8>>,
    /// Bytes allocated on the device.
    pub vram_bytes: usize,
}

const F: usize = std::mem::size_of::<f32>();

fn err(e: impl std::fmt::Debug) -> String {
    format!("CUDA: {e:?}")
}

impl GpuModel {
    /// `with_cache` with the oracle's 4800 MiB expert cache.
    pub fn new(g: Gguf, n_ctx: usize) -> Result<Self, String> {
        Self::with_cache(g, n_ctx, 4800)
    }

    /// Load the dense weights to GPU 0, allocate a KV cache for `n_ctx` positions and up to
    /// `cache_mib` of expert slots (capped by free VRAM, so `usize::MAX >> 20` = all that fits),
    /// split so each group caches the same fraction of its experts.
    pub fn with_cache(g: Gguf, n_ctx: usize, cache_mib: usize) -> Result<Self, String> {
        let w = Weights::load(g)?;
        let ctx = CudaContext::new(0).map_err(err)?;
        let stream = ctx.default_stream();
        let m = ctx.load_module(Ptx::from_binary(FATBIN.to_vec())).map_err(err)?;
        let f = |n: &str| m.load_function(n).map_err(err);
        let k = Kernels {
            mm_f32: f("mm_f32")?,
            mm_q4_k: f("mm_q4_k")?,
            mm_q6_k: f("mm_q6_k")?,
            mm_q4_k_ids: f("mm_q4_k_ids")?,
            mm_q6_k_ids: f("mm_q6_k_ids")?,
            rmsnorm: f("rmsnorm")?,
            qk_norm_rope: f("qk_norm_rope")?,
            attention: f("attention")?,
            add_inplace: f("add_inplace")?,
            gather_rows: f("gather_rows")?,
            swiglu: f("swiglu")?,
            moe_combine: f("moe_combine")?,
        };

        let mut bufs = vec![];
        let vram = std::cell::Cell::new(0usize);
        let mut alloc = |bytes: usize| -> Result<Ptr, String> {
            let b = stream.alloc_zeros::<u8>(bytes.max(1)).map_err(err)?;
            let p = b.device_ptr(&stream).0;
            bufs.push(b);
            vram.set(vram.get() + bytes);
            Ok(p)
        };
        let mut upload = |t: &TensorInfo| -> Result<Ptr, String> {
            let p = alloc(t.size)?;
            unsafe { result::memcpy_htod_sync(p, w.g.data(t)) }.map_err(err)?;
            Ok(p)
        };
        let mat = |t: &TensorInfo, p: Ptr| -> Result<Mat, String> {
            check_ty(t)?;
            Ok(Mat { p, ty: t.ty, row_bytes: t.row_bytes(), k: t.dims[0] as usize, n_rows: t.size / t.row_bytes() })
        };
        let mut layers = vec![];
        for l in &w.layers {
            layers.push(DLayer {
                attn_norm: upload(&l.attn_norm)?,
                wq: mat(&l.wq, upload(&l.wq)?)?,
                wk: mat(&l.wk, upload(&l.wk)?)?,
                wv: mat(&l.wv, upload(&l.wv)?)?,
                wo: mat(&l.wo, upload(&l.wo)?)?,
                q_norm: upload(&l.q_norm)?,
                k_norm: upload(&l.k_norm)?,
                ffn_norm: upload(&l.ffn_norm)?,
                router: mat(&l.router, upload(&l.router)?)?,
            });
            for t in [&l.gate, &l.up, &l.down] {
                if !matches!(t.ty, GgmlType::Q4_K | GgmlType::Q6_K) {
                    return Err(format!("{}: {:?} experts have no CUDA kernel yet", t.name, t.ty));
                }
            }
            // gate and up are read as one [gate | up] matrix of each slot.
            if l.gate.ty != l.up.ty || l.gate.dims != l.up.dims {
                return Err(format!("{}: gate and up experts differ in type or shape", l.gate.name));
            }
        }
        let out_norm = upload(&w.out_norm)?;
        let output = mat(&w.output, upload(&w.output)?)?;
        let dense = vram.get();

        let hp = &w.hp;
        if hp.head_dim % 32 != 0 || hp.head_dim > 256 {
            return Err(format!("head_dim {} unsupported by the attention kernel", hp.head_dim));
        }
        let (e, kvd, nq, kk) = (hp.n_embd, hp.n_head_kv * hp.head_dim, hp.n_head * hp.head_dim, hp.n_expert_used);
        let kv = alloc(hp.n_layer * 2 * n_ctx * kvd * F)?;
        let ebytes = |l: &crate::qwen3moe::Layer| w.expert(l, 0).iter().map(|b| b.len()).sum::<usize>();
        let layout = |l: &crate::qwen3moe::Layer| (l.gate.ty, l.up.ty, l.down.ty, ebytes(l));
        let mut layouts = vec![];
        let group_of: Vec<usize> = w
            .layers
            .iter()
            .map(|l| layouts.iter().position(|x| *x == layout(l)).unwrap_or_else(|| {
                layouts.push(layout(l));
                layouts.len() - 1
            }))
            .collect();
        let host_bytes: usize = w.layers.iter().map(|l| ebytes(l) * hp.n_expert).sum();
        let stage_bytes = layouts.iter().map(|l| l.3).max().unwrap_or(0);
        let host = pin_experts(&ctx, &w, host_bytes);
        let stage = alloc(STAGE * stage_bytes)?;
        let b = n_ctx.min(512);
        let s = Scratch {
            x: alloc(b * e * F)?,
            h: alloc(b * e * F)?,
            q: alloc(b * nq * F)?,
            attn: alloc(b * nq * F)?,
            o: alloc(b * e * F)?,
            rl: alloc(b * hp.n_expert * F)?,
            xg: alloc(b * kk * e * F)?,
            gu: alloc(b * kk * 2 * hp.n_ff_exp * F)?,
            act: alloc(b * kk * hp.n_ff_exp * F)?,
            db: alloc(b * kk * e * F)?,
            logits: alloc(b * hp.n_vocab * F)?,
            meta: alloc(meta_bytes(b * kk))?,
        };
        let meta_host = unsafe { ctx.alloc_pinned_with_flags::<u8>(meta_bytes(b * kk), 0) }.map_err(err)?;
        // The expert banks get what is left: the requested MiB, capped by free VRAM.
        let (free, _) = result::mem_get_info().map_err(err)?;
        let budget = (cache_mib << 20).min(free.saturating_sub(VRAM_RESERVE));
        if budget < cache_mib << 20 {
            if cache_mib != usize::MAX >> 20 { log::warn!("gpu: expert cache capped at {} MiB of free VRAM (asked {cache_mib})", budget >> 20); }
        }
        let frac = (budget as f64 / host_bytes as f64).min(1.0);
        let mut groups = vec![];
        for (gi, lay) in layouts.iter().enumerate() {
            let n_layers = group_of.iter().filter(|&&x| x == gi).count();
            // At least one cached ubatch's worth of distinct experts.
            let min_slots = hp.n_expert.min(hp.n_expert_used * CACHE_MAX_BATCH);
            let n_slots = ((frac * (n_layers * hp.n_expert) as f64) as usize).max(min_slots);
            groups.push(Group { cache: ExpertCache::new(hp.n_layer, hp.n_expert, n_slots), bank: alloc(n_slots * lay.3)?, slot_bytes: lay.3 });
        }
        log::info!(
            "gpu: dense weights {} MiB, expert cache {} MiB ({:?} slots per group), total {} MiB (n_ctx {n_ctx}, batch {b})",
            dense >> 20,
            groups.iter().map(|g| g.cache.n_slots() * g.slot_bytes).sum::<usize>() >> 20,
            groups.iter().map(|g| g.cache.n_slots()).collect::<Vec<_>>(),
            vram.get() >> 20
        );
        Ok(Self {
            stream,
            k,
            layers,
            out_norm,
            output,
            kv,
            groups,
            group_of,
            stage,
            stage_bytes,
            stats: CacheStats::default(),
            host,
            meta_host,
            s,
            n_ctx,
            max_batch: b,
            n_past: 0,
            _bufs: bufs,
            vram_bytes: vram.get(),
            w,
        })
    }

    /// Expert cache counters since load (cached ubatches only).
    pub fn cache_stats(&self) -> CacheStats {
        self.stats
    }

    /// Queue a copy of expert `ex` of layer `li` (gate | up | down) to `dst`; returns its bytes.
    fn upload_expert(&self, li: usize, ex: usize, dst: Ptr) -> usize {
        let parts = self.w.expert(&self.w.layers[li], ex);
        let n: usize = parts.iter().map(|b| b.len()).sum();
        match &self.host {
            Some(h) => {
                let src = &h.buf.as_slice().expect("pinned")[h.base[li] + ex * n..][..n];
                unsafe { result::memcpy_htod_async(dst, src, self.stream.cu_stream()) }.expect("htod async");
            }
            None => {
                let mut off = 0;
                for b in parts {
                    self.htod(dst + off as u64, b);
                    off += b.len();
                }
            }
        }
        n
    }

    fn mm(&self, m: &Mat, x: Ptr, y: Ptr, ntok: usize) {
        // Quantized kernels: one warp per row, MM_ROWS (= 4) rows per block.
        let (f, gx, block) = match m.ty {
            GgmlType::F32 => (&self.k.mm_f32, m.n_rows, 256),
            GgmlType::Q4_K => (&self.k.mm_q4_k, m.n_rows.div_ceil(4), 128),
            GgmlType::Q6_K => (&self.k.mm_q6_k, m.n_rows.div_ceil(4), 128),
            t => unreachable!("checked at load: {t:?}"),
        };
        let (p, rb, k, n) = (m.p, m.row_bytes as u64, m.k as i32, m.n_rows as i32);
        launch!(self, f, (gx as u32, ntok as u32, 1), block, [p, rb, k, x, y, n]);
    }

    /// Grouped expert matvec: gathered row j reads the weights at wp[j] + off.
    #[allow(clippy::too_many_arguments)]
    fn mm_ids(&self, ty: GgmlType, wp: Ptr, off: u64, row_bytes: usize, k: usize, x: Ptr, y: Ptr, n_rows: usize, ntok: usize) {
        let f = match ty {
            GgmlType::Q4_K => &self.k.mm_q4_k_ids,
            GgmlType::Q6_K => &self.k.mm_q6_k_ids,
            t => unreachable!("checked at load: {t:?}"),
        };
        let (rb, k, n) = (row_bytes as u64, k as i32, n_rows as i32);
        launch!(self, f, (n_rows.div_ceil(4) as u32, ntok as u32, 1), 128, [wp, off, rb, k, x, y, n]);
    }

    fn htod<T>(&self, dst: Ptr, src: &[T]) {
        unsafe { result::memcpy_htod_sync(dst, src) }.expect("htod");
    }

    fn dtoh(&self, src: Ptr, n: usize) -> Vec<f32> {
        let mut v = vec![0f32; n];
        self.stream.synchronize().expect("sync");
        unsafe { result::memcpy_dtoh_sync(&mut v, src) }.expect("dtoh");
        v
    }

    fn check_ctx(&self, n: usize) {
        assert!(self.n_past + n <= self.n_ctx, "context overflow: {} + {n} > {}", self.n_past, self.n_ctx);
    }

    /// Run one ubatch; returns the logits of every token, or of the last only.
    fn forward_chunk(&mut self, tokens: &[u32], all_logits: bool) -> Vec<f32> {
        let hp = &self.w.hp;
        let (e, hd, nh, nkv, ff, kk) = (hp.n_embd, hp.head_dim, hp.n_head, hp.n_head_kv, hp.n_ff_exp, hp.n_expert_used);
        let (kvd, t) = (nkv * hd, tokens.len());
        let s = self.s;
        let nv = hp.n_vocab;
        let n_expert = hp.n_expert;

        let mut xe = vec![0f32; t * e];
        for (row, &tok) in xe.chunks_exact_mut(e).zip(tokens) {
            dequant_row(self.w.tok_embd.ty, self.w.g.row(&self.w.tok_embd, tok as usize), row);
        }
        self.htod(s.x, &xe);

        let (ei, eps, base, pos0) = (e as i32, hp.eps, hp.rope_base, self.n_past as i32);
        let (nhi, nkvi, hdi, tn) = (nh as i32, nkv as i32, hd as i32, t as u32);
        let scale = 1.0 / (hd as f32).sqrt();
        for (li, l) in self.layers.iter().enumerate() {
            let (x, h, q, attn, o) = (s.x, s.h, s.q, s.attn, s.o);
            let kc = self.kv + (li * 2 * self.n_ctx * kvd * F) as u64;
            let vc = kc + (self.n_ctx * kvd * F) as u64;
            let (kd, vd) = (kc + (self.n_past * kvd * F) as u64, vc + (self.n_past * kvd * F) as u64);

            // --- attention ---
            let wn = l.attn_norm;
            launch!(self, self.k.rmsnorm, (tn, 1, 1), 256, [h, x, wn, ei, eps]);
            self.mm(&l.wq, h, q, t);
            self.mm(&l.wk, h, kd, t);
            self.mm(&l.wv, h, vd, t);
            let (qn, kn) = (l.q_norm, l.k_norm);
            launch!(self, self.k.qk_norm_rope, (tn * nh as u32, 1, 1), hd as u32, [q, qn, nhi, hdi, pos0, eps, base]);
            launch!(self, self.k.qk_norm_rope, (tn * nkv as u32, 1, 1), hd as u32, [kd, kn, nkvi, hdi, pos0, eps, base]);
            launch!(self, self.k.attention, (nh as u32, tn, 1), 32 * 8, [q, kc, vc, attn, nhi, nkvi, hdi, pos0, scale]);
            self.mm(&l.wo, attn, o, t);
            let n = (t * e) as i32;
            launch!(self, self.k.add_inplace, ((t * e).div_ceil(256) as u32, 1, 1), 256, [x, o, n]);

            // --- MoE FFN: route on the host, stage the selected experts, run them ---
            let wn = l.ffn_norm;
            launch!(self, self.k.rmsnorm, (tn, 1, 1), 256, [h, x, wn, ei, eps]);
            self.mm(&l.router, h, s.rl, t);
            let sel = route(&self.dtoh(s.rl, t * n_expert), n_expert, kk);
            let mut by_expert: Vec<Vec<(usize, f32)>> = vec![vec![]; n_expert];
            for (ti, ts) in sel.iter().enumerate() {
                for &(ex, wgt) in ts {
                    by_expert[ex].push((ti, wgt));
                }
            }
            // Rows grouped by expert (ascending); per token, its k rows in expert order.
            let (mut gidx, mut segs) = (vec![], vec![]);
            let (mut trow, mut tw) = (vec![vec![]; t], vec![vec![]; t]);
            for (ex, toks) in by_expert.iter().enumerate().filter(|(_, v)| !v.is_empty()) {
                segs.push((ex, gidx.len(), toks.len()));
                for &(ti, wgt) in toks {
                    trow[ti].push(gidx.len() as i32);
                    tw[ti].push(wgt);
                    gidx.push(ti as i32);
                }
            }
            let r = gidx.len();
            // wp[j]: the slot (gate | up | down) of the expert of gathered row j.
            let mut wp = vec![0u64; r];
            let gi = self.group_of[li];
            let cached = t <= CACHE_MAX_BATCH;
            if cached {
                let ids: Vec<u32> = sel.iter().flatten().map(|&(e, _)| e as u32).collect();
                let p = self.groups[gi].cache.plan(li, &ids, t);
                let (bank, sb) = (self.groups[gi].bank, self.groups[gi].slot_bytes);
                for &(ex, slot) in &p.fills {
                    self.stats.bytes += self.upload_expert(li, ex, bank + (slot * sb) as u64) as u64;
                }
                self.stats.hits += p.hits as u64;
                self.stats.misses += p.fills.len() as u64;
                for &(ex, r0, nr) in &segs {
                    let slot = self.groups[gi].cache.slot(li, ex).expect("planned expert resident");
                    wp[r0..r0 + nr].fill(bank + (slot * sb) as u64);
                }
            }
            // One async copy of the routing. The pinned buffer is free again: the router dtoh
            // above waited for the previous layer's copy.
            let (o_tr, o_tw, o_wp) = meta_offsets(r);
            {
                let m = self.meta_host.as_mut_slice().expect("pinned");
                let put = |m: &mut [u8], o: usize, v: [u8; 4]| m[o..o + 4].copy_from_slice(&v);
                for (j, &v) in gidx.iter().enumerate() {
                    put(m, 4 * j, v.to_le_bytes());
                }
                for (j, &v) in trow.concat().iter().enumerate() {
                    put(m, o_tr + 4 * j, v.to_le_bytes());
                }
                for (j, &v) in tw.concat().iter().enumerate() {
                    put(m, o_tw + 4 * j, v.to_le_bytes());
                }
                for (j, &v) in wp.iter().enumerate() {
                    m[o_wp + 8 * j..o_wp + 8 * j + 8].copy_from_slice(&v.to_le_bytes());
                }
                unsafe { result::memcpy_htod_async(s.meta, &m[..o_wp + 8 * r], self.stream.cu_stream()) }.expect("htod async");
            }
            let (xg, gu, act, dbuf) = (s.xg, s.gu, s.act, s.db);
            let (gi_, tr, twp, wpd) = (s.meta, s.meta + o_tr as u64, s.meta + o_tw as u64, s.meta + o_wp as u64);
            launch!(self, self.k.gather_rows, (r as u32, 1, 1), 256, [xg, h, gi_, ei]);
            let wl = &self.w.layers[li];
            let down_off = (self.w.expert(wl, 0)[0].len() * 2) as u64;
            // Cached ubatches run all experts at once; uncached ones in waves of STAGE, the
            // misses of each wave streamed through the staging slots.
            for wave in segs.chunks(if cached { segs.len().max(1) } else { STAGE }) {
                let (r0, rn) = (wave[0].1, wave.iter().map(|x| x.2).sum::<usize>());
                if !cached {
                    let g = &self.groups[gi];
                    for (j, &(ex, w0, nr)) in wave.iter().enumerate() {
                        let p = match g.cache.slot(li, ex) {
                            Some(slot) => g.bank + (slot * g.slot_bytes) as u64,
                            None => {
                                let p = self.stage + (j * self.stage_bytes) as u64;
                                self.upload_expert(li, ex, p);
                                p
                            }
                        };
                        wp[w0..w0 + nr].fill(p);
                    }
                    self.htod(wpd + (r0 * 8) as u64, &wp[r0..r0 + rn]);
                }
                let (wpw, xin) = (wpd + (r0 * 8) as u64, xg + (r0 * e * F) as u64);
                let (guw, aw) = (gu + (r0 * 2 * ff * F) as u64, act + (r0 * ff * F) as u64);
                self.mm_ids(wl.gate.ty, wpw, 0, wl.gate.row_bytes(), e, xin, guw, 2 * ff, rn);
                let (ffi, rni) = (ff as i32, rn as i32);
                launch!(self, self.k.swiglu, ((rn * ff).div_ceil(256) as u32, 1, 1), 256, [aw, guw, ffi, rni]);
                self.mm_ids(wl.down.ty, wpw, down_off, wl.down.row_bytes(), ff, aw, dbuf + (r0 * e * F) as u64, e, rn);
            }
            let ki = kk as i32;
            launch!(self, self.k.moe_combine, (tn, 1, 1), 256, [x, dbuf, tr, twp, ki, ei]);
        }
        self.n_past += t;

        let (r0, nr) = if all_logits { (0, t) } else { (t - 1, 1) };
        let (x, h, wn, nrn) = (s.x + (r0 * e * F) as u64, s.h, self.out_norm, nr as u32);
        launch!(self, self.k.rmsnorm, (nrn, 1, 1), 256, [h, x, wn, ei, eps]);
        self.mm(&self.output, h, s.logits, nr);
        self.dtoh(s.logits, nr * nv)
    }
}

/// Byte offsets of trow, tw and wp in the routing buffer for r gathered rows (gidx at 0).
fn meta_offsets(r: usize) -> (usize, usize, usize) {
    (4 * r, 8 * r, (12 * r).next_multiple_of(8))
}

fn meta_bytes(r: usize) -> usize {
    meta_offsets(r).2 + 8 * r
}

/// Copy every expert into pinned RAM in slot layout; None (pageable fallback) if pinning fails.
fn pin_experts(ctx: &Arc<CudaContext>, w: &Weights, total: usize) -> Option<HostExperts> {
    let t0 = std::time::Instant::now();
    let mut buf = match unsafe { ctx.alloc_pinned_with_flags::<u8>(total, 0) } {
        Ok(b) => b,
        Err(e) => {
            log::warn!("gpu: pinning {} MiB of experts failed ({e:?}), using pageable copies", total >> 20);
            return None;
        }
    };
    let (mut base, mut off) = (vec![], 0);
    for l in &w.layers {
        base.push(off);
        off += w.expert(l, 0).iter().map(|b| b.len()).sum::<usize>() * w.hp.n_expert;
    }
    let dst = buf.as_mut_slice().expect("pinned");
    let mut chunks = vec![];
    let mut rest = &mut dst[..];
    for l in &w.layers {
        for ex in 0..w.hp.n_expert {
            let parts = w.expert(l, ex);
            let (c, r) = rest.split_at_mut(parts.iter().map(|b| b.len()).sum());
            chunks.push((c, parts));
            rest = r;
        }
    }
    chunks.into_par_iter().for_each(|(c, parts)| {
        let mut o = 0;
        for b in parts {
            c[o..o + b.len()].copy_from_slice(b);
            o += b.len();
        }
    });
    log::info!("gpu: pinned {} MiB of experts in {:.1}s", total >> 20, t0.elapsed().as_secs_f32());
    Some(HostExperts { buf, base })
}

fn check_ty(t: &TensorInfo) -> Result<(), String> {
    match t.ty {
        GgmlType::F32 | GgmlType::Q4_K | GgmlType::Q6_K => Ok(()),
        ty => Err(format!("{}: {ty:?} has no CUDA matmul kernel yet", t.name)),
    }
}

impl Engine for GpuModel {
    fn n_vocab(&self) -> usize {
        self.w.hp.n_vocab
    }

    fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    fn truncate(&mut self, n_past: usize) {
        self.n_past = self.n_past.min(n_past);
    }

    /// Panics if the KV cache would overflow n_ctx.
    fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        self.check_ctx(tokens.len());
        tokens.chunks(self.max_batch).flat_map(|c| self.forward_chunk(c, true)).collect()
    }

    fn forward_last(&mut self, tokens: &[u32]) -> Vec<f32> {
        self.check_ctx(tokens.len());
        let n = tokens.len().div_ceil(self.max_batch);
        let mut last = vec![];
        for (i, c) in tokens.chunks(self.max_batch).enumerate() {
            last = self.forward_chunk(c, false);
            if i + 1 < n {
                last.clear();
            }
        }
        last
    }
}
