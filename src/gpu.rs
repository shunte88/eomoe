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
use cudarc::driver::{result, sys::CUdeviceptr, CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PinnedHostSlice, PushKernelArg};
use rayon::prelude::*;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/eomoe.fatbin"));

type Ptr = CUdeviceptr;

/// Larger ubatches touch most experts of a layer: caching them would evict the decode set.
const CACHE_MAX_BATCH: usize = 32;
/// Staging slots for the misses of uncached (large) ubatches, used in waves.
const STAGE: usize = 32;
/// Batches of at least this many tokens (per expert, for experts) use the tiled GEMM.
const GEMM_MIN: usize = 16;
/// Tokens per layer-major prefill pass: the experts move once per SPAN tokens. Costs
/// 2 x SPAN x n_embd x 4 bytes of VRAM (64 MiB) for the hidden states.
const SPAN: usize = 4096;
/// Logit rows per output matmul (all-logits calls): 64 x n_vocab x 4 bytes = 39 MiB.
/// Positions per split-K block of decode attention.
const ATT_SPLIT: usize = 128;
const LOGIT_ROWS: usize = 64;
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
    mm_q4_k_t8: CudaFunction,
    mm_q6_k_t8: CudaFunction,
    mm_q4_k_ids_t8: CudaFunction,
    mm_q6_k_ids_t8: CudaFunction,
    gemm_q4_k: CudaFunction,
    gemm_q6_k: CudaFunction,
    gemm_q4_k_ids: CudaFunction,
    gemm_q6_k_ids: CudaFunction,
    rmsnorm: CudaFunction,
    qk_norm_rope: CudaFunction,
    attention: CudaFunction,
    attention_dec: CudaFunction,
    attention_tc: CudaFunction,
    attn_merge: CudaFunction,
    store_f16: CudaFunction,
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
    /// This chunk's K | V (f32) before roping and storing into the f16 cache.
    kvs: Ptr,
    o: Ptr,
    rl: Ptr,
    /// Gathered rows: inputs, [gate | up] outputs, swiglu activations, down outputs.
    xg: Ptr,
    gu: Ptr,
    act: Ptr,
    db: Ptr,
    logits: Ptr,
    /// Per layer routing, see `forward_span` and `meta_bytes`.
    meta: Ptr,
}

pub struct GpuModel {
    w: Weights,
    stream: Arc<CudaStream>,
    k: Kernels,
    layers: Vec<DLayer>,
    out_norm: Ptr,
    output: Mat,
    /// Per layer: K then V, each n_ctx x (n_head_kv * head_dim) f16.
    kv: Ptr,
    groups: Vec<Group>,
    group_of: Vec<usize>,
    /// STAGE slots of the largest expert, for uncached ubatches: two halves, so one wave's
    /// misses upload on `copy` while the previous wave computes on `stream`.
    stage: Ptr,
    copy: Arc<CudaStream>,
    /// Per stage half: uploads done (copy stream) / last reader done (main stream).
    ready: [CudaEvent; 2],
    free: [CudaEvent; 2],
    /// Decode prefetch of the next layer's predicted experts done (copy stream).
    pf_ready: CudaEvent,
    stage_bytes: usize,
    stats: CacheStats,
    host: Option<HostExperts>,
    /// Page-locked staging for the per-layer routing upload.
    meta_host: PinnedHostSlice<u8>,
    s: Scratch,
    n_ctx: usize,
    max_batch: usize,
    /// Tokens per layer-major pass (`forward_span`).
    span: usize,
    n_past: usize,
    /// Owns every device allocation the raw pointers above point into.
    _bufs: Vec<CudaSlice<u8>>,
    /// Bytes allocated on the device.
    pub vram_bytes: usize,
}

const F: usize = std::mem::size_of::<f32>();
/// Decode prefetches the next layer's top-5 predicted experts: the low-ranked predictions
/// miss often enough that their bytes cost more PCIe time than they save (sweep on the 19K
/// gate: top-2 24.3, 4 25.3, 5 25.4, 6 25.3, 8 24.0 tok/s).
const PREFETCH_TOP: usize = 5;
const H: usize = 2; // f16

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
            mm_q4_k_t8: f("mm_q4_k_t8")?,
            mm_q6_k_t8: f("mm_q6_k_t8")?,
            mm_q4_k_ids_t8: f("mm_q4_k_ids_t8")?,
            mm_q6_k_ids_t8: f("mm_q6_k_ids_t8")?,
            gemm_q4_k: f("gemm_q4_k")?,
            gemm_q6_k: f("gemm_q6_k")?,
            gemm_q4_k_ids: f("gemm_q4_k_ids")?,
            gemm_q6_k_ids: f("gemm_q6_k_ids")?,
            rmsnorm: f("rmsnorm")?,
            qk_norm_rope: f("qk_norm_rope")?,
            attention: f("attention")?,
            attention_dec: f("attention_dec")?,
            attention_tc: f("attention_tc")?,
            attn_merge: f("attn_merge")?,
            store_f16: f("store_f16")?,
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
        let kv = alloc(hp.n_layer * 2 * n_ctx * kvd * H)?;
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
        let (b, span) = (n_ctx.min(512), n_ctx.min(SPAN));
        let s = Scratch {
            x: alloc(span * e * F)?,
            h: alloc(span * e * F)?,
            q: alloc(b * nq * F)?,
            attn: alloc(b * nq * F)?,
            kvs: alloc(2 * b * kvd * F)?,
            o: alloc(b * e * F)?,
            rl: alloc(b * hp.n_expert * F)?,
            xg: alloc(b * kk * e * F)?,
            gu: alloc(b * kk * 2 * hp.n_ff_exp * F)?,
            act: alloc(b * kk * hp.n_ff_exp * F)?,
            db: alloc(b * kk * e * F)?,
            logits: alloc(LOGIT_ROWS.min(b) * hp.n_vocab * F)?,
            meta: alloc(meta_bytes(span, b, kk, hp.n_expert))?,
        };
        let meta_host = unsafe { ctx.alloc_pinned_with_flags::<u8>(meta_bytes(span, b, kk, hp.n_expert), 0) }.map_err(err)?;
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
        let ev = || ctx.new_event(None).map_err(err);
        Ok(Self {
            copy: ctx.new_stream().map_err(err)?,
            ready: [ev()?, ev()?],
            free: [ev()?, ev()?],
            pf_ready: ev()?,
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
            span,
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

    /// Expert cache slots per layer group.
    pub fn cache_slots(&self) -> Vec<usize> {
        self.groups.iter().map(|g| g.cache.n_slots()).collect()
    }

    /// Queue a copy of expert `ex` of layer `li` (gate | up | down) to `dst`; returns its bytes.
    fn upload_expert(&self, li: usize, ex: usize, dst: Ptr, st: &CudaStream) -> usize {
        let parts = self.w.expert(&self.w.layers[li], ex);
        let n: usize = parts.iter().map(|b| b.len()).sum();
        match &self.host {
            Some(h) => {
                let src = &h.buf.as_slice().expect("pinned")[h.base[li] + ex * n..][..n];
                unsafe { result::memcpy_htod_async(dst, src, st.cu_stream()) }.expect("htod async");
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
        // Quantized kernels: one warp per row, MM_ROWS (= 4) rows per block; tiles of 8 tokens
        // (prefill) or one token (decode) per grid row.
        let (p, rb, k, n, ti) = (m.p, m.row_bytes as u64, m.k as i32, m.n_rows as i32, ntok as i32);
        if m.ty == GgmlType::F32 {
            return launch!(self, self.k.mm_f32, (m.n_rows as u32, ntok as u32, 1), 256, [p, rb, k, x, y, n]);
        }
        let (f, gx, gy) = match (m.ty, ntok) {
            (GgmlType::Q4_K, 1) => (&self.k.mm_q4_k, m.n_rows.div_ceil(4), 1),
            (GgmlType::Q6_K, 1) => (&self.k.mm_q6_k, m.n_rows.div_ceil(4), 1),
            (GgmlType::Q4_K, ..GEMM_MIN) => (&self.k.mm_q4_k_t8, m.n_rows.div_ceil(4), ntok.div_ceil(8)),
            (GgmlType::Q6_K, ..GEMM_MIN) => (&self.k.mm_q6_k_t8, m.n_rows.div_ceil(4), ntok.div_ceil(8)),
            (GgmlType::Q4_K, _) => (&self.k.gemm_q4_k, m.n_rows.div_ceil(128), ntok.div_ceil(32)),
            (GgmlType::Q6_K, _) => (&self.k.gemm_q6_k, m.n_rows.div_ceil(128), ntok.div_ceil(32)),
            (t, _) => unreachable!("checked at load: {t:?}"),
        };
        launch!(self, f, (gx as u32, gy as u32, 1), 128, [p, rb, k, x, y, n, ti]);
    }

    /// Grouped expert matvec: gathered row j reads the weights at wp[j] + off. The kernel is
    /// picked by rows per expert: tiles share decoded weights only within one expert (the
    /// GEMM re-decodes them per expert a tile straddles; the 8-token tile falls back to one
    /// token at a time), so decode's ~1 row per expert runs one row per grid row.
    #[allow(clippy::too_many_arguments)]
    fn mm_ids(&self, ty: GgmlType, wp: Ptr, off: u64, row_bytes: usize, k: usize, x: Ptr, y: Ptr, n_rows: usize, ntok: usize, per_expert: usize) {
        let (f, gx, gy) = match (ty, per_expert) {
            (GgmlType::Q4_K, ..4) => (&self.k.mm_q4_k_ids, n_rows.div_ceil(4), ntok),
            (GgmlType::Q6_K, ..4) => (&self.k.mm_q6_k_ids, n_rows.div_ceil(4), ntok),
            (GgmlType::Q4_K, ..GEMM_MIN) => (&self.k.mm_q4_k_ids_t8, n_rows.div_ceil(4), ntok.div_ceil(8)),
            (GgmlType::Q6_K, ..GEMM_MIN) => (&self.k.mm_q6_k_ids_t8, n_rows.div_ceil(4), ntok.div_ceil(8)),
            (GgmlType::Q4_K, _) => (&self.k.gemm_q4_k_ids, n_rows.div_ceil(128), ntok.div_ceil(32)),
            (GgmlType::Q6_K, _) => (&self.k.gemm_q6_k_ids, n_rows.div_ceil(128), ntok.div_ceil(32)),
            (t, _) => unreachable!("checked at load: {t:?}"),
        };
        let (rb, k, n, ti) = (row_bytes as u64, k as i32, n_rows as i32, ntok as i32);
        launch!(self, f, (gx as u32, gy as u32, 1), 128, [wp, off, rb, k, x, y, n, ti]);
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

    /// Run up to `span` tokens layer by layer: attention in ubatches of `max_batch`, then the
    /// MoE of the whole span in expert waves, so each expert is uploaded at most once per layer
    /// (a 4096-token prefill moves the experts once, not once per ubatch). Returns the logits of
    /// every token, or of the last only.
    fn forward_span(&mut self, tokens: &[u32], all_logits: bool) -> Vec<f32> {
        let hp = &self.w.hp;
        let (e, hd, nh, nkv, ff, kk) = (hp.n_embd, hp.head_dim, hp.n_head, hp.n_head_kv, hp.n_ff_exp, hp.n_expert_used);
        let (kvd, t, b) = (nkv * hd, tokens.len(), self.max_batch);
        let s = self.s;
        let nv = hp.n_vocab;
        let n_expert = hp.n_expert;
        let chunks: Vec<(usize, usize)> = (0..t).step_by(b).map(|c0| (c0, b.min(t - c0))).collect();
        let at = |p: Ptr, c0: usize, width: usize| p + (c0 * width * F) as u64;

        let mut xe = vec![0f32; t * e];
        for (row, &tok) in xe.chunks_exact_mut(e).zip(tokens) {
            dequant_row(self.w.tok_embd.ty, self.w.g.row(&self.w.tok_embd, tok as usize), row);
        }
        self.htod(s.x, &xe);

        let (ei, eps, base) = (e as i32, hp.eps, hp.rope_base);
        let (nhi, nkvi, hdi) = (nh as i32, nkv as i32, hd as i32);
        let scale = 1.0 / (hd as f32).sqrt();
        for (li, l) in self.layers.iter().enumerate() {
            let kc = self.kv + (li * 2 * self.n_ctx * kvd * H) as u64;
            let vc = kc + (self.n_ctx * kvd * H) as u64;
            // --- attention, per ubatch; leaves each ubatch's normed FFN input in s.h ---
            let mut sels = vec![];
            let mut pred: Vec<usize> = vec![];
            for &(c0, tc) in &chunks {
                let (x, h, q, attn, o) = (at(s.x, c0, e), at(s.h, c0, e), s.q, s.attn, s.o);
                let pos = self.n_past + c0;
                let (pos0, tn, ti) = (pos as i32, tc as u32, tc as i32);
                let (kd, vd) = (kc + (pos * kvd * H) as u64, vc + (pos * kvd * H) as u64);
                let (ks, vs) = (s.kvs, s.kvs + (tc * kvd * F) as u64);
                let wn = l.attn_norm;
                launch!(self, self.k.rmsnorm, (tn, 1, 1), 256, [h, x, wn, ei, eps]);
                self.mm(&l.wq, h, q, tc);
                self.mm(&l.wk, h, ks, tc);
                self.mm(&l.wv, h, vs, tc);
                let (qn, kn) = (l.q_norm, l.k_norm);
                launch!(self, self.k.qk_norm_rope, (tn * nh as u32, 1, 1), hd as u32, [q, qn, nhi, hdi, pos0, eps, base]);
                launch!(self, self.k.qk_norm_rope, (tn * nkv as u32, 1, 1), hd as u32, [ks, kn, nkvi, hdi, pos0, eps, base]);
                let nkv_t = (tc * kvd) as i32;
                let g = ((tc * kvd).div_ceil(256) as u32, 1, 1);
                launch!(self, self.k.store_f16, g, 256, [kd, ks, nkv_t]);
                launch!(self, self.k.store_f16, g, 256, [vd, vs, nkv_t]);
                let gq = nh / nkv;
                if hd == 128 && 8 % gq == 0 {
                    // Decode (<= 4 tokens): attention_dec, one row per warp. Prefill: tensor-core
                    // attention_tc. Both split the positions when blocks are few, partials in s.gu
                    // (idle during attention).
                    let small = tc <= 4;
                    let cap = self.max_batch * hp.n_expert_used * 2 * hp.n_ff_exp / (tc * nh * 130);
                    let (f, gy, bs) = if small { (&self.k.attention_dec, (tc * gq).div_ceil(8), 256) }
                        else { (&self.k.attention_tc, (tc * gq).div_ceil(64), 128) };
                    // Fill the GPU: split positions until ~240 blocks (decode: every ATT_SPLIT).
                    let want = if small { usize::MAX } else { (240 / (nkv * gy)).max(1) };
                    let sk = (pos + tc).div_ceil(if small { ATT_SPLIT } else { 512 }).min(want).min(cap).max(1);
                    let (part, ski) = (s.gu, sk as i32);
                    launch!(self, *f, (nkv as u32, gy as u32, sk as u32), bs, [q, kc, vc, attn, nhi, nkvi, pos0, ti, scale, part]);
                    if sk > 1 {
                        let rows = (tc * nh) as i32;
                        launch!(self, self.k.attn_merge, ((tc * nh) as u32, 1, 1), 128, [part, attn, rows, ski]);
                    }
                } else {
                    launch!(self, self.k.attention, (nh as u32, tn, 1), 32 * 8, [q, kc, vc, attn, nhi, nkvi, hdi, pos0, scale]);
                }
                self.mm(&l.wo, attn, o, tc);
                let n = (tc * e) as i32;
                launch!(self, self.k.add_inplace, ((tc * e).div_ceil(256) as u32, 1, 1), 256, [x, o, n]);
                let wn = l.ffn_norm;
                launch!(self, self.k.rmsnorm, (tn, 1, 1), 256, [h, x, wn, ei, eps]);
                self.mm(&l.router, h, s.rl, tc);
                // Decode: also predict layer li+1's experts, its router on this post-attention x
                // (top-8 recall ~84%). Read back with the router logits; prefetched below.
                let predict = t == 1 && li + 1 < self.layers.len();
                if predict {
                    let (o, wn, rl2) = (s.o, self.layers[li + 1].ffn_norm, s.rl + (n_expert * F) as u64);
                    launch!(self, self.k.rmsnorm, (1, 1, 1), 256, [o, x, wn, ei, eps]);
                    self.mm(&self.layers[li + 1].router, o, rl2, 1);
                }
                let mut r = route(&self.dtoh(s.rl, (tc + predict as usize) * n_expert), n_expert, kk);
                if predict {
                    pred = r.pop().unwrap().iter().take(PREFETCH_TOP).map(|p| p.0).collect();
                }
                sels.push(r);
            }

            // --- MoE: experts in waves; each wave runs over every ubatch ---
            let mut by_expert: Vec<Vec<(usize, f32)>> = vec![vec![]; n_expert]; // (span token, weight)
            for (ti, ts) in sels.iter().flatten().enumerate() {
                for &(ex, wgt) in ts {
                    by_expert[ex].push((ti, wgt));
                }
            }
            let used: Vec<usize> = (0..n_expert).filter(|&ex| !by_expert[ex].is_empty()).collect();
            let gi = self.group_of[li];
            let (bank, sb) = (self.groups[gi].bank, self.groups[gi].slot_bytes);
            // ptr[ex]: where expert ex's (gate | up | down) will be when its wave runs.
            let mut ptr = vec![0u64; n_expert];
            // Cached ubatches go through the cache policy and run as one wave; larger spans
            // bypass it (as in the oracle): resident experts are read in place (wave 0), misses
            // are streamed through the staging halves in waves of HALF, alternating halves.
            let cached = t <= CACHE_MAX_BATCH;
            const HALF: usize = STAGE / 2;
            let mut half: Vec<Option<usize>> = vec![None]; // staging half each wave uses
            let waves: Vec<Vec<usize>> = if cached {
                let ids: Vec<u32> = sels[0].iter().flatten().map(|&(e, _)| e as u32).collect();
                self.stream.wait(&self.pf_ready).expect("wait"); // the prefetch's slots are written
                let p = self.groups[gi].cache.plan(li, &ids, t);
                for &(ex, slot) in &p.fills {
                    self.stats.bytes += self.upload_expert(li, ex, bank + (slot * sb) as u64, &self.stream) as u64;
                }
                // Prefetch li+1's predicted misses on the copy stream, overlapping this layer's
                // MoE and the next attention. Victims are never this layer's experts (still to be
                // read); earlier layers are finished (the router readback synchronized).
                if !pred.is_empty() {
                    let g2 = self.group_of[li + 1];
                    let (bank2, sb2) = (self.groups[g2].bank, self.groups[g2].slot_bytes);
                    for (ex, slot) in self.groups[g2].cache.prefetch(li + 1, &pred, li) {
                        self.stats.bytes += self.upload_expert(li + 1, ex, bank2 + (slot * sb2) as u64, &self.copy) as u64;
                    }
                    self.pf_ready.record(&self.copy).expect("record");
                }
                self.stats.hits += p.hits as u64;
                self.stats.misses += p.fills.len() as u64;
                for &ex in &used {
                    ptr[ex] = bank + (self.groups[gi].cache.slot(li, ex).expect("planned expert resident") * sb) as u64;
                }
                vec![used]
            } else {
                let (res, miss): (Vec<usize>, Vec<usize>) = used.iter().partition(|&&ex| self.groups[gi].cache.slot(li, ex).is_some());
                for &ex in &res {
                    ptr[ex] = bank + (self.groups[gi].cache.slot(li, ex).unwrap() * sb) as u64;
                }
                let mut waves = vec![res];
                for (k, wave) in miss.chunks(HALF).enumerate() {
                    for (j, &ex) in wave.iter().enumerate() {
                        ptr[ex] = self.stage + (((k % 2) * HALF + j) * self.stage_bytes) as u64;
                    }
                    half.push(Some(k % 2));
                    waves.push(wave.to_vec());
                }
                waves
            };
            // Routing for every pass, one async copy per layer: a pass is up to `b * kk` gathered
            // rows of one wave, from the whole span (so each expert sees all its span tokens).
            // The pinned buffer is free: the router dtoh above waited for the previous layer's copy.
            let mut passes = vec![]; // (wave, rows, offset of this pass's routing, rows per expert)
            let mut blob: Vec<u8> = vec![];
            for (wi, wave) in waves.iter().enumerate() {
                // (span token, expert, weight), grouped by expert
                let rows: Vec<(usize, usize, f32)> = wave
                    .iter()
                    .flat_map(|&ex| by_expert[ex].iter().map(move |r| (r.0, ex, r.1)))
                    .collect();
                for rows in rows.chunks(b * kk) {
                    let mut off = vec![0i32; t + 1];
                    for r in rows {
                        off[r.0 + 1] += 1;
                    }
                    for i in 0..t {
                        off[i + 1] += off[i];
                    }
                    let (mut crow, mut cw, mut fill) = (vec![0i32; rows.len()], vec![0f32; rows.len()], off.clone());
                    for (j, r) in rows.iter().enumerate() {
                        let k = fill[r.0] as usize;
                        (crow[k], cw[k]) = (j as i32, r.2);
                        fill[r.0] += 1;
                    }
                    let n_ex = 1 + rows.windows(2).filter(|w| w[0].1 != w[1].1).count();
                    passes.push((wi, rows.len(), blob.len(), rows.len() / n_ex));
                    blob.extend(rows.iter().flat_map(|r| (r.0 as i32).to_le_bytes()));
                    blob.extend(off.iter().flat_map(|v| v.to_le_bytes()));
                    blob.extend(crow.iter().flat_map(|v| v.to_le_bytes()));
                    blob.extend(cw.iter().flat_map(|v| v.to_le_bytes()));
                    blob.resize(blob.len().next_multiple_of(8), 0);
                    blob.extend(rows.iter().flat_map(|r| ptr[r.1].to_le_bytes()));
                }
            }
            {
                let m = self.meta_host.as_mut_slice().expect("pinned");
                m[..blob.len()].copy_from_slice(&blob);
                unsafe { result::memcpy_htod_async(s.meta, &m[..blob.len()], self.stream.cu_stream()) }.expect("htod async");
            }
            let wl = &self.w.layers[li];
            let down_off = (self.w.expert(wl, 0)[0].len() * 2) as u64;
            let (xg, gu, act, dbuf) = (s.xg, s.gu, s.act, s.db);
            // Copy stream: wave k's misses into its half once the half's last reader is done.
            let upload = |k: usize| {
                if let Some(Some(hf)) = half.get(k) {
                    self.copy.wait(&self.free[*hf]).expect("wait");
                    for (j, &ex) in waves[k].iter().enumerate() {
                        self.upload_expert(li, ex, self.stage + ((hf * HALF + j) * self.stage_bytes) as u64, &self.copy);
                    }
                    self.ready[*hf].record(&self.copy).expect("record");
                }
            };
            upload(1);
            upload(2);
            for (wi, _) in waves.iter().enumerate() {
                if let Some(hf) = half[wi] {
                    self.stream.wait(&self.ready[hf]).expect("wait");
                }
                for &(_, r, o, per_ex) in passes.iter().filter(|p| p.0 == wi) {
                    let (x, h) = (s.x, s.h);
                    let gidx = s.meta + o as u64;
                    let off = gidx + (4 * r) as u64;
                    let (crow, cw) = (off + (4 * (t + 1)) as u64, off + (4 * (t + 1) + 4 * r) as u64);
                    let wp = s.meta + (o + 4 * r + 4 * (t + 1) + 8 * r).next_multiple_of(8) as u64;
                    launch!(self, self.k.gather_rows, (r as u32, 1, 1), 256, [xg, h, gidx, ei]);
                    self.mm_ids(wl.gate.ty, wp, 0, wl.gate.row_bytes(), e, xg, gu, 2 * ff, r, per_ex);
                    let (ffi, ri) = (ff as i32, r as i32);
                    launch!(self, self.k.swiglu, ((r * ff).div_ceil(256) as u32, 1, 1), 256, [act, gu, ffi, ri]);
                    self.mm_ids(wl.down.ty, wp, down_off, wl.down.row_bytes(), ff, act, dbuf, e, r, per_ex);
                    launch!(self, self.k.moe_combine, (t as u32, 1, 1), 256, [x, dbuf, off, crow, cw, ei]);
                }
                if let Some(hf) = half[wi] {
                    self.free[hf].record(&self.stream).expect("record");
                    upload(wi + 2);
                }
            }
        }
        self.n_past += t;

        let rows: Vec<(usize, usize)> =
            if all_logits { (0..t).step_by(LOGIT_ROWS).map(|r0| (r0, LOGIT_ROWS.min(t - r0))).collect() } else { vec![(t - 1, 1)] };
        let mut out = Vec::with_capacity(rows.iter().map(|r| r.1).sum::<usize>() * nv);
        for (r0, nr) in rows {
            let (x, h, wn, nrn) = (at(s.x, r0, e), s.h, self.out_norm, nr as u32);
            launch!(self, self.k.rmsnorm, (nrn, 1, 1), 256, [h, x, wn, ei, eps]);
            self.mm(&self.output, h, s.logits, nr);
            out.extend(self.dtoh(s.logits, nr * nv));
        }
        out
    }
}

/// Bytes of one layer's routing for a span of `t` tokens in ubatches of `b` (see
/// `forward_span`): per pass, gidx | off | crow | cw (4 bytes each; off has t + 1 entries),
/// padding to 8, then wp (8 bytes per row). Passes: at most one partial per wave plus
/// t * kk / (b * kk) full ones.
fn meta_bytes(t: usize, b: usize, kk: usize, n_expert: usize) -> usize {
    let passes = 1 + n_expert.div_ceil(STAGE / 2) + t.div_ceil(b);
    t * kk * 20 + passes * (4 * (t + 1) + 8)
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
        tokens.chunks(self.span).flat_map(|c| self.forward_span(c, true)).collect()
    }

    fn forward_last(&mut self, tokens: &[u32]) -> Vec<f32> {
        self.check_ctx(tokens.len());
        let n = tokens.len().div_ceil(self.span);
        let mut last = vec![];
        for (i, c) in tokens.chunks(self.span).enumerate() {
            last = self.forward_span(c, false);
            if i + 1 < n {
                last.clear();
            }
        }
        last
    }
}
