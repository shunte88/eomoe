//! CUDA engine for qwen3moe. Same numerics as the CPU reference (f32 activations and KV
//! cache, weights dequantized on the fly), with the dense weights resident in VRAM and the
//! router-selected experts of each layer copied from host RAM into staging slots.
//!
//! Kernel seam: kernels are loaded by name from an embedded fatbin built from
//! kernels/eomoe.cu (build.rs); an all-Rust module exporting the same names can replace it.

use crate::gguf::{GgmlType, Gguf, TensorInfo};
use crate::quant::dequant_row;
use crate::qwen3moe::{route, Engine, Weights};
use cudarc::driver::{result, sys::CUdeviceptr, CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/eomoe.fatbin"));

type Ptr = CUdeviceptr;

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
    xg: Ptr,
    gb: Ptr,
    ub: Ptr,
    db: Ptr,
    logits: Ptr,
    gidx: Ptr,
    trow: Ptr,
    tw: Ptr,
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
    /// One staging slot per expert of a layer: gate | up | down bytes.
    slots: Ptr,
    slot_bytes: usize,
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
    /// Load the dense weights to GPU 0 and allocate a KV cache for `n_ctx` positions.
    pub fn new(g: Gguf, n_ctx: usize) -> Result<Self, String> {
        let w = Weights::load(g)?;
        let ctx = CudaContext::new(0).map_err(err)?;
        let stream = ctx.default_stream();
        let m = ctx.load_module(Ptx::from_binary(FATBIN.to_vec())).map_err(err)?;
        let f = |n: &str| m.load_function(n).map_err(err);
        let k = Kernels {
            mm_f32: f("mm_f32")?,
            mm_q4_k: f("mm_q4_k")?,
            mm_q6_k: f("mm_q6_k")?,
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
                check_ty(t)?;
            }
        }
        let out_norm = upload(&w.out_norm)?;
        let output = mat(&w.output, upload(&w.output)?)?;
        let dense = vram.get();

        let hp = &w.hp;
        let (e, kvd, nq, kk) = (hp.n_embd, hp.n_head_kv * hp.head_dim, hp.n_head * hp.head_dim, hp.n_expert_used);
        let kv = alloc(hp.n_layer * 2 * n_ctx * kvd * F)?;
        let slot_bytes = w.layers.iter().map(|l| w.expert(l, 0).iter().map(|b| b.len()).sum::<usize>()).max().unwrap_or(0);
        let slots = alloc(hp.n_expert * slot_bytes)?;
        let b = n_ctx.min(512);
        let s = Scratch {
            x: alloc(b * e * F)?,
            h: alloc(b * e * F)?,
            q: alloc(b * nq * F)?,
            attn: alloc(b * nq * F)?,
            o: alloc(b * e * F)?,
            rl: alloc(b * hp.n_expert * F)?,
            xg: alloc(b * kk * e * F)?,
            gb: alloc(b * kk * hp.n_ff_exp * F)?,
            ub: alloc(b * kk * hp.n_ff_exp * F)?,
            db: alloc(b * kk * e * F)?,
            logits: alloc(b * hp.n_vocab * F)?,
            gidx: alloc(b * kk * 4)?,
            trow: alloc(b * kk * 4)?,
            tw: alloc(b * kk * F)?,
        };
        log::info!(
            "gpu: dense weights {} MiB, expert slots {} MiB, total {} MiB (n_ctx {n_ctx}, batch {b})",
            dense >> 20,
            (hp.n_expert * slot_bytes) >> 20,
            vram.get() >> 20
        );
        Ok(Self {
            stream,
            k,
            layers,
            out_norm,
            output,
            kv,
            slots,
            slot_bytes,
            s,
            n_ctx,
            max_batch: b,
            n_past: 0,
            _bufs: bufs,
            vram_bytes: vram.get(),
            w,
        })
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

    fn htod<T>(&self, dst: Ptr, src: &[T]) {
        unsafe { result::memcpy_htod_sync(dst, src) }.expect("htod");
    }

    fn dtoh(&self, src: Ptr, n: usize) -> Vec<f32> {
        let mut v = vec![0f32; n];
        self.stream.synchronize().expect("sync");
        unsafe { result::memcpy_dtoh_sync(&mut v, src) }.expect("dtoh");
        v
    }

    fn forward_chunk(&mut self, tokens: &[u32]) -> Vec<f32> {
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
            launch!(self, self.k.attention, (nh as u32, tn, 1), hd as u32, [q, kc, vc, attn, nhi, nkvi, hdi, pos0, scale]);
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
            let wl = &self.w.layers[li];
            for (ex, toks) in by_expert.iter().enumerate().filter(|(_, v)| !v.is_empty()) {
                let slot = self.slots + (segs.len() * self.slot_bytes) as u64;
                let [gb, ub, db] = self.w.expert(wl, ex);
                self.htod(slot, gb);
                self.htod(slot + gb.len() as u64, ub);
                self.htod(slot + (gb.len() + ub.len()) as u64, db);
                segs.push((slot, gb.len() as u64, (gb.len() + ub.len()) as u64, gidx.len(), toks.len()));
                for &(ti, wgt) in toks {
                    trow[ti].push(gidx.len() as i32);
                    tw[ti].push(wgt);
                    gidx.push(ti as i32);
                }
            }
            self.htod(s.gidx, &gidx);
            self.htod(s.trow, &trow.concat());
            self.htod(s.tw, &tw.concat());
            let (xg, gbuf, ubuf, dbuf, gi, tr, twp) = (s.xg, s.gb, s.ub, s.db, s.gidx, s.trow, s.tw);
            launch!(self, self.k.gather_rows, (gidx.len() as u32, 1, 1), 256, [xg, h, gi, ei]);
            let mat = |p: Ptr, ti: &TensorInfo, k: usize, n_rows: usize| Mat { p, ty: ti.ty, row_bytes: ti.row_bytes(), k, n_rows };
            for &(slot, uo, _, r0, nr) in &segs {
                let xin = xg + (r0 * e * F) as u64;
                self.mm(&mat(slot, &wl.gate, e, ff), xin, gbuf + (r0 * ff * F) as u64, nr);
                self.mm(&mat(slot + uo, &wl.up, e, ff), xin, ubuf + (r0 * ff * F) as u64, nr);
            }
            let n = (gidx.len() * ff) as i32;
            launch!(self, self.k.swiglu, ((gidx.len() * ff).div_ceil(256) as u32, 1, 1), 256, [gbuf, ubuf, n]);
            for &(slot, _, dn, r0, nr) in &segs {
                self.mm(&mat(slot + dn, &wl.down, ff, e), gbuf + (r0 * ff * F) as u64, dbuf + (r0 * e * F) as u64, nr);
            }
            let ki = kk as i32;
            launch!(self, self.k.moe_combine, (tn, 1, 1), 256, [x, dbuf, tr, twp, ki, ei]);
        }
        self.n_past += t;

        let (x, h, wn) = (s.x, s.h, self.out_norm);
        launch!(self, self.k.rmsnorm, (tn, 1, 1), 256, [h, x, wn, ei, eps]);
        self.mm(&self.output, h, s.logits, t);
        self.dtoh(s.logits, t * nv)
    }
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

    fn reset(&mut self) {
        self.n_past = 0;
    }

    /// Panics if the KV cache would overflow n_ctx.
    fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        assert!(self.n_past + tokens.len() <= self.n_ctx, "context overflow: {} + {} > {}", self.n_past, tokens.len(), self.n_ctx);
        tokens.chunks(self.max_batch).flat_map(|c| self.forward_chunk(c)).collect()
    }
}
