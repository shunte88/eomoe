//! CUDA engine for qwen3moe, gpt-oss (biases, sinks, sliding window, YaRN; `Hparams::oai`) and
//! GLM-4.7-Flash (absorbed MLA over a shared latent cache, dense / shared-expert FFNs) and Qwen3.6
//! (qwen35moe: Gated DeltaNet layers with per-sequence recurrent state, gated attention). Same numerics as the CPU reference (f32 activations and KV
//! cache, weights dequantized on the fly), with the dense weights resident in VRAM and the
//! experts in host RAM, cached in VRAM slots by `ExpertCache` (ubatches up to 32 tokens; larger
//! ones read the cached experts and stream the rest through staging slots, as llama.cpp).
//!
//! Kernel seam: kernels are loaded by name from an embedded fatbin built from
//! kernels/eomoe.cu (build.rs); an all-Rust module exporting the same names can replace it.

use crate::expert_cache::ExpertCache;
use crate::gguf::{GgmlType, Gguf, TensorInfo};
use crate::quant::dequant_row;
use crate::qwen3moe::{Attn, Engine, Hparams, Weights};
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
/// Tokens per tensor-core GEMM tile (GB_T in kernels/eomoe.cu).
const GEMM_T: usize = 128;
/// Tokens per expert (_ids) GEMM tile (GB_TI).
const GEMM_TI: usize = 64;
/// Tokens per layer-major prefill pass: the experts move once per SPAN tokens. Costs
/// 2 x SPAN x n_embd x 4 bytes of VRAM (64 MiB) for the hidden states.
const SPAN: usize = 4096;
/// Positions per split-K block of decode attention.
const ATT_SPLIT: usize = 128;
/// Logit rows per output matmul (all-logits calls): 64 x n_vocab x 4 bytes = 39 MiB.
const LOGIT_ROWS: usize = 64;
/// KV positions per page (kernels/eomoe.cu KV_PAGE). Sequences take pages from one shared pool.
const KV_PAGE: usize = 256;
/// Hybrid: the fraction of a layer's cache misses uploaded and run on the GPU; the rest run on
/// the CPU meanwhile. ~0.2 ms to upload an expert vs ~0.15 ms on the CPU (4 decode rows).
const ADMIT_NUM: usize = 1;
const ADMIT_DEN: usize = 5;
/// Sequences the page tables are sized for (`set_n_seq`).
pub const MAX_SEQ: usize = 16;
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
    mm_q8_0: CudaFunction,
    mm_q8_0_t8: CudaFunction,
    mm_q8_0_ids: CudaFunction,
    mm_q8_0_ids_t8: CudaFunction,
    mm_mxfp4: CudaFunction,
    mm_mxfp4_t8: CudaFunction,
    mm_mxfp4_ids: CudaFunction,
    mm_mxfp4_ids_t8: CudaFunction,
    rmsnorm: CudaFunction,
    qk_norm_rope: CudaFunction,
    attention: CudaFunction,
    attention_dec: CudaFunction,
    attention_tc: CudaFunction,
    attn_merge: CudaFunction,
    attention_mla: CudaFunction,
    rmsnorm_strided: CudaFunction,
    rope_norm: CudaFunction,
    mm_heads_q4_k: CudaFunction,
    mm_heads_q6_k: CudaFunction,
    mm_heads_q8_0: CudaFunction,
    store_f16: CudaFunction,
    add_inplace: CudaFunction,
    gather_rows: CudaFunction,
    swiglu: CudaFunction,
    swiglu_oai: CudaFunction,
    add_bias: CudaFunction,
    bias_ids: CudaFunction,
    moe_combine: CudaFunction,
    ssm_conv: CudaFunction,
    l2norm_heads: CudaFunction,
    gated_delta: CudaFunction,
    gated_rmsnorm: CudaFunction,
    split_qgate: CudaFunction,
    mul_sigmoid: CudaFunction,
    sigmoid_dot_scale: CudaFunction,
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

enum DAttn {
    Gqa { wq: Mat, wk: Mat, wv: Mat },
    /// k_b and v_b are per-head stacks of lat (k_b) / vd (v_b) rows.
    Mla { q_a: Mat, q_a_norm: Ptr, q_b: Mat, kv_a: Mat, kv_a_norm: Ptr, k_b: Mat, v_b: Mat },
    /// qwen35 DeltaNet: one matrix [qkv | z | beta | alpha] per token, dt_bias, ssm_a, conv, ssm_norm.
    Delta { w: Mat, dt: Ptr, a: Ptr, conv: Ptr, norm: Ptr },
}

struct DLayer {
    attn_norm: Ptr,
    attn: DAttn,
    wo: Mat,
    /// 0: none (gpt-oss).
    q_norm: Ptr,
    k_norm: Ptr,
    ffn_norm: Ptr,
    /// None: a dense layer (GLM layer 0).
    router: Option<Mat>,
    /// Dense FFN or shared expert: [gate | up] (2 x ff rows), down.
    ffn: Option<(Mat, Mat)>,
    /// Biases of q, k, v, o, router (0: none), attention sinks (0: none), sliding window (0: none).
    bias: [Ptr; 5],
    sinks: Ptr,
    window: i32,
    /// qwen35: the shared expert's sigmoid gate (0: none).
    shared_gate: Ptr,
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
    /// qwen35: the DeltaNet projections ([qkv | z | beta | alpha] rows) / gated attention's q|gate.
    dn: Ptr,
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
    /// YaRN rope frequencies (0: base^(-2i/hd)) and cos/sin scale.
    rope_freq: Ptr,
    rope_mscale: f32,
    /// Bytes from an expert slot's start to its gate | up bias, then down bias (gpt-oss).
    bias_off: usize,
    /// Per layer: K then V, each n_ctx x (n_head_kv * head_dim) f16, in KV_PAGE pages that the
    /// sequences' page tables map their positions to.
    kv: Ptr,
    /// Per layer: its index among the layers of its kind (attention: KV pool, DeltaNet: state).
    slot: Vec<usize>,
    /// Layers with a KV cache.
    n_kv_layers: usize,
    /// qwen35: per sequence, every DeltaNet layer's conv then ssm state (`rs_layer` floats each).
    rs: Vec<Ptr>,
    rs_layer: usize,
    /// Page tables, MAX_SEQ rows of n_ctx / KV_PAGE i32 (device copy of `seqs[i].pages`).
    pt: Ptr,
    seqs: Vec<SeqKv>,
    free_pages: Vec<i32>,
    /// The sequence `forward` and `truncate` act on.
    cur: usize,
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
    /// Hybrid decode: experts that miss the cache run on the CPU (`cpu_moe`) while the GPU runs
    /// the resident ones and a share of the misses (ADMIT_NUM/ADMIT_DEN), uploaded to become
    /// resident. Off: misses are uploaded and run on the GPU, batch-invariant.
    hybrid: bool,
    /// Page-locked staging for the CPU experts' output rows.
    cpu_out: std::cell::RefCell<PinnedHostSlice<f32>>,
    s: Scratch,
    n_ctx: usize,
    max_batch: usize,
    /// Tokens per layer-major pass (`forward_span`).
    span: usize,
    /// Floats in s.gu (attention split partials live there).
    gu_len: usize,
    /// Owns every device allocation the raw pointers above point into.
    _bufs: Vec<CudaSlice<u8>>,
    /// Bytes allocated on the device.
    pub vram_bytes: usize,
}

/// One sequence's share of the KV pool.
#[derive(Default)]
struct SeqKv {
    n_past: usize,
    pages: Vec<i32>,
}

const F: usize = std::mem::size_of::<f32>();
/// Decode prefetches the next layer's top-5 predicted experts: the low-ranked predictions
/// miss often enough that their bytes cost more PCIe time than they save (sweep on the 19K
/// gate: top-2 24.3, 4 25.3, 5 25.4, 6 25.3, 8 24.0 tok/s).
const PREFETCH_TOP: usize = 5;
const H: usize = 2; // f16

/// qwen35: floats per token of the DeltaNet projection [qkv | z | beta | alpha].
fn dn_width(q: &crate::qwen3moe::Qwen35) -> usize {
    2 * q.nk * q.dk + 2 * q.nv * q.dk + 2 * q.nv
}

/// KV arrays per layer: K and V, or MLA's one shared latent row per position.
fn kv_arrays(hp: &Hparams) -> usize {
    if hp.glm.is_some() { 1 } else { 2 }
}

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
            mm_q8_0: f("mm_q8_0")?,
            mm_q8_0_t8: f("mm_q8_0_t8")?,
            mm_q8_0_ids: f("mm_q8_0_ids")?,
            mm_q8_0_ids_t8: f("mm_q8_0_ids_t8")?,
            mm_mxfp4: f("mm_mxfp4")?,
            mm_mxfp4_t8: f("mm_mxfp4_t8")?,
            mm_mxfp4_ids: f("mm_mxfp4_ids")?,
            mm_mxfp4_ids_t8: f("mm_mxfp4_ids_t8")?,
            rmsnorm: f("rmsnorm")?,
            qk_norm_rope: f("qk_norm_rope")?,
            attention: f("attention")?,
            attention_dec: f("attention_dec")?,
            attention_tc: f("attention_tc")?,
            attn_merge: f("attn_merge")?,
            attention_mla: f("attention_mla")?,
            rmsnorm_strided: f("rmsnorm_strided")?,
            rope_norm: f("rope_norm")?,
            mm_heads_q4_k: f("mm_heads_q4_k")?,
            mm_heads_q6_k: f("mm_heads_q6_k")?,
            mm_heads_q8_0: f("mm_heads_q8_0")?,
            store_f16: f("store_f16")?,
            add_inplace: f("add_inplace")?,
            gather_rows: f("gather_rows")?,
            swiglu: f("swiglu")?,
            swiglu_oai: f("swiglu_oai")?,
            add_bias: f("add_bias")?,
            bias_ids: f("bias_ids")?,
            moe_combine: f("moe_combine")?,
            ssm_conv: f("ssm_conv")?,
            l2norm_heads: f("l2norm_heads")?,
            gated_delta: f("gated_delta")?,
            gated_rmsnorm: f("gated_rmsnorm")?,
            split_qgate: f("split_qgate")?,
            mul_sigmoid: f("mul_sigmoid")?,
            sigmoid_dot_scale: f("sigmoid_dot_scale")?,
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
        // Upload tensors back to back as one matrix (rows of the same type and length).
        let mut up = |ts: &[&TensorInfo]| -> Result<Mat, String> {
            let t = ts[0];
            check_ty(t)?;
            if ts.iter().any(|u| u.ty != t.ty || u.dims[0] != t.dims[0]) {
                return Err(format!("{}: concatenated tensors differ in type or row length", t.name));
            }
            let n: usize = ts.iter().map(|u| u.size).sum();
            let p = alloc(n)?;
            let mut o = 0;
            for u in ts {
                unsafe { result::memcpy_htod_sync(p + o as u64, w.g.data(u)) }.map_err(err)?;
                o += u.size;
            }
            Ok(Mat { p, ty: t.ty, row_bytes: t.row_bytes(), k: t.dims[0] as usize, n_rows: n / t.row_bytes() })
        };
        let mut layers = vec![];
        for (li, l) in w.layers.iter().enumerate() {
            let attn = match &l.attn {
                Attn::Gqa { wq, wk, wv } => DAttn::Gqa { wq: up(&[wq])?, wk: up(&[wk])?, wv: up(&[wv])? },
                Attn::Mla { q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b } => {
                    if let Some(t) = [k_b, v_b].into_iter().find(|t| !matches!(t.ty, GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::Q8_0)) {
                        return Err(format!("{}: {:?} has no per-head CUDA kernel yet", t.name, t.ty));
                    }
                    DAttn::Mla { q_a: up(&[q_a])?, q_a_norm: up(&[q_a_norm])?.p, q_b: up(&[q_b])?, kv_a: up(&[kv_a])?, kv_a_norm: up(&[kv_a_norm])?.p, k_b: up(&[k_b])?, v_b: up(&[v_b])? }
                }
                Attn::Delta { qkv, z, beta, alpha, dt_bias, a, conv, norm } => {
                    DAttn::Delta { w: up(&[qkv, z, beta, alpha])?, dt: up(&[dt_bias])?.p, a: up(&[a])?.p, conv: up(&[conv])?.p, norm: up(&[norm])?.p }
                }
            };
            let ffn = match (&l.dense, l.moe.as_ref().and_then(|m| m.shared.as_ref())) {
                (Some([g, u, d]), _) | (None, Some([g, u, d])) => Some((up(&[g, u])?, up(&[d])?)),
                _ => None,
            };
            let router = l.moe.as_ref().map(|m| up(&[&m.router])).transpose()?;
            let mut opt = |t: Option<&TensorInfo>| t.map_or(Ok(0), |t| up(&[t]).map(|m| m.p));
            let bias = [0, 1, 2, 3, 4].map(|i| l.bias.as_ref().map(|b| &b[i]));
            layers.push(DLayer {
                q_norm: opt(l.q_norm.as_ref())?,
                k_norm: opt(l.k_norm.as_ref())?,
                bias: [opt(bias[0])?, opt(bias[1])?, opt(bias[2])?, opt(bias[3])?, opt(bias[4])?],
                sinks: opt(l.sinks.as_ref())?,
                window: if li % 2 == 0 { w.hp.swa as i32 } else { 0 },
                shared_gate: opt(l.moe.as_ref().and_then(|m| m.shared_gate.as_ref()))?,
                attn_norm: up(&[&l.attn_norm])?.p,
                attn,
                wo: up(&[&l.wo])?,
                ffn_norm: up(&[&l.ffn_norm])?.p,
                router,
                ffn,
            });
            if let Some(m) = &l.moe {
                for t in [&m.gate, &m.up, &m.down] {
                    if !matches!(t.ty, GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::Q8_0 | GgmlType::MXFP4) {
                        return Err(format!("{}: {:?} experts have no CUDA kernel yet", t.name, t.ty));
                    }
                }
                // gate and up are read as one [gate | up] matrix of each slot.
                if m.gate.ty != m.up.ty || m.gate.dims != m.up.dims {
                    return Err(format!("{}: gate and up experts differ in type or shape", m.gate.name));
                }
            }
        }
        let out_norm = up(&[&w.out_norm])?.p;
        let output = up(&[&w.output])?;
        let rope_freq = match &w.hp.rope_yarn {
            Some((f, _)) => {
                let p = alloc(f.len() * F)?;
                unsafe { result::memcpy_htod_sync(p, &f[..]) }.map_err(err)?;
                p
            }
            None => 0,
        };
        let dense = vram.get();

        let hp = &w.hp;
        if hp.glm.is_none() && (hp.head_dim % 32 != 0 || hp.head_dim > 256) {
            return Err(format!("head_dim {} unsupported by the attention kernel", hp.head_dim));
        }
        let (e, kvd, nq, kk) = (hp.n_embd, hp.n_head_kv * hp.head_dim, hp.n_head * hp.head_dim, hp.n_expert_used);
        let n_ctx = n_ctx.next_multiple_of(KV_PAGE);
        let is_delta: Vec<bool> = layers.iter().map(|l| matches!(l.attn, DAttn::Delta { .. })).collect();
        let slot: Vec<usize> = (0..is_delta.len()).map(|i| is_delta[..i].iter().filter(|&&d| d == is_delta[i]).count()).collect();
        let n_kv_layers = is_delta.iter().filter(|&&d| !d).count();
        let n_delta = is_delta.len() - n_kv_layers;
        let rs_layer = hp.qwen35.as_ref().map_or(0, |q| {
            let ch = 2 * q.nk * q.dk + q.nv * q.dk;
            (q.d_conv - 1) * ch + q.nv * q.dk * q.dk
        });
        if let Some(q) = &hp.qwen35 {
            if q.dk != 128 || q.d_conv > 8 || hp.n_rot > hp.head_dim {
                return Err(format!("qwen35: state_size {} / conv_kernel {} unsupported by the DeltaNet kernels", q.dk, q.d_conv));
            }
        }
        let rs = if n_delta > 0 { vec![alloc(n_delta * rs_layer * F)?] } else { vec![] };
        let kv = alloc(n_kv_layers * kv_arrays(hp) * n_ctx * kvd * H)?;
        let pt = alloc(MAX_SEQ * (n_ctx / KV_PAGE) * 4)?;
        let ebytes = |l: &crate::qwen3moe::Layer| w.expert(l, 0).iter().map(|b| b.len()).sum::<usize>();
        let layout = |m: &crate::qwen3moe::Moe, l| (m.gate.ty, m.up.ty, m.down.ty, ebytes(l));
        let mut layouts = vec![];
        // Dense layers: no group (usize::MAX).
        let group_of: Vec<usize> = w
            .layers
            .iter()
            .map(|l| l.moe.as_ref().map_or(usize::MAX, |m| layouts.iter().position(|x| *x == layout(m, l)).unwrap_or_else(|| {
                layouts.push(layout(m, l));
                layouts.len() - 1
            })))
            .collect();
        // Expert parts are packed back to back in cache slots; the GEMM's 16-byte Q4_K loads
        // need each part (and so each slot) to stay 16-aligned.
        if let Some(l) = w.layers.iter().find(|l| w.expert(l, 0).iter().any(|b| b.len() % 16 != 0)) {
            return Err(format!("expert tensor sizes not 16-byte multiples ({:?})", layout(l.moe.as_ref().unwrap(), l)));
        }
        let host_bytes: usize = w.layers.iter().map(|l| ebytes(l) * hp.n_expert).sum();
        let stage_bytes = layouts.iter().map(|l| l.3).max().unwrap_or(0);
        let host = pin_experts(&ctx, &w, host_bytes);
        let stage = alloc(STAGE * stage_bytes)?;
        let (b, span) = (n_ctx.min(512), n_ctx.min(SPAN));
        // Per token FFN widths: the routed experts' or the widest dense / shared FFN.
        let ff_tok = layers.iter().filter_map(|l| l.ffn.map(|f| f.0.n_rows / 2)).fold(kk * hp.n_ff_exp, usize::max);
        let gu_len = b * 2 * ff_tok;
        let s = Scratch {
            x: alloc(span * e * F)?,
            h: alloc(span * e * F)?,
            q: alloc(b * nq * F)?,
            attn: alloc(b * nq * F)?,
            kvs: alloc(2 * b * kvd * F)?,
            o: alloc(b * e * F)?,
            rl: alloc(b * hp.n_expert * F)?,
            xg: alloc(b * kk * e * F)?,
            gu: alloc(gu_len * F)?,
            act: alloc(b * ff_tok * F)?,
            db: alloc(b * kk * e * F)?,
            logits: alloc(LOGIT_ROWS.min(b) * hp.n_vocab * F)?,
            meta: alloc(meta_bytes(span, b, kk, hp.n_expert))?,
            dn: match &hp.qwen35 {
                // [b rows of [qkv | z | beta | alpha] (gated attention: q|gate, gate)][b conv rows]
                Some(q) => alloc(b * (dn_width(q).max(3 * nq) + 2 * q.nk * q.dk + q.nv * q.dk) * F)?,
                None => 0,
            },
        };
        let cpu_out = unsafe { ctx.alloc_pinned_with_flags::<f32>(CACHE_MAX_BATCH * e, 0) }.map_err(err)?;
        let meta_host = unsafe { ctx.alloc_pinned_with_flags::<u8>(meta_bytes(span, b, kk, hp.n_expert), 0) }.map_err(err)?;
        // The expert banks get what is left: the requested MiB, capped by free VRAM.
        let (free, _) = result::mem_get_info().map_err(err)?;
        let budget = (cache_mib << 20).min(free.saturating_sub(VRAM_RESERVE));
        if budget < cache_mib << 20 && cache_mib != usize::MAX >> 20 {
            log::warn!("gpu: expert cache capped at {} MiB of free VRAM (asked {cache_mib})", budget >> 20);
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
        let bias_off = w.layers.iter().find(|l| l.moe.is_some()).map_or(0, |l| w.expert(l, 0)[..3].iter().map(|b| b.len()).sum());
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
            rope_freq,
            rope_mscale: w.hp.rope_yarn.as_ref().map_or(1.0, |y| y.1),
            bias_off,
            kv,
            slot,
            n_kv_layers,
            rs,
            rs_layer,
            pt,
            seqs: vec![SeqKv::default()],
            free_pages: (0..(n_ctx / KV_PAGE) as i32).rev().collect(),
            cur: 0,
            groups,
            group_of,
            stage,
            stage_bytes,
            stats: CacheStats::default(),
            host,
            meta_host,
            hybrid: true,
            cpu_out: cpu_out.into(),
            s,
            n_ctx,
            max_batch: b,
            span,
            gu_len,
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
                // Pageable fallback (pinning failed). Still async on `st`, so the staging-half
                // events order it after the wave that last read `dst`.
                for b in parts {
                    unsafe { result::memcpy_htod_async(dst + off as u64, b, st.cu_stream()) }.expect("htod async");
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
            (GgmlType::Q4_K, _) => (&self.k.gemm_q4_k, m.n_rows.div_ceil(128), ntok.div_ceil(GEMM_T)),
            (GgmlType::Q6_K, _) => (&self.k.gemm_q6_k, m.n_rows.div_ceil(128), ntok.div_ceil(GEMM_T)),
            // ponytail: no tensor-core GEMM for Q8_0 / MXFP4 yet; prefill runs the 8-token tiles.
            (GgmlType::Q8_0, 1) => (&self.k.mm_q8_0, m.n_rows.div_ceil(4), 1),
            (GgmlType::MXFP4, 1) => (&self.k.mm_mxfp4, m.n_rows.div_ceil(4), 1),
            (GgmlType::Q8_0, _) => (&self.k.mm_q8_0_t8, m.n_rows.div_ceil(4), ntok.div_ceil(8)),
            (GgmlType::MXFP4, _) => (&self.k.mm_mxfp4_t8, m.n_rows.div_ceil(4), ntok.div_ceil(8)),
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
            (GgmlType::Q4_K, _) => (&self.k.gemm_q4_k_ids, n_rows.div_ceil(128), ntok.div_ceil(GEMM_TI)),
            (GgmlType::Q6_K, _) => (&self.k.gemm_q6_k_ids, n_rows.div_ceil(128), ntok.div_ceil(GEMM_TI)),
            (GgmlType::Q8_0, ..4) => (&self.k.mm_q8_0_ids, n_rows.div_ceil(4), ntok),
            (GgmlType::MXFP4, ..4) => (&self.k.mm_mxfp4_ids, n_rows.div_ceil(4), ntok),
            (GgmlType::Q8_0, _) => (&self.k.mm_q8_0_ids_t8, n_rows.div_ceil(4), ntok.div_ceil(8)),
            (GgmlType::MXFP4, _) => (&self.k.mm_mxfp4_ids_t8, n_rows.div_ceil(4), ntok.div_ceil(8)),
            (t, _) => unreachable!("checked at load: {t:?}"),
        };
        let (rb, k, n, ti) = (row_bytes as u64, k as i32, n_rows as i32, ntok as i32);
        launch!(self, f, (gx as u32, gy as u32, 1), 128, [wp, off, rb, k, x, y, n, ti]);
    }

    /// Per-head matvec: head h's n_rows rows of m (at h * n_rows) against the k floats at
    /// x + t * x_ts + h * x_hs, into y[t * y_ts + h * y_hs + r].
    #[allow(clippy::too_many_arguments)]
    fn mm_heads(&self, m: &Mat, n_rows: usize, x: Ptr, x_ts: usize, x_hs: usize, y: Ptr, y_ts: usize, y_hs: usize, ntok: usize, heads: usize) {
        let f = match m.ty {
            GgmlType::Q4_K => &self.k.mm_heads_q4_k,
            GgmlType::Q6_K => &self.k.mm_heads_q6_k,
            GgmlType::Q8_0 => &self.k.mm_heads_q8_0,
            t => unreachable!("checked at load: {t:?}"),
        };
        let (p, rb, hs, k, nr) = (m.p, m.row_bytes as u64, (n_rows * m.row_bytes) as u64, m.k as i32, n_rows as i32);
        let (xt, xh, yt, yh) = (x_ts as i32, x_hs as i32, y_ts as i32, y_hs as i32);
        launch!(self, f, (n_rows.div_ceil(4) as u32, ntok as u32, heads as u32), 128, [p, rb, hs, k, x, xt, xh, y, yt, yh, nr]);
    }

    /// GLM's absorbed MLA (src/glm.rs) for one ubatch: h (tc normed rows) -> s.q, tc x nh x vd,
    /// the attn_output input. Stores the ubatch's latent cache rows into kc.
    fn mla_attention(&self, a: &DAttn, h: Ptr, tc: usize, ps: &[(usize, usize, usize, usize)], kc: Ptr) -> Ptr {
        let DAttn::Mla { q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b } = a else { unreachable!() };
        let hp = &self.w.hp;
        let g = hp.glm.as_ref().expect("MLA is GLM");
        let (s, nh, eps, base) = (self.s, hp.n_head, hp.eps, hp.rope_base);
        let (lat, kd, row, vd) = (g.lat, g.nope + g.rope, g.lat + g.rope, g.vd);
        let at = |p: Ptr, floats: usize| p + (floats * F) as u64;
        let tn = tc as u32;
        self.mm(q_a, h, s.o, tc);
        let (o, qn, qal) = (s.o, *q_a_norm, q_a.n_rows as i32);
        launch!(self, self.k.rmsnorm, (tn, 1, 1), 256, [o, o, qn, qal, eps]);
        self.mm(q_b, s.o, s.attn, tc); // tc x nh x [nope | rope]
        self.mm(kv_a, h, s.kvs, tc); // tc x [latent | k_pe]
        let (kvs, kn, lati, rowi, kdi, nhi, rdi) = (s.kvs, *kv_a_norm, lat as i32, row as i32, kd as i32, nh as i32, g.rope as i32);
        launch!(self, self.k.rmsnorm_strided, (tn, 1, 1), 256, [kvs, kvs, kn, lati, rowi, eps]);
        // Absorbed q: [q_nope . k_b[h] (lat) | roped q_pe], row floats per head.
        self.mm_heads(k_b, lat, s.attn, nh * kd, kd, s.q, nh * row, row, tc, nh);
        // The pieces' roping and stores first: attention below overwrites s.attn.
        for &(sq, r0, n, pos) in ps {
            let (pos0, one, half) = (pos as i32, 1i32, (g.rope / 2) as u32);
            let (src, dst) = (at(s.attn, r0 * nh * kd + g.nope), at(s.q, r0 * nh * row + lat));
            launch!(self, self.k.rope_norm, ((n * nh) as u32, 1, 1), half, [dst, rowi, src, kdi, nhi, pos0, base, rdi]);
            let kp = at(s.kvs, r0 * row + lat);
            launch!(self, self.k.rope_norm, (n as u32, 1, 1), half, [kp, rowi, kp, rowi, one, pos0, base, rdi]);
            let pages = &self.seqs[sq].pages;
            let mut p = pos;
            while p < pos + n {
                let m = (KV_PAGE - p % KV_PAGE).min(pos + n - p);
                let slot = pages[p / KV_PAGE] as usize * KV_PAGE + p % KV_PAGE;
                let (kd16, src, cnt) = (kc + (slot * row * H) as u64, at(s.kvs, (r0 + p - pos) * row), (m * row) as i32);
                launch!(self, self.k.store_f16, ((m * row).div_ceil(256) as u32, 1, 1), 256, [kd16, src, cnt]);
                p += m;
            }
        }
        let scale = 1.0 / (kd as f32).sqrt();
        for &(sq, r0, n, pos) in ps {
            let pt = self.pt + (sq * (self.n_ctx / KV_PAGE) * 4) as u64;
            // Decode: split the positions (partials in s.gu, idle during attention).
            let sk = if n <= 4 { (pos + n).div_ceil(ATT_SPLIT).min(self.gu_len / (n * nh * (lat + 2))).max(1) } else { 1 };
            let (q, out, part) = (at(s.q, r0 * nh * row), at(s.attn, r0 * nh * lat), s.gu);
            let (pos0, ni, per, ski, rows) = (pos as i32, n as i32, (pos + n).div_ceil(sk) as i32, sk as i32, (n * nh) as i32);
            launch!(self, self.k.attention_mla, (n as u32, sk as u32, 1), (32 * nh) as u32, [q, kc, pt, out, nhi, pos0, ni, scale, per, part]);
            if sk > 1 {
                launch!(self, self.k.attn_merge, ((n * nh) as u32, 1, 1), lat as u32, [part, out, rows, ski]);
            }
        }
        self.mm_heads(v_b, vd, s.attn, nh * lat, lat, s.q, nh * vd, vd, tc, nh);
        s.q
    }

    /// y (ntok rows of n_cols) += bias b; no-op for b = 0.
    fn bias(&self, b: Ptr, y: Ptr, n_cols: usize, ntok: usize) {
        if b != 0 {
            let (c, n) = (n_cols as i32, (n_cols * ntok) as i32);
            launch!(self, self.k.add_bias, ((n_cols * ntok).div_ceil(256) as u32, 1, 1), 256, [y, b, c, n]);
        }
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

    /// Hybrid CPU/GPU expert compute for decode (default on). Off is batch-invariant: a stream's
    /// tokens do not depend on what else is batched with it.
    pub fn set_hybrid(&mut self, on: bool) {
        self.hybrid = on;
    }

    /// Run `experts` of layer `li` on the CPU for their rows of `h` (the span's FFN input, read
    /// back) while the GPU runs the rest, then add their output to the hidden state.
    fn cpu_experts(&self, li: usize, experts: &[usize], by_expert: &[Vec<(usize, f32)>], h: &[f32]) {
        let hp = &self.w.hp;
        let (e, t) = (hp.n_embd, h.len() / hp.n_embd);
        let wl = &self.w.layers[li];
        let m = wl.moe.as_ref().expect("MoE layer");
        let ty = [m.gate.ty, m.up.ty, m.down.ty];
        let jobs: Vec<crate::cpu_moe::ExpertJob> = experts
            .iter()
            .map(|&ex| {
                let mut parts = self.w.expert(wl, ex);
                if let Some(hx) = &self.host {
                    // The same parts, from the pinned copy (slot layout).
                    let mut o = hx.base[li] + ex * parts.iter().map(|b| b.len()).sum::<usize>();
                    let all = hx.buf.as_slice().expect("pinned");
                    for p in &mut parts {
                        (*p, o) = (&all[o..o + p.len()], o + p.len());
                    }
                }
                let bias = (parts.len() == 6).then(|| [parts[3], parts[4], parts[5]]);
                crate::cpu_moe::ExpertJob { w: [parts[0], parts[1], parts[2]], ty, bias, rows: by_expert[ex].clone() }
            })
            .collect();
        let mut buf = self.cpu_out.borrow_mut();
        let out = &mut buf.as_mut_slice().expect("pinned")[..t * e];
        out.fill(0.0);
        crate::cpu_moe::moe(&jobs, h, e, hp.n_ff_exp, hp.oai, out);
        let (x, o, n) = (self.s.x, self.s.o, (t * e) as i32);
        unsafe { result::memcpy_htod_async(o, &*out, self.stream.cu_stream()) }.expect("htod async");
        launch!(self, self.k.add_inplace, ((t * e).div_ceil(256) as u32, 1, 1), 256, [x, o, n]);
    }

    fn rs_bytes(&self) -> usize {
        (self.slot.len() - self.n_kv_layers) * self.rs_layer * F
    }

    /// qwen35 Gated DeltaNet over one ubatch of `tc` rows of h (pieces as in forward_span), each
    /// piece continuing its sequence's conv and ssm state. Returns the ssm_out input (s.attn).
    fn delta_net(&self, a: &DAttn, di: usize, h: Ptr, tc: usize, ps: &[(usize, usize, usize, usize)]) -> Ptr {
        let DAttn::Delta { w, dt, a, conv, norm } = a else { unreachable!() };
        let (hp, s) = (&self.w.hp, self.s);
        let q = hp.qwen35.as_ref().unwrap();
        let (wd, vd) = (dn_width(q), q.nv * q.dk);
        let ch = 2 * q.nk * q.dk + vd;
        let cv = s.dn + (self.max_batch * wd.max(3 * hp.n_head * hp.head_dim) * F) as u64;
        self.mm(w, h, s.dn, tc);
        let (wdi, chi, kc, dk, eps, nki, nvi, vdi) = (wd as i32, ch as i32, q.d_conv as i32, q.dk as i32, hp.eps, q.nk as i32, q.nv as i32, vd as i32);
        let (boff, aoff) = ((ch + vd) as i32, (ch + vd + q.nv) as i32);
        let (dt, a, conv, norm) = (*dt, *a, *conv, *norm);
        for &(sq, r0, n, _) in ps {
            let cst = self.rs[sq] + (di * self.rs_layer * F) as u64;
            let sst = cst + ((q.d_conv - 1) * ch * F) as u64;
            let (inp, c, out, dnr) = (s.dn + (r0 * wd * F) as u64, cv + (r0 * ch * F) as u64, s.attn + (r0 * vd * F) as u64, s.dn + (r0 * wd * F) as u64);
            let ni = n as i32;
            launch!(self, self.k.ssm_conv, (ch.div_ceil(256) as u32, 1, 1), 256, [c, inp, wdi, cst, conv, chi, ni, kc]);
            let (qk_heads, l2eps) = ((2 * q.nk) as i32, hp.eps);
            launch!(self, self.k.l2norm_heads, ((n * 2 * q.nk) as u32, 1, 1), q.dk as u32, [c, chi, qk_heads, dk, l2eps]);
            launch!(self, self.k.gated_delta, ((q.dk / 4) as u32, q.nv as u32, 1), 128, [c, chi, dnr, wdi, boff, aoff, dt, a, sst, out, vdi, nki, nvi, ni]);
        }
        let (attn, z) = (s.attn, s.dn + (ch * F) as u64);
        launch!(self, self.k.gated_rmsnorm, ((tc * q.nv) as u32, 1, 1), q.dk as u32, [attn, z, wdi, norm, nvi, dk, eps]);
        s.attn
    }

    /// Serve up to `n` sequences at once (`Engine::n_seq`); drops every cached sequence.
    pub fn set_n_seq(&mut self, n: usize) {
        assert!((1..=MAX_SEQ).contains(&n), "1..={MAX_SEQ} sequences");
        for s in 0..self.seqs.len() {
            self.select(s);
            self.truncate(0);
        }
        self.seqs.resize_with(n, SeqKv::default);
        while !self.rs.is_empty() && self.rs.len() < n {
            let b = self.stream.alloc_zeros::<u8>(self.rs_bytes()).expect("recurrent state alloc");
            self.rs.push(b.device_ptr(&self.stream).0);
            self._bufs.push(b);
        }
        self.cur = 0;
    }

    /// Route each token to its top `n` experts (renormalized over those) instead of the model's
    /// own count; 1..=that count, since the expert buffers were sized for it.
    pub fn set_n_expert_used(&mut self, n: usize) -> Result<(), String> {
        let arch = self.w.g.str("general.architecture").unwrap_or("");
        let max = self.w.g.u64(&format!("{arch}.expert_used_count")).unwrap_or(0) as usize;
        if !(1..=max).contains(&n) {
            return Err(format!("--moeexperts must be 1..={max}"));
        }
        self.w.hp.n_expert_used = n;
        Ok(())
    }

    /// Give sequence `seq` pages for `n` more positions and upload its page table.
    /// Panics if its context or the shared pool would overflow.
    fn grow(&mut self, seq: usize, n: usize) {
        let sq = &mut self.seqs[seq];
        assert!(sq.n_past + n <= self.n_ctx, "context overflow: {} + {n} > {}", sq.n_past, self.n_ctx);
        let need = (sq.n_past + n).div_ceil(KV_PAGE);
        if need > sq.pages.len() {
            while sq.pages.len() < need {
                sq.pages.push(self.free_pages.pop().expect("KV page pool exhausted"));
            }
            let row = self.pt + (seq * (self.n_ctx / KV_PAGE) * 4) as u64;
            self.stream.synchronize().expect("sync"); // the previous pass may still read the table
            let pages = sq.pages.clone();
            self.htod(row, &pages);
        }
    }

    /// Run up to `span` tokens layer by layer: attention in ubatches of `max_batch`, then the
    /// MoE of the whole span in expert waves, so each expert is uploaded at most once per layer
    /// (a 4096-token prefill moves the experts once, not once per ubatch). Returns the logits of
    /// every token, or of each segment's last.
    ///
    /// The span is segments (sequence, tokens), each continuing its own sequence: projections,
    /// routing and experts run over all their tokens together; RoPE, the KV store and attention
    /// run per segment, through its sequence's page table.
    fn forward_span(&mut self, segs: &[(usize, &[u32])], all_logits: bool) -> Vec<f32> {
        for &(sq, toks) in segs {
            self.grow(sq, toks.len());
        }
        let tokens: Vec<u32> = segs.iter().flat_map(|s| s.1.iter().copied()).collect();
        let hp = &self.w.hp;
        let (e, hd, nh, nkv, ff, kk) = (hp.n_embd, hp.head_dim, hp.n_head, hp.n_head_kv, hp.n_ff_exp, hp.n_expert_used);
        let (kvd, t, b) = (nkv * hd, tokens.len(), self.max_batch);
        // Per ubatch: its pieces of segments, (sequence, first row in the ubatch, rows, position).
        let mut pieces: Vec<Vec<(usize, usize, usize, usize)>> = vec![];
        let (mut row, mut c0) = (0, 0);
        for &(sq, toks) in segs {
            let mut done = 0;
            while done < toks.len() {
                if row == 0 || row - c0 == b {
                    c0 = row;
                    pieces.push(vec![]);
                }
                let n = (toks.len() - done).min(b - (row - c0));
                pieces.last_mut().unwrap().push((sq, row - c0, n, self.seqs[sq].n_past + done));
                (row, done) = (row + n, done + n);
            }
        }
        let s = self.s;
        let nv = hp.n_vocab;
        let n_expert = hp.n_expert;
        let chunks: Vec<(usize, usize)> = (0..t).step_by(b).map(|c0| (c0, b.min(t - c0))).collect();
        let at = |p: Ptr, c0: usize, width: usize| p + (c0 * width * F) as u64;

        let mut xe = vec![0f32; t * e];
        for (row, &tok) in xe.chunks_exact_mut(e).zip(&tokens) {
            dequant_row(self.w.tok_embd.ty, self.w.g.row(&self.w.tok_embd, tok as usize), row);
        }
        self.htod(s.x, &xe);

        let (ei, eps, base) = (e as i32, hp.eps, hp.rope_base);
        let (nhi, nkvi, hdi, nrot) = (nh as i32, nkv as i32, hd as i32, hp.n_rot as i32);
        let gated = hp.qwen35.is_some();
        let scale = 1.0 / (hd as f32).sqrt();
        let pt_row = |sq: usize| self.pt + (sq * (self.n_ctx / KV_PAGE) * 4) as u64;
        for (li, l) in self.layers.iter().enumerate() {
            let kc = self.kv + (self.slot[li] * kv_arrays(hp) * self.n_ctx * kvd * H) as u64;
            let vc = kc + (self.n_ctx * kvd * H) as u64;
            // --- attention, per ubatch; leaves each ubatch's normed FFN input in s.h ---
            let mut sels = vec![];
            let mut pred: Vec<usize> = vec![];
            let mut hrows: Vec<f32> = vec![];
            let mut cpu_ex: Vec<usize> = vec![];
            for (&(c0, tc), ps) in chunks.iter().zip(&pieces) {
                let (x, h, o) = (at(s.x, c0, e), at(s.h, c0, e), s.o);
                let tn = tc as u32;
                let (ks, vs) = (s.kvs, s.kvs + (tc * kvd * F) as u64);
                let wn = l.attn_norm;
                launch!(self, self.k.rmsnorm, (tn, 1, 1), 256, [h, x, wn, ei, eps]);
                let wo_in = if let DAttn::Delta { .. } = &l.attn {
                    self.delta_net(&l.attn, self.slot[li], h, tc, ps)
                } else if let DAttn::Gqa { wq, wk, wv } = &l.attn {
                let gate = s.dn + (tc * 2 * nh * hd * F) as u64;
                if gated {
                    // attn_q rows are per head [q | gate]; split, gate the output after attention.
                    self.mm(wq, h, s.dn, tc);
                    let (q, dn, hdi) = (s.q, s.dn, hd as i32);
                    launch!(self, self.k.split_qgate, ((tc * nh) as u32, 1, 1), hd as u32, [q, gate, dn, hdi]);
                } else {
                    self.mm(wq, h, s.q, tc);
                }
                self.mm(wk, h, ks, tc);
                self.mm(wv, h, vs, tc);
                self.bias(l.bias[0], s.q, nh * hd, tc);
                self.bias(l.bias[1], ks, kvd, tc);
                self.bias(l.bias[2], vs, kvd, tc);
                for &(sq, r0, tc, pos) in ps {
                  let pages = &self.seqs[sq].pages;
                  let pt = pt_row(sq);
                  let (q, attn, ks, vs) = (at(s.q, r0, nh * hd), at(s.attn, r0, nh * hd), at(ks, r0, kvd), at(vs, r0, kvd));
                  let (pos0, tn, ti) = (pos as i32, tc as u32, tc as i32);
                  let (qn, kn, fr, ms) = (l.q_norm, l.k_norm, self.rope_freq, self.rope_mscale);
                  launch!(self, self.k.qk_norm_rope, (tn * nh as u32, 1, 1), hd as u32, [q, qn, nhi, hdi, pos0, eps, base, fr, ms, nrot]);
                  launch!(self, self.k.qk_norm_rope, (tn * nkv as u32, 1, 1), hd as u32, [ks, kn, nkvi, hdi, pos0, eps, base, fr, ms, nrot]);
                  // Store K and V, one launch per page the rows land in.
                  let mut p = pos;
                  while p < pos + tc {
                      let n = (KV_PAGE - p % KV_PAGE).min(pos + tc - p);
                      let slot = pages[p / KV_PAGE] as usize * KV_PAGE + p % KV_PAGE;
                      let (kd, vd) = (kc + (slot * kvd * H) as u64, vc + (slot * kvd * H) as u64);
                      let (ksp, vsp) = (at(ks, p - pos, kvd), at(vs, p - pos, kvd));
                      let nkv_t = (n * kvd) as i32;
                      let g = ((n * kvd).div_ceil(256) as u32, 1, 1);
                      launch!(self, self.k.store_f16, g, 256, [kd, ksp, nkv_t]);
                      launch!(self, self.k.store_f16, g, 256, [vd, vsp, nkv_t]);
                      p += n;
                  }
                  let gq = nh / nkv;
                  if hd == 128 && 8 % gq == 0 && l.sinks == 0 && l.window == 0 {
                    // Decode (<= 4 tokens): attention_dec, one row per warp. Prefill: tensor-core
                    // attention_tc. Both split the positions when blocks are few, partials in s.gu
                    // (idle during attention).
                    let small = tc <= 4;
                    let cap = self.gu_len / (tc * nh * 130);
                    let (f, gy, bs) = if small { (&self.k.attention_dec, (tc * gq).div_ceil(8), 256) }
                        else { (&self.k.attention_tc, (tc * gq).div_ceil(64), 128) };
                    // Fill the GPU: split positions until ~240 blocks (decode: every ATT_SPLIT).
                    let want = if small { usize::MAX } else { (240 / (nkv * gy)).max(1) };
                    let sk = (pos + tc).div_ceil(if small { ATT_SPLIT } else { 512 }).min(want).min(cap).max(1);
                    let (part, ski) = (s.gu, sk as i32);
                    launch!(self, *f, (nkv as u32, gy as u32, sk as u32), bs, [q, kc, vc, pt, attn, nhi, nkvi, pos0, ti, scale, part]);
                    if sk > 1 {
                        let rows = (tc * nh) as i32;
                        launch!(self, self.k.attn_merge, ((tc * nh) as u32, 1, 1), 128, [part, attn, rows, ski]);
                    }
                  } else {
                    let (sinks, win) = (l.sinks, l.window);
                    launch!(self, self.k.attention, (nh as u32, tn, 1), 32 * 8, [q, kc, vc, pt, attn, nhi, nkvi, hdi, pos0, scale, sinks, win]);
                  }
                }
                if gated {
                    let (attn, n) = (s.attn, (tc * nh * hd) as i32);
                    launch!(self, self.k.mul_sigmoid, ((tc * nh * hd).div_ceil(256) as u32, 1, 1), 256, [attn, gate, n]);
                }
                s.attn
                } else {
                    self.mla_attention(&l.attn, h, tc, ps, kc)
                };
                self.mm(&l.wo, wo_in, o, tc);
                self.bias(l.bias[3], o, e, tc);
                let n = (tc * e) as i32;
                launch!(self, self.k.add_inplace, ((tc * e).div_ceil(256) as u32, 1, 1), 256, [x, o, n]);
                let wn = l.ffn_norm;
                launch!(self, self.k.rmsnorm, (tn, 1, 1), 256, [h, x, wn, ei, eps]);
                if let Some((gu, down)) = &l.ffn {
                    // Dense FFN or shared expert, straight into the residual.
                    let (sgu, act, ffi, ti) = (s.gu, s.act, (gu.n_rows / 2) as i32, tc as i32);
                    self.mm(gu, h, sgu, tc);
                    launch!(self, self.k.swiglu, ((tc * gu.n_rows / 2).div_ceil(256) as u32, 1, 1), 256, [act, sgu, ffi, ti]);
                    self.mm(down, act, o, tc);
                    if l.shared_gate != 0 {
                        let (sg, h) = (l.shared_gate, h);
                        launch!(self, self.k.sigmoid_dot_scale, (tn, 1, 1), 256, [o, h, sg, ei]);
                    }
                    launch!(self, self.k.add_inplace, ((tc * e).div_ceil(256) as u32, 1, 1), 256, [x, o, n]);
                }
                let Some(router) = &l.router else { continue };
                self.mm(router, h, s.rl, tc);
                self.bias(l.bias[4], s.rl, n_expert, tc);
                // Decode: also predict layer li+1's experts, its router on this post-attention x
                // (top-8 recall ~84%). Read back with the router logits; prefetched below.
                let next = self.layers.get(li + 1).and_then(|n| n.router.as_ref().map(|r| (n, r)));
                let predict = t == 1 && next.is_some();
                if let (true, Some((nl, nr))) = (predict, next) {
                    let (o, wn, rl2) = (s.o, nl.ffn_norm, s.rl + (n_expert * F) as u64);
                    launch!(self, self.k.rmsnorm, (1, 1, 1), 256, [o, x, wn, ei, eps]);
                    self.mm(nr, o, rl2, 1);
                    self.bias(nl.bias[4], rl2, n_expert, 1);
                }
                let rl = self.dtoh(s.rl, (tc + predict as usize) * n_expert);
                let mut r = self.w.route(li, &rl[..tc * n_expert]);
                if predict {
                    r.extend(self.w.route(li + 1, &rl[tc * n_expert..]));
                }
                if self.hybrid && t <= CACHE_MAX_BATCH {
                    hrows = self.dtoh(h, t * e); // the CPU experts' input (stream just synced)
                }
                if predict {
                    pred = r.pop().unwrap().iter().take(PREFETCH_TOP).map(|p| p.0).collect();
                }
                sels.push(r);
            }

            if l.router.is_none() {
                continue;
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
                // Hybrid: split the misses so PCIe (upload, then GPU) and the CPU finish together.
                let admit = if self.hybrid {
                    let mut seen = vec![false; n_expert];
                    let miss = ids.iter().filter(|&&ex| !std::mem::replace(&mut seen[ex as usize], true) && self.groups[gi].cache.slot(li, ex as usize).is_none()).count();
                    (miss * ADMIT_NUM).div_ceil(ADMIT_DEN)
                } else {
                    usize::MAX
                };
                let p = self.groups[gi].cache.plan(li, &ids, t, admit);
                let mut copied = false;
                for &(ex, slot) in &p.fills {
                    self.stats.bytes += self.upload_expert(li, ex, bank + (slot * sb) as u64, &self.stream) as u64;
                }
                cpu_ex.extend(&p.rest);
                // Prefetch li+1's predicted misses on the copy stream, overlapping this layer's
                // MoE and the next attention. Victims are never this layer's experts (still to be
                // read); earlier layers are finished (the router readback synchronized).
                if !pred.is_empty() {
                    let g2 = self.group_of[li + 1];
                    let (bank2, sb2) = (self.groups[g2].bank, self.groups[g2].slot_bytes);
                    for (ex, slot) in self.groups[g2].cache.prefetch(li + 1, &pred, li) {
                        self.stats.bytes += self.upload_expert(li + 1, ex, bank2 + (slot * sb2) as u64, &self.copy) as u64;
                        copied = true;
                    }
                }
                if copied {
                    self.pf_ready.record(&self.copy).expect("record");
                }
                self.stats.hits += p.hits as u64;
                self.stats.misses += (p.fills.len() + p.rest.len()) as u64;
                let gpu: Vec<usize> = used.iter().copied().filter(|ex| !cpu_ex.contains(ex)).collect();
                for &ex in &gpu {
                    ptr[ex] = bank + (self.groups[gi].cache.slot(li, ex).expect("planned expert resident") * sb) as u64;
                }
                vec![gpu]
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
            let wm = wl.moe.as_ref().unwrap();
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
                    self.mm_ids(wm.gate.ty, wp, 0, wm.gate.row_bytes(), e, xg, gu, 2 * ff, r, per_ex);
                    let (ffi, ri, oai) = (ff as i32, r as i32, hp.oai);
                    let (bo, bdo, ff2, ei2) = (self.bias_off as u64, (self.bias_off + 2 * ff * F) as u64, 2 * ff as i32, e as i32);
                    if oai {
                        launch!(self, self.k.bias_ids, (r as u32, 1, 1), 256, [gu, wp, bo, ff2]);
                    }
                    let glu = if oai { &self.k.swiglu_oai } else { &self.k.swiglu };
                    launch!(self, *glu, ((r * ff).div_ceil(256) as u32, 1, 1), 256, [act, gu, ffi, ri]);
                    self.mm_ids(wm.down.ty, wp, down_off, wm.down.row_bytes(), ff, act, dbuf, e, r, per_ex);
                    if oai {
                        launch!(self, self.k.bias_ids, (r as u32, 1, 1), 256, [dbuf, wp, bdo, ei2]);
                    }
                    launch!(self, self.k.moe_combine, (t as u32, 1, 1), 256, [x, dbuf, off, crow, cw, ei]);
                }
                if let Some(hf) = half[wi] {
                    self.free[hf].record(&self.stream).expect("record");
                    upload(wi + 2);
                }
            }
            if !cpu_ex.is_empty() {
                self.cpu_experts(li, &cpu_ex, &by_expert, &hrows);
            }
        }
        for &(sq, toks) in segs {
            self.seqs[sq].n_past += toks.len();
        }

        // Logit rows: all, or each segment's last, in runs of consecutive rows.
        let want: Vec<usize> = if all_logits {
            (0..t).collect()
        } else {
            segs.iter().scan(0, |r, s| { *r += s.1.len(); Some(*r - 1) }).collect()
        };
        let mut rows: Vec<(usize, usize)> = vec![];
        for r in want {
            match rows.last_mut() {
                Some(l) if l.0 + l.1 == r && l.1 < LOGIT_ROWS => l.1 += 1,
                _ => rows.push((r, 1)),
            }
        }
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
        GgmlType::F32 | GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::Q8_0 | GgmlType::MXFP4 => Ok(()),
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

    fn n_seq(&self) -> usize {
        self.seqs.len()
    }

    fn kv_page(&self) -> usize {
        KV_PAGE
    }

    /// The sequence's pages of every layer's K and V, page-major per (layer, K|V).
    fn kv_save(&mut self) -> Vec<u8> {
        let pages = self.seqs[self.cur].pages.clone();
        assert!(self.rs.is_empty(), "kv_save: recurrent state is not snapshotted");
        let (kvd, nl) = (self.w.hp.n_head_kv * self.w.hp.head_dim, self.n_kv_layers);
        let pb = KV_PAGE * kvd * H;
        self.stream.synchronize().expect("sync");
        let mut out = vec![0u8; nl * kv_arrays(&self.w.hp) * pages.len() * pb];
        for (i, dst) in out.chunks_mut(pb).enumerate() {
            let src = self.kv + ((i / pages.len()) * self.n_ctx * kvd * H + pages[i % pages.len()] as usize * pb) as u64;
            unsafe { result::memcpy_dtoh_sync(dst, src) }.expect("kv dtoh");
        }
        out
    }

    fn kv_restore(&mut self, data: &[u8], n: usize) {
        assert!(self.rs.is_empty(), "kv_restore: recurrent state is not snapshotted");
        let (kvd, nl) = (self.w.hp.n_head_kv * self.w.hp.head_dim, self.n_kv_layers);
        let pb = KV_PAGE * kvd * H;
        let nl = nl * kv_arrays(&self.w.hp);
        let np_img = data.len() / (nl * pb);
        assert!(self.seqs[self.cur].n_past == 0 && n <= np_img * KV_PAGE, "kv_restore into a used sequence or past the image");
        self.grow(self.cur, n);
        let pages = self.seqs[self.cur].pages.clone();
        for lh in 0..nl {
            for (j, &p) in pages.iter().enumerate() {
                let dst = self.kv + (lh * self.n_ctx * kvd * H + p as usize * pb) as u64;
                self.htod(dst, &data[(lh * np_img + j) * pb..][..pb]);
            }
        }
        self.seqs[self.cur].n_past = n;
    }

    fn select(&mut self, seq: usize) {
        assert!(seq < self.seqs.len());
        self.cur = seq;
    }

    /// Recurrent models: only a rewind to 0 (which zeroes the state) until state checkpoints.
    fn truncate(&mut self, n_past: usize) {
        if !self.rs.is_empty() && n_past < self.seqs[self.cur].n_past {
            assert!(n_past == 0, "recurrent state can only be reset");
            unsafe { result::memset_d8_async(self.rs[self.cur], 0, self.rs_bytes(), self.stream.cu_stream()) }.expect("memset");
        }
        let sq = &mut self.seqs[self.cur];
        sq.n_past = sq.n_past.min(n_past);
        let keep = sq.n_past.div_ceil(KV_PAGE);
        self.free_pages.extend(sq.pages.drain(keep..));
    }

    /// Panics if the KV cache would overflow n_ctx.
    fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        let sq = self.cur;
        assert!(self.seqs[sq].n_past + tokens.len() <= self.n_ctx, "context overflow");
        tokens.chunks(self.span).flat_map(|c| self.forward_span(&[(sq, c)], true)).collect()
    }

    fn forward_last(&mut self, tokens: &[u32]) -> Vec<f32> {
        let sq = self.cur;
        assert!(self.seqs[sq].n_past + tokens.len() <= self.n_ctx, "context overflow");
        let n = tokens.len().div_ceil(self.span);
        let mut last = vec![];
        for (i, c) in tokens.chunks(self.span).enumerate() {
            last = self.forward_span(&[(sq, c)], false);
            if i + 1 < n {
                last.clear();
            }
        }
        last
    }

    fn decode_batch(&mut self, rows: &[(usize, u32)]) -> Vec<f32> {
        let toks: Vec<[u32; 1]> = rows.iter().map(|r| [r.1]).collect();
        let segs: Vec<(usize, &[u32])> = rows.iter().zip(&toks).map(|(r, t)| (r.0, &t[..])).collect();
        assert!(segs.len() <= self.span);
        self.forward_span(&segs, false)
    }
}
