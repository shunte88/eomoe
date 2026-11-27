//! GLM-4.7-Flash CPU reference forward (llama.cpp arch `deepseek2`, src/models/deepseek2.cpp,
//! MLA branch). f32 activations, f16 latent KV cache. Per layer:
//! - MLA attention, absorbed: q = q_b(norm(q_a(x))) per head is [nope 192 | rope 64]; the nope
//!   part is absorbed through k_b into the 512-wide latent, so every head attends over one shared
//!   cache row per position, [norm(kv_a(x)[..512]) | rope(kv_a(x)[512..])] (576 wide). The latent
//!   half is also V; the per-head output goes back up through v_b (512 -> 256), then attn_output.
//! - RoPE is NORM (adjacent pairs), on the 64 rope dims of q and the shared k_pe.
//! - FFN: layer 0 is a dense SwiGLU; the others are 64 routed experts (sigmoid gate, top-4 chosen
//!   by sigmoid + exp_probs_b, weights the plain sigmoids renormalised and scaled by 1.8) plus one
//!   shared expert.

use crate::cpu::rmsnorm;
use crate::gguf::{Gguf, TensorInfo};
use crate::quant::{dequant_row, dot, matmul};
use crate::qwen3moe::{glu, route_sigmoid, Engine};
use rayon::prelude::*;

struct Moe {
    router: TensorInfo,
    bias: TensorInfo,
    gate: TensorInfo,
    up: TensorInfo,
    down: TensorInfo,
    shared: [TensorInfo; 3],
}

struct Layer {
    attn_norm: TensorInfo,
    q_a: TensorInfo,
    q_a_norm: TensorInfo,
    q_b: TensorInfo,
    kv_a: TensorInfo,
    kv_a_norm: TensorInfo,
    k_b: TensorInfo,
    v_b: TensorInfo,
    wo: TensorInfo,
    ffn_norm: TensorInfo,
    /// gate, up, down of the dense FFN (leading layers).
    dense: Option<[TensorInfo; 3]>,
    moe: Option<Moe>,
}

pub struct CpuModel {
    g: Gguf,
    layers: Vec<Layer>,
    tok_embd: TensorInfo,
    out_norm: TensorInfo,
    output: TensorInfo,
    n_embd: usize,
    n_head: usize,
    /// kv_lora_rank (latent width), qk_nope and qk_rope head dims, v head dim.
    lat: usize,
    nope: usize,
    rope: usize,
    vd: usize,
    n_expert: usize,
    n_expert_used: usize,
    n_ff_exp: usize,
    w_scale: f32,
    eps: f32,
    rope_base: f32,
    /// Per layer: n_past x (lat + rope) cache rows.
    kv: Vec<Vec<f32>>,
    n_past: usize,
}

impl CpuModel {
    pub fn new(g: Gguf) -> Result<Self, String> {
        let arch = g.str("general.architecture").unwrap_or("");
        if arch != "deepseek2" {
            return Err(format!("not a deepseek2 model: {arch:?}"));
        }
        let u = |k: &str| g.u64(&format!("deepseek2.{k}")).map(|v| v as usize).ok_or(format!("missing deepseek2.{k}"));
        let f = |k: &str| g.f64(&format!("deepseek2.{k}")).map(|v| v as f32).ok_or(format!("missing deepseek2.{k}"));
        let t = |n: &str| g.tensor(n).cloned().ok_or(format!("missing tensor {n}"));
        // ponytail: GLM-4.7-Flash only (sigmoid gating, the llama.cpp default for this shape when the
        // GGUF has no expert_gating_func; normalised weights; one expert group; no YaRN).
        if g.u64("deepseek2.expert_gating_func").is_some_and(|v| v != 2) || u("expert_group_count").unwrap_or(1) != 1 || g.f64("deepseek2.rope.scaling.factor").is_some() {
            return Err("deepseek2: only the GLM-4.7-Flash variant is supported".into());
        }
        let (n_head, rope, kl) = (u("attention.head_count")?, u("rope.dimension_count")?, u("attention.key_length_mla")?);
        let n_dense = u("leading_dense_block_count").unwrap_or(0);
        let layers = (0..u("block_count")?)
            .map(|i| {
                let w = |s: &str| t(&format!("blk.{i}.{s}.weight"));
                Ok(Layer {
                    attn_norm: w("attn_norm")?,
                    q_a: w("attn_q_a")?,
                    q_a_norm: w("attn_q_a_norm")?,
                    q_b: w("attn_q_b")?,
                    kv_a: w("attn_kv_a_mqa")?,
                    kv_a_norm: w("attn_kv_a_norm")?,
                    k_b: w("attn_k_b")?,
                    v_b: w("attn_v_b")?,
                    wo: w("attn_output")?,
                    ffn_norm: w("ffn_norm")?,
                    dense: if i < n_dense { Some([w("ffn_gate")?, w("ffn_up")?, w("ffn_down")?]) } else { None },
                    moe: if i < n_dense {
                        None
                    } else {
                        Some(Moe {
                            router: w("ffn_gate_inp")?,
                            bias: t(&format!("blk.{i}.exp_probs_b.bias"))?,
                            gate: w("ffn_gate_exps")?,
                            up: w("ffn_up_exps")?,
                            down: w("ffn_down_exps")?,
                            shared: [w("ffn_gate_shexp")?, w("ffn_up_shexp")?, w("ffn_down_shexp")?],
                        })
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            tok_embd: t("token_embd.weight")?,
            out_norm: t("output_norm.weight")?,
            output: t("output.weight").or_else(|_| t("token_embd.weight"))?,
            n_embd: u("embedding_length")?,
            n_head,
            lat: u("attention.kv_lora_rank")?,
            nope: kl - rope,
            rope,
            vd: u("attention.value_length_mla")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            w_scale: f("expert_weights_scale").unwrap_or(1.0),
            eps: f("attention.layer_norm_rms_epsilon")?,
            rope_base: f("rope.freq_base")?,
            kv: layers.iter().map(|_| vec![]).collect(),
            layers,
            n_past: 0,
            g,
        })
    }

    /// x (T x k) times rows [r0, r1) of `w` (a 2-D or flattened 3-D tensor).
    fn mm(&self, w: &TensorInfo, r: std::ops::Range<usize>, x: &[f32]) -> Vec<f32> {
        let (k, rb) = (w.dims[0] as usize, w.row_bytes());
        let mut out = vec![0f32; x.len() / k * r.len()];
        matmul(w.ty, &self.g.data(w)[r.start * rb..r.end * rb], k, x, &mut out);
        out
    }

    fn mm_all(&self, w: &TensorInfo, x: &[f32]) -> Vec<f32> {
        self.mm(w, 0..w.size / w.row_bytes(), x)
    }

    /// NORM rope: adjacent pairs (2i, 2i+1) rotate by pos * base^(-2i/d).
    fn rope(&self, v: &mut [f32], pos: usize) {
        let d = v.len();
        for i in 0..d / 2 {
            let (s, c) = (pos as f32 * self.rope_base.powf(-2.0 * i as f32 / d as f32)).sin_cos();
            let (a, b) = (v[2 * i], v[2 * i + 1]);
            v[2 * i] = a * c - b * s;
            v[2 * i + 1] = a * s + b * c;
        }
    }

    /// SwiGLU FFN on rows [r0, r1) of gate/up (and the matching down rows) of each tensor.
    fn swiglu(&self, [gate, up, down]: [&TensorInfo; 3], ff: std::ops::Range<usize>, out: std::ops::Range<usize>, x: &[f32]) -> Vec<f32> {
        let (g, u) = (self.mm(gate, ff.clone(), x), self.mm(up, ff, x));
        let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| glu(false, g, u)).collect();
        self.mm(down, out, &act)
    }
}

impl Engine for CpuModel {
    fn n_vocab(&self) -> usize {
        self.output.dims[1] as usize
    }

    fn n_ctx(&self) -> usize {
        usize::MAX
    }

    fn truncate(&mut self, n_past: usize) {
        let row = self.lat + self.rope;
        self.kv.iter_mut().for_each(|k| k.truncate(n_past * row));
        self.n_past = self.n_past.min(n_past);
    }

    fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        let (e, nh, lat, nope, rd, vd) = (self.n_embd, self.n_head, self.lat, self.nope, self.rope, self.vd);
        let (kd, row) = (nope + rd, lat + rd);
        let t_n = tokens.len();
        let g = &self.g;

        let mut x = vec![0f32; t_n * e];
        for (r, &tok) in x.chunks_exact_mut(e).zip(tokens) {
            dequant_row(self.tok_embd.ty, g.row(&self.tok_embd, tok as usize), r);
        }

        for li in 0..self.layers.len() {
            let l = &self.layers[li];

            // --- MLA attention ---
            let mut h = x.clone();
            h.chunks_exact_mut(e).for_each(|r| rmsnorm(r, g.f32s(&l.attn_norm), self.eps));
            let mut qa = self.mm_all(&l.q_a, &h);
            let qal = qa.len() / t_n;
            qa.chunks_exact_mut(qal).for_each(|r| rmsnorm(r, g.f32s(&l.q_a_norm), self.eps));
            let mut q = self.mm_all(&l.q_b, &qa); // T x nh x [nope | rope]
            let mut kv = self.mm_all(&l.kv_a, &h); // T x [latent | k_pe]
            for (ti, (qr, kr)) in q.chunks_exact_mut(nh * kd).zip(kv.chunks_exact_mut(row)).enumerate() {
                let pos = self.n_past + ti;
                qr.chunks_exact_mut(kd).for_each(|hq| self.rope(&mut hq[nope..], pos));
                self.rope(&mut kr[lat..], pos);
                rmsnorm(&mut kr[..lat], g.f32s(&l.kv_a_norm), self.eps);
            }
            // Absorb q_nope through k_b, per head: (T x nope) x k_b[h] -> T x lat.
            let mut qf = vec![0f32; t_n * nh * row];
            for hi in 0..nh {
                let qn: Vec<f32> = (0..t_n).flat_map(|ti| q[(ti * nh + hi) * kd..][..nope].iter().copied()).collect();
                let qabs = self.mm(&l.k_b, hi * lat..(hi + 1) * lat, &qn);
                for ti in 0..t_n {
                    let dst = &mut qf[(ti * nh + hi) * row..][..row];
                    dst[..lat].copy_from_slice(&qabs[ti * lat..][..lat]);
                    dst[lat..].copy_from_slice(&q[(ti * nh + hi) * kd + nope..][..rd]);
                }
            }
            let f16 = |x: &f32| half::f16::from_f32(*x).to_f32();
            self.kv[li].extend(kv.iter().map(f16));
            let kc = &self.kv[li];

            let scale = 1.0 / (kd as f32).sqrt();
            let n_past = self.n_past;
            let mut att = vec![0f32; t_n * nh * lat];
            att.par_chunks_mut(lat).enumerate().for_each(|(idx, out)| {
                let qv = &qf[idx * row..][..row];
                let pos = n_past + idx / nh;
                let mut sc: Vec<f32> = (0..=pos).map(|p| dot(qv, &kc[p * row..][..row]) * scale).collect();
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in &mut sc {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                for (p, s) in sc.iter().enumerate() {
                    out.iter_mut().zip(&kc[p * row..][..lat]).for_each(|(o, v)| *o += s / sum * v);
                }
            });
            // Up through v_b per head: (T x lat) x v_b[h] -> T x vd.
            let mut o_in = vec![0f32; t_n * nh * vd];
            for hi in 0..nh {
                let a: Vec<f32> = (0..t_n).flat_map(|ti| att[(ti * nh + hi) * lat..][..lat].iter().copied()).collect();
                let v = self.mm(&l.v_b, hi * vd..(hi + 1) * vd, &a);
                for ti in 0..t_n {
                    o_in[(ti * nh + hi) * vd..][..vd].copy_from_slice(&v[ti * vd..][..vd]);
                }
            }
            let o = self.mm_all(&l.wo, &o_in);
            x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);

            // --- FFN ---
            let mut h = x.clone();
            h.chunks_exact_mut(e).for_each(|r| rmsnorm(r, g.f32s(&l.ffn_norm), self.eps));
            let ffn = if let Some([gt, up, dn]) = &l.dense {
                let ff = gt.dims[1] as usize;
                self.swiglu([gt, up, dn], 0..ff, 0..e, &h)
            } else {
                let m = l.moe.as_ref().unwrap();
                let ff = self.n_ff_exp;
                let mut out = self.swiglu([&m.shared[0], &m.shared[1], &m.shared[2]], 0..ff, 0..e, &h);
                let logits = self.mm_all(&m.router, &h);
                let mut assign: Vec<Vec<(usize, f32)>> = vec![vec![]; self.n_expert];
                for (ti, sel) in route_sigmoid(&logits, g.f32s(&m.bias), self.n_expert_used, self.w_scale).into_iter().enumerate() {
                    sel.into_iter().for_each(|(i, w)| assign[i].push((ti, w)));
                }
                for (ex, toks) in assign.iter().enumerate().filter(|(_, t)| !t.is_empty()) {
                    let xin: Vec<f32> = toks.iter().flat_map(|&(ti, _)| h[ti * e..][..e].iter().copied()).collect();
                    let d = self.swiglu([&m.gate, &m.up, &m.down], ex * ff..(ex + 1) * ff, ex * e..(ex + 1) * e, &xin);
                    for (j, &(ti, w)) in toks.iter().enumerate() {
                        out[ti * e..][..e].iter_mut().zip(&d[j * e..][..e]).for_each(|(a, b)| *a += w * b);
                    }
                }
                out
            };
            x.iter_mut().zip(&ffn).for_each(|(a, b)| *a += b);
        }
        self.n_past += t_n;

        x.chunks_exact_mut(e).for_each(|r| rmsnorm(r, g.f32s(&self.out_norm), self.eps));
        self.mm_all(&self.output, &x)
    }
}
