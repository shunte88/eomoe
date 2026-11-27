//! gpt-oss CPU reference forward (mirrors llama.cpp src/models/openai-moe.cpp). f32 activations,
//! f16 KV cache, like cpu.rs for qwen3moe. Differences from qwen3moe: biases on q/k/v/o, the
//! router and every expert matrix; no QK-norm; YaRN NeoX rope; per-head attention sinks; even
//! layers use a 128-token sliding window; top-k by router logit, softmax over the chosen k;
//! clamped SwiGLU (alpha 1.702, limit 7, up + 1); a post-attention norm feeds the MoE.

use crate::cpu::rmsnorm;
use crate::gguf::{Gguf, TensorInfo};
use crate::quant::{dequant_row, dot, matmul};
use crate::qwen3moe::Engine;
use rayon::prelude::*;

struct Layer {
    attn_norm: TensorInfo,
    q: (TensorInfo, TensorInfo),
    k: (TensorInfo, TensorInfo),
    v: (TensorInfo, TensorInfo),
    o: (TensorInfo, TensorInfo),
    sinks: TensorInfo,
    post_norm: TensorInfo,
    router: (TensorInfo, TensorInfo),
    gate: (TensorInfo, TensorInfo),
    up: (TensorInfo, TensorInfo),
    down: (TensorInfo, TensorInfo),
}

pub struct CpuModel {
    g: Gguf,
    layers: Vec<Layer>,
    tok_embd: TensorInfo,
    out_norm: TensorInfo,
    output: TensorInfo,
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    n_expert: usize,
    n_expert_used: usize,
    n_ff: usize,
    eps: f32,
    swa: usize,
    /// Per rotated pair i: angle per position (YaRN-blended), and the cos/sin magnitude.
    rope_freq: Vec<f32>,
    rope_mscale: f32,
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    n_past: usize,
}

impl CpuModel {
    pub fn new(g: Gguf) -> Result<Self, String> {
        let arch = g.str("general.architecture").unwrap_or("");
        if arch != "gpt-oss" {
            return Err(format!("not a gpt-oss model: {arch:?}"));
        }
        let u = |k: &str| g.u64(&format!("gpt-oss.{k}")).map(|v| v as usize).ok_or(format!("missing gpt-oss.{k}"));
        let f = |k: &str| g.f64(&format!("gpt-oss.{k}")).map(|v| v as f32).ok_or(format!("missing gpt-oss.{k}"));
        let t = |n: &str| g.tensor(n).cloned().ok_or(format!("missing tensor {n}"));
        let head_dim = u("attention.key_length")?;
        let base = f("rope.freq_base")?;

        // YaRN (ggml rope_yarn with ext_factor 1): pair i interpolates (angle / factor) below the
        // correction range, extrapolates (plain angle) above it, ramps in between; cos/sin carry
        // mscale = 1 + 0.1 ln(factor) (llama.cpp's attn_factor cancels to 1 for gpt-oss).
        let factor = f("rope.scaling.factor")?;
        let n_orig = u("rope.scaling.original_context_length")? as f32;
        let corr = |beta: f32| head_dim as f32 * (n_orig / (beta * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln());
        let lo = corr(f("rope.scaling.yarn_beta_fast")?).floor().max(0.0);
        let hi = corr(f("rope.scaling.yarn_beta_slow")?).ceil().min(head_dim as f32 - 1.0);
        let rope_freq = (0..head_dim / 2)
            .map(|i| {
                let ramp = 1.0 - ((i as f32 - lo) / (hi - lo).max(0.001)).clamp(0.0, 1.0);
                base.powf(-2.0 * i as f32 / head_dim as f32) * (ramp + (1.0 - ramp) / factor)
            })
            .collect();

        let n_layer = u("block_count")?;
        let layers = (0..n_layer)
            .map(|i| {
                let w = |s: &str| t(&format!("blk.{i}.{s}.weight"));
                let wb = |s: &str| Ok::<_, String>((w(s)?, t(&format!("blk.{i}.{s}.bias"))?));
                Ok(Layer {
                    attn_norm: w("attn_norm")?,
                    q: wb("attn_q")?,
                    k: wb("attn_k")?,
                    v: wb("attn_v")?,
                    o: wb("attn_output")?,
                    sinks: w("attn_sinks")?,
                    post_norm: w("post_attention_norm")?,
                    router: wb("ffn_gate_inp")?,
                    gate: wb("ffn_gate_exps")?,
                    up: wb("ffn_up_exps")?,
                    down: wb("ffn_down_exps")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            tok_embd: t("token_embd.weight")?,
            out_norm: t("output_norm.weight")?,
            output: t("output.weight")?,
            n_embd: u("embedding_length")?,
            n_head: u("attention.head_count")?,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff: u("expert_feed_forward_length")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            swa: u("attention.sliding_window")?,
            rope_freq,
            rope_mscale: 1.0 + 0.1 * factor.ln(),
            kv: (0..n_layer).map(|_| (vec![], vec![])).collect(),
            layers,
            n_past: 0,
            g,
        })
    }

    /// x (T x k) times rows [r0, r1) of `w`, plus the matching slice of `b`.
    fn linear(&self, (w, b): &(TensorInfo, TensorInfo), r: std::ops::Range<usize>, x: &[f32]) -> Vec<f32> {
        let k = w.dims[0] as usize;
        let rb = w.row_bytes();
        let mut out = vec![0f32; x.len() / k * r.len()];
        matmul(w.ty, &self.g.data(w)[r.start * rb..r.end * rb], k, x, &mut out);
        let bias = &self.g.f32s(b)[r.clone()];
        out.chunks_exact_mut(r.len()).for_each(|o| o.iter_mut().zip(bias).for_each(|(a, b)| *a += b));
        out
    }

    fn rope(&self, v: &mut [f32], pos: usize) {
        let half = self.head_dim / 2;
        for (i, f) in self.rope_freq.iter().enumerate() {
            let (s, c) = (pos as f32 * f).sin_cos();
            let (s, c) = (s * self.rope_mscale, c * self.rope_mscale);
            let (a, b) = (v[i], v[i + half]);
            v[i] = a * c - b * s;
            v[i + half] = a * s + b * c;
        }
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
        let kvd = self.n_head_kv * self.head_dim;
        for (k, v) in &mut self.kv {
            k.truncate(n_past * kvd);
            v.truncate(n_past * kvd);
        }
        self.n_past = self.n_past.min(n_past);
    }

    fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        let (e, hd, nh, nkv, ff) = (self.n_embd, self.head_dim, self.n_head, self.n_head_kv, self.n_ff);
        let t_n = tokens.len();
        let g = &self.g;

        let mut x = vec![0f32; t_n * e];
        for (row, &tok) in x.chunks_exact_mut(e).zip(tokens) {
            dequant_row(self.tok_embd.ty, g.row(&self.tok_embd, tok as usize), row);
        }

        for li in 0..self.layers.len() {
            let l = &self.layers[li];

            // --- attention ---
            let mut h = x.clone();
            h.chunks_exact_mut(e).for_each(|r| rmsnorm(r, g.f32s(&l.attn_norm), self.eps));
            let mut q = self.linear(&l.q, 0..nh * hd, &h);
            let mut k = self.linear(&l.k, 0..nkv * hd, &h);
            let v = self.linear(&l.v, 0..nkv * hd, &h);
            for (ti, (qr, kr)) in q.chunks_exact_mut(nh * hd).zip(k.chunks_exact_mut(nkv * hd)).enumerate() {
                qr.chunks_exact_mut(hd).for_each(|hv| self.rope(hv, self.n_past + ti));
                kr.chunks_exact_mut(hd).for_each(|hv| self.rope(hv, self.n_past + ti));
            }
            let f16 = |x: &f32| half::f16::from_f32(*x).to_f32();
            let (kc, vc) = &mut self.kv[li];
            kc.extend(k.iter().map(f16));
            vc.extend(v.iter().map(f16));
            let (kc, vc) = (&self.kv[li].0, &self.kv[li].1);

            // Even layers see only the last `swa` positions (llama.cpp swa pattern 2, standard).
            let window = if li % 2 == 0 { self.swa } else { usize::MAX };
            let sinks = g.f32s(&l.sinks);
            let scale = 1.0 / (hd as f32).sqrt();
            let n_past = self.n_past;
            let mut attn = vec![0f32; t_n * nh * hd];
            attn.par_chunks_mut(hd).enumerate().for_each(|(idx, out)| {
                let (ti, hq) = (idx / nh, idx % nh);
                let hk = hq / (nh / nkv);
                let qv = &q[(ti * nh + hq) * hd..][..hd];
                let pos = n_past + ti;
                let p0 = (pos + 1).saturating_sub(window);
                let mut sc: Vec<f32> = (p0..=pos).map(|p| dot(qv, &kc[(p * nkv + hk) * hd..][..hd]) * scale).collect();
                // The sink joins the softmax denominator but has no value row.
                let m = sc.iter().cloned().fold(sinks[hq], f32::max);
                let mut sum = (sinks[hq] - m).exp();
                for s in &mut sc {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                for (p, s) in (p0..=pos).zip(&sc) {
                    let vv = &vc[(p * nkv + hk) * hd..][..hd];
                    out.iter_mut().zip(vv).for_each(|(o, vx)| *o += s / sum * vx);
                }
            });
            let o = self.linear(&l.o, 0..e, &attn);
            x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);

            // --- MoE FFN ---
            let mut h = x.clone();
            h.chunks_exact_mut(e).for_each(|r| rmsnorm(r, g.f32s(&l.post_norm), self.eps));
            let logits = self.linear(&l.router, 0..self.n_expert, &h);
            let mut assign: Vec<Vec<(usize, f32)>> = vec![vec![]; self.n_expert];
            for (ti, lr) in logits.chunks_exact(self.n_expert).enumerate() {
                let mut idx: Vec<usize> = (0..self.n_expert).collect();
                idx.sort_by(|a, b| lr[*b].total_cmp(&lr[*a]));
                let top = &idx[..self.n_expert_used];
                let m = lr[top[0]];
                let s: f32 = top.iter().map(|&i| (lr[i] - m).exp()).sum();
                top.iter().for_each(|&i| assign[i].push((ti, (lr[i] - m).exp() / s)));
            }
            let mut moe = vec![0f32; t_n * e];
            for (ex, toks) in assign.iter().enumerate().filter(|(_, t)| !t.is_empty()) {
                let xin: Vec<f32> = toks.iter().flat_map(|&(ti, _)| h[ti * e..][..e].iter().copied()).collect();
                let gate = self.linear(&l.gate, ex * ff..(ex + 1) * ff, &xin);
                let up = self.linear(&l.up, ex * ff..(ex + 1) * ff, &xin);
                let act: Vec<f32> = gate
                    .iter()
                    .zip(&up)
                    .map(|(&g, &u)| {
                        let (g, u) = (g.min(7.0), u.clamp(-7.0, 7.0));
                        g / (1.0 + (-1.702 * g).exp()) * (u + 1.0)
                    })
                    .collect();
                let down = self.linear(&l.down, ex * e..(ex + 1) * e, &act);
                for (j, &(ti, wgt)) in toks.iter().enumerate() {
                    moe[ti * e..][..e].iter_mut().zip(&down[j * e..][..e]).for_each(|(a, b)| *a += wgt * b);
                }
            }
            x.iter_mut().zip(&moe).for_each(|(a, b)| *a += b);
        }
        self.n_past += t_n;

        x.chunks_exact_mut(e).for_each(|r| rmsnorm(r, g.f32s(&self.out_norm), self.eps));
        let k = self.output.dims[0] as usize;
        let mut out = vec![0f32; t_n * self.n_vocab()];
        matmul(self.output.ty, g.data(&self.output), k, &x, &mut out);
        out
    }
}
