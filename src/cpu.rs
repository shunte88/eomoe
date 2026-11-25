//! CPU reference forward pass for qwen3moe (mirrors llama.cpp src/models/qwen3moe.cpp).
//! f32 activations and KV cache. This is the correctness baseline the CUDA engine is
//! checked against, and later the CPU fallback for experts missing from VRAM.

use crate::gguf::{Gguf, TensorInfo};
use crate::quant::{dequant_row, dot, matmul};
use rayon::prelude::*;

pub struct Hparams {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub n_vocab: usize,
}

struct Layer {
    attn_norm: TensorInfo,
    wq: TensorInfo,
    wk: TensorInfo,
    wv: TensorInfo,
    wo: TensorInfo,
    q_norm: TensorInfo,
    k_norm: TensorInfo,
    ffn_norm: TensorInfo,
    router: TensorInfo,
    gate: TensorInfo,
    up: TensorInfo,
    down: TensorInfo,
}

pub struct CpuModel {
    g: Gguf,
    pub hp: Hparams,
    layers: Vec<Layer>,
    tok_embd: TensorInfo,
    out_norm: TensorInfo,
    output: TensorInfo,
    /// Per layer: K and V, each n_past x (n_head_kv * head_dim).
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    n_past: usize,
}

fn rmsnorm(x: &mut [f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    for (v, w) in x.iter_mut().zip(w) {
        *v = *v * s * w;
    }
}

impl CpuModel {
    pub fn new(g: Gguf) -> Result<Self, String> {
        let arch = g.str("general.architecture").unwrap_or("");
        if arch != "qwen3moe" {
            return Err(format!("unsupported architecture {arch:?}"));
        }
        let u = |k: &str| g.u64(&format!("qwen3moe.{k}")).map(|v| v as usize).ok_or(format!("missing qwen3moe.{k}"));
        let n_embd = u("embedding_length")?;
        let n_head = u("attention.head_count")?;
        let hp = Hparams {
            n_layer: u("block_count")?,
            n_embd,
            n_head,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length").unwrap_or(n_embd / n_head),
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            eps: g.f64("qwen3moe.attention.layer_norm_rms_epsilon").ok_or("missing rms eps")? as f32,
            rope_base: g.f64("qwen3moe.rope.freq_base").unwrap_or(10000.0) as f32,
            n_vocab: 0,
        };
        let t = |n: String| g.tensor(&n).cloned().ok_or(format!("missing tensor {n}"));
        let layers = (0..hp.n_layer)
            .map(|i| {
                let b = |s: &str| t(format!("blk.{i}.{s}.weight"));
                Ok(Layer {
                    attn_norm: b("attn_norm")?,
                    wq: b("attn_q")?,
                    wk: b("attn_k")?,
                    wv: b("attn_v")?,
                    wo: b("attn_output")?,
                    q_norm: b("attn_q_norm")?,
                    k_norm: b("attn_k_norm")?,
                    ffn_norm: b("ffn_norm")?,
                    router: b("ffn_gate_inp")?,
                    gate: b("ffn_gate_exps")?,
                    up: b("ffn_up_exps")?,
                    down: b("ffn_down_exps")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let tok_embd = t("token_embd.weight".into())?;
        let output = t("output.weight".into()).or_else(|_| t("token_embd.weight".into()))?;
        let hp = Hparams { n_vocab: output.dims[1] as usize, ..hp };
        Ok(Self {
            out_norm: t("output_norm.weight".into())?,
            kv: (0..hp.n_layer).map(|_| (vec![], vec![])).collect(),
            g,
            hp,
            layers,
            tok_embd,
            output,
            n_past: 0,
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.hp.n_vocab
    }

    pub fn reset(&mut self) {
        for (k, v) in &mut self.kv {
            k.clear();
            v.clear();
        }
        self.n_past = 0;
    }

    /// Bytes of rows [r0, r1) of a (possibly 3-D, flattened) tensor.
    fn rows(&self, t: &TensorInfo, r0: usize, r1: usize) -> &[u8] {
        let rb = t.row_bytes();
        &self.g.data(t)[r0 * rb..r1 * rb]
    }

    fn mm(&self, t: &TensorInfo, x: &[f32]) -> Vec<f32> {
        let k = t.dims[0] as usize;
        let n = t.size / t.row_bytes();
        let mut out = vec![0f32; x.len() / k * n];
        matmul(t.ty, self.g.data(t), k, x, &mut out);
        out
    }

    /// Run `tokens` at positions n_past.. and return logits for every token (T x n_vocab).
    pub fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        let hp = &self.hp;
        let (e, hd, nh, nkv) = (hp.n_embd, hp.head_dim, hp.n_head, hp.n_head_kv);
        let t_n = tokens.len();

        let mut x = vec![0f32; t_n * e];
        for (row, &tok) in x.chunks_exact_mut(e).zip(tokens) {
            dequant_row(self.tok_embd.ty, self.g.row(&self.tok_embd, tok as usize), row);
        }

        // NEOX rope tables: theta_i = base^(-2i/hd), rotating pairs (i, i + hd/2).
        let half = hd / 2;
        let inv_freq: Vec<f32> = (0..half).map(|i| hp.rope_base.powf(-2.0 * i as f32 / hd as f32)).collect();
        let rope = |v: &mut [f32], pos: usize| {
            for i in 0..half {
                let (s, c) = (pos as f32 * inv_freq[i]).sin_cos();
                let (a, b) = (v[i], v[i + half]);
                v[i] = a * c - b * s;
                v[i + half] = a * s + b * c;
            }
        };

        for li in 0..hp.n_layer {
            let l = &self.layers[li];

            // --- attention ---
            let mut h = x.clone();
            let w = self.g.f32s(&l.attn_norm);
            h.chunks_exact_mut(e).for_each(|r| rmsnorm(r, w, hp.eps));
            let mut q = self.mm(&l.wq, &h);
            let mut k = self.mm(&l.wk, &h);
            let v = self.mm(&l.wv, &h);
            let (qn, kn) = (self.g.f32s(&l.q_norm), self.g.f32s(&l.k_norm));
            for (ti, (qr, kr)) in q.chunks_exact_mut(nh * hd).zip(k.chunks_exact_mut(nkv * hd)).enumerate() {
                let pos = self.n_past + ti;
                for hv in qr.chunks_exact_mut(hd) {
                    rmsnorm(hv, qn, hp.eps);
                    rope(hv, pos);
                }
                for hv in kr.chunks_exact_mut(hd) {
                    rmsnorm(hv, kn, hp.eps);
                    rope(hv, pos);
                }
            }
            let (kc, vc) = &mut self.kv[li];
            kc.extend_from_slice(&k);
            vc.extend_from_slice(&v);
            let (kc, vc) = (&*kc, &*vc);

            let scale = 1.0 / (hd as f32).sqrt();
            let group = nh / nkv;
            let n_past = self.n_past;
            let mut attn = vec![0f32; t_n * nh * hd];
            attn.par_chunks_mut(hd).enumerate().for_each(|(idx, out)| {
                let (ti, hq) = (idx / nh, idx % nh);
                let hk = hq / group;
                let qv = &q[(ti * nh + hq) * hd..][..hd];
                let n_ctx = n_past + ti + 1; // causal
                let mut sc: Vec<f32> = (0..n_ctx).map(|p| dot(qv, &kc[(p * nkv + hk) * hd..][..hd]) * scale).collect();
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in &mut sc {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                for (p, s) in sc.iter().enumerate() {
                    let vv = &vc[(p * nkv + hk) * hd..][..hd];
                    let wgt = s / sum;
                    for (o, vx) in out.iter_mut().zip(vv) {
                        *o += wgt * vx;
                    }
                }
            });
            let o = self.mm(&l.wo, &attn);
            x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);

            // --- MoE FFN ---
            let mut h = x.clone();
            let w = self.g.f32s(&l.ffn_norm);
            h.chunks_exact_mut(e).for_each(|r| rmsnorm(r, w, hp.eps));
            let logits = self.mm(&l.router, &h); // T x n_expert
            let mut assign: Vec<Vec<(usize, f32)>> = vec![vec![]; hp.n_expert]; // expert -> (token, weight)
            for (ti, lr) in logits.chunks_exact(hp.n_expert).enumerate() {
                let m = lr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let p: Vec<f32> = lr.iter().map(|v| (v - m).exp()).collect();
                let s: f32 = p.iter().sum();
                let mut idx: Vec<usize> = (0..hp.n_expert).collect();
                idx.sort_by(|a, b| p[*b].total_cmp(&p[*a]));
                let top = &idx[..hp.n_expert_used];
                // softmax then renormalise over the selected experts (norm_w = true).
                let wsum: f32 = top.iter().map(|&i| p[i] / s).sum::<f32>().max(6.103515625e-5);
                for &i in top {
                    assign[i].push((ti, p[i] / s / wsum));
                }
            }
            let ff = hp.n_ff_exp;
            let mut moe = vec![0f32; t_n * e];
            for (ex, toks) in assign.iter().enumerate() {
                if toks.is_empty() {
                    continue;
                }
                let xin: Vec<f32> = toks.iter().flat_map(|&(ti, _)| h[ti * e..(ti + 1) * e].iter().copied()).collect();
                let n = toks.len();
                let mut g_ = vec![0f32; n * ff];
                let mut u_ = vec![0f32; n * ff];
                matmul(l.gate.ty, self.rows(&l.gate, ex * ff, (ex + 1) * ff), e, &xin, &mut g_);
                matmul(l.up.ty, self.rows(&l.up, ex * ff, (ex + 1) * ff), e, &xin, &mut u_);
                for (gv, uv) in g_.iter_mut().zip(&u_) {
                    *gv = *gv / (1.0 + (-*gv).exp()) * uv; // silu(gate) * up
                }
                let mut d_ = vec![0f32; n * e];
                matmul(l.down.ty, self.rows(&l.down, ex * e, (ex + 1) * e), ff, &g_, &mut d_);
                for (j, &(ti, wgt)) in toks.iter().enumerate() {
                    for (a, b) in moe[ti * e..(ti + 1) * e].iter_mut().zip(&d_[j * e..(j + 1) * e]) {
                        *a += wgt * b;
                    }
                }
            }
            x.iter_mut().zip(&moe).for_each(|(a, b)| *a += b);
        }
        self.n_past += t_n;

        let w = self.g.f32s(&self.out_norm);
        x.chunks_exact_mut(e).for_each(|r| rmsnorm(r, w, self.hp.eps));
        self.mm(&self.output, &x)
    }
}
