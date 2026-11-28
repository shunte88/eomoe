//! CPU reference forward pass for qwen3moe (mirrors llama.cpp src/models/qwen3moe.cpp).
//! f32 activations and KV cache. This is the correctness baseline the CUDA engine is
//! checked against, and later the CPU fallback for experts missing from VRAM.

use crate::gguf::{Gguf, TensorInfo};
use crate::quant::{dequant_row, dot, matmul};
use crate::qwen3moe::{route, Attn, Engine, Weights};
use rayon::prelude::*;

pub struct CpuModel {
    pub w: Weights,
    /// Per layer: K and V, each n_past x (n_head_kv * head_dim).
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    n_past: usize,
}

pub fn rmsnorm(x: &mut [f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    for (v, w) in x.iter_mut().zip(w) {
        *v = *v * s * w;
    }
}

impl CpuModel {
    pub fn new(g: Gguf) -> Result<Self, String> {
        let w = Weights::load(g)?;
        if w.hp.oai {
            return Err("gpt-oss: use gptoss::CpuModel".into());
        }
        if w.hp.glm.is_some() {
            return Err("GLM: use glm::CpuModel".into());
        }
        if w.hp.qwen35.is_some() {
            return Err("Qwen3.6: use qwen35::CpuModel".into());
        }
        Ok(Self { kv: (0..w.hp.n_layer).map(|_| (vec![], vec![])).collect(), w, n_past: 0 })
    }

    fn mm(&self, t: &TensorInfo, x: &[f32]) -> Vec<f32> {
        let k = t.dims[0] as usize;
        let n = t.size / t.row_bytes();
        let mut out = vec![0f32; x.len() / k * n];
        matmul(t.ty, self.w.g.data(t), k, x, &mut out);
        out
    }
}

impl Engine for CpuModel {
    fn n_vocab(&self) -> usize {
        self.w.hp.n_vocab
    }

    fn n_ctx(&self) -> usize {
        usize::MAX
    }

    fn truncate(&mut self, n_past: usize) {
        let kvd = self.w.hp.n_head_kv * self.w.hp.head_dim;
        for (k, v) in &mut self.kv {
            k.truncate(n_past * kvd);
            v.truncate(n_past * kvd);
        }
        self.n_past = self.n_past.min(n_past);
    }

    fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        let (g, hp) = (&self.w.g, &self.w.hp);
        let (e, hd, nh, nkv) = (hp.n_embd, hp.head_dim, hp.n_head, hp.n_head_kv);
        let t_n = tokens.len();

        let mut x = vec![0f32; t_n * e];
        for (row, &tok) in x.chunks_exact_mut(e).zip(tokens) {
            dequant_row(self.w.tok_embd.ty, g.row(&self.w.tok_embd, tok as usize), row);
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
            let l = &self.w.layers[li];

            // --- attention ---
            let mut h = x.clone();
            let w = g.f32s(&l.attn_norm);
            h.chunks_exact_mut(e).for_each(|r| rmsnorm(r, w, hp.eps));
            let Attn::Gqa { wq, wk, wv } = &l.attn else { unreachable!("qwen3moe only") };
            let (mut q, mut k, v) = (self.mm(wq, &h), self.mm(wk, &h), self.mm(wv, &h));
            let (qn, kn) = (g.f32s(l.q_norm.as_ref().unwrap()), g.f32s(l.k_norm.as_ref().unwrap()));
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
            // The KV cache holds f16, as llama.cpp's default (and the GPU engine) does.
            let f16 = |x: &f32| half::f16::from_f32(*x).to_f32();
            let (kc, vc) = &mut self.kv[li];
            kc.extend(k.iter().map(f16));
            vc.extend(v.iter().map(f16));
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
            let w = g.f32s(&l.ffn_norm);
            h.chunks_exact_mut(e).for_each(|r| rmsnorm(r, w, hp.eps));
            let m = l.moe.as_ref().expect("qwen3moe: every layer is MoE");
            let logits = self.mm(&m.router, &h); // T x n_expert
            let mut assign: Vec<Vec<(usize, f32)>> = vec![vec![]; hp.n_expert]; // expert -> (token, weight)
            for (ti, sel) in route(&logits, hp.n_expert, hp.n_expert_used).into_iter().enumerate() {
                for (ex, wgt) in sel {
                    assign[ex].push((ti, wgt));
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
                let [gb, ub, db] = self.w.expert(l, ex)[..] else { unreachable!() };
                let mut g_ = vec![0f32; n * ff];
                let mut u_ = vec![0f32; n * ff];
                matmul(m.gate.ty, gb, e, &xin, &mut g_);
                matmul(m.up.ty, ub, e, &xin, &mut u_);
                for (gv, uv) in g_.iter_mut().zip(&u_) {
                    *gv = *gv / (1.0 + (-*gv).exp()) * uv; // silu(gate) * up
                }
                let mut d_ = vec![0f32; n * e];
                matmul(m.down.ty, db, ff, &g_, &mut d_);
                for (j, &(ti, wgt)) in toks.iter().enumerate() {
                    for (a, b) in moe[ti * e..(ti + 1) * e].iter_mut().zip(&d_[j * e..(j + 1) * e]) {
                        *a += wgt * b;
                    }
                }
            }
            x.iter_mut().zip(&moe).for_each(|(a, b)| *a += b);
        }
        self.n_past += t_n;

        let w = g.f32s(&self.w.out_norm);
        x.chunks_exact_mut(e).for_each(|r| rmsnorm(r, w, hp.eps));
        self.mm(&self.w.output, &x)
    }
}
