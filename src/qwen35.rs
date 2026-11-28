//! Qwen3.6-35B-A3B CPU reference forward (llama.cpp arch `qwen35moe`, src/models/qwen35moe.cpp).
//! f32 activations, f16 KV cache on the full-attention layers. Per layer, x += mixer(norm(x));
//! x += ffn(post_attention_norm(x)), where the mixer is one of:
//! - Gated DeltaNet (3 of every 4 layers): qkv -> causal depthwise conv (kernel 4, over the last 3
//!   raw inputs) -> SiLU -> q, k (16 heads, L2-normed), v (32 heads, 128 wide). Each v head h reads
//!   k head h % 16 and keeps a 128x128 state S: S *= exp(g); S += k (x) beta (v - S^T k);
//!   out = S^T q / sqrt(128), with g = softplus(alpha + dt_bias) * ssm_a, beta = sigmoid(beta).
//!   Then rmsnorm(out, ssm_norm) * silu(z) per head, and ssm_out.
//! - Gated attention: attn_q gives [q | gate] per head (256 each); RMS-normed q/k, NEOX rope on the
//!   first 64 dims (IMRoPE with equal text positions), GQA 16/2, output times sigmoid(gate).
//!
//! FFN: 256 routed experts (softmax, top-8, normalised) plus a shared expert scaled by
//! sigmoid(ffn_gate_inp_shexp . x).

use crate::cpu::rmsnorm;
use crate::gguf::{Gguf, TensorInfo, Value};
use crate::quant::{dequant_row, dot, matmul};
use crate::qwen3moe::{glu, route, Engine};
use rayon::prelude::*;

struct Delta {
    qkv: TensorInfo,
    z: TensorInfo,
    beta: TensorInfo,
    alpha: TensorInfo,
    dt_bias: TensorInfo,
    a: TensorInfo,
    conv: TensorInfo,
    norm: TensorInfo,
    out: TensorInfo,
}

struct Attn {
    q: TensorInfo,
    k: TensorInfo,
    v: TensorInfo,
    q_norm: TensorInfo,
    k_norm: TensorInfo,
    wo: TensorInfo,
}

enum Mixer {
    Delta(Box<Delta>),
    Attn(Box<Attn>),
}

struct Layer {
    attn_norm: TensorInfo,
    mixer: Mixer,
    post_norm: TensorInfo,
    router: TensorInfo,
    gate: TensorInfo,
    up: TensorInfo,
    down: TensorInfo,
    shared: [TensorInfo; 3],
    shared_gate: TensorInfo,
}

/// Per-layer sequence state: conv (3 raw qkv rows) + ssm (heads x 128 x 128, stored S^T) for
/// DeltaNet layers, f16-rounded k/v rows for attention layers.
#[derive(Default, Clone)]
struct State {
    conv: Vec<f32>,
    ssm: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
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
    n_rot: usize,
    /// DeltaNet: k heads (group_count), v heads (time_step_rank), head width (state_size), conv kernel.
    nk: usize,
    nv: usize,
    dk: usize,
    d_conv: usize,
    n_expert: usize,
    n_expert_used: usize,
    n_ff_exp: usize,
    eps: f32,
    rope_base: f32,
    state: Vec<State>,
    n_past: usize,
}

impl CpuModel {
    pub fn new(g: Gguf) -> Result<Self, String> {
        let arch = g.str("general.architecture").unwrap_or("");
        if arch != "qwen35moe" {
            return Err(format!("not a qwen35moe model: {arch:?}"));
        }
        let u = |k: &str| g.u64(&format!("qwen35moe.{k}")).map(|v| v as usize).ok_or(format!("missing qwen35moe.{k}"));
        let f = |k: &str| g.f64(&format!("qwen35moe.{k}")).map(|v| v as f32).ok_or(format!("missing qwen35moe.{k}"));
        let t = |n: &str| g.tensor(n).cloned().ok_or(format!("missing tensor {n}"));
        let n_layer = u("block_count")?;
        let recurrent: Vec<bool> = match g.arr("qwen35moe.attention.recurrent_layers") {
            Some(a) => a.iter().map(|v| matches!(v, Value::Bool(true))).collect(),
            None => (0..n_layer).map(|i| (i + 1) % u("full_attention_interval").unwrap_or(4) != 0).collect(),
        };
        let layers = (0..n_layer)
            .map(|i| {
                let w = |s: &str| t(&format!("blk.{i}.{s}.weight"));
                Ok(Layer {
                    attn_norm: w("attn_norm")?,
                    mixer: if recurrent[i] {
                        Mixer::Delta(Box::new(Delta {
                            qkv: w("attn_qkv")?,
                            z: w("attn_gate")?,
                            beta: w("ssm_beta")?,
                            alpha: w("ssm_alpha")?,
                            dt_bias: t(&format!("blk.{i}.ssm_dt.bias"))?,
                            a: t(&format!("blk.{i}.ssm_a"))?,
                            conv: w("ssm_conv1d")?,
                            norm: w("ssm_norm")?,
                            out: w("ssm_out")?,
                        }))
                    } else {
                        Mixer::Attn(Box::new(Attn { q: w("attn_q")?, k: w("attn_k")?, v: w("attn_v")?, q_norm: w("attn_q_norm")?, k_norm: w("attn_k_norm")?, wo: w("attn_output")? }))
                    },
                    post_norm: w("post_attention_norm")?,
                    router: w("ffn_gate_inp")?,
                    gate: w("ffn_gate_exps")?,
                    up: w("ffn_up_exps")?,
                    down: w("ffn_down_exps")?,
                    shared: [w("ffn_gate_shexp")?, w("ffn_up_shexp")?, w("ffn_down_shexp")?],
                    shared_gate: w("ffn_gate_inp_shexp")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            tok_embd: t("token_embd.weight")?,
            out_norm: t("output_norm.weight")?,
            output: t("output.weight").or_else(|_| t("token_embd.weight"))?,
            n_embd: u("embedding_length")?,
            n_head: u("attention.head_count")?,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_rot: u("rope.dimension_count")?,
            nk: u("ssm.group_count")?,
            nv: u("ssm.time_step_rank")?,
            dk: u("ssm.state_size")?,
            d_conv: u("ssm.conv_kernel")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            rope_base: f("rope.freq_base")?,
            state: vec![State::default(); layers.len()],
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

    /// NEOX rope on the first n_rot dims: pairs (i, i + n_rot/2) rotate by pos * base^(-2i/n_rot).
    fn rope(&self, v: &mut [f32], pos: usize) {
        let half = self.n_rot / 2;
        for i in 0..half {
            let (s, c) = (pos as f32 * self.rope_base.powf(-2.0 * i as f32 / self.n_rot as f32)).sin_cos();
            let (a, b) = (v[i], v[i + half]);
            v[i] = a * c - b * s;
            v[i + half] = a * s + b * c;
        }
    }

    fn swiglu(&self, [gate, up, down]: [&TensorInfo; 3], ff: std::ops::Range<usize>, out: std::ops::Range<usize>, x: &[f32]) -> Vec<f32> {
        let (g, u) = (self.mm(gate, ff.clone(), x), self.mm(up, ff, x));
        let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| glu(false, g, u)).collect();
        self.mm(down, out, &act)
    }

    fn delta_net(&self, d: &Delta, st: &mut State, h: &[f32]) -> Vec<f32> {
        let g = &self.g;
        let (nk, nv, dk) = (self.nk, self.nv, self.dk);
        let ch = 2 * nk * dk + nv * dk;
        let t_n = h.len() / self.n_embd;
        let qkv = self.mm_all(&d.qkv, h);
        let z = self.mm_all(&d.z, h);
        let beta = self.mm_all(&d.beta, h);
        let alpha = self.mm_all(&d.alpha, h);

        // Causal depthwise conv over [cached raw rows | new rows], then SiLU.
        let hist = self.d_conv - 1;
        if st.conv.is_empty() {
            st.conv = vec![0.0; hist * ch];
            st.ssm = vec![0.0; nv * dk * dk];
        }
        let buf = [&st.conv[..], &qkv[..]].concat();
        let w = g.f32s(&d.conv);
        let mut c = vec![0f32; t_n * ch];
        for (ti, row) in c.chunks_exact_mut(ch).enumerate() {
            for (j, o) in row.iter_mut().enumerate() {
                let s: f32 = (0..self.d_conv).map(|k| buf[(ti + k) * ch + j] * w[j * self.d_conv + k]).sum();
                *o = s / (1.0 + (-s).exp());
            }
            // L2-norm q and k per head: x / sqrt(sum x^2 + eps).
            for hd in row[..2 * nk * dk].chunks_exact_mut(dk) {
                let s = 1.0 / (hd.iter().map(|v| v * v).sum::<f32>() + self.eps).sqrt();
                hd.iter_mut().for_each(|v| *v *= s);
            }
        }
        st.conv = buf[t_n * ch..].to_vec();

        // Gated delta rule, one v head per task. m[j*dk + i] = S[i][j].
        let (dt, a) = (g.f32s(&d.dt_bias), g.f32s(&d.a));
        let scale = 1.0 / (dk as f32).sqrt();
        let heads: Vec<Vec<f32>> = st
            .ssm
            .par_chunks_mut(dk * dk)
            .enumerate()
            .map(|(hi, m)| {
                let kh = hi % nk;
                let mut out = vec![0f32; t_n * dk];
                let mut delta = vec![0f32; dk];
                for ti in 0..t_n {
                    let row = &c[ti * ch..][..ch];
                    let (q, k, v) = (&row[kh * dk..][..dk], &row[(nk + kh) * dk..][..dk], &row[(2 * nk + hi) * dk..][..dk]);
                    let x = alpha[ti * nv + hi] + dt[hi];
                    let sp = if x > 20.0 { x } else { x.exp().ln_1p() };
                    let decay = (sp * a[hi]).exp();
                    let b = 1.0 / (1.0 + (-beta[ti * nv + hi]).exp());
                    m.iter_mut().for_each(|s| *s *= decay);
                    for j in 0..dk {
                        delta[j] = (v[j] - dot(&m[j * dk..][..dk], k)) * b;
                    }
                    for j in 0..dk {
                        m[j * dk..][..dk].iter_mut().zip(k).for_each(|(s, k)| *s += delta[j] * k);
                    }
                    for j in 0..dk {
                        out[ti * dk + j] = dot(&m[j * dk..][..dk], q) * scale;
                    }
                }
                out
            })
            .collect();
        let mut o = vec![0f32; t_n * nv * dk];
        for (hi, ho) in heads.iter().enumerate() {
            for ti in 0..t_n {
                let dst = &mut o[(ti * nv + hi) * dk..][..dk];
                dst.copy_from_slice(&ho[ti * dk..][..dk]);
                rmsnorm(dst, g.f32s(&d.norm), self.eps);
                dst.iter_mut().zip(&z[(ti * nv + hi) * dk..][..dk]).for_each(|(v, z)| *v *= z / (1.0 + (-z).exp()));
            }
        }
        self.mm_all(&d.out, &o)
    }

    fn attention(&self, at: &Attn, st: &mut State, h: &[f32]) -> Vec<f32> {
        let g = &self.g;
        let (nh, nkv, hd) = (self.n_head, self.n_head_kv, self.head_dim);
        let t_n = h.len() / self.n_embd;
        let mut qg = self.mm_all(&at.q, h); // T x nh x [q | gate]
        let mut k = self.mm_all(&at.k, h);
        let v = self.mm_all(&at.v, h);
        for ti in 0..t_n {
            let pos = self.n_past + ti;
            for q in qg[ti * nh * 2 * hd..][..nh * 2 * hd].chunks_exact_mut(2 * hd) {
                rmsnorm(&mut q[..hd], g.f32s(&at.q_norm), self.eps);
                self.rope(&mut q[..hd], pos);
            }
            for kr in k[ti * nkv * hd..][..nkv * hd].chunks_exact_mut(hd) {
                rmsnorm(kr, g.f32s(&at.k_norm), self.eps);
                self.rope(kr, pos);
            }
        }
        let f16 = |x: &f32| half::f16::from_f32(*x).to_f32();
        st.k.extend(k.iter().map(f16));
        st.v.extend(v.iter().map(f16));
        let (kc, vc) = (&st.k, &st.v);

        let scale = 1.0 / (hd as f32).sqrt();
        let (n_past, grp) = (self.n_past, nh / nkv);
        let mut att = vec![0f32; t_n * nh * hd];
        att.par_chunks_mut(hd).enumerate().for_each(|(idx, out)| {
            let (ti, hi) = (idx / nh, idx % nh);
            let q = &qg[(ti * nh + hi) * 2 * hd..][..2 * hd];
            let kv = hi / grp;
            let mut sc: Vec<f32> = (0..=n_past + ti).map(|p| dot(&q[..hd], &kc[(p * nkv + kv) * hd..][..hd]) * scale).collect();
            let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in &mut sc {
                *s = (*s - m).exp();
                sum += *s;
            }
            for (p, s) in sc.iter().enumerate() {
                out.iter_mut().zip(&vc[(p * nkv + kv) * hd..][..hd]).for_each(|(o, v)| *o += s / sum * v);
            }
            out.iter_mut().zip(&q[hd..]).for_each(|(o, gt)| *o /= 1.0 + (-gt).exp());
        });
        self.mm_all(&at.wo, &att)
    }
}

impl Engine for CpuModel {
    fn n_vocab(&self) -> usize {
        self.output.dims[1] as usize
    }

    fn n_ctx(&self) -> usize {
        usize::MAX
    }

    fn rewind(&mut self, n: usize) -> usize {
        let n = if n < self.n_past { 0 } else { n };
        self.truncate(n);
        n
    }

    /// Recurrent state can't rewind: only "keep everything" or a full reset.
    fn truncate(&mut self, n_past: usize) {
        if n_past >= self.n_past {
            return;
        }
        assert_eq!(n_past, 0, "qwen35 CPU reference: recurrent state can only be reset");
        self.state.iter_mut().for_each(|s| *s = State::default());
        self.n_past = 0;
    }

    fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        let e = self.n_embd;
        let t_n = tokens.len();
        let g = &self.g;

        let mut x = vec![0f32; t_n * e];
        for (r, &tok) in x.chunks_exact_mut(e).zip(tokens) {
            dequant_row(self.tok_embd.ty, g.row(&self.tok_embd, tok as usize), r);
        }

        let mut state = std::mem::take(&mut self.state);
        for (l, st) in self.layers.iter().zip(&mut state) {
            let mut h = x.clone();
            h.chunks_exact_mut(e).for_each(|r| rmsnorm(r, g.f32s(&l.attn_norm), self.eps));
            let o = match &l.mixer {
                Mixer::Delta(d) => self.delta_net(d, st, &h),
                Mixer::Attn(a) => self.attention(a, st, &h),
            };
            x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);

            // --- FFN ---
            let mut h = x.clone();
            h.chunks_exact_mut(e).for_each(|r| rmsnorm(r, g.f32s(&l.post_norm), self.eps));
            let ff = self.n_ff_exp;
            let mut out = self.swiglu([&l.shared[0], &l.shared[1], &l.shared[2]], 0..l.shared[0].dims[1] as usize, 0..e, &h);
            let sg = g.f32s(&l.shared_gate);
            for (o, hr) in out.chunks_exact_mut(e).zip(h.chunks_exact(e)) {
                let s = 1.0 / (1.0 + (-dot(sg, hr)).exp());
                o.iter_mut().for_each(|v| *v *= s);
            }
            let logits = self.mm_all(&l.router, &h);
            let mut assign: Vec<Vec<(usize, f32)>> = vec![vec![]; self.n_expert];
            for (ti, sel) in route(&logits, self.n_expert, self.n_expert_used).into_iter().enumerate() {
                sel.into_iter().for_each(|(i, w)| assign[i].push((ti, w)));
            }
            for (ex, toks) in assign.iter().enumerate().filter(|(_, t)| !t.is_empty()) {
                let xin: Vec<f32> = toks.iter().flat_map(|&(ti, _)| h[ti * e..][..e].iter().copied()).collect();
                let d = self.swiglu([&l.gate, &l.up, &l.down], ex * ff..(ex + 1) * ff, ex * e..(ex + 1) * e, &xin);
                for (j, &(ti, w)) in toks.iter().enumerate() {
                    out[ti * e..][..e].iter_mut().zip(&d[j * e..][..e]).for_each(|(a, b)| *a += w * b);
                }
            }
            x.iter_mut().zip(&out).for_each(|(a, b)| *a += b);
        }
        self.state = state;
        self.n_past += t_n;

        x.chunks_exact_mut(e).for_each(|r| rmsnorm(r, g.f32s(&self.out_norm), self.eps));
        self.mm_all(&self.output, &x)
    }
}
