//! qwen3moe weight map and routing, shared by the CPU reference and the CUDA engine
//! (mirrors llama.cpp src/models/qwen3moe.cpp). The CUDA engine also loads gpt-oss through it
//! (`Hparams::oai`): biases, sinks, sliding window, YaRN rope, see src/gptoss.rs.

use crate::gguf::{Gguf, TensorInfo};

/// A forward-pass engine: run tokens at positions n_past.. and return T x n_vocab logits.
pub trait Engine {
    fn n_vocab(&self) -> usize;
    /// Positions the KV cache can hold.
    fn n_ctx(&self) -> usize;
    /// Keep the first `n_past` positions of the KV cache, drop the rest.
    fn truncate(&mut self, n_past: usize);
    fn reset(&mut self) {
        self.truncate(0)
    }
    /// Sequences the KV cache can hold at once. They share the `n_ctx` positions.
    fn n_seq(&self) -> usize {
        1
    }
    /// KV is allocated to sequences in pages of this many positions.
    fn kv_page(&self) -> usize {
        1
    }
    /// Make `seq` (< n_seq) the sequence `forward` and `truncate` act on.
    fn select(&mut self, seq: usize) {
        assert_eq!(seq, 0, "single-sequence engine");
    }
    /// One token for each of several sequences, in one pass where the engine can:
    /// rows (seq, token) -> rows x n_vocab logits.
    fn decode_batch(&mut self, rows: &[(usize, u32)]) -> Vec<f32> {
        let mut out = vec![];
        for &(s, t) in rows {
            self.select(s);
            out.extend(self.forward_last(&[t]));
        }
        out
    }
    /// Copy of the selected sequence's whole KV cache, for `kv_restore`. Empty: unsupported.
    fn kv_save(&mut self) -> Vec<u8> {
        vec![]
    }
    /// Load the first `n` positions of a `kv_save` image into the selected, empty sequence.
    fn kv_restore(&mut self, _data: &[u8], _n: usize) {
        unimplemented!("kv_restore without kv_save")
    }
    fn forward(&mut self, tokens: &[u32]) -> Vec<f32>;
    /// Logits of the last token only: prefill without T x n_vocab logits.
    fn forward_last(&mut self, tokens: &[u32]) -> Vec<f32> {
        let (l, nv) = (self.forward(tokens), self.n_vocab());
        l[l.len() - nv..].to_vec()
    }
}

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
    /// gpt-oss: top-k by router logit then softmax over the k, clamped SwiGLU, biases, sinks.
    pub oai: bool,
    /// Sliding window of the even layers (0: none).
    pub swa: usize,
    /// YaRN per-pair rope frequencies and cos/sin scale (None: base^(-2i/hd), scale 1).
    pub rope_yarn: Option<(Vec<f32>, f32)>,
}

pub struct Layer {
    pub attn_norm: TensorInfo,
    pub wq: TensorInfo,
    pub wk: TensorInfo,
    pub wv: TensorInfo,
    pub wo: TensorInfo,
    /// qwen3moe only.
    pub q_norm: Option<TensorInfo>,
    pub k_norm: Option<TensorInfo>,
    /// The MoE input norm (gpt-oss: post_attention_norm).
    pub ffn_norm: TensorInfo,
    /// gpt-oss only: biases of q, k, v, o, router, gate/up/down experts; attention sinks.
    pub bias: Option<[TensorInfo; 8]>,
    pub sinks: Option<TensorInfo>,
    pub router: TensorInfo,
    pub gate: TensorInfo,
    pub up: TensorInfo,
    pub down: TensorInfo,
}

pub struct Weights {
    pub g: Gguf,
    pub hp: Hparams,
    pub layers: Vec<Layer>,
    pub tok_embd: TensorInfo,
    pub out_norm: TensorInfo,
    pub output: TensorInfo,
}

impl Weights {
    pub fn load(g: Gguf) -> Result<Self, String> {
        let arch = g.str("general.architecture").unwrap_or("").to_string();
        if arch != "qwen3moe" && arch != "gpt-oss" {
            return Err(format!("unsupported architecture {arch:?}"));
        }
        let oai = arch == "gpt-oss";
        let u = |k: &str| g.u64(&format!("{arch}.{k}")).map(|v| v as usize).ok_or(format!("missing {arch}.{k}"));
        let n_embd = u("embedding_length")?;
        let n_head = u("attention.head_count")?;
        let t = |n: String| g.tensor(&n).cloned().ok_or(format!("missing tensor {n}"));
        let output = t("output.weight".into()).or_else(|_| t("token_embd.weight".into()))?;
        let hp = Hparams {
            n_layer: u("block_count")?,
            n_embd,
            n_head,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length").unwrap_or(n_embd / n_head),
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            eps: g.f64(&format!("{arch}.attention.layer_norm_rms_epsilon")).ok_or("missing rms eps")? as f32,
            rope_base: g.f64(&format!("{arch}.rope.freq_base")).unwrap_or(10000.0) as f32,
            n_vocab: output.dims[1] as usize,
            oai,
            swa: if oai { u("attention.sliding_window")? } else { 0 },
            rope_yarn: if oai { Some(crate::gptoss::yarn(&g)?) } else { None },
        };
        let layers = (0..hp.n_layer)
            .map(|i| {
                let b = |s: &str| t(format!("blk.{i}.{s}.weight"));
                let bias = |s: &str| t(format!("blk.{i}.{s}.bias"));
                let qwen = |s: &str| if oai { Ok(None) } else { b(s).map(Some) };
                Ok(Layer {
                    attn_norm: b("attn_norm")?,
                    wq: b("attn_q")?,
                    wk: b("attn_k")?,
                    wv: b("attn_v")?,
                    wo: b("attn_output")?,
                    q_norm: qwen("attn_q_norm")?,
                    k_norm: qwen("attn_k_norm")?,
                    ffn_norm: b(if oai { "post_attention_norm" } else { "ffn_norm" })?,
                    bias: if oai {
                        let n = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
                        Some(n.map(bias).into_iter().collect::<Result<Vec<_>, _>>()?.try_into().unwrap())
                    } else {
                        None
                    },
                    sinks: if oai { Some(b("attn_sinks")?) } else { None },
                    router: b("ffn_gate_inp")?,
                    gate: b("ffn_gate_exps")?,
                    up: b("ffn_up_exps")?,
                    down: b("ffn_down_exps")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            tok_embd: t("token_embd.weight".into())?,
            out_norm: t("output_norm.weight".into())?,
            output,
            g,
            hp,
            layers,
        })
    }

    /// Bytes of rows [r0, r1) of a (possibly 3-D, flattened) tensor.
    pub fn rows(&self, t: &TensorInfo, r0: usize, r1: usize) -> &[u8] {
        let rb = t.row_bytes();
        &self.g.data(t)[r0 * rb..r1 * rb]
    }

    /// Raw bytes of expert `ex` in layer `l`: gate, up, down, then (gpt-oss) their f32 biases.
    pub fn expert(&self, l: &Layer, ex: usize) -> Vec<&[u8]> {
        let (e, ff) = (self.hp.n_embd, self.hp.n_ff_exp);
        let mut v = vec![
            self.rows(&l.gate, ex * ff, (ex + 1) * ff),
            self.rows(&l.up, ex * ff, (ex + 1) * ff),
            self.rows(&l.down, ex * e, (ex + 1) * e),
        ];
        if let Some(b) = &l.bias {
            v.extend(b[5..].iter().map(|t| self.rows(t, ex, ex + 1)));
        }
        v
    }

    /// Router logits -> per token (expert, weight) pairs, as this architecture routes.
    pub fn route(&self, logits: &[f32]) -> Vec<Vec<(usize, f32)>> {
        let (n, k) = (self.hp.n_expert, self.hp.n_expert_used);
        if self.hp.oai { route_oai(logits, n, k) } else { route(logits, n, k) }
    }
}

/// Router logits (T x n_expert) -> per token, the selected (expert, weight) pairs:
/// softmax, top-k, renormalised over the selected experts (norm_w = true, no scale).
pub fn route(logits: &[f32], n_expert: usize, k: usize) -> Vec<Vec<(usize, f32)>> {
    logits
        .chunks_exact(n_expert)
        .map(|lr| {
            let m = lr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let p: Vec<f32> = lr.iter().map(|v| (v - m).exp()).collect();
            let s: f32 = p.iter().sum();
            let mut idx: Vec<usize> = (0..n_expert).collect();
            idx.sort_by(|a, b| p[*b].total_cmp(&p[*a]));
            let top = &idx[..k];
            let wsum: f32 = top.iter().map(|&i| p[i] / s).sum::<f32>().max(6.103_515_6e-5);
            top.iter().map(|&i| (i, p[i] / s / wsum)).collect()
        })
        .collect()
}

/// gpt-oss routing (SOFTMAX_WEIGHT): top-k by router logit, softmax over the chosen k.
pub fn route_oai(logits: &[f32], n_expert: usize, k: usize) -> Vec<Vec<(usize, f32)>> {
    logits
        .chunks_exact(n_expert)
        .map(|lr| {
            let mut idx: Vec<usize> = (0..n_expert).collect();
            idx.sort_by(|a, b| lr[*b].total_cmp(&lr[*a]));
            let top = &idx[..k];
            let m = lr[top[0]];
            let s: f32 = top.iter().map(|&i| (lr[i] - m).exp()).sum();
            top.iter().map(|&i| (i, (lr[i] - m).exp() / s)).collect()
        })
        .collect()
}

/// gpt-oss clamped SwiGLU (swiglu_oai: alpha 1.702, limit 7, up + 1); `silu(g) * u` otherwise.
pub fn glu(oai: bool, g: f32, u: f32) -> f32 {
    if oai {
        let (g, u) = (g.min(7.0), u.clamp(-7.0, 7.0));
        g / (1.0 + (-1.702 * g).exp()) * (u + 1.0)
    } else {
        g / (1.0 + (-g).exp()) * u
    }
}
