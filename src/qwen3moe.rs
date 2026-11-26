//! qwen3moe weight map and routing, shared by the CPU reference and the CUDA engine
//! (mirrors llama.cpp src/models/qwen3moe.cpp).

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
}

pub struct Layer {
    pub attn_norm: TensorInfo,
    pub wq: TensorInfo,
    pub wk: TensorInfo,
    pub wv: TensorInfo,
    pub wo: TensorInfo,
    pub q_norm: TensorInfo,
    pub k_norm: TensorInfo,
    pub ffn_norm: TensorInfo,
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
        let arch = g.str("general.architecture").unwrap_or("");
        if arch != "qwen3moe" {
            return Err(format!("unsupported architecture {arch:?}"));
        }
        let u = |k: &str| g.u64(&format!("qwen3moe.{k}")).map(|v| v as usize).ok_or(format!("missing qwen3moe.{k}"));
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
            eps: g.f64("qwen3moe.attention.layer_norm_rms_epsilon").ok_or("missing rms eps")? as f32,
            rope_base: g.f64("qwen3moe.rope.freq_base").unwrap_or(10000.0) as f32,
            n_vocab: output.dims[1] as usize,
        };
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

    /// Raw (gate, up, down) bytes of expert `ex` in layer `l`.
    pub fn expert(&self, l: &Layer, ex: usize) -> [&[u8]; 3] {
        let (e, ff) = (self.hp.n_embd, self.hp.n_ff_exp);
        [
            self.rows(&l.gate, ex * ff, (ex + 1) * ff),
            self.rows(&l.up, ex * ff, (ex + 1) * ff),
            self.rows(&l.down, ex * e, (ex + 1) * e),
        ]
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
            let wsum: f32 = top.iter().map(|&i| p[i] / s).sum::<f32>().max(6.103515625e-5);
            top.iter().map(|&i| (i, p[i] / s / wsum)).collect()
        })
        .collect()
}
