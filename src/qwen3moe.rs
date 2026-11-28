//! qwen3moe weight map and routing, shared by the CPU reference and the CUDA engine
//! (mirrors llama.cpp src/models/qwen3moe.cpp). The CUDA engine also loads gpt-oss through it
//! (`Hparams::oai`): biases, sinks, sliding window, YaRN rope, see src/gptoss.rs; and
//! GLM-4.7-Flash (`Hparams::glm`): MLA attention, a dense first layer, sigmoid routing with
//! exp_probs_b and a shared expert, see src/glm.rs.

use crate::gguf::{Gguf, TensorInfo};

/// A forward-pass engine: run tokens at positions n_past.. and return T x n_vocab logits.
pub trait Engine {
    fn n_vocab(&self) -> usize;
    /// Positions the KV cache can hold.
    fn n_ctx(&self) -> usize;
    /// Keep the first `n_past` positions of the KV cache, drop the rest.
    fn truncate(&mut self, n_past: usize);
    /// Keep at most the first `n` positions; returns how many were kept. Recurrent engines
    /// (qwen35) keep all or none: their state can't rewind.
    fn rewind(&mut self, n: usize) -> usize {
        self.truncate(n);
        n
    }
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
    /// GLM-4.7-Flash (deepseek2). head_dim is then the absorbed MLA cache row (lat + rope) and
    /// n_head_kv 1: every head attends over one shared latent row per position.
    pub glm: Option<Glm>,
    /// Qwen3.6 (qwen35moe): Gated DeltaNet layers, gated attention (attn_q gives [q | gate] per
    /// head), roped dims n_rot of head_dim, sigmoid-gated shared expert (src/qwen35.rs).
    pub qwen35: Option<Qwen35>,
    /// Dims of each head that RoPE rotates (head_dim but for qwen35).
    pub n_rot: usize,
}

/// DeltaNet widths of Qwen3.6: k heads (group_count), v heads (time_step_rank), head width
/// (state_size), conv kernel.
pub struct Qwen35 {
    pub nk: usize,
    pub nv: usize,
    pub dk: usize,
    pub d_conv: usize,
}

/// MLA widths and the routed experts' weight scale of GLM-4.7-Flash.
pub struct Glm {
    /// kv_lora_rank (the latent, also V), qk_nope and qk_rope head dims, v head dim.
    pub lat: usize,
    pub nope: usize,
    pub rope: usize,
    pub vd: usize,
    pub w_scale: f32,
}

/// Grouped-query attention (qwen3moe, gpt-oss, qwen35 full-attention layers), absorbed MLA (GLM,
/// src/glm.rs) or a Gated DeltaNet mixer (qwen35; its ssm_out is the layer's `wo`).
#[allow(clippy::large_enum_variant)] // one per layer
pub enum Attn {
    Gqa { wq: TensorInfo, wk: TensorInfo, wv: TensorInfo },
    Mla { q_a: TensorInfo, q_a_norm: TensorInfo, q_b: TensorInfo, kv_a: TensorInfo, kv_a_norm: TensorInfo, k_b: TensorInfo, v_b: TensorInfo },
    Delta { qkv: TensorInfo, z: TensorInfo, beta: TensorInfo, alpha: TensorInfo, dt_bias: TensorInfo, a: TensorInfo, conv: TensorInfo, norm: TensorInfo },
}

/// A layer's routed experts (3-D gate/up/down tensors).
pub struct Moe {
    pub router: TensorInfo,
    pub gate: TensorInfo,
    pub up: TensorInfo,
    pub down: TensorInfo,
    /// GLM: selection bias (exp_probs_b). GLM, qwen35: the shared expert's gate, up, down;
    /// qwen35: its sigmoid gate (ffn_gate_inp_shexp).
    pub sel_bias: Option<TensorInfo>,
    pub shared: Option<[TensorInfo; 3]>,
    pub shared_gate: Option<TensorInfo>,
}

pub struct Layer {
    pub attn_norm: TensorInfo,
    pub attn: Attn,
    pub wo: TensorInfo,
    /// qwen3moe only.
    pub q_norm: Option<TensorInfo>,
    pub k_norm: Option<TensorInfo>,
    /// The MoE input norm (gpt-oss: post_attention_norm).
    pub ffn_norm: TensorInfo,
    /// gpt-oss only: biases of q, k, v, o, router, gate/up/down experts; attention sinks.
    pub bias: Option<[TensorInfo; 8]>,
    pub sinks: Option<TensorInfo>,
    /// None: a dense SwiGLU layer (GLM's first), `dense` holds its gate, up, down.
    pub moe: Option<Moe>,
    pub dense: Option<[TensorInfo; 3]>,
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
        if !["qwen3moe", "gpt-oss", "deepseek2", "qwen35moe"].contains(&arch.as_str()) {
            return Err(format!("unsupported architecture {arch:?}"));
        }
        let oai = arch == "gpt-oss";
        let u = |k: &str| g.u64(&format!("{arch}.{k}")).map(|v| v as usize).ok_or(format!("missing {arch}.{k}"));
        let glm = if arch == "deepseek2" {
            // ponytail: GLM-4.7-Flash only (sigmoid gating, the llama.cpp default for this shape
            // when the GGUF has no expert_gating_func; one expert group; no YaRN).
            if g.u64("deepseek2.expert_gating_func").is_some_and(|v| v != 2) || u("expert_group_count").unwrap_or(1) != 1 || g.f64("deepseek2.rope.scaling.factor").is_some() {
                return Err("deepseek2: only the GLM-4.7-Flash variant is supported".into());
            }
            let rope = u("rope.dimension_count")?;
            Some(Glm {
                lat: u("attention.kv_lora_rank")?,
                nope: u("attention.key_length_mla")? - rope,
                rope,
                vd: u("attention.value_length_mla")?,
                w_scale: g.f64("deepseek2.expert_weights_scale").unwrap_or(1.0) as f32,
            })
        } else {
            None
        };
        let qwen35 = if arch == "qwen35moe" {
            Some(Qwen35 { nk: u("ssm.group_count")?, nv: u("ssm.time_step_rank")?, dk: u("ssm.state_size")?, d_conv: u("ssm.conv_kernel")? })
        } else {
            None
        };
        let recurrent: Vec<bool> = match g.arr(&format!("{arch}.attention.recurrent_layers")) {
            Some(a) => a.iter().map(|v| matches!(v, crate::gguf::Value::Bool(true))).collect(),
            None => (0..u("block_count")?).map(|i| qwen35.is_some() && (i + 1) % u("full_attention_interval").unwrap_or(4) != 0).collect(),
        };
        let n_dense = if glm.is_some() { u("leading_dense_block_count").unwrap_or(0) } else { 0 };
        let n_embd = u("embedding_length")?;
        let n_head = u("attention.head_count")?;
        let t = |n: String| g.tensor(&n).cloned().ok_or(format!("missing tensor {n}"));
        let output = t("output.weight".into()).or_else(|_| t("token_embd.weight".into()))?;
        let head_dim = glm.as_ref().map_or_else(|| u("attention.key_length").unwrap_or(n_embd / n_head), |m| m.lat + m.rope);
        let hp = Hparams {
            n_layer: u("block_count")?,
            n_embd,
            n_head,
            n_head_kv: if glm.is_some() { 1 } else { u("attention.head_count_kv")? },
            head_dim,
            n_rot: if qwen35.is_some() { u("rope.dimension_count")? } else { head_dim },
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            eps: g.f64(&format!("{arch}.attention.layer_norm_rms_epsilon")).ok_or("missing rms eps")? as f32,
            rope_base: g.f64(&format!("{arch}.rope.freq_base")).unwrap_or(10000.0) as f32,
            n_vocab: output.dims[1] as usize,
            oai,
            swa: if oai { u("attention.sliding_window")? } else { 0 },
            rope_yarn: if oai { Some(crate::gptoss::yarn(&g)?) } else { None },
            glm,
            qwen35,
        };
        let layers = (0..hp.n_layer)
            .map(|i| {
                let b = |s: &str| t(format!("blk.{i}.{s}.weight"));
                let bias = |s: &str| t(format!("blk.{i}.{s}.bias"));
                let (is_glm, is_q35, delta) = (hp.glm.is_some(), hp.qwen35.is_some(), recurrent[i]);
                let qwen = |s: &str| if oai || is_glm || delta { Ok(None) } else { b(s).map(Some) };
                Ok(Layer {
                    attn_norm: b("attn_norm")?,
                    attn: if delta {
                        Attn::Delta {
                            qkv: b("attn_qkv")?,
                            z: b("attn_gate")?,
                            beta: b("ssm_beta")?,
                            alpha: b("ssm_alpha")?,
                            dt_bias: bias("ssm_dt")?,
                            a: t(format!("blk.{i}.ssm_a"))?,
                            conv: b("ssm_conv1d")?,
                            norm: b("ssm_norm")?,
                        }
                    } else if is_glm {
                        Attn::Mla { q_a: b("attn_q_a")?, q_a_norm: b("attn_q_a_norm")?, q_b: b("attn_q_b")?, kv_a: b("attn_kv_a_mqa")?, kv_a_norm: b("attn_kv_a_norm")?, k_b: b("attn_k_b")?, v_b: b("attn_v_b")? }
                    } else {
                        Attn::Gqa { wq: b("attn_q")?, wk: b("attn_k")?, wv: b("attn_v")? }
                    },
                    wo: b(if delta { "ssm_out" } else { "attn_output" })?,
                    q_norm: qwen("attn_q_norm")?,
                    k_norm: qwen("attn_k_norm")?,
                    ffn_norm: b(if oai || is_q35 { "post_attention_norm" } else { "ffn_norm" })?,
                    bias: if oai {
                        let n = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
                        Some(n.map(bias).into_iter().collect::<Result<Vec<_>, _>>()?.try_into().unwrap())
                    } else {
                        None
                    },
                    sinks: if oai { Some(b("attn_sinks")?) } else { None },
                    moe: if i < n_dense {
                        None
                    } else {
                        Some(Moe {
                            router: b("ffn_gate_inp")?,
                            gate: b("ffn_gate_exps")?,
                            up: b("ffn_up_exps")?,
                            down: b("ffn_down_exps")?,
                            sel_bias: if is_glm { Some(bias("exp_probs_b")?) } else { None },
                            shared: if is_glm || is_q35 { Some([b("ffn_gate_shexp")?, b("ffn_up_shexp")?, b("ffn_down_shexp")?]) } else { None },
                            shared_gate: if is_q35 { Some(b("ffn_gate_inp_shexp")?) } else { None },
                        })
                    },
                    dense: if i < n_dense { Some([b("ffn_gate")?, b("ffn_up")?, b("ffn_down")?]) } else { None },
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
    /// Empty for a dense layer.
    pub fn expert(&self, l: &Layer, ex: usize) -> Vec<&[u8]> {
        let (e, ff) = (self.hp.n_embd, self.hp.n_ff_exp);
        let Some(m) = &l.moe else { return vec![] };
        let mut v = vec![
            self.rows(&m.gate, ex * ff, (ex + 1) * ff),
            self.rows(&m.up, ex * ff, (ex + 1) * ff),
            self.rows(&m.down, ex * e, (ex + 1) * e),
        ];
        if let Some(b) = &l.bias {
            v.extend(b[5..].iter().map(|t| self.rows(t, ex, ex + 1)));
        }
        v
    }

    /// Layer `li`'s router logits -> per token (expert, weight) pairs, as this architecture routes.
    pub fn route(&self, li: usize, logits: &[f32]) -> Vec<Vec<(usize, f32)>> {
        let (n, k) = (self.hp.n_expert, self.hp.n_expert_used);
        match (&self.hp.glm, self.layers[li].moe.as_ref().and_then(|m| m.sel_bias.as_ref())) {
            (Some(glm), Some(b)) => route_sigmoid(logits, self.g.f32s(b), k, glm.w_scale),
            _ if self.hp.oai => route_oai(logits, n, k),
            _ => route(logits, n, k),
        }
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

/// GLM routing (llama.cpp build_moe_ffn: SIGMOID, exp_probs_b, norm_w, scale): top-k by
/// sigmoid + selection bias; weights the plain sigmoids, renormalised over the k, times `scale`.
pub fn route_sigmoid(logits: &[f32], bias: &[f32], k: usize, scale: f32) -> Vec<Vec<(usize, f32)>> {
    logits
        .chunks_exact(bias.len())
        .map(|lr| {
            let p: Vec<f32> = lr.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
            let mut idx: Vec<usize> = (0..p.len()).collect();
            idx.sort_by(|&a, &b| (p[b] + bias[b]).total_cmp(&(p[a] + bias[a])));
            let top = &idx[..k];
            let sum = top.iter().map(|&i| p[i]).sum::<f32>().max(6.103_515_6e-5);
            top.iter().map(|&i| (i, p[i] / sum * scale)).collect()
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
