//! Byte-level BPE tokenizer (GPT-2 style, qwen2 pre-tokenizer) matching llama.cpp's llama-vocab.

use crate::gguf::{Gguf, Value};
use fancy_regex::Regex;
use std::collections::HashMap;

// llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN2.
const QWEN2_PRE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// o200k (gpt-oss), the original regex from its tokenizer.json, as HF `tokenizers` runs it.
const GPT4O_PRE: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35 (Qwen3.5/3.6): qwen2 with combining marks kept in words.
const QWEN35_PRE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// glm4 (GLM-4.x), the regex from its tokenizer.json, as HF `tokenizers` runs it.
const GLM4_PRE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

// GGUF tokenizer.ggml.token_type values.
const TT_UNKNOWN: i64 = 2;
const TT_CONTROL: i64 = 3;
const TT_USER_DEFINED: i64 = 4;

pub struct Tokenizer {
    tokens: Vec<String>,
    types: Vec<i64>,
    ids: HashMap<String, u32>,
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Special tokens, longest text first: llama.cpp's cache_special_tokens order.
    specials: Vec<u32>,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    pre: Regex,
    ignore_merges: bool,
    pub eos: u32,
}

/// GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to U+0100 upward.
fn bytes_to_unicode() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        out[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    out
}

enum Frag<'a> {
    Raw(&'a str),
    Tok(u32),
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self, String> {
        let model = g.str("tokenizer.ggml.model").unwrap_or("");
        let pre = g.str("tokenizer.ggml.pre").unwrap_or("");
        // ignore_merges (llama-vocab.cpp): a pre-token that is itself in the vocab is one token.
        let (pre_re, ignore_merges) = match (model, pre) {
            ("gpt2", "qwen2") => (QWEN2_PRE, false),
            ("gpt2", "qwen35") => (QWEN35_PRE, false),
            ("gpt2", "gpt-4o") => (GPT4O_PRE, false),
            ("gpt2", "glm4") => (GLM4_PRE, true),
            _ => return Err(format!("unsupported tokenizer {model}/{pre} (gpt2 with qwen2, qwen35, gpt-4o or glm4)")),
        };
        let strs = |k: &str| -> Result<Vec<String>, String> {
            g.arr(k)
                .ok_or(format!("missing {k}"))?
                .iter()
                .map(|v| match v {
                    Value::Str(s) => Ok(s.clone()),
                    _ => Err(format!("{k}: non-string entry")),
                })
                .collect()
        };
        let tokens = strs("tokenizer.ggml.tokens")?;
        let types: Vec<i64> = match g.arr("tokenizer.ggml.token_type") {
            Some(a) => a.iter().map(|v| if let Value::I64(t) = v { *t } else { 1 }).collect(),
            None => vec![1; tokens.len()],
        };
        let ids: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();

        let mut merges = HashMap::new();
        for (rank, m) in strs("tokenizer.ggml.merges")?.iter().enumerate() {
            let (l, r) = m.split_once(' ').ok_or(format!("bad merge {m:?}"))?;
            if let (Some(&a), Some(&b), Some(&c)) = (ids.get(l), ids.get(r), ids.get(&format!("{l}{r}"))) {
                merges.entry((a, b)).or_insert((rank as u32, c));
            }
        }

        // ponytail: llama.cpp also force-marks EOG/FIM-looking texts as CONTROL; every such
        // token in the Qwen vocab is already type 3, so that override list is not ported.
        let mut specials: Vec<u32> = (0..tokens.len() as u32)
            .filter(|&i| matches!(types[i as usize], TT_UNKNOWN | TT_CONTROL | TT_USER_DEFINED))
            .collect();
        specials.sort_by_key(|&i| std::cmp::Reverse(tokens[i as usize].len()));

        let byte_enc = bytes_to_unicode();
        let byte_dec = byte_enc.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let eos = g.u64("tokenizer.ggml.eos_token_id").ok_or("missing eos_token_id")? as u32;

        Ok(Self {
            tokens,
            types,
            ids,
            merges,
            specials,
            byte_enc,
            byte_dec,
            pre: Regex::new(pre_re).map_err(|e| e.to_string())?,
            ignore_merges,
            eos,
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_text(&self, id: u32) -> &str {
        &self.tokens[id as usize]
    }

    pub fn is_control(&self, id: u32) -> bool {
        self.types[id as usize] == TT_CONTROL
    }

    /// Tokenize like llama.cpp `llama_tokenize(add_special=false, parse_special)`.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        // tokenizer_st_partition: split raw fragments on every special token, longest first.
        let mut frags = vec![Frag::Raw(text)];
        for &id in &self.specials {
            let ty = self.types[id as usize];
            if !parse_special && (ty == TT_CONTROL || ty == TT_UNKNOWN) {
                continue;
            }
            let pat = self.tokens[id as usize].as_str();
            if pat.is_empty() {
                continue;
            }
            let mut next = Vec::with_capacity(frags.len());
            for f in frags {
                let Frag::Raw(s) = f else {
                    next.push(f);
                    continue;
                };
                let mut rest = s;
                while let Some(p) = rest.find(pat) {
                    if p > 0 {
                        next.push(Frag::Raw(&rest[..p]));
                    }
                    next.push(Frag::Tok(id));
                    rest = &rest[p + pat.len()..];
                }
                if !rest.is_empty() {
                    next.push(Frag::Raw(rest));
                }
            }
            frags = next;
        }

        let mut out = Vec::new();
        for f in frags {
            match f {
                Frag::Tok(id) => out.push(id),
                Frag::Raw(s) => {
                    for m in self.pre.find_iter(s) {
                        self.bpe_word(m.expect("pre-tokenizer regex").as_str(), &mut out);
                    }
                }
            }
        }
        out
    }

    fn bpe_word(&self, word: &str, out: &mut Vec<u32>) {
        let mapped: String = word.bytes().map(|b| self.byte_enc[b as usize]).collect();
        if let Some(&id) = self.ignore_merges.then(|| self.ids.get(&mapped)).flatten() {
            return out.push(id);
        }
        let mut syms: Vec<u32> = mapped
            .chars()
            .map(|c| self.ids[c.encode_utf8(&mut [0; 4]) as &str])
            .collect();
        // ponytail: O(n^2) per word; fine for normal words, switch to a heap if huge words show up in profiles.
        loop {
            let best = syms
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| self.merges.get(&(w[0], w[1])).map(|&(rank, id)| (rank, i, id)))
                .min();
            let Some((_, i, id)) = best else { break };
            syms[i] = id;
            syms.remove(i + 1);
        }
        out.extend(syms);
    }

    /// Raw bytes of a token sequence (special-token text included).
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            let t = &self.tokens[id as usize];
            match self.types[id as usize] {
                TT_CONTROL | TT_USER_DEFINED | TT_UNKNOWN => out.extend_from_slice(t.as_bytes()),
                _ => out.extend(t.chars().filter_map(|c| self.byte_dec.get(&c))),
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.ids.get(text).copied()
    }
}
