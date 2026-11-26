//! Token generation over an `Engine`: KV prefix reuse across requests, sampling, stop tokens.

use crate::qwen3moe::Engine;

#[derive(Clone, Debug)]
pub struct Sampling {
    /// <= 0: greedy.
    pub temperature: f32,
    /// 0: off.
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
}

impl Default for Sampling {
    /// llama.cpp's defaults (common/common.h).
    fn default() -> Self {
        Self { temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05, seed: 0x5eed }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// An end-of-generation token.
    Stop,
    /// max_tokens or the context is full.
    Length,
    /// The caller's callback asked to stop (e.g. a stop string or a closed connection).
    Cancelled,
}

/// An engine plus the tokens its KV cache holds, so a request sharing a prefix with the
/// previous one (an agent resending its conversation) only computes the new suffix.
pub struct Generator<E: Engine> {
    pub engine: E,
    cached: Vec<u32>,
}

/// Why a prompt could not be run.
#[derive(Debug, PartialEq, Eq)]
pub struct ContextExceeded {
    pub n_prompt: usize,
    pub n_ctx: usize,
}

impl<E: Engine> Generator<E> {
    pub fn new(engine: E) -> Self {
        Self { engine, cached: vec![] }
    }

    /// Prompt tokens `generate` will take from the KV cache: the common prefix with what it
    /// holds, but never the last prompt token (recomputed for its logits).
    pub fn reusable(&self, prompt: &[u32]) -> usize {
        self.cached.iter().zip(prompt).take_while(|(a, b)| a == b).count().min(prompt.len().saturating_sub(1))
    }

    /// Generate up to `max_tokens` after `prompt`. `on_token` gets each sampled token that is
    /// not end-of-generation, and returns false to stop. Returns the finish reason and how many
    /// prompt tokens were reused from the KV cache.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        s: &Sampling,
        is_eog: impl Fn(u32) -> bool,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<(Finish, usize), ContextExceeded> {
        let n_ctx = self.engine.n_ctx();
        if prompt.is_empty() || prompt.len() >= n_ctx {
            return Err(ContextExceeded { n_prompt: prompt.len(), n_ctx });
        }
        let reuse = self.reusable(prompt);
        self.engine.truncate(reuse);
        self.cached.truncate(reuse);
        let mut logits = self.engine.forward_last(&prompt[reuse..]);
        self.cached.extend_from_slice(&prompt[reuse..]);

        let mut rng = s.seed;
        let mut n = 0;
        let finish = loop {
            if max_tokens == 0 {
                break Finish::Length;
            }
            let tok = sample(&mut logits, s, &mut rng);
            if is_eog(tok) {
                break Finish::Stop;
            }
            if !on_token(tok) {
                break Finish::Cancelled;
            }
            n += 1;
            if n >= max_tokens || self.cached.len() >= n_ctx {
                break Finish::Length;
            }
            logits = self.engine.forward_last(&[tok]);
            self.cached.push(tok);
        };
        Ok((finish, reuse))
    }
}

/// xorshift64*: uniform in [0, 1).
fn next_f32(state: &mut u64) -> f32 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    (state.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32
}

/// llama.cpp's default chain order: top_k, top_p, min_p, temperature, then draw.
fn sample(logits: &mut [f32], s: &Sampling, rng: &mut u64) -> u32 {
    let argmax = || logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    if s.temperature <= 0.0 {
        return argmax();
    }
    let mut c: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
    let k = if s.top_k == 0 { c.len() } else { s.top_k.min(c.len()) };
    c.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    c.truncate(k);
    c.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let softmax = |c: &[(u32, f32)], t: f32| -> Vec<f32> {
        let m = c[0].1;
        let e: Vec<f32> = c.iter().map(|x| ((x.1 - m) / t).exp()).collect();
        let z: f32 = e.iter().sum();
        e.into_iter().map(|v| v / z).collect()
    };
    let p = softmax(&c, 1.0);
    let mut cum = 0.0;
    let keep = p.iter().position(|&pi| {
        cum += pi;
        cum >= s.top_p
    });
    c.truncate(keep.map_or(c.len(), |i| i + 1));
    let p0 = p[0];
    let keep = p.iter().take(c.len()).take_while(|&&pi| pi >= s.min_p * p0).count().max(1);
    c.truncate(keep);
    let p = softmax(&c, s.temperature);
    let mut r = next_f32(rng);
    for (i, pi) in p.iter().enumerate() {
        r -= pi;
        if r < 0.0 {
            return c[i].0;
        }
    }
    c[c.len() - 1].0
}

/// Turns a growing token stream into valid UTF-8 text deltas, holding back split characters.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            let e = match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out += s;
                    self.pending.clear();
                    return out;
                }
                Err(e) => e,
            };
            let v = e.valid_up_to();
            out += std::str::from_utf8(&self.pending[..v]).unwrap();
            // An incomplete trailing character waits; an invalid sequence is replaced.
            let Some(bad) = e.error_len() else {
                self.pending.drain(..v);
                return out;
            };
            out.push('\u{FFFD}');
            self.pending.drain(..v + bad);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Emits logits favouring (last input token + 1) % 8; records what it was asked to compute.
    struct Fake {
        n_past: usize,
        computed: Vec<u32>,
    }

    impl Engine for Fake {
        fn n_vocab(&self) -> usize {
            8
        }
        fn n_ctx(&self) -> usize {
            16
        }
        fn truncate(&mut self, n: usize) {
            self.n_past = self.n_past.min(n);
        }
        fn forward(&mut self, t: &[u32]) -> Vec<f32> {
            self.n_past += t.len();
            self.computed.extend_from_slice(t);
            t.iter().flat_map(|&x| (0..8).map(move |i| if i == (x + 1) % 8 { 1.0 } else { 0.0 })).collect()
        }
    }

    fn run(g: &mut Generator<Fake>, prompt: &[u32], max: usize) -> (Vec<u32>, Finish, usize) {
        let mut out = vec![];
        let greedy = Sampling { temperature: 0.0, ..Default::default() };
        let (f, reused) = g.generate(prompt, max, &greedy, |t| t == 7, |t| {
            out.push(t);
            true
        }).unwrap();
        (out, f, reused)
    }

    #[test]
    fn generates_until_eog_or_length_and_reuses_prefix() {
        let mut g = Generator::new(Fake { n_past: 0, computed: vec![] });
        assert_eq!(run(&mut g, &[1, 2], 10), (vec![3, 4, 5, 6], Finish::Stop, 0));
        // Same conversation plus a new turn: only the suffix is computed.
        g.engine.computed.clear();
        assert_eq!(run(&mut g, &[1, 2, 3, 4, 0], 2), (vec![1, 2], Finish::Length, 4));
        assert_eq!(g.engine.computed, vec![0, 1]);
        // Identical prompt: the last token is recomputed for its logits.
        g.engine.computed.clear();
        let (_, _, reused) = run(&mut g, &[1, 2, 3, 4, 0], 1);
        assert_eq!((reused, g.engine.computed[0]), (4, 0));
        // max_tokens 0: no tokens at all.
        assert_eq!(run(&mut g, &[1, 2], 0).0, Vec::<u32>::new());
        assert_eq!(g.generate(&[0; 16], 1, &Sampling::default(), |_| false, |_| true), Err(ContextExceeded { n_prompt: 16, n_ctx: 16 }));
    }

    #[test]
    fn sampling_respects_filters() {
        let s = Sampling { temperature: 1.0, top_k: 2, top_p: 1.0, min_p: 0.0, seed: 1 };
        let mut rng = 1;
        for _ in 0..200 {
            let t = sample(&mut [5.0, 0.0, 4.9, 4.8], &s, &mut rng);
            assert!(t == 0 || t == 2, "top_k=2 drew {t}");
        }
        let s = Sampling { min_p: 0.5, top_k: 0, ..s };
        assert!((0..200).all(|_| sample(&mut [5.0, 0.0, 1.0], &s, &mut rng) == 0));
    }

    #[test]
    fn utf8_stream_holds_split_chars() {
        let mut u = Utf8Stream::default();
        let e = "é".as_bytes();
        assert_eq!(u.push(&[b'a', e[0]]), "a");
        assert_eq!(u.push(&e[1..]), "é");
        // An invalid byte does not swallow a split character after it.
        assert_eq!(u.push(&[0xff, b'a', e[0]]), "\u{FFFD}a");
        assert_eq!(u.push(&e[1..]), "é");
    }
}
