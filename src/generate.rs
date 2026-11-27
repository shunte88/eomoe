//! Token generation over an `Engine`: continuous batching of requests over its KV sequences,
//! prefix reuse, sampling, stop tokens.

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
    /// Constrain the output to JSON matching this schema (`{}`: any JSON); see `constrain`.
    pub schema: Option<serde_json::Value>,
}

impl Default for Sampling {
    /// llama.cpp's defaults (common/common.h).
    fn default() -> Self {
        Self { temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05, seed: 0x5eed, schema: None }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// An end-of-generation token.
    Stop,
    /// max_tokens or the context is full.
    Length,
}

/// What a request's sink sees, in order: Started once, Token per generated token, Done last
/// (no Done if the sink returned false: the client is gone).
pub enum Event {
    Started { n_cached: usize },
    Token(u32),
    Done(Finish),
}

pub struct Request {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: Sampling,
    /// Returns false when nobody is listening any more; the request is then dropped.
    pub sink: Box<dyn FnMut(Event) -> bool + Send>,
}

/// Prefill chunk while other streams are decoding, so they keep getting tokens.
const PREFILL_CHUNK: usize = 512;

/// A running request: its slot holds the KV for `prompt[..cached.len()]` and output so far.
struct Job {
    req: Request,
    /// Assignment order; the oldest unfinished prefill goes first.
    order: u64,
    n_out: usize,
    /// Sampled, emitted, not yet in the KV cache.
    pending: Option<u32>,
    /// Output bytes so far, for a schema-constrained request.
    text: Vec<u8>,
    rng: u64,
}

/// One KV sequence: the tokens it holds, and the request using it (if any).
#[derive(Default)]
struct Slot {
    cached: Vec<u32>,
    job: Option<Job>,
    last_used: u64,
}

/// Continuous batching over an `Engine`: requests are assigned to KV sequences (preferring the
/// one already holding their prefix, so an agent's resent conversation only computes the new
/// suffix), prefilled one chunk per step, and decoded together one token per step.
///
/// The sequences share the engine's `n_ctx` positions in `kv_page` pages. Idle caches are evicted
/// least-recently-used when a stream needs room; a stream that still can't grow ends with Length.
/// Sampling uses each request's own seed, so output does not depend on what else is running.
pub struct Batcher<E: Engine> {
    pub engine: E,
    eog: Vec<u32>,
    slots: Vec<Slot>,
    queue: std::collections::VecDeque<Request>,
    tick: u64,
    /// Host-RAM prompt cache limit in bytes (llama-server --cache-ram); 0: off.
    pub ram_cache: usize,
    /// Each token's bytes, for `Request::schema`.
    pub token_bytes: Vec<Vec<u8>>,
    snaps: std::collections::VecDeque<Snap>,
}

/// A displaced sequence's KV, kept in host RAM until a prompt that shares its prefix returns.
struct Snap {
    tokens: Vec<u32>,
    kv: Vec<u8>,
}

impl<E: Engine> Batcher<E> {
    pub fn new(engine: E, eog: Vec<u32>) -> Self {
        let slots = (0..engine.n_seq()).map(|_| Slot::default()).collect();
        Self { engine, eog, slots, queue: Default::default(), tick: 0, ram_cache: 0, token_bytes: vec![], snaps: Default::default() }
    }

    pub fn submit(&mut self, r: Request) {
        self.queue.push_back(r);
    }

    /// Nothing queued or running: `step` would do nothing.
    pub fn idle(&self) -> bool {
        self.queue.is_empty() && self.slots.iter().all(|s| s.job.is_none())
    }

    /// Start queued requests on free sequences, run one prefill chunk, then one decode token for
    /// every stream that has one.
    pub fn step(&mut self) {
        self.assign();
        self.prefill();
        self.decode();
    }

    fn assign(&mut self) {
        while self.slots.iter().any(|s| s.job.is_none()) {
            let Some(mut req) = self.queue.pop_front() else { return };
            if req.prompt.is_empty() || req.prompt.len() >= self.engine.n_ctx() {
                (req.sink)(Event::Done(Finish::Length));
                continue;
            }
            let reuse = |s: &Slot| s.cached.iter().zip(&req.prompt).take_while(|(a, b)| a == b).count().min(req.prompt.len() - 1);
            let (si, n) = (self.slots.iter().enumerate())
                .filter(|(_, s)| s.job.is_none())
                .map(|(i, s)| (i, reuse(s)))
                .max_by_key(|&(i, n)| (n, std::cmp::Reverse(self.slots[i].last_used)))
                .unwrap();
            let n = self.ram_swap(si, &req.prompt).unwrap_or(n);
            self.tick += 1;
            let slot = &mut self.slots[si];
            slot.last_used = self.tick;
            slot.cached.truncate(n);
            self.engine.select(si);
            self.engine.truncate(n);
            if (req.sink)(Event::Started { n_cached: n }) {
                let rng = req.sampling.seed;
                slot.job = Some(Job { req, order: self.tick, n_out: 0, pending: None, text: vec![], rng });
            }
        }
    }

    fn prefill(&mut self) {
        let Some(si) = (0..self.slots.len())
            .filter(|&i| self.slots[i].job.as_ref().is_some_and(|j| self.slots[i].cached.len() < j.req.prompt.len()))
            .min_by_key(|&i| self.slots[i].job.as_ref().unwrap().order)
        else {
            return;
        };
        let decoding = self.slots.iter().any(|s| s.job.as_ref().is_some_and(|j| j.pending.is_some()));
        let (from, len) = (self.slots[si].cached.len(), self.slots[si].job.as_ref().unwrap().req.prompt.len());
        let to = if decoding { len.min(from + PREFILL_CHUNK) } else { len };
        if !self.reserve(si, to) {
            // Wait for running streams to free the pool; with nothing else running it never fits.
            if !self.slots.iter().enumerate().any(|(i, s)| i != si && s.job.is_some()) {
                self.finish(si, Finish::Length);
            }
            return;
        }
        let slot = &mut self.slots[si];
        let chunk = &slot.job.as_ref().unwrap().req.prompt[from..to];
        self.engine.select(si);
        let mut logits = self.engine.forward_last(chunk);
        slot.cached.extend_from_slice(chunk);
        if to == len {
            self.emit(si, &mut logits);
        }
    }

    fn decode(&mut self) {
        let mut rows = vec![];
        for si in 0..self.slots.len() {
            let Some(t) = self.slots[si].job.as_ref().and_then(|j| j.pending) else { continue };
            if self.reserve(si, self.slots[si].cached.len() + 1) {
                rows.push((si, t));
            } else {
                self.finish(si, Finish::Length);
            }
        }
        if rows.is_empty() {
            return;
        }
        let mut logits = self.engine.decode_batch(&rows);
        let nv = self.engine.n_vocab();
        for (r, &(si, t)) in rows.iter().enumerate() {
            self.slots[si].cached.push(t);
            self.emit(si, &mut logits[r * nv..(r + 1) * nv]);
        }
    }

    /// Sample the next token of the stream in `si` and send it, or finish the stream.
    fn emit(&mut self, si: usize, logits: &mut [f32]) {
        let n_ctx = self.engine.n_ctx();
        let slot = &mut self.slots[si];
        let job = slot.job.as_mut().unwrap();
        job.pending = None;
        if job.n_out >= job.req.max_tokens {
            return self.finish(si, Finish::Length);
        }
        if let Some(sch) = &job.req.sampling.schema {
            if !allow_json(logits, sch, &job.text, &self.token_bytes, &self.eog) {
                return self.finish(si, Finish::Stop); // no token can continue the JSON
            }
        }
        let t = sample(logits, &job.req.sampling, &mut job.rng);
        if job.req.sampling.schema.is_some() && !self.eog.contains(&t) {
            job.text.extend_from_slice(&self.token_bytes[t as usize]);
        }
        if self.eog.contains(&t) {
            return self.finish(si, Finish::Stop);
        }
        if !(job.req.sink)(Event::Token(t)) {
            slot.job = None; // client gone
            return;
        }
        job.n_out += 1;
        if job.n_out >= job.req.max_tokens || slot.cached.len() >= n_ctx {
            return self.finish(si, Finish::Length);
        }
        job.pending = Some(t);
    }

    fn finish(&mut self, si: usize, f: Finish) {
        if let Some(mut j) = self.slots[si].job.take() {
            (j.req.sink)(Event::Done(f));
        }
    }

    /// llama-server's prompt cache (server-slot-sched.cpp): before `prompt` replaces most of the
    /// slot's tokens, or when a saved prompt beats them, save the slot to host RAM and load the best
    /// saved prompt: one keeping >= 1/4 of itself and more of both itself and the new prompt than the
    /// slot does. Returns the positions the slot now reuses, if it changed.
    fn ram_swap(&mut self, si: usize, prompt: &[u32]) -> Option<usize> {
        if self.ram_cache == 0 {
            return None;
        }
        let lcp = |a: &[u32]| a.iter().zip(prompt).take_while(|(x, y)| x == y).count();
        let better = |snaps: &std::collections::VecDeque<Snap>, cached: &[u32]| {
            let n = lcp(cached);
            let mut best = (if cached.is_empty() { -1.0 } else { n as f32 / cached.len() as f32 }, n as f32 / prompt.len() as f32, None);
            for (i, s) in snaps.iter().enumerate() {
                let n = lcp(&s.tokens);
                let (keep, sim) = (n as f32 / s.tokens.len() as f32, n as f32 / prompt.len() as f32);
                if keep >= 0.25 && best.0 < keep && best.1 < sim {
                    best = (keep, sim, Some((i, n)));
                }
            }
            best.2
        };
        let cached = &self.slots[si].cached;
        let keep = if cached.is_empty() { 0.0 } else { lcp(cached) as f32 / cached.len() as f32 };
        if keep >= 0.5 && better(&self.snaps, cached).is_none() {
            return None;
        }
        self.engine.select(si);
        self.ram_save(si);
        let (i, n) = better(&self.snaps, &self.slots[si].cached)?;
        let n = n.min(prompt.len() - 1);
        self.slots[si].cached.clear();
        self.engine.truncate(0);
        // Out first: making room may save other slots, which reorders the cache.
        let s = self.snaps.remove(i).unwrap();
        if !self.reserve(si, n) {
            self.snaps.push_back(s);
            return Some(0);
        }
        self.engine.select(si);
        self.engine.kv_restore(&s.kv, n);
        self.slots[si].cached = s.tokens[..n].to_vec();
        Some(n)
    }

    /// Save the selected sequence `si` to the RAM cache (llama-server's prompt_cache.alloc).
    fn ram_save(&mut self, si: usize) {
        let tokens = &self.slots[si].cached;
        let lcp = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
        if self.ram_cache == 0 || tokens.is_empty() || self.snaps.iter().any(|s| lcp(&s.tokens, tokens) == tokens.len()) {
            return;
        }
        let kv = self.engine.kv_save();
        if kv.is_empty() || kv.len() > self.ram_cache {
            return;
        }
        self.snaps.retain(|s| lcp(&s.tokens, tokens) < s.tokens.len());
        while self.snaps.iter().map(|s| s.kv.len()).sum::<usize>() + kv.len() > self.ram_cache {
            self.snaps.pop_front();
        }
        self.snaps.push_back(Snap { tokens: tokens.clone(), kv });
    }

    /// Make room for sequence `si` to hold `len` positions, evicting idle caches LRU.
    fn reserve(&mut self, si: usize, len: usize) -> bool {
        let page = self.engine.kv_page();
        let pages = |n: usize| n.div_ceil(page);
        let total = self.engine.n_ctx() / page;
        loop {
            let used: usize = self.slots.iter().enumerate().map(|(i, s)| if i == si { pages(len) } else { pages(s.cached.len()) }).sum();
            if used <= total {
                return true;
            }
            let Some(v) = (0..self.slots.len())
                .filter(|&i| i != si && self.slots[i].job.is_none() && !self.slots[i].cached.is_empty())
                .min_by_key(|&i| self.slots[i].last_used)
            else {
                return false;
            };
            self.engine.select(v);
            self.ram_save(v);
            self.slots[v].cached.clear();
            self.engine.truncate(0);
        }
    }
}

/// Mask `logits` to the best-scoring tokens (up to 64) that keep `text` a valid prefix of JSON
/// matching `schema`; once the JSON is complete only end-of-generation tokens remain. False if
/// nothing is allowed.
/// ponytail: sorts the whole vocab per token (~ms at 151K); a top-k partial sort if it shows up.
fn allow_json(logits: &mut [f32], schema: &serde_json::Value, text: &[u8], bytes: &[Vec<u8>], eog: &[u32]) -> bool {
    use crate::constrain::{check, State};
    let done = check(schema, text) == State::Done;
    // A top-level number is never Done (more digits may follow) but may end here.
    let can_end = done || check(schema, &[text, b" "].concat()) == State::Done;
    let ok = |t: usize| {
        if eog.contains(&(t as u32)) {
            return can_end;
        }
        let b = bytes.get(t).map_or(&[][..], |b| b);
        !done && !b.is_empty() && check(schema, &[text, b].concat()) != State::Invalid
    };
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let keep: Vec<(usize, f32)> = idx.into_iter().filter(|&t| ok(t)).take(64).map(|t| (t, logits[t])).collect();
    logits.fill(f32::NEG_INFINITY);
    for &(t, l) in &keep {
        logits[t] = l;
    }
    !keep.is_empty()
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

    use serde_json::json;
    use std::sync::{Arc, Mutex};

    /// Emits logits favouring (last input token + 1) % 8. Up to `n_seq` sequences share a pool of
    /// 16 positions in pages of 4, and it panics if the scheduler overfills it (as the GPU does).
    /// Records what it computed per sequence and the row count of every decode batch.
    struct Fake {
        n_seq: usize,
        cur: usize,
        n_past: Vec<usize>,
        computed: Vec<Vec<u32>>,
        batches: Vec<usize>,
    }

    fn fake(n_seq: usize) -> Fake {
        Fake { n_seq, cur: 0, n_past: vec![0; n_seq], computed: vec![vec![]; n_seq], batches: vec![] }
    }

    impl Engine for Fake {
        fn n_vocab(&self) -> usize {
            8
        }
        fn n_ctx(&self) -> usize {
            16
        }
        fn n_seq(&self) -> usize {
            self.n_seq
        }
        fn kv_page(&self) -> usize {
            4
        }
        fn select(&mut self, seq: usize) {
            self.cur = seq;
        }
        fn truncate(&mut self, n: usize) {
            self.n_past[self.cur] = self.n_past[self.cur].min(n);
        }
        fn forward(&mut self, t: &[u32]) -> Vec<f32> {
            self.n_past[self.cur] += t.len();
            let pages: usize = self.n_past.iter().map(|n| n.div_ceil(4)).sum();
            assert!(pages <= 4, "KV pool overflow: {:?}", self.n_past);
            self.computed[self.cur].extend_from_slice(t);
            t.iter().flat_map(|&x| (0..8).map(move |i| if i == (x + 1) % 8 { 1.0 } else { 0.0 })).collect()
        }
        fn kv_save(&mut self) -> Vec<u8> {
            vec![1; self.n_past[self.cur]]
        }
        fn kv_restore(&mut self, data: &[u8], n: usize) {
            assert!(self.n_past[self.cur] == 0 && n <= data.len());
            self.n_past[self.cur] = n;
        }
        fn decode_batch(&mut self, rows: &[(usize, u32)]) -> Vec<f32> {
            self.batches.push(rows.len());
            let mut out = vec![];
            for &(s, t) in rows {
                self.select(s);
                out.extend(self.forward(&[t]));
            }
            out
        }
    }

    /// What one request saw: tokens, finish (None if it never got one), cached prompt tokens.
    type Seen = Arc<Mutex<(Vec<u32>, Option<Finish>, Option<usize>)>>;

    fn req(prompt: &[u32], max: usize) -> (Request, Seen) {
        req_cancel(prompt, max, usize::MAX)
    }

    /// A request whose client goes away after `keep` tokens.
    fn req_cancel(prompt: &[u32], max: usize, keep: usize) -> (Request, Seen) {
        let seen: Seen = Default::default();
        let s2 = seen.clone();
        let sink = Box::new(move |ev: Event| {
            let mut s = s2.lock().unwrap();
            match ev {
                Event::Started { n_cached } => s.2 = Some(n_cached),
                Event::Token(t) => {
                    if s.0.len() == keep {
                        return false;
                    }
                    s.0.push(t)
                }
                Event::Done(f) => s.1 = Some(f),
            }
            true
        });
        (Request { prompt: prompt.to_vec(), max_tokens: max, sampling: Sampling { temperature: 0.0, ..Default::default() }, sink }, seen)
    }

    fn run_all(b: &mut Batcher<Fake>, reqs: Vec<(Request, Seen)>) -> Vec<(Vec<u32>, Option<Finish>, Option<usize>)> {
        let seen: Vec<Seen> = reqs.iter().map(|r| r.1.clone()).collect();
        for (r, _) in reqs {
            b.submit(r);
        }
        let mut steps = 0;
        while !b.idle() {
            b.step();
            steps += 1;
            assert!(steps < 1000, "scheduler stuck");
        }
        seen.iter().map(|s| s.lock().unwrap().clone()).collect()
    }

    #[test]
    fn generates_until_eog_or_length_and_reuses_prefix() {
        let mut b = Batcher::new(fake(1), vec![7]);
        assert_eq!(run_all(&mut b, vec![req(&[1, 2], 10)]), vec![(vec![3, 4, 5, 6], Some(Finish::Stop), Some(0))]);
        // Same conversation plus a new turn: only the suffix is computed.
        b.engine.computed[0].clear();
        assert_eq!(run_all(&mut b, vec![req(&[1, 2, 3, 4, 0], 2)]), vec![(vec![1, 2], Some(Finish::Length), Some(4))]);
        assert_eq!(b.engine.computed[0], vec![0, 1]);
        // Identical prompt: the last token is recomputed for its logits.
        b.engine.computed[0].clear();
        assert_eq!(run_all(&mut b, vec![req(&[1, 2, 3, 4, 0], 1)])[0].2, Some(4));
        assert_eq!(b.engine.computed[0][0], 0);
        // max_tokens 0: no tokens at all.
        assert_eq!(run_all(&mut b, vec![req(&[1, 2], 0)])[0].0, Vec::<u32>::new());
    }

    #[test]
    fn concurrent_streams_decode_batched_and_match_solo() {
        let mut b = Batcher::new(fake(2), vec![]);
        let got = run_all(&mut b, vec![req(&[1], 3), req(&[3], 3), req(&[0], 2)]);
        assert_eq!(got[0].0, vec![2, 3, 4]);
        assert_eq!(got[1].0, vec![4, 5, 6]);
        assert_eq!(got[2].0, vec![1, 2], "the third request waits for a free slot");
        assert!(got.iter().all(|g| g.1 == Some(Finish::Length)));
        assert!(b.engine.batches.contains(&2), "two streams share decode steps: {:?}", b.engine.batches);
    }

    #[test]
    fn request_lands_in_the_slot_holding_its_prefix() {
        let mut b = Batcher::new(fake(2), vec![]);
        run_all(&mut b, vec![req(&[1], 3), req(&[3], 3)]);
        // Slot caches: [1, 2, 3] and [3, 4, 5].
        assert_eq!(run_all(&mut b, vec![req(&[3, 4, 5, 6, 0], 1)])[0].2, Some(3));
        assert_eq!(run_all(&mut b, vec![req(&[1, 2, 3, 0], 1)])[0].2, Some(3));
    }

    #[test]
    fn cancelled_stream_frees_its_slot() {
        let mut b = Batcher::new(fake(1), vec![]);
        let got = run_all(&mut b, vec![req_cancel(&[1], 10, 2), req(&[5], 2)]);
        assert_eq!((got[0].0.clone(), got[0].1), (vec![2, 3], None));
        assert_eq!(got[1].0, vec![6, 7]);
    }

    #[test]
    fn full_pool_evicts_idle_caches_then_ends_streams_that_cannot_grow() {
        let mut b = Batcher::new(fake(2), vec![]);
        run_all(&mut b, vec![req(&[1, 1, 1, 1, 1, 1], 1)]); // 6 positions cached, 2 pages, idle
        // 12 tokens need 3 pages: the idle cache in the other slot is evicted.
        let got = run_all(&mut b, vec![req(&[2; 12], 3)]);
        assert_eq!(got[0].0.len(), 3);
        // Two streams each wanting 2+ pages of growth: one ends when the pool is full.
        let got = run_all(&mut b, vec![req(&[3; 6], 20), req(&[4; 6], 20)]);
        assert!(got.iter().all(|g| g.1 == Some(Finish::Length)));
        assert!(got.iter().all(|g| g.0.len() >= 2), "both streams ran: {got:?}");
    }

    #[test]
    fn displaced_prompt_comes_back_from_the_ram_cache() {
        let a = [1, 2, 3, 4, 5];
        let a2 = [1, 2, 3, 4, 5, 6];
        for (limit, want) in [(1 << 20, 5), (0, 0)] {
            let mut b = Batcher::new(fake(1), vec![]);
            b.ram_cache = limit;
            run_all(&mut b, vec![req(&a, 1)]);
            run_all(&mut b, vec![req(&[0, 0, 0], 1)]); // displaces A from the only slot
            b.engine.computed[0].clear();
            assert_eq!(run_all(&mut b, vec![req(&a2, 1)])[0].2, Some(want), "limit {limit}");
            assert_eq!(b.engine.computed[0].len(), a2.len() - want, "only the uncached suffix is computed");
        }
        // B was saved when A2 displaced it; it is restored the same way.
        let mut b = Batcher::new(fake(1), vec![]);
        b.ram_cache = 1 << 20;
        run_all(&mut b, vec![req(&[0, 0, 0], 1)]);
        run_all(&mut b, vec![req(&a, 1)]);
        assert_eq!(run_all(&mut b, vec![req(&[0, 0, 0], 1)])[0].2, Some(2));
    }

    #[test]
    fn schema_constrains_output_to_complete_json() {
        let bytes: Vec<Vec<u8>> = ["{", "\"a\"", ":", "1", "}", "x", ",", ""].iter().map(|t| t.as_bytes().to_vec()).collect();
        let mut b = Batcher::new(fake(1), vec![7]);
        b.token_bytes = bytes.clone();
        // Unconstrained, the fake ends at once (token 7 is EOG).
        assert_eq!(run_all(&mut b, vec![req(&[6], 30)])[0].0, Vec::<u32>::new());
        for schema in [json!({}), json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]})] {
            // The fake's favourite after 4 is "x": masked, it must start the JSON instead.
            let (mut r, seen) = req(&[4], 40);
            r.sampling.schema = Some(schema.clone());
            let got = run_all(&mut b, vec![(r, seen)]).remove(0);
            let text: Vec<u8> = got.0.iter().flat_map(|&t| bytes[t as usize].clone()).collect();
            let done = crate::constrain::check(&schema, &[&text[..], b" "].concat());
            assert_eq!((done, got.1), (crate::constrain::State::Done, Some(Finish::Stop)), "{schema} -> {:?}", String::from_utf8_lossy(&text));
            println!("{schema} -> {}", String::from_utf8_lossy(&text));
        }
    }

    #[test]
    fn sampling_respects_filters() {
        let s = Sampling { temperature: 1.0, top_k: 2, top_p: 1.0, min_p: 0.0, seed: 1, schema: None };
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
