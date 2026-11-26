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
}

impl<E: Engine> Batcher<E> {
    pub fn new(engine: E, eog: Vec<u32>) -> Self {
        let slots = (0..engine.n_seq()).map(|_| Slot::default()).collect();
        Self { engine, eog, slots, queue: Default::default(), tick: 0 }
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
            self.tick += 1;
            let slot = &mut self.slots[si];
            slot.last_used = self.tick;
            slot.cached.truncate(n);
            self.engine.select(si);
            self.engine.truncate(n);
            if (req.sink)(Event::Started { n_cached: n }) {
                let rng = req.sampling.seed;
                slot.job = Some(Job { req, order: self.tick, n_out: 0, pending: None, rng });
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
        let t = sample(logits, &job.req.sampling, &mut job.rng);
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
            self.slots[v].cached.clear();
            self.engine.select(v);
            self.engine.truncate(0);
        }
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
