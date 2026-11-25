//! Which experts live in the VRAM expert slots: a pure policy, no CUDA, so it can be replayed
//! against llama.cpp's MCT1 traces (tests/phase4_golden.rs).
//!
//! ClassLRU, as llama.cpp's MoE cache (src/llama-moe-cache.cpp):
//! - Each expert of each layer has a CRF (combined recency and frequency): its uses, decayed with
//!   a half-life of 128 tokens of its layer. The CRF is kept across evictions.
//! - Slots sit in one LRU list per class, where class = floor(log2(CRF)).
//! - On a miss the victim is an empty slot, or else the least recently used slot of each class
//!   list that the current lookup does not select, with the largest age / 2^class.
//!
//! One cache serves a group of layers whose experts have the same byte size.

const N_CLASS: usize = 8;
const HALF_LIFE: f32 = 128.0;
const NIL: usize = usize::MAX;

/// What one lookup needs: experts already resident, and experts to upload into a slot.
pub struct Plan {
    pub hits: usize,
    /// (expert, slot), ascending by expert.
    pub fills: Vec<(usize, usize)>,
}

pub struct ExpertCache {
    n_expert: usize,
    /// [n_layer * n_expert] slot of each expert, NIL if not resident.
    slot_of: Vec<usize>,
    /// [n_slots] key (layer * n_expert + expert) held by each slot, NIL if empty.
    key_of: Vec<usize>,
    prev: Vec<usize>,
    next: Vec<usize>,
    head: [usize; N_CLASS],
    tail: [usize; N_CLASS],
    list_of: Vec<u8>,
    crf: Vec<f32>,
    last_use: Vec<i64>,
    /// [n_layer] tokens seen by each layer.
    clock: Vec<i64>,
    /// Experts selected by the current lookup.
    seen: Vec<bool>,
}

impl ExpertCache {
    pub fn new(n_layer: usize, n_expert: usize, n_slots: usize) -> Self {
        assert!(n_slots > 0);
        let mut head = [NIL; N_CLASS];
        let mut tail = [NIL; N_CLASS];
        (head[0], tail[0]) = (0, n_slots - 1);
        Self {
            n_expert,
            slot_of: vec![NIL; n_layer * n_expert],
            key_of: vec![NIL; n_slots],
            prev: (0..n_slots).map(|s| s.wrapping_sub(1)).collect(),
            next: (0..n_slots).map(|s| if s + 1 < n_slots { s + 1 } else { NIL }).collect(),
            head,
            tail,
            list_of: vec![0; n_slots],
            crf: vec![0.0; n_layer * n_expert],
            last_use: vec![-1; n_layer * n_expert],
            clock: vec![0; n_layer],
            seen: vec![false; n_expert],
        }
    }

    pub fn n_slots(&self) -> usize {
        self.key_of.len()
    }

    /// The slot holding expert `e` of layer `il`, if resident.
    pub fn slot(&self, il: usize, e: usize) -> Option<usize> {
        Some(self.slot_of[il * self.n_expert + e]).filter(|&s| s != NIL)
    }

    /// Give a slot to every expert selected by `ids` (n_tokens x n_used) in layer `il`.
    /// The returned fills must be uploaded before the experts are computed.
    /// Panics if the lookup selects more distinct experts than there are slots.
    pub fn plan(&mut self, il: usize, ids: &[u32], n_tokens: usize) -> Plan {
        self.seen.fill(false);
        let mut uniq = vec![];
        for &e in ids {
            let e = e as usize;
            if !self.seen[e] {
                self.seen[e] = true;
                uniq.push(e);
            }
        }
        assert!(uniq.len() <= self.n_slots(), "layer {il}: {} experts > {} slots", uniq.len(), self.n_slots());

        // Update every selected CRF and promote the hits before choosing victims.
        let (mut hits, mut misses) = (0, vec![]);
        for &e in &uniq {
            let key = il * self.n_expert + e;
            self.crf[key] = self.decayed_crf(key) + 1.0;
            self.last_use[key] = self.clock[il];
            match self.slot_of[key] {
                NIL => misses.push(e),
                s => {
                    self.move_to(s, self.class_of(key));
                    hits += 1;
                }
            }
        }
        misses.sort_unstable();
        let mut fills = vec![];
        for e in misses {
            let s = self.victim(il);
            if self.key_of[s] != NIL {
                self.slot_of[self.key_of[s]] = NIL;
            }
            let key = il * self.n_expert + e;
            self.key_of[s] = key;
            self.slot_of[key] = s;
            self.move_to(s, self.class_of(key));
            fills.push((e, s));
        }
        self.clock[il] += n_tokens as i64;
        Plan { hits, fills }
    }

    fn decayed_crf(&self, key: usize) -> f32 {
        let il = key / self.n_expert;
        self.crf[key] * (-((self.clock[il] - self.last_use[key]) as f32) / HALF_LIFE).exp2()
    }

    /// After a use the CRF is >= 1, so the cast floors log2.
    fn class_of(&self, key: usize) -> usize {
        (self.crf[key].log2() as usize).min(N_CLASS - 1)
    }

    /// Selected by the current lookup: must not be evicted (nor slots filled by it).
    fn is_selected(&self, il: usize, key: usize) -> bool {
        key != NIL && key / self.n_expert == il && self.seen[key % self.n_expert]
    }

    fn victim(&self, il: usize) -> usize {
        let mut best = (NIL, -1.0f64);
        for c in 0..N_CLASS {
            let mut s = self.head[c];
            while s != NIL && self.is_selected(il, self.key_of[s]) {
                s = self.next[s];
            }
            if s == NIL {
                continue;
            }
            let key = self.key_of[s];
            if key == NIL {
                return s;
            }
            let age = self.clock[key / self.n_expert] - self.last_use[key];
            let score = age as f64 / (1u64 << c) as f64;
            if score > best.1 {
                best = (s, score);
            }
        }
        assert!(best.0 != NIL, "no expert cache victim");
        best.0
    }

    /// Move slot s to the most recently used end of list c.
    fn move_to(&mut self, s: usize, c: usize) {
        let from = self.list_of[s] as usize;
        if from == c && s == self.tail[from] {
            return;
        }
        let (p, n) = (self.prev[s], self.next[s]);
        if p != NIL { self.next[p] = n } else { self.head[from] = n }
        if n != NIL { self.prev[n] = p } else { self.tail[from] = p }
        self.prev[s] = self.tail[c];
        self.next[s] = NIL;
        if self.tail[c] != NIL { self.next[self.tail[c]] = s } else { self.head[c] = s }
        self.tail[c] = s;
        self.list_of[s] = c as u8;
    }
}
