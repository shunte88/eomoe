//! Phase 4 gate: VRAM expert cache.
//! `cargo test --release --test phase4_golden -- --include-ignored --nocapture --test-threads 1`

mod common;
use common::{golden, model_path};
use eomoe::expert_cache::ExpertCache;
use eomoe::gguf::Gguf;
use eomoe::gpu::GpuModel;
use eomoe::qwen3moe::Engine;

/// Oracle (llama.cpp ClassLRU, 4800 MiB = 879 slots per group) on the same run, ubatch <= 8:
/// hits = 117973, misses = 36917 (76.17%), from its log.
const ORACLE_HITS: usize = 117973;
const ORACLE_LOOKUPS: usize = 117973 + 36917;
const ORACLE_SLOTS: usize = 879;
/// Oracle decode: prompt 1 of greedy.json, 256 greedy tokens, cache warm from the greedy cases.
const ORACLE_DECODE_TOK_S: f32 = 27.2;

/// One expert-selection lookup: layer, experts per token, selected ids (n_tokens * n_used).
struct Lookup {
    layer: usize,
    n_used: usize,
    ids: Vec<u32>,
}

/// Parse llama.cpp's MCT1 trace: groups of layers sharing a cache, then the lookups.
fn read_trace() -> (Vec<Vec<usize>>, usize, Vec<Lookup>) {
    let b = std::fs::read(format!("{}/tests/golden/ref_trace.mct", env!("CARGO_MANIFEST_DIR"))).unwrap();
    assert_eq!(&b[..4], b"MCT1");
    let o = std::cell::Cell::new(4);
    let u32_ = || {
        let v = u32::from_le_bytes(b[o.get()..o.get() + 4].try_into().unwrap());
        o.set(o.get() + 4);
        v
    };
    let (_n_layer, _n_used, n_group) = (u32_(), u32_(), u32_());
    let mut groups = vec![];
    let mut n_expert = 0;
    for _ in 0..n_group {
        let (_dev, ne, nl) = (u32_(), u32_(), u32_());
        n_expert = ne as usize;
        groups.push((0..nl).map(|_| u32_() as usize).collect::<Vec<_>>());
        (0..2 * 2 + 3 * 2).for_each(|_| { u32_(); }); // f64 split, split_sum; u64 host bytes x2, alignment
        let nt = u32_();
        (0..2 * nt).for_each(|_| { u32_(); }); // u64 bytes per tensor
    }
    let mut lookups = vec![];
    while o.get() < b.len() {
        let (layer, n_used, n) = (u32_() as usize, u32_() as usize, u32_() as usize);
        lookups.push(Lookup { layer, n_used, ids: (0..n).map(|_| u32_()).collect() });
    }
    (groups, n_expert, lookups)
}

#[test]
fn cache_hit_rate_on_ref_trace_beats_oracle() {
    let (groups, n_expert, lookups) = read_trace();
    let n_layer = groups.iter().flatten().max().unwrap() + 1;
    let mut caches: Vec<ExpertCache> = groups.iter().map(|_| ExpertCache::new(n_layer, n_expert, ORACLE_SLOTS)).collect();
    let group_of = |l: usize| groups.iter().position(|g| g.contains(&l)).unwrap();
    let (mut hits, mut total) = (0usize, 0usize);
    for lk in &lookups {
        let p = caches[group_of(lk.layer)].plan(lk.layer, &lk.ids, lk.ids.len() / lk.n_used);
        if lk.ids.len() <= 8 * lk.n_used {
            hits += p.hits;
            total += p.hits + p.fills.len();
        }
    }
    println!("ref_trace ubatch <= 8: hits {hits} / {total} = {:.2}% (oracle {ORACLE_HITS} / {ORACLE_LOOKUPS})", 100.0 * hits as f64 / total as f64);
    assert_eq!(total, ORACLE_LOOKUPS, "replay must see the oracle's lookups");
    assert!(hits >= ORACLE_HITS, "hits {hits} < oracle {ORACLE_HITS}");
}

/// The oracle's perf run: the 4 greedy cases (warming the cache), then prompt 1 + 256 greedy
/// tokens. Decode tok/s must reach the oracle's, with the oracle's cache budget.
#[test]
#[ignore = "needs model file + GPU"]
fn gpu_decode_beats_oracle_tok_s() {
    let mut m = GpuModel::with_cache(Gguf::open(model_path()).unwrap(), 512, 4800).unwrap();
    println!("VRAM allocated: {:.0} MiB", m.vram_bytes as f64 / 1048576.0);
    let nv = m.n_vocab();
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    let cases = golden("greedy.json");
    let mut decode = |prompt: &[u32], n: usize| -> (Vec<u32>, f32) {
        m.reset();
        let l = m.forward(prompt);
        let mut got = vec![argmax(&l[(prompt.len() - 1) * nv..])];
        let t = std::time::Instant::now();
        while got.len() < n {
            let l = m.forward(&[*got.last().unwrap()]);
            got.push(argmax(&l));
        }
        (got, (n - 1) as f32 / t.elapsed().as_secs_f32())
    };
    for (ci, case) in cases.as_array().unwrap().iter().enumerate() {
        let prompt: Vec<u32> = serde_json::from_value(case["prompt_tokens"].clone()).unwrap();
        let want: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let (got, tps) = decode(&prompt, want.len());
        println!("warm-up case {ci}: {tps:.1} tok/s");
        assert_eq!(got, want, "case {ci}: cached engine must still reproduce greedy.json");
    }
    let prompt: Vec<u32> = serde_json::from_value(cases[1]["prompt_tokens"].clone()).unwrap();
    let (_, tps) = decode(&prompt, 257);
    let st = m.cache_stats();
    println!(
        "decode 256 tok: {tps:.1} tok/s (oracle {ORACLE_DECODE_TOK_S}); cache hits {} misses {} = {:.2}%, uploaded {:.0} MiB",
        st.hits,
        st.misses,
        100.0 * st.hits as f64 / (st.hits + st.misses) as f64,
        st.bytes as f64 / 1048576.0
    );
    assert!(tps >= ORACLE_DECODE_TOK_S, "decode {tps:.1} tok/s < oracle {ORACLE_DECODE_TOK_S}");
}
