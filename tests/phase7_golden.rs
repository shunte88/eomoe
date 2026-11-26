//! Phase 7 gate: continuous batching.
//! `cargo test --release --test phase7_golden -- --include-ignored --nocapture --test-threads 1`

mod common;
use common::{golden, model_path};
use eomoe::generate::{Batcher, Event, Request, Sampling};
use eomoe::gguf::Gguf;
use eomoe::gpu::GpuModel;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const N_OUT: usize = 64;

fn greedy(prompt: Vec<u32>, out: Arc<Mutex<Vec<u32>>>) -> Request {
    let sink = Box::new(move |ev: Event| {
        if let Event::Token(t) = ev {
            out.lock().unwrap().push(t);
        }
        true
    });
    Request { prompt, max_tokens: N_OUT, sampling: Sampling { temperature: 0.0, ..Default::default() }, sink }
}

/// Run `prompts` together on fresh slots; tokens per prompt and wall seconds.
fn run(m: GpuModel, prompts: &[Vec<u32>]) -> (GpuModel, Vec<Vec<u32>>, f32) {
    let mut b = Batcher::new(m, vec![]); // no EOG: every stream decodes N_OUT tokens
    let outs: Vec<_> = prompts.iter().map(|_| Arc::new(Mutex::new(vec![]))).collect();
    for (p, o) in prompts.iter().zip(&outs) {
        b.submit(greedy(p.clone(), o.clone()));
    }
    let (t0, c0) = (Instant::now(), b.engine.cache_stats());
    while !b.idle() {
        b.step();
    }
    let dt = t0.elapsed().as_secs_f32();
    let c = b.engine.cache_stats();
    let (h, mi) = (c.hits - c0.hits, c.misses - c0.misses);
    println!("  {} streams: {dt:.2}s, hits {:.1}%, {} MiB uploaded", prompts.len(), 100.0 * h as f64 / (h + mi) as f64, (c.bytes - c0.bytes) >> 20);
    (b.engine, outs.iter().map(|o| o.lock().unwrap().clone()).collect(), dt)
}

fn setup(hybrid: bool) -> (GpuModel, Vec<Vec<u32>>) {
    let mut m = GpuModel::with_cache(Gguf::open(model_path()).unwrap(), 4096, usize::MAX >> 20).unwrap();
    m.set_n_seq(4);
    m.set_hybrid(hybrid);
    let prompts = (golden("greedy.json").as_array().unwrap().iter())
        .map(|c| serde_json::from_value(c["prompt_tokens"].clone()).unwrap())
        .collect();
    (m, prompts)
}

/// Each prompt alone, then all together: (solo tokens, solo seconds, together tokens, seconds).
fn solo_and_together(hybrid: bool) -> (Vec<Vec<u32>>, f32, Vec<Vec<u32>>, f32) {
    let (mut m, prompts) = setup(hybrid);
    let (mut solo, mut t_solo) = (vec![], 0.0);
    for p in &prompts {
        let (m2, mut out, dt) = run(m, std::slice::from_ref(p));
        m = m2;
        solo.push(out.remove(0));
        t_solo += dt;
    }
    let (_, together, t_all) = run(m, &prompts);
    let n = (prompts.len() * N_OUT) as f32;
    println!("{} streams x {N_OUT} tok: one at a time {:.1} tok/s, together {:.1} tok/s", prompts.len(), n / t_solo, n / t_all);
    (solo, t_solo, together, t_all)
}

/// Four streams decoded together give each exactly the tokens it gets alone: batched rows use
/// the same per-row kernels, and paged attention reads each sequence's own KV. (Hybrid off: with
/// it, which experts run on the CPU depends on cache state, so results differ by rounding.)
#[test]
#[ignore]
fn concurrent_greedy_matches_solo() {
    let (solo, _, together, _) = solo_and_together(false);
    for (i, (s, t)) in solo.iter().zip(&together).enumerate() {
        assert_eq!(s.len(), N_OUT);
        assert_eq!(s, t, "stream {i}: batched tokens differ from solo");
    }
}

/// Batching must pay: four streams together finish at >= 1.3x the aggregate rate of running
/// them one at a time (hybrid on, the server's all-free-VRAM cache). Gate revised from 1.5x
/// (STATUS.md, Phase 7): CPU experts and expert uploads share one DDR4 bus, which bounds it.
#[test]
#[ignore]
fn batching_beats_one_at_a_time() {
    let (_, t_solo, _, t_all) = solo_and_together(true);
    assert!(t_solo / t_all >= 1.3, "batched speedup {:.2}x < 1.3x", t_solo / t_all);
}
