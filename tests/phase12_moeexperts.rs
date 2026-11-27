//! Phase 12a gate: `--moeexperts N` routes each token to the top N experts instead of the
//! model's n_expert_used (koboldcpp's knob; llama.cpp: --override-kv <arch>.expert_used_count).
//! N = the model's own count changes nothing (bit-identical logits); fewer experts must take
//! effect, keep most greedy choices, and is reported with its decode speed.
//! `cargo test --release --test phase12_moeexperts -- --include-ignored --nocapture --test-threads 1`

mod common;
use common::{golden, model_path};
use eomoe::gguf::Gguf;
use eomoe::gpu::GpuModel;
use eomoe::qwen3moe::Engine;
use std::time::Instant;

fn argmax(r: &[f32]) -> usize {
    (0..r.len()).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap()
}

/// Teacher-forced logits for every greedy.json case.
fn logits(m: &mut GpuModel) -> Vec<Vec<f32>> {
    golden("greedy.json")
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let mut seq: Vec<u32> = serde_json::from_value(case["prompt_tokens"].clone()).unwrap();
            let gen: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
            seq.extend_from_slice(&gen[..gen.len() - 1]);
            m.reset();
            m.forward(&seq)
        })
        .collect()
}

/// Decode tok/s over 64 single-token steps.
fn tok_s(m: &mut GpuModel) -> f64 {
    m.reset();
    let l = m.forward(&[9707, 11, 1879, 0]);
    let mut t = argmax(&l[l.len() - m.n_vocab()..]) as u32;
    let t0 = Instant::now();
    for _ in 0..64 {
        t = argmax(&m.forward(&[t])) as u32;
    }
    64.0 / t0.elapsed().as_secs_f64()
}

#[test]
#[ignore = "needs model file + GPU"]
fn moe_experts_override() {
    let mut m = GpuModel::with_cache(Gguf::open(model_path()).unwrap(), 512, usize::MAX >> 20).unwrap();
    let nv = m.n_vocab();
    // Which experts sit in the VRAM cache sets the summation order (streamed vs cached), so
    // logits depend on what ran before: up to ~1 at a routing near-tie (see phase3_golden).
    // Repeating the same pass back to back is bit-identical, so compare like that.
    logits(&mut m);
    let base = logits(&mut m);
    assert!(m.set_n_expert_used(0).is_err() && m.set_n_expert_used(9).is_err(), "N outside 1..=8 must be refused");
    m.set_n_expert_used(8).unwrap();
    assert!(logits(&mut m) == base, "N = n_expert_used must be bit-identical");
    let full = tok_s(&mut m);

    for n in [6, 4] {
        m.set_n_expert_used(8).unwrap();
        logits(&mut m);
        let base = logits(&mut m);
        m.set_n_expert_used(n).unwrap();
        let l = logits(&mut m);
        assert!(l != base, "N={n} changed nothing");
        let (mut agree, mut total) = (0, 0);
        for (a, b) in base.iter().zip(&l) {
            for (ra, rb) in a.chunks_exact(nv).zip(b.chunks_exact(nv)) {
                agree += (argmax(ra) == argmax(rb)) as usize;
                total += 1;
            }
        }
        let frac = agree as f64 / total as f64;
        println!("moeexperts {n}: top-1 agreement {agree}/{total} = {frac:.3}; decode {:.1} tok/s (8 experts: {full:.1})", tok_s(&mut m));
        assert!(frac >= 0.6, "N={n} top-1 agreement {frac:.3} < 0.6");
    }
}
