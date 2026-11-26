//! Phase 3 gate: CUDA engine vs the CPU reference and the llama.cpp oracle.
//! `cargo test --release --test phase3_golden -- --include-ignored --nocapture --test-threads 1`

mod common;
use common::{golden, model_path};
use eomoe::cpu::CpuModel;
use eomoe::gguf::Gguf;
use eomoe::gpu::GpuModel;
use eomoe::qwen3moe::Engine;

fn gpu() -> GpuModel {
    GpuModel::new(Gguf::open(model_path()).unwrap(), 256).unwrap()
}

fn log_softmax(l: &[f32]) -> Vec<f32> {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + l.iter().map(|x| ((x - m) as f64).exp()).sum::<f64>().ln() as f32;
    l.iter().map(|x| x - lse).collect()
}

/// Both engines compute in f32 and store KV in f16 (llama.cpp's default); only summation order
/// differs. End to end, that is chaotic: when two experts tie for the 8th routing slot (router
/// logit gap 5.9e-5 at greedy.json case 2, position 8, layer 4), any reordering can flip the
/// choice, and every later position of that case then moves by up to ~1. Positions before such a
/// flip agree to <= 0.003. A kernel bug moves every position of every case, so the gate is on the
/// median (and a sanity max); accuracy against llama.cpp is gated by the oracle floor test.
#[test]
#[ignore = "needs model file + GPU"]
fn gpu_matches_cpu_reference() {
    const P50_TOL: f32 = 0.001;
    const MAX_TOL: f32 = 2.0;
    let mut c = CpuModel::new(Gguf::open(model_path()).unwrap()).unwrap();
    let mut gm = gpu();
    let nv = c.n_vocab();
    let (mut worst, mut diffs) = (0f32, vec![]);
    for case in golden("greedy.json").as_array().unwrap() {
        let mut seq: Vec<u32> = serde_json::from_value(case["prompt_tokens"].clone()).unwrap();
        let gen: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        seq.extend_from_slice(&gen[..gen.len() - 1]);
        c.reset();
        gm.reset();
        let (a, b) = (c.forward(&seq), gm.forward(&seq));
        assert_eq!(a.len(), b.len());
        for (ra, rb) in a.chunks_exact(nv).zip(b.chunks_exact(nv)) {
            let (la, lb) = (log_softmax(ra), log_softmax(rb));
            let mut idx: Vec<usize> = (0..nv).collect();
            idx.select_nth_unstable_by(4, |x, y| la[*y].total_cmp(&la[*x]));
            for &i in &idx[..5] {
                let d = (la[i] - lb[i]).abs();
                worst = worst.max(d);
                diffs.push(d);
            }
        }
    }
    diffs.sort_by(f32::total_cmp);
    let q = |f: f64| diffs[((diffs.len() - 1) as f64 * f) as usize];
    println!("GPU vs CPU: max |dlogprob| over CPU top-5 = {worst:.5}; mean {:.5}, p50 {:.5}, p99 {:.5}, {} of {} > 0.02",
        diffs.iter().sum::<f32>() / diffs.len() as f32, q(0.5), q(0.99), diffs.iter().filter(|&&d| d > 0.02).count(), diffs.len());
    assert!(q(0.5) <= P50_TOL && worst <= MAX_TOL, "GPU diverges from CPU reference: p50 {} max {worst}", q(0.5));
}

#[test]
#[ignore = "needs model file + GPU"]
fn gpu_forward_matches_oracle_greedy() {
    common::greedy_floor_gate(&mut gpu());
}

/// Incremental decode through the KV cache: prefill the prompt, then one token per call.
/// Greedy tokens must reproduce the oracle's (min top-1/top-2 gap in greedy.json is 0.58).
#[test]
#[ignore = "needs model file + GPU"]
fn gpu_incremental_decode_matches_oracle_tokens() {
    let mut m = gpu();
    println!("VRAM allocated: {:.0} MiB", m.vram_bytes as f64 / 1048576.0);
    let nv = m.n_vocab();
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    for (ci, case) in golden("greedy.json").as_array().unwrap().iter().enumerate() {
        let prompt: Vec<u32> = serde_json::from_value(case["prompt_tokens"].clone()).unwrap();
        let want: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        m.reset();
        let t0 = std::time::Instant::now();
        let l = m.forward(&prompt);
        let pf = t0.elapsed().as_secs_f32();
        let mut got = vec![argmax(&l[(prompt.len() - 1) * nv..])];
        let t1 = std::time::Instant::now();
        while got.len() < want.len() {
            let l = m.forward(&[*got.last().unwrap()]);
            got.push(argmax(&l));
        }
        let dt = t1.elapsed().as_secs_f32();
        println!(
            "case {ci}: prefill {} tok {:.1} tok/s, decode {} tok {:.1} tok/s",
            prompt.len(),
            prompt.len() as f32 / pf,
            want.len() - 1,
            (want.len() - 1) as f32 / dt
        );
        assert_eq!(got, want, "case {ci}");
    }
}

/// Hybrid decode (Phase 7b): four streams decoded together, cache misses computed on the CPU
/// (`cpu_moe`) from the first step (a small cache forces many). Each row's logits are gated like
/// gpu_matches_cpu_reference against the CPU reference over the same sequence.
#[test]
#[ignore = "needs model file + GPU"]
fn gpu_hybrid_batched_decode_matches_cpu_reference() {
    const P50_TOL: f32 = 0.001;
    const MAX_TOL: f32 = 2.0;
    let mut c = CpuModel::new(Gguf::open(model_path()).unwrap()).unwrap();
    let mut gm = GpuModel::with_cache(Gguf::open(model_path()).unwrap(), 1024, 1000).unwrap();
    gm.set_n_seq(4);
    gm.set_hybrid(true);
    let nv = c.n_vocab();
    let cases: Vec<(Vec<u32>, Vec<u32>)> = (golden("greedy.json").as_array().unwrap().iter())
        .map(|k| (serde_json::from_value(k["prompt_tokens"].clone()).unwrap(), serde_json::from_value(k["tokens"].clone()).unwrap()))
        .collect();
    let steps = cases.iter().map(|(_, g)| g.len() - 1).min().unwrap();
    // CPU reference: logits of every position of prompt + generated tokens.
    let refs: Vec<Vec<f32>> = cases.iter().map(|(p, g)| {
        c.reset();
        c.forward(&[&p[..], &g[..steps]].concat())
    }).collect();
    for (i, (p, _)) in cases.iter().enumerate() {
        gm.select(i);
        gm.forward_last(p);
    }
    let (mut diffs, mut worst, mut top1) = (vec![], 0f32, 0);
    for k in 0..steps {
        let rows: Vec<(usize, u32)> = cases.iter().enumerate().map(|(i, (_, g))| (i, g[k])).collect();
        let l = gm.decode_batch(&rows);
        for (i, (p, _)) in cases.iter().enumerate() {
            let pos = p.len() + k; // the row that predicts token k+1
            let (la, lb) = (log_softmax(&refs[i][pos * nv..(pos + 1) * nv]), log_softmax(&l[i * nv..(i + 1) * nv]));
            let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            top1 += (am(&la) == am(&lb)) as usize;
            let mut idx: Vec<usize> = (0..nv).collect();
            idx.select_nth_unstable_by(4, |x, y| la[*y].total_cmp(&la[*x]));
            for &j in &idx[..5] {
                let d = (la[j] - lb[j]).abs();
                worst = worst.max(d);
                diffs.push(d);
            }
        }
    }
    diffs.sort_by(f32::total_cmp);
    let p50 = diffs[(diffs.len() - 1) / 2];
    let s = gm.cache_stats();
    println!("hybrid batched decode vs CPU: p50 {p50:.5}, max {worst:.5}, top-1 {top1}/{}, cache hits {:.1}%",
        steps * cases.len(), 100.0 * s.hits as f64 / (s.hits + s.misses) as f64);
    assert!(p50 <= P50_TOL && worst <= MAX_TOL, "hybrid decode diverges: p50 {p50} max {worst}");
    assert!(top1 * 100 >= steps * cases.len() * 95, "hybrid top-1 agreement {top1}/{}", steps * cases.len());
}
