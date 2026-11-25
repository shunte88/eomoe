//! Phase 2 gate: CPU reference forward pass vs the llama.cpp oracle.
//! `cargo test --release --test phase2_golden -- --include-ignored --nocapture`

use eomoe::gguf::Gguf;

fn model_path() -> String {
    std::env::var("EOMOE_TEST_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/.local/share/local-code-llm/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
            std::env::var("HOME").unwrap()
        )
    })
}

fn golden(name: &str) -> serde_json::Value {
    let p = format!("{}/tests/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// Rows dequantized by llama.cpp's gguf-py (tools/capture_dequant.py).
#[test]
#[ignore = "needs model file"]
fn dequant_matches_gguf_py() {
    let g = Gguf::open(model_path()).unwrap();
    for case in golden("dequant.json").as_array().unwrap() {
        let t = g.tensor(case["tensor"].as_str().unwrap()).unwrap();
        let row = case["row"].as_u64().unwrap() as usize;
        let want: Vec<f32> = serde_json::from_value(case["values"].clone()).unwrap();
        let mut got = vec![0f32; want.len()];
        eomoe::quant::dequant_row(t.ty, g.row(t, row), &mut got);
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() <= 1e-6 * b.abs().max(1e-3), "{} row {row} [{i}]: {a} vs {b}", t.name);
        }
    }
}

/// Near-tie: oracle top-1/top-2 gap below this exempts the top-1 check.
const TIE: f32 = 0.05;
/// llama.cpp's own CPU backend vs its CUDA greedy.json: mean |dlogprob| over the oracle's
/// top-5 (tools/measure_backend_floor.py, 2026-11-25). Activation quantization (Q8_K/q8_1)
/// makes a tighter per-token tolerance unreachable even for llama.cpp across backends.
const FLOOR_MEAN: f64 = 0.2414;

fn log_softmax(l: &[f32]) -> Vec<f32> {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + l.iter().map(|x| ((x - m) as f64).exp()).sum::<f64>().ln() as f32;
    l.iter().map(|x| x - lse).collect()
}

/// Teacher-forced: feed prompt + the oracle's own greedy tokens in one pass and compare
/// every step with the oracle (so one divergence can't derail the rest).
/// Gate: top-1 equal on every non-tie step; mean |dlogprob| <= llama.cpp's backend floor.
#[test]
#[ignore = "needs model file"]
fn forward_matches_oracle_greedy() {
    let mut m = eomoe::cpu::CpuModel::new(Gguf::open(model_path()).unwrap()).unwrap();
    let mut failures = vec![];
    let (mut worst, mut dsum, mut dn, mut top5_diff, mut steps) = (0f32, 0f64, 0u32, 0u32, 0u32);
    for (ci, case) in golden("greedy.json").as_array().unwrap().iter().enumerate() {
        let prompt: Vec<u32> = serde_json::from_value(case["prompt_tokens"].clone()).unwrap();
        let gen: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let mut seq = prompt.clone();
        seq.extend_from_slice(&gen[..gen.len() - 1]);

        m.reset();
        let t0 = std::time::Instant::now();
        let logits = m.forward(&seq);
        let el = t0.elapsed().as_secs_f32();
        let nv = m.n_vocab();
        let mut case_worst = 0f32;
        for (k, step) in case["probs"].as_array().unwrap().iter().enumerate() {
            let lp = log_softmax(&logits[(prompt.len() - 1 + k) * nv..][..nv]);
            let top: Vec<(u32, f32)> = step["top_logprobs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| (e["id"].as_u64().unwrap() as u32, e["logprob"].as_f64().unwrap() as f32))
                .collect();
            let mut ours: Vec<u32> = (0..nv as u32).collect();
            ours.select_nth_unstable_by(4, |a, b| lp[*b as usize].total_cmp(&lp[*a as usize]));
            ours.truncate(5);
            ours.sort_by(|a, b| lp[*b as usize].total_cmp(&lp[*a as usize]));

            steps += 1;
            if ours[0] != top[0].0 && top[0].1 - top[1].1 >= TIE {
                failures.push(format!("case {ci} step {k}: top1 {} vs oracle {}", ours[0], top[0].0));
            }
            for &(id, olp) in &top {
                let d = (lp[id as usize] - olp).abs();
                case_worst = case_worst.max(d);
                dsum += d as f64;
                dn += 1;
            }
            let mut b: Vec<u32> = top.iter().map(|t| t.0).collect();
            ours.sort();
            b.sort();
            top5_diff += (ours != b) as u32;
        }
        worst = worst.max(case_worst);
        println!(
            "case {ci}: {} tokens in {el:.1}s ({:.1} tok/s), max |dlogprob| {case_worst:.4}",
            seq.len(),
            seq.len() as f32 / el
        );
    }
    let mean = dsum / dn as f64;
    println!(
        "top1 fails {}/{steps}, mean |dlogprob| {mean:.4} (floor {FLOOR_MEAN}), max {worst:.4}, top5 set differs on {top5_diff} steps",
        failures.len()
    );
    if mean > FLOOR_MEAN {
        failures.push(format!("mean |dlogprob| {mean:.4} > llama.cpp backend floor {FLOOR_MEAN}"));
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
