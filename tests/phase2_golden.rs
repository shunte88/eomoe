//! Phase 2 gate: CPU reference forward pass vs the llama.cpp oracle.
//! `cargo test --release --test phase2_golden -- --include-ignored --nocapture`

mod common;
use common::{golden, model_path};
use eomoe::gguf::Gguf;

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

#[test]
#[ignore = "needs model file"]
fn forward_matches_oracle_greedy() {
    let mut m = eomoe::cpu::CpuModel::new(Gguf::open(model_path()).unwrap()).unwrap();
    common::greedy_floor_gate(&mut m);
}
