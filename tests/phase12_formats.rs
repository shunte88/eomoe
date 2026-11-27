//! Phase 12d gate: Q8_0 and MXFP4 rows of gpt-oss-20b dequantize to llama.cpp's gguf-py values
//! (tests/golden/dequant_gpt-oss.json from tools/capture_dequant.py).
//! `cargo test --release --test phase12_formats -- --include-ignored`

mod common;
use common::{golden, gptoss_path};
use eomoe::gguf::Gguf;

#[test]
#[ignore = "needs the gpt-oss-20b model file"]
fn dequant_matches_gguf_py() {
    let g = Gguf::open(gptoss_path()).unwrap();
    for case in golden("dequant_gpt-oss.json").as_array().unwrap() {
        let t = g.tensor(case["tensor"].as_str().unwrap()).unwrap();
        let row = case["row"].as_u64().unwrap() as usize;
        let want: Vec<f32> = serde_json::from_value(case["values"].clone()).unwrap();
        let mut got = vec![0f32; want.len()];
        eomoe::quant::dequant_row(t.ty, g.row(t, row), &mut got);
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() <= 1e-6 * b.abs().max(1e-3), "{} {} row {row} [{i}]: {a} vs {b}", t.name, case["type"]);
        }
    }
}
