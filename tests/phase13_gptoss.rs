//! Phase 13 gates: gpt-oss-20b.
//! (a) Tokenizer: o200k (`tokenizer.ggml.pre` = gpt-4o) matches HF `tokenizers` running the
//!     model's own tokenizer.json (tests/golden/tokenize_gpt-oss.json, tools/capture_tokenize_hf.py).
//! `cargo test --release --test phase13_gptoss -- --include-ignored`

mod common;
use common::{golden, gptoss_path};
use eomoe::{gguf::Gguf, tokenizer::Tokenizer};

#[test]
#[ignore = "needs the gpt-oss-20b model file"]
fn tokenize_matches_hf() {
    let tok = Tokenizer::from_gguf(&Gguf::open(gptoss_path()).unwrap()).unwrap();
    let mut bad = vec![];
    for case in golden("tokenize_gpt-oss.json").as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let want: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let got = tok.encode(text, true);
        if got != want {
            bad.push(format!("{text:?}\n  want {want:?}\n  got  {got:?}"));
        }
        assert_eq!(tok.decode(&want), text, "decode round trip");
    }
    assert!(bad.is_empty(), "{} mismatches:\n{}", bad.len(), bad.join("\n"));
}
