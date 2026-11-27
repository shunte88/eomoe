//! Phase 15 gates: Qwen3.6-35B-A3B (qwen35moe arch: Gated DeltaNet + gated full attention, ggml-org Q4_K_M).
//! (a) Tokenizer: qwen35 pre-tokenizer matches HF `tokenizers` running the model's own
//!     tokenizer.json (tests/golden/tokenize_qwen36.json, tools/capture_tokenize_hf.py). The GGUF
//!     sets normalizer.nfc, but llama.cpp applies NFC only to UGM vocabs, so neither do we; every
//!     golden text is already NFC.
//! (b) Chat template: the GGUF's template renders as Python jinja2 configured like HF transformers
//!     (tests/golden/template_qwen35moe.json, tools/capture_template_jinja.py); its transformers-only
//!     `{% generation %}` tag is a no-op, as in llama.cpp.
//! `cargo test --release --test phase15_qwen36 -- --include-ignored --nocapture --test-threads=1`

mod common;
use common::{golden, qwen36_path};
use eomoe::{gguf::Gguf, template::ChatTemplate, tokenizer::Tokenizer};

#[test]
#[ignore = "needs the Qwen3.6-35B-A3B model file"]
fn tokenize_matches_hf() {
    let tok = Tokenizer::from_gguf(&Gguf::open(qwen36_path()).unwrap()).unwrap();
    let mut bad = vec![];
    for case in golden("tokenize_qwen36.json").as_array().unwrap() {
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

#[test]
#[ignore = "needs the Qwen3.6-35B-A3B model file"]
fn template_matches_jinja() {
    let tpl = ChatTemplate::from_gguf(&Gguf::open(qwen36_path()).unwrap()).unwrap();
    for case in golden("template_qwen35moe.json").as_array().unwrap() {
        let got = tpl.render(&case["request"]);
        match case["prompt"].as_str() {
            Some(want) => assert_eq!(got.unwrap(), want, "request: {}", case["request"]),
            None => assert!(got.is_err(), "template error {} expected, got {got:?}", case["error"]),
        }
    }
}
