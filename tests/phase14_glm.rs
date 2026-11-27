//! Phase 14 gates: GLM-4.7-Flash (deepseek2 arch, ggml-org Q4_K).
//! (a) Tokenizer: glm4 pre-tokenizer with ignore_merges matches HF `tokenizers` running the
//!     model's own tokenizer.json (tests/golden/tokenize_glm.json, tools/capture_tokenize_hf.py).
//! (b) Chat template: the GGUF's template renders as Python jinja2 configured like HF transformers
//!     (tests/golden/template_deepseek2.json, tools/capture_template_jinja.py).
//! `cargo test --release --test phase14_glm -- --include-ignored --nocapture --test-threads=1`

mod common;
use common::{glm_path, golden};
use eomoe::{gguf::Gguf, template::ChatTemplate, tokenizer::Tokenizer};

#[test]
#[ignore = "needs the GLM-4.7-Flash model file"]
fn tokenize_matches_hf() {
    let tok = Tokenizer::from_gguf(&Gguf::open(glm_path()).unwrap()).unwrap();
    let mut bad = vec![];
    for case in golden("tokenize_glm.json").as_array().unwrap() {
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
#[ignore = "needs the GLM-4.7-Flash model file"]
fn template_matches_jinja() {
    let tpl = ChatTemplate::from_gguf(&Gguf::open(glm_path()).unwrap()).unwrap();
    for case in golden("template_deepseek2.json").as_array().unwrap() {
        let got = tpl.render(&case["request"]);
        match case["prompt"].as_str() {
            Some(want) => assert_eq!(got.unwrap(), want, "request: {}", case["request"]),
            None => assert!(got.is_err(), "template error {} expected, got {got:?}", case["error"]),
        }
    }
}
