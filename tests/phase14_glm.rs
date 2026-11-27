//! Phase 14 gates: GLM-4.7-Flash (deepseek2 arch, ggml-org Q4_K).
//! (a) Tokenizer: glm4 pre-tokenizer with ignore_merges matches HF `tokenizers` running the
//!     model's own tokenizer.json (tests/golden/tokenize_glm.json, tools/capture_tokenize_hf.py).
//! (b) Chat template: the GGUF's template renders as Python jinja2 configured like HF transformers
//!     (tests/golden/template_deepseek2.json, tools/capture_template_jinja.py).
//! `cargo test --release --test phase14_glm -- --include-ignored --nocapture --test-threads=1`

mod common;
use common::{glm_path, golden};
use eomoe::qwen3moe::Engine;
use eomoe::{glm, gguf::Gguf, template::ChatTemplate, tokenizer::Tokenizer};
use serde_json::json;

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

/// Greedy chat answers through the real tokenizer + template (thinking off): a short question and
/// a fact stated ~250 tokens before the question.
fn answers(m: &mut impl Engine) {
    let g = Gguf::open(glm_path()).unwrap();
    let (tok, tpl) = (Tokenizer::from_gguf(&g).unwrap(), ChatTemplate::from_gguf(&g).unwrap());
    let filler = "The weather that spring was mild, with long grey mornings that burned off into bright afternoons. \
        Farmers in the valley planted barley and beans, and the river ran high with snowmelt from the hills. \
        In the town, the bakery opened an hour earlier than usual, and the library extended its hours on Thursdays. \
        Children walked to school along the canal path, counting herons and arguing about football. \
        By May the orchards were in blossom and the market square filled with stalls selling cheese, honey and wool.";
    let cases = [
        ("What is the capital of France? Answer in one word.".to_string(), "Paris"),
        (format!("Remember this: the vault code is 4817. {filler} {filler}\n\nWhat is the vault code?"), "4817"),
    ];
    let eog = ["<|endoftext|>", "<|user|>", "<|observation|>"].map(|s| tok.token_id(s).unwrap());
    for (q, want) in cases {
        let req = json!({"messages": [{"role": "user", "content": q}], "chat_template_kwargs": {"enable_thinking": false}});
        let ids = tok.encode(&tpl.render(&req).unwrap(), true);
        m.reset();
        let mut logits = m.forward_last(&ids);
        let mut out = vec![];
        for _ in 0..32 {
            let t = (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap() as u32;
            if eog.contains(&t) {
                break;
            }
            out.push(t);
            logits = m.forward_last(&[t]);
        }
        let text = tok.decode(&out);
        println!("{} prompt tokens -> {text:?}", ids.len());
        assert!(text.contains(want), "expected {want:?} in {text:?}");
    }
}

/// (c) coherence through the real tokenizer + template.
#[test]
#[ignore = "needs the GLM-4.7-Flash model file (CPU, minutes)"]
fn cpu_reference_answers() {
    answers(&mut glm::CpuModel::new(Gguf::open(glm_path()).unwrap()).unwrap());
}

/// (c) gate: teacher-forced vs llama.cpp greedy_glm.json, within llama.cpp's own floor.
#[test]
#[ignore = "needs the GLM-4.7-Flash model file (CPU, minutes)"]
fn cpu_reference_matches_oracle() {
    let mut m = glm::CpuModel::new(Gguf::open(glm_path()).unwrap()).unwrap();
    common::greedy_floor_gate(&mut m, "greedy_glm.json", common::GLM_TIE, common::GLM_FLOOR_MEAN);
}
