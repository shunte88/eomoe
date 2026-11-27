//! Phase 13 gates: gpt-oss-20b.
//! (a) Tokenizer: o200k (`tokenizer.ggml.pre` = gpt-4o) matches HF `tokenizers` running the
//!     model's own tokenizer.json (tests/golden/tokenize_gpt-oss.json, tools/capture_tokenize_hf.py).
//! (b) Harmony chat template: the GGUF's template renders as Python jinja2 configured like HF
//!     transformers (tests/golden/template_gpt-oss.json, tools/capture_template_jinja.py);
//!     "<DATE:fmt>" there is strftime_now(fmt); a case the template itself fails must fail here.
//! (c) CPU reference forward (provisional until the llama.cpp greedy oracle is captured): greedy
//!     decoding through the real tokenizer + template answers a short question, and recalls a
//!     fact stated >128 tokens (one sliding window) before the question.
//! `cargo test --release --test phase13_gptoss -- --include-ignored --nocapture`

mod common;
use common::{golden, gptoss_path};
use eomoe::qwen3moe::Engine;
use eomoe::{gguf::Gguf, gptoss, template::ChatTemplate, tokenizer::Tokenizer};
use serde_json::json;

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

#[test]
#[ignore = "needs the gpt-oss-20b model file"]
fn template_matches_jinja() {
    let tpl = ChatTemplate::from_gguf(&Gguf::open(gptoss_path()).unwrap()).unwrap();
    for case in golden("template_gpt-oss.json").as_array().unwrap() {
        let got = tpl.render(&case["request"]);
        match case["prompt"].as_str() {
            Some(want) => assert_eq!(got.unwrap(), want.replace("<DATE:%Y-%m-%d>", &eomoe::template::strftime_now("%Y-%m-%d")), "request: {}", case["request"]),
            None => assert!(got.is_err(), "template error {} expected, got {got:?}", case["error"]),
        }
    }
}

#[test]
#[ignore = "needs the gpt-oss-20b model file (CPU, minutes)"]
fn cpu_reference_answers() {
    let path = gptoss_path();
    let g = Gguf::open(&path).unwrap();
    let (tok, tpl) = (Tokenizer::from_gguf(&g).unwrap(), ChatTemplate::from_gguf(&g).unwrap());
    let mut m = gptoss::CpuModel::new(Gguf::open(&path).unwrap()).unwrap();
    let filler = "The weather that spring was mild, with long grey mornings that burned off into bright afternoons. \
        Farmers in the valley planted barley and beans, and the river ran high with snowmelt from the hills. \
        In the town, the bakery opened an hour earlier than usual, and the library extended its hours on Thursdays. \
        Children walked to school along the canal path, counting herons and arguing about football. \
        By May the orchards were in blossom and the market square filled with stalls selling cheese, honey and wool.";
    let cases = [
        ("What is the capital of France? Answer in one word.".to_string(), "Paris"),
        (format!("Remember this: the vault code is 4817. {filler} {filler}\n\nWhat is the vault code?"), "4817"),
    ];
    for (q, want) in cases {
        let prompt = tpl.render(&json!({"messages": [{"role": "user", "content": q}]})).unwrap();
        let ids = tok.encode(&prompt, true);
        assert!(want != "4817" || ids.len() > 200, "recall prompt must span more than one 128-token window");
        m.reset();
        let mut logits = m.forward_last(&ids);
        let mut out = vec![];
        for _ in 0..48 {
            let t = (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap() as u32;
            if t == 200002 {
                break; // <|return|>
            }
            out.push(t);
            logits = m.forward_last(&[t]);
        }
        let text = tok.decode(&out);
        println!("{} prompt tokens -> {text:?}", ids.len());
        assert!(text.contains(want), "expected {want:?} in {text:?}");
    }
}

/// (c) proper: teacher-forced CPU logits vs the llama.cpp greedy oracle (tools/capture_greedy_gptoss.py),
/// same floor gate as phase 2 with gpt-oss's own measured llama.cpp floor (GPTOSS_TIE / GPTOSS_FLOOR_MEAN).
#[test]
#[ignore = "needs the gpt-oss-20b model file (CPU, minutes)"]
fn cpu_reference_matches_oracle() {
    let mut m = gptoss::CpuModel::new(Gguf::open(gptoss_path()).unwrap()).unwrap();
    common::greedy_floor_gate(&mut m, "greedy_gpt-oss.json", common::GPTOSS_TIE, common::GPTOSS_FLOOR_MEAN);
}
