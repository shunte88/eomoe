//! Phase 1 gate: GGUF loader, tokenizer and chat template vs the llama.cpp oracle.
//! Needs the real model: `cargo test --test phase1_golden -- --include-ignored`
//! (override the path with EOMOE_TEST_MODEL).

use eomoe::gguf::{Gguf, GgmlType};

fn model_path() -> String {
    std::env::var("EOMOE_TEST_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/.local/share/local-code-llm/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
            std::env::var("HOME").unwrap()
        )
    })
}

// Expected values from an independent stdlib-Python GGUF reader (scratch gguf_probe.py).
#[test]
#[ignore = "needs model file"]
fn gguf_reads_metadata_and_tensors() {
    let g = Gguf::open(model_path()).unwrap();
    assert_eq!(g.str("general.architecture"), Some("qwen3moe"));
    assert_eq!(g.u64("qwen3moe.block_count"), Some(48));
    assert_eq!(g.tensors.len(), 579);

    let t = g.tensor("token_embd.weight").unwrap();
    assert_eq!((t.ty, t.dims.as_slice()), (GgmlType::Q4_K, &[2048u64, 151936][..]));
    assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::Q6_K);

    let q = g.tensor("blk.0.attn_q_norm.weight").unwrap();
    let f = g.f32s(q);
    assert_eq!(&f[..2], &[0.4453125, 1.8359375]);
    assert_eq!(g.f32s(g.tensor("output_norm.weight").unwrap())[0], 2.59375);
}

fn golden(name: &str) -> serde_json::Value {
    let p = format!("{}/tests/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn tokenizer() -> eomoe::tokenizer::Tokenizer {
    eomoe::tokenizer::Tokenizer::from_gguf(&Gguf::open(model_path()).unwrap()).unwrap()
}

#[test]
#[ignore = "needs model file"]
fn tokenize_matches_oracle() {
    let tok = tokenizer();
    let mut bad = vec![];
    for case in golden("tokenize.json").as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let want: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let got = tok.encode(text, case["parse_special"].as_bool().unwrap());
        if got != want {
            bad.push(format!("{text:?}\n  want {want:?}\n  got  {got:?}"));
        }
        assert_eq!(tok.decode(&want), text, "decode round trip");
    }
    assert!(bad.is_empty(), "{} mismatches:\n{}", bad.len(), bad.join("\n"));
}

/// Detokenizing the oracle's greedy tokens must reproduce its streamed content exactly.
#[test]
#[ignore = "needs model file"]
fn decode_matches_oracle_greedy_content() {
    let tok = tokenizer();
    for case in golden("greedy.json").as_array().unwrap() {
        let toks: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        assert_eq!(tok.decode(&toks), case["content"].as_str().unwrap());
        let prompt_toks: Vec<u32> = serde_json::from_value(case["prompt_tokens"].clone()).unwrap();
        assert_eq!(tok.encode(case["prompt"].as_str().unwrap(), true), prompt_toks);
    }
}

#[test]
#[ignore = "needs model file"]
fn chat_template_matches_oracle() {
    let tpl = eomoe::template::ChatTemplate::from_gguf(&Gguf::open(model_path()).unwrap()).unwrap();
    for case in golden("template.json").as_array().unwrap() {
        let got = tpl.render(&case["request"]).unwrap();
        assert_eq!(got, case["prompt"].as_str().unwrap(), "request: {}", case["request"]);
    }
}
