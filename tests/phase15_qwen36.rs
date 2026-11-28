//! Phase 15 gates: Qwen3.6-35B-A3B (qwen35moe arch: Gated DeltaNet + gated full attention, ggml-org Q4_K_M).
//! (a) Tokenizer: qwen35 pre-tokenizer matches HF `tokenizers` running the model's own
//!     tokenizer.json (tests/golden/tokenize_qwen36.json, tools/capture_tokenize_hf.py). The GGUF
//!     sets normalizer.nfc, but llama.cpp applies NFC only to UGM vocabs, so neither do we; every
//!     golden text is already NFC.
//! (b) Chat template: the GGUF's template renders as Python jinja2 configured like HF transformers
//!     (tests/golden/template_qwen35moe.json, tools/capture_template_jinja.py); its transformers-only
//!     `{% generation %}` tag is a no-op, as in llama.cpp.
//! (c) CPU reference forward vs llama.cpp's greedy oracle (tests/golden/greedy_qwen36.json), within
//!     llama.cpp's own `-fa off` backend floor (gate revision: QWEN36_TIE/QWEN36_FLOOR_MEAN).
//! `cargo test --release --test phase15_qwen36 -- --include-ignored --nocapture --test-threads=1`

mod common;
use common::{golden, qwen36_path};
use eomoe::{gguf::Gguf, gpu::GpuModel, qwen35, qwen3moe::Engine, template::ChatTemplate, tokenizer::Tokenizer};
use serde_json::json;

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

/// Greedy chat answers through the real tokenizer + template (thinking off): a short question and
/// a fact stated ~250 tokens before the question.
fn answers(m: &mut impl Engine) {
    let g = Gguf::open(qwen36_path()).unwrap();
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
    let eog = ["<|im_end|>", "<|endoftext|>"].map(|s| tok.token_id(s).unwrap());
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
#[ignore = "needs the Qwen3.6-35B-A3B model file (CPU, minutes)"]
fn cpu_reference_answers() {
    answers(&mut qwen35::CpuModel::new(Gguf::open(qwen36_path()).unwrap()).unwrap());
}

/// (c) gate: teacher-forced vs llama.cpp greedy_qwen36.json, within llama.cpp's own floor.
#[test]
#[ignore = "needs the Qwen3.6-35B-A3B model file (CPU, minutes)"]
fn cpu_reference_matches_oracle() {
    let mut m = qwen35::CpuModel::new(Gguf::open(qwen36_path()).unwrap()).unwrap();
    common::greedy_floor_gate(&mut m, "greedy_qwen36.json", common::QWEN36_TIE, common::QWEN36_FLOOR_MEAN);
}

/// (d) CUDA engine, prefill (DeltaNet + gated attention kernels, staged expert waves).
#[test]
#[ignore = "needs the Qwen3.6-35B-A3B model file and a CUDA GPU"]
fn gpu_matches_oracle() {
    let mut m = GpuModel::new(Gguf::open(qwen36_path()).unwrap(), 4096).unwrap();
    common::greedy_floor_gate(&mut m, "greedy_qwen36.json", common::QWEN36_TIE, common::QWEN36_FLOOR_MEAN);
}

/// (d) CUDA engine token by token through a small expert cache (recurrent state carried per token).
#[test]
#[ignore = "needs the Qwen3.6-35B-A3B model file and a CUDA GPU"]
fn gpu_decode_matches_oracle() {
    let m = GpuModel::with_cache(Gguf::open(qwen36_path()).unwrap(), 4096, 1000).unwrap();
    common::greedy_floor_gate(&mut common::Decode(m), "greedy_qwen36.json", common::QWEN36_TIE, common::QWEN36_FLOOR_MEAN);
}

/// (d) the CUDA engine answers coherently (Paris; vault code recall over 240 tokens).
#[test]
#[ignore = "needs the Qwen3.6-35B-A3B model file and a CUDA GPU"]
fn gpu_answers() {
    answers(&mut GpuModel::new(Gguf::open(qwen36_path()).unwrap(), 4096).unwrap());
}

/// (d) three sequences decoded together (`decode_batch`, recurrent state per sequence, lazily
/// allocated by `set_n_seq`) match each sequence run alone, teacher-forced on the golden tokens.
#[test]
#[ignore = "needs the Qwen3.6-35B-A3B model file and a CUDA GPU"]
fn gpu_batched_sequences_match_solo() {
    let cases: Vec<(Vec<u32>, Vec<u32>)> = (golden("greedy_qwen36.json").as_array().unwrap().iter().skip(1).take(3))
        .map(|k| (serde_json::from_value(k["prompt_tokens"].clone()).unwrap(), serde_json::from_value(k["tokens"].clone()).unwrap()))
        .collect();
    let steps = cases.iter().map(|(_, g)| g.len() - 1).min().unwrap();
    let mut m = GpuModel::new(Gguf::open(qwen36_path()).unwrap(), 4096).unwrap();
    let nv = m.n_vocab();
    let solo: Vec<Vec<f32>> = cases.iter().map(|(p, g)| {
        m.reset();
        let mut l = vec![];
        m.forward_last(p);
        for &t in &g[..steps] {
            l.extend(m.forward_last(&[t]));
        }
        l
    }).collect();
    m.set_n_seq(3);
    for (i, (p, _)) in cases.iter().enumerate() {
        m.select(i);
        m.forward_last(p);
    }
    let lsm = |v: &[f32]| { let mx = v.iter().fold(f32::MIN, |a, &b| a.max(b)); let s = v.iter().map(|x| (x - mx).exp()).sum::<f32>().ln() + mx; v.iter().map(|x| x - s).collect::<Vec<f32>>() };
    let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let mut worst = 0f32;
    for k in 0..steps {
        let rows: Vec<(usize, u32)> = cases.iter().enumerate().map(|(i, (_, g))| (i, g[k])).collect();
        let l = m.decode_batch(&rows);
        for i in 0..cases.len() {
            let (a, b) = (lsm(&solo[i][k * nv..][..nv]), lsm(&l[i * nv..][..nv]));
            assert_eq!(am(&a), am(&b), "seq {i} step {k}: top-1 differs");
            worst = worst.max((a[am(&a)] - b[am(&a)]).abs());
        }
    }
    println!("{} sequences x {steps} steps batched vs solo: max top-1 |dlogprob| {worst:.4}", cases.len());
    assert!(worst < 0.05, "batched decode drifts from solo: {worst}");
}

/// (e) `eomoe serve` defaults: the expert cache takes all free VRAM, then `--parallel 4`
/// sequences each get recurrent state; the cache must leave room for them, and they must work.
#[test]
#[ignore = "needs the Qwen3.6-35B-A3B model file and a CUDA GPU"]
fn gpu_full_cache_fits_four_sequences() {
    let mut m = GpuModel::with_cache(Gguf::open(qwen36_path()).unwrap(), 32768, usize::MAX >> 20).unwrap();
    m.set_n_seq(4);
    m.select(3);
    answers(&mut m);
}
