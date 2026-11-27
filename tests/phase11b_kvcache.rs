//! Phase 11 gate (b): host-RAM prompt cache (llama-server --cache-ram).
//!
//! No oracle capture: the CPU-only llama-server locked this machine up twice (STATUS.md). The
//! gate checks what the oracle's cache promises instead:
//! - a KV image saved, dropped and restored resumes bit-identical to a sequence that stayed on
//!   the GPU (hybrid off, so the expert path is batch-invariant), for a whole image and a prefix
//!   that ends mid-page;
//! - over HTTP with one slot, a displaced prompt comes back with the cached-token counts the
//!   oracle's rules give (server-slot-sched.cpp: save on displacement, restore the best entry)
//!   and its prompt is processed in a fraction of the cold time.
//!
//! cargo test --release --test phase11b_kvcache -- --ignored --test-threads=1 --nocapture

mod common;
use common::model_path;
use eomoe::gguf::Gguf;
use eomoe::gpu::GpuModel;
use eomoe::qwen3moe::Engine;
use eomoe::tokenizer::Tokenizer;
use serde_json::{json, Value};

const PORT: u16 = 8095;

#[test]
#[ignore = "needs model file + GPU"]
fn kv_restore_resumes_bit_identical() {
    let mut g = GpuModel::with_cache(Gguf::open(model_path()).unwrap(), 1024, 1000).unwrap();
    g.set_n_seq(2);
    g.set_hybrid(false);
    let a: Vec<u32> = (0..300u32).map(|i| 100 + i * 7919 % 30000).collect();
    let s: Vec<u32> = (0..20u32).map(|i| 200 + i * 104729 % 30000).collect();
    let tail = |from: usize| [&a[from..], &s[..]].concat();
    let run = |g: &mut GpuModel, seq: usize, toks: &[u32]| {
        g.select(seq);
        let mut l = g.forward_last(toks);
        let t = l.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u32;
        l.extend(g.decode_batch(&[(seq, t)]));
        l
    };
    // Whole image: seq 0 stays hot; seq 1 saves, drops and restores (onto other pages).
    for sq in [0, 1] {
        g.select(sq);
        g.forward_last(&a);
    }
    g.select(1);
    let img = g.kv_save();
    g.truncate(0);
    g.kv_restore(&img, a.len());
    let (hot, restored) = (run(&mut g, 0, &s), run(&mut g, 1, &s));
    assert!(hot == restored, "whole-image restore diverges");

    // A 200-position prefix of a 300-position image (ends mid-page).
    for sq in [0, 1] {
        g.select(sq);
        g.truncate(0);
        g.forward_last(&a[..200]);
    }
    g.select(1);
    g.forward_last(&a[200..]);
    let img = g.kv_save();
    g.truncate(0);
    g.kv_restore(&img, 200);
    let (hot, restored) = (run(&mut g, 0, &tail(200)), run(&mut g, 1, &tail(200)));
    assert!(hot == restored, "prefix restore diverges");
    println!("kv restore: whole image and mid-page prefix bit-identical ({} MiB image)", img.len() >> 20);
}

#[test]
#[ignore = "needs model file + GPU"]
fn ram_cache_restores_displaced_prompts() {
    let tok = Tokenizer::from_gguf(&Gguf::open(model_path()).unwrap()).unwrap();
    let a: String = (0..60).map(|i| format!("def f{i}(x):\n    return x * {i} + {} - {}\n\n", i * i, i % 7)).collect();
    let b: String = (0..40).map(|i| format!("class C{i}:\n    value = '{}'\n\n", ((b'a' + i % 26) as char).to_string().repeat(i as usize % 5 + 1))).collect();
    let t = "# Now write a function g that sums f0..f9:\n";
    let (a2, b2) = (format!("{a}{t}"), format!("{b}{t}"));
    let lcp = |x: &str, y: &str| tok.encode(x, true).iter().zip(tok.encode(y, true)).take_while(|(p, q)| **p == *q).count();
    let n = |x: &str| tok.encode(x, true).len();
    // (prompt, cached tokens by the oracle's rules)
    let steps = [
        ("A", &a, 0),
        ("B", &b, 0),                    // A saved to RAM
        ("A2", &a2, lcp(&a, &a2)),       // A restored, B saved
        ("A2_hot", &a2, n(&a2) - 1),     // same prompt on the slot: all but the last token
        ("B2", &b2, lcp(&b, &b2)),       // B restored, A2 saved
        ("A2_restored", &a2, n(&a2) - 1), // A2 restored
    ];
    let _s = common::start_with(PORT, &["--parallel", "1", "--cache-ram", "4096"]);
    let mut ms = std::collections::HashMap::new();
    let mut fails = vec![];
    for (name, p, want) in steps {
        let (st, body) = common::http(PORT, "POST", "/completion", Some(&json!({"prompt": p, "n_predict": 8, "temperature": 0})));
        assert_eq!(st, 200, "{name}: {body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        let (got, pms) = (v["timings"]["cache_n"].as_u64().unwrap() as usize, v["timings"]["prompt_ms"].as_f64().unwrap());
        println!("{name}: prompt {} cached {got} (want {want}) prompt_ms {pms:.1} text {:?}", v["tokens_evaluated"], v["content"]);
        if got != want {
            fails.push(name);
        }
        ms.insert(name, pms);
    }
    assert!(fails.is_empty(), "cached-token mismatches: {fails:?}");
    // A2 restored computes 1 token; cold A computed all of its ~1.5K.
    assert!(ms["A2_restored"] < 0.25 * ms["A"], "restore not faster than cold prefill: {ms:?}");
}
