//! Phase 6 gates (beyond parity).
//! `cargo test --release --test phase6_golden -- --include-ignored --nocapture --test-threads 1`

mod common;
use common::{golden, model_path};
use eomoe::gguf::Gguf;
use eomoe::gpu::GpuModel;
use eomoe::qwen3moe::Engine;

/// Expert slots per group with the Phase 5 default (16K ctx, all free VRAM, f32 KV).
const PHASE5_SLOTS: usize = 539;

/// A stock Claude Code prompt (~19K tokens) needs 32K ctx. With an f16 KV cache, 32K costs what
/// 16K did in f32, so the expert cache must not shrink, and decoding must still be exact.
#[test]
#[ignore]
fn ctx_32k_keeps_cache_and_greedy_tokens() {
    let mut m = GpuModel::with_cache(Gguf::open(model_path()).unwrap(), 32768, usize::MAX >> 20).unwrap();
    let slots = m.cache_slots();
    println!("32K ctx: VRAM {:.0} MiB, slots per group {slots:?}", m.vram_bytes as f64 / 1048576.0);
    assert!(slots.iter().all(|&s| s >= PHASE5_SLOTS), "expert cache shrank: {slots:?} < {PHASE5_SLOTS}");
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    for (ci, case) in golden("greedy.json").as_array().unwrap().iter().enumerate() {
        let prompt: Vec<u32> = serde_json::from_value(case["prompt_tokens"].clone()).unwrap();
        let want: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        m.reset();
        let mut got = vec![argmax(&m.forward_last(&prompt))];
        while got.len() < want.len() {
            got.push(argmax(&m.forward_last(&[*got.last().unwrap()])));
        }
        assert_eq!(got, want, "case {ci}: greedy tokens at 32K ctx");
    }
}

/// Oracle (llama-server, same model and GPU) prefill on a 4096-token prompt: STATUS.md Phase 0.
const ORACLE_PREFILL_4K: f32 = 392.9;

/// Prefill must beat the oracle: a 19K-token Claude Code first turn is pure prefill.
/// The prompt is real code (this repo's sources), so expert routing looks like a client session.
#[test]
#[ignore]
fn prefill_4k_beats_oracle() {
    let g = Gguf::open(model_path()).unwrap();
    let tok = eomoe::tokenizer::Tokenizer::from_gguf(&g).unwrap();
    let src: String = ["src/gpu.rs", "src/server.rs", "src/cpu.rs", "kernels/eomoe.cu", "src/generate.rs"]
        .iter().map(|p| std::fs::read_to_string(p).unwrap()).collect();
    let prompt = &tok.encode(&src, false)[..4096];
    let mut m = GpuModel::with_cache(g, 32768, usize::MAX >> 20).unwrap();
    m.forward_last(&prompt[..512]); // warm up: kernel load, pinned staging
    m.reset();
    let t0 = std::time::Instant::now();
    m.forward_last(prompt);
    let tps = prompt.len() as f32 / t0.elapsed().as_secs_f32();
    println!("prefill 4096 tok: {tps:.1} tok/s (oracle {ORACLE_PREFILL_4K})");
    assert!(tps > ORACLE_PREFILL_4K, "prefill {tps:.1} tok/s <= oracle {ORACLE_PREFILL_4K}");
}
