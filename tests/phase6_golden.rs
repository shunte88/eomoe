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
/// The prompt is real code (frozen repo sources), so expert routing looks like a client session.
#[test]
#[ignore]
fn prefill_4k_beats_oracle() {
    let g = Gguf::open(model_path()).unwrap();
    let tok = eomoe::tokenizer::Tokenizer::from_gguf(&g).unwrap();
    let src = std::fs::read_to_string("tests/golden/long_prompt.txt").unwrap();
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

/// Oracle decode right after a 4096-token prompt (STATUS.md Phase 0). Claude Code decodes after
/// ~19K tokens, where attention reads ~1.9 GB of KV per token; we must still beat the oracle's
/// 4K-context number there, and aim for 20.
const ORACLE_DECODE_AFTER_4K: f32 = 16.0;
const DECODE_19K_TARGET: f32 = 20.0;

#[test]
#[ignore]
fn decode_after_19k_prompt_beats_oracle() {
    // Teacher-forced: decode the source's next 72 tokens, so the routed experts (and the cache
    // hit rate) do not depend on what greedy decoding happens to produce after numeric changes.
    let (mut m, all) = long_prompt();
    let (prompt, cont) = all.split_at(19000);
    m.forward_last(prompt);
    for &tk in &cont[..8] { m.forward_last(&[tk]); } // warm the expert cache for decode
    let n = 64;
    let s0 = m.cache_stats();
    let t0 = std::time::Instant::now();
    for &tk in &cont[8..8 + n] { m.forward_last(&[tk]); }
    let tps = n as f32 / t0.elapsed().as_secs_f32();
    let s1 = m.cache_stats();
    let (h, mi) = (s1.hits - s0.hits, s1.misses - s0.misses);
    println!("decode after 19K prompt: {tps:.1} tok/s (oracle after 4K: {ORACLE_DECODE_AFTER_4K}, target {DECODE_19K_TARGET}); \
        cache hits {:.1}%, {:.0} MiB uploaded", 100.0 * h as f64 / (h + mi) as f64, (s1.bytes - s0.bytes) as f64 / 1048576.0);
    assert!(tps >= DECODE_19K_TARGET, "decode {tps:.1} tok/s < {DECODE_19K_TARGET}");
}

/// 19072 tokens of a frozen snapshot of this repo's sources (tests/golden/long_prompt.txt; frozen
/// because routing, and so cache hits and speed, depends on the exact text) (a Claude Code first turn is ~19K) and a 32K-ctx model.
fn long_prompt() -> (GpuModel, Vec<u32>) {
    let g = Gguf::open(model_path()).unwrap();
    let tok = eomoe::tokenizer::Tokenizer::from_gguf(&g).unwrap();
    let src = std::fs::read_to_string("tests/golden/long_prompt.txt").unwrap();
    let mut all = tok.encode(&src, false);
    assert!(all.len() >= 19072, "prompt source too short: {}", all.len());
    all.truncate(19072); // 19000-token prompt + 72 continuation tokens
    (GpuModel::with_cache(g, 32768, usize::MAX >> 20).unwrap(), all)
}

/// Follow-up turns (100-200 new tokens on a 19K prefix) ran at 74-99 tok/s in the server log.
/// 128 tokens route to nearly every expert of every layer: ~13.9 GB of cache misses, ~1.0 s at
/// PCIe 4.0 x8 (~14 GB/s), so ~128 tok/s is the ceiling until the cache hit rate improves.
const FOLLOWUP_TARGET: f32 = 100.0;

#[test]
#[ignore]
fn followup_prefill_on_19k_prefix() {
    let (mut m, prompt) = long_prompt();
    m.forward_last(&prompt[..18872]);
    let t0 = std::time::Instant::now();
    m.forward_last(&prompt[18872..19000]);
    let tps = 128.0 / t0.elapsed().as_secs_f32();
    println!("follow-up prefill 128 tok on 18.9K: {tps:.1} tok/s (target {FOLLOWUP_TARGET})");
    assert!(tps >= FOLLOWUP_TARGET, "follow-up prefill {tps:.1} tok/s < {FOLLOWUP_TARGET}");
}

fn log_softmax(l: &[f32]) -> Vec<f32> {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + l.iter().map(|x| ((x - m) as f64).exp()).sum::<f64>().ln() as f32;
    l.iter().map(|x| x - lse).collect()
}

/// p50 |dlogprob| over the CPU's top-5, per position row of logits.
fn top5_diffs(cpu: &[f32], gpu: &[f32], nv: usize) -> Vec<f32> {
    let mut v = vec![];
    for (ra, rb) in cpu.chunks_exact(nv).zip(gpu.chunks_exact(nv)) {
        let (la, lb) = (log_softmax(ra), log_softmax(rb));
        let mut idx: Vec<usize> = (0..nv).collect();
        idx.select_nth_unstable_by(4, |x, y| la[*y].total_cmp(&la[*x]));
        v.extend(idx[..5].iter().map(|&i| (la[i] - lb[i]).abs()));
    }
    v
}

fn p50(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// The greedy.json cases are <= 55 tokens, so they never reach the expert GEMM (>= 16 rows per
/// expert), split-K attention (decode: past 128 positions; prefill: a short batch on a long
/// context) or `attn_merge`. Against the f32 CPU reference: a 640-token prefill of real code, a
/// 16-token follow-up prefill (attention_tc split in 2) and 16 decode steps (attention_dec split
/// in 6).
///
/// Routing near-ties make long contexts chaotic: any change in summation order flips a top-8 choice
/// somewhere, and attention spreads it to every later position. Prefilling the same 640 tokens in
/// chunks of 1, 3, 4, 8, 32 or 640 (identical math, different split points) moves the p50 at
/// positions 512+ anywhere from 0.0005 to 0.01. So the strict 0.001 holds only before chaos sets in
/// (positions 0-127, p50 ~0.0002); later segments are gated at 0.02, well above chaos and far below
/// a broken kernel (dropping one attn_merge partial gives > 0.1).
#[test]
#[ignore]
fn long_context_matches_cpu_reference() {
    const N: usize = 640;
    const FOLLOW: usize = 16;
    const DEC: usize = 16;
    let g = Gguf::open(model_path()).unwrap();
    let tok = eomoe::tokenizer::Tokenizer::from_gguf(&g).unwrap();
    let src = std::fs::read_to_string("tests/golden/long_prompt.txt").unwrap();
    let all = tok.encode(&src, false);
    let (prompt, follow, dec) = (&all[..N], &all[N..N + FOLLOW], &all[N + FOLLOW..N + FOLLOW + DEC]);
    let mut c = eomoe::cpu::CpuModel::new(Gguf::open(model_path()).unwrap()).unwrap();
    let mut m = GpuModel::new(g, 4096).unwrap();
    let nv = m.n_vocab();
    let pre = top5_diffs(&c.forward(prompt), &m.forward(prompt), nv);
    let fol = top5_diffs(&c.forward(follow), &m.forward(follow), nv);
    let mut de = vec![];
    for &t in dec {
        de.extend(top5_diffs(&c.forward(&[t]), &m.forward(&[t]), nv));
    }
    let max = pre.iter().chain(&fol).chain(&de).cloned().fold(0f32, f32::max);
    let (early, late, fol, de) = (p50(pre[..128 * 5].to_vec()), p50(pre[128 * 5..].to_vec()), p50(fol), p50(de));
    println!("long ctx GPU vs CPU p50: prefill 0-127 {early:.5}, 128-{N} {late:.5}; follow-up {FOLLOW} {fol:.5}; decode {DEC} {de:.5}; max {max:.4}");
    assert!(early <= 0.001, "prefill before chaos: p50 {early}");
    for (name, v) in [("prefill 128+", late), ("follow-up", fol), ("decode", de)] {
        assert!(v <= 0.02, "{name}: p50 {v}");
    }
    assert!(max <= 2.0, "max {max}");
}
