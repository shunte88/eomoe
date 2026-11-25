//! Phase 3 gate: CUDA engine vs the CPU reference and the llama.cpp oracle.
//! `cargo test --release --test phase3_golden -- --include-ignored --nocapture --test-threads 1`

mod common;
use common::{golden, model_path};
use eomoe::cpu::CpuModel;
use eomoe::gguf::Gguf;
use eomoe::gpu::GpuModel;
use eomoe::qwen3moe::Engine;

fn gpu() -> GpuModel {
    GpuModel::new(Gguf::open(model_path()).unwrap(), 256).unwrap()
}

fn log_softmax(l: &[f32]) -> Vec<f32> {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + l.iter().map(|x| ((x - m) as f64).exp()).sum::<f64>().ln() as f32;
    l.iter().map(|x| x - lse).collect()
}

/// Both engines are f32 end to end, so only summation order differs: the GPU must agree
/// with the CPU reference far more tightly than either agrees with llama.cpp.
#[test]
#[ignore = "needs model file + GPU"]
fn gpu_matches_cpu_reference() {
    const TOL: f32 = 0.02;
    let mut c = CpuModel::new(Gguf::open(model_path()).unwrap()).unwrap();
    let mut gm = gpu();
    let nv = c.n_vocab();
    let mut worst = 0f32;
    for case in golden("greedy.json").as_array().unwrap() {
        let mut seq: Vec<u32> = serde_json::from_value(case["prompt_tokens"].clone()).unwrap();
        let gen: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        seq.extend_from_slice(&gen[..gen.len() - 1]);
        c.reset();
        gm.reset();
        let (a, b) = (c.forward(&seq), gm.forward(&seq));
        assert_eq!(a.len(), b.len());
        for (ra, rb) in a.chunks_exact(nv).zip(b.chunks_exact(nv)) {
            let (la, lb) = (log_softmax(ra), log_softmax(rb));
            let mut idx: Vec<usize> = (0..nv).collect();
            idx.select_nth_unstable_by(4, |x, y| la[*y].total_cmp(&la[*x]));
            for &i in &idx[..5] {
                worst = worst.max((la[i] - lb[i]).abs());
            }
        }
    }
    println!("GPU vs CPU: max |dlogprob| over CPU top-5 = {worst:.5}");
    assert!(worst <= TOL, "GPU diverges from CPU reference: {worst} > {TOL}");
}

#[test]
#[ignore = "needs model file + GPU"]
fn gpu_forward_matches_oracle_greedy() {
    common::greedy_floor_gate(&mut gpu());
}

/// Incremental decode through the KV cache: prefill the prompt, then one token per call.
/// Greedy tokens must reproduce the oracle's (min top-1/top-2 gap in greedy.json is 0.58).
#[test]
#[ignore = "needs model file + GPU"]
fn gpu_incremental_decode_matches_oracle_tokens() {
    let mut m = gpu();
    println!("VRAM allocated: {:.0} MiB", m.vram_bytes as f64 / 1048576.0);
    let nv = m.n_vocab();
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    for (ci, case) in golden("greedy.json").as_array().unwrap().iter().enumerate() {
        let prompt: Vec<u32> = serde_json::from_value(case["prompt_tokens"].clone()).unwrap();
        let want: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        m.reset();
        let t0 = std::time::Instant::now();
        let l = m.forward(&prompt);
        let pf = t0.elapsed().as_secs_f32();
        let mut got = vec![argmax(&l[(prompt.len() - 1) * nv..])];
        let t1 = std::time::Instant::now();
        while got.len() < want.len() {
            let l = m.forward(&[*got.last().unwrap()]);
            got.push(argmax(&l));
        }
        let dt = t1.elapsed().as_secs_f32();
        println!(
            "case {ci}: prefill {} tok {:.1} tok/s, decode {} tok {:.1} tok/s",
            prompt.len(),
            prompt.len() as f32 / pf,
            want.len() - 1,
            (want.len() - 1) as f32 / dt
        );
        assert_eq!(got, want, "case {ci}");
    }
}
