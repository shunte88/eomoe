//! Shared gate helpers for the golden tests (`mod common;`).
#![allow(dead_code)]

use eomoe::qwen3moe::Engine;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};

pub fn model_path() -> String {
    std::env::var("EOMOE_TEST_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/.local/share/local-code-llm/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
            std::env::var("HOME").unwrap()
        )
    })
}

/// gpt-oss-20b MXFP4 (ggml-org/gpt-oss-20b-GGUF rev 41e8a94d, sha256 27cd6c43...).
pub fn gptoss_path() -> String {
    std::env::var("EOMOE_TEST_GPTOSS").unwrap_or_else(|_| format!("{}/.local/share/eomoe/models/gpt-oss-20b-MXFP4.gguf", std::env::var("HOME").unwrap()))
}

pub fn golden(name: &str) -> serde_json::Value {
    let p = format!("{}/tests/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// Near-tie: oracle top-1/top-2 gap below this exempts the top-1 check.
const TIE: f32 = 0.05;
/// llama.cpp's own CPU backend vs its CUDA greedy.json: mean |dlogprob| over the oracle's
/// top-5 (tools/measure_backend_floor.py, 2026-11-25). Activation quantization (Q8_K/q8_1)
/// makes a tighter per-token tolerance unreachable even for llama.cpp across backends.
const FLOOR_MEAN: f64 = 0.2414;

fn log_softmax(l: &[f32]) -> Vec<f32> {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + l.iter().map(|x| ((x - m) as f64).exp()).sum::<f64>().ln() as f32;
    l.iter().map(|x| x - lse).collect()
}

/// Teacher-forced: feed prompt + the oracle's own greedy tokens in one pass and compare
/// every step with the oracle (so one divergence can't derail the rest).
/// Gate: top-1 equal on every non-tie step; mean |dlogprob| <= llama.cpp's backend floor.
pub fn greedy_floor_gate(m: &mut impl Engine) {
    let mut failures = vec![];
    let (mut worst, mut dsum, mut dn, mut top5_diff, mut steps) = (0f32, 0f64, 0u32, 0u32, 0u32);
    for (ci, case) in golden("greedy.json").as_array().unwrap().iter().enumerate() {
        let prompt: Vec<u32> = serde_json::from_value(case["prompt_tokens"].clone()).unwrap();
        let gen: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let mut seq = prompt.clone();
        seq.extend_from_slice(&gen[..gen.len() - 1]);

        m.reset();
        let t0 = std::time::Instant::now();
        let logits = m.forward(&seq);
        let el = t0.elapsed().as_secs_f32();
        let nv = m.n_vocab();
        let mut case_worst = 0f32;
        for (k, step) in case["probs"].as_array().unwrap().iter().enumerate() {
            let lp = log_softmax(&logits[(prompt.len() - 1 + k) * nv..][..nv]);
            let top: Vec<(u32, f32)> = step["top_logprobs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| (e["id"].as_u64().unwrap() as u32, e["logprob"].as_f64().unwrap() as f32))
                .collect();
            let mut ours: Vec<u32> = (0..nv as u32).collect();
            ours.select_nth_unstable_by(4, |a, b| lp[*b as usize].total_cmp(&lp[*a as usize]));
            ours.truncate(5);
            ours.sort_by(|a, b| lp[*b as usize].total_cmp(&lp[*a as usize]));

            steps += 1;
            if ours[0] != top[0].0 && top[0].1 - top[1].1 >= TIE {
                failures.push(format!("case {ci} step {k}: top1 {} vs oracle {}", ours[0], top[0].0));
            }
            for &(id, olp) in &top {
                let d = (lp[id as usize] - olp).abs();
                case_worst = case_worst.max(d);
                dsum += d as f64;
                dn += 1;
            }
            let mut b: Vec<u32> = top.iter().map(|t| t.0).collect();
            ours.sort();
            b.sort();
            top5_diff += (ours != b) as u32;
        }
        worst = worst.max(case_worst);
        println!(
            "case {ci}: {} tokens in {el:.1}s ({:.1} tok/s), max |dlogprob| {case_worst:.4}",
            seq.len(),
            seq.len() as f32 / el
        );
    }
    let mean = dsum / dn as f64;
    println!(
        "top1 fails {}/{steps}, mean |dlogprob| {mean:.4} (floor {FLOOR_MEAN}), max {worst:.4}, top5 set differs on {top5_diff} steps",
        failures.len()
    );
    if mean > FLOOR_MEAN {
        failures.push(format!("mean |dlogprob| {mean:.4} > llama.cpp backend floor {FLOOR_MEAN}"));
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

pub struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        self.0.kill().ok();
        self.0.wait().ok();
    }
}

/// `eomoe serve` on `port`, returned once it is listening; killed on drop.
pub fn start(port: u16) -> Server {
    start_with(port, &[])
}

/// `eomoe serve` on `port` with extra flags.
pub fn start_with(port: u16, extra: &[&str]) -> Server {
    let mut c = Command::new(env!("CARGO_BIN_EXE_eomoe"))
        .args(["serve", "--port", &port.to_string()])
        .args(extra)
        .env("RUST_LOG", "info")
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let err = c.stderr.take().unwrap();
    let s = Server(c);
    for l in BufReader::new(err).lines() {
        let l = l.unwrap();
        eprintln!("{l}");
        if l.contains("listening on") {
            return s;
        }
    }
    panic!("server exited before listening");
}

/// (status, body) over HTTP/1.0, so the body is close-delimited.
pub fn http(port: u16, method: &str, path: &str, body: Option<&Value>) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    let b = body.map(Value::to_string).unwrap_or_default();
    write!(s, "{method} {path} HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len()).unwrap();
    let mut r = String::new();
    s.read_to_string(&mut r).unwrap();
    let (head, body) = r.split_once("\r\n\r\n").unwrap();
    (head[9..12].parse().unwrap(), body.to_owned())
}

/// SSE events as (event line, data JSON).
pub fn events(body: &str) -> Vec<(Option<String>, Value)> {
    let mut out = vec![];
    let mut name = None;
    for l in body.lines() {
        if let Some(n) = l.strip_prefix("event: ") {
            name = Some(n.to_owned());
        } else if let Some(d) = l.strip_prefix("data: ") {
            out.push((name.take(), if d == "[DONE]" { json!("[DONE]") } else { serde_json::from_str(d).unwrap() }));
        }
    }
    out
}

