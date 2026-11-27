//! Phase 12b gate: request limits (koboldcpp's maxrequestsize / queuelimit / reqtimeout).
//! - `--max-request-mb`: a larger body gets 413.
//! - `--queue-limit Q`: past `--parallel` + Q requests in flight, 503 with a JSON error (Ollama's
//!   OLLAMA_MAX_QUEUE answers 503 too).
//! - `--req-timeout S`: generation stops after S seconds, finish_reason "length".
//!
//! The engine is a slow stub (20 ms per pass, never EOG); the tokenizer comes from the model file.
//! `cargo test --release --test phase12_limits -- --include-ignored`

mod common;
use common::model_path;
use eomoe::qwen3moe::Engine;
use eomoe::{gguf::Gguf, server, template::ChatTemplate, tokenizer::Tokenizer};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

struct Slow;
impl Engine for Slow {
    fn n_vocab(&self) -> usize {
        1
    }
    fn n_ctx(&self) -> usize {
        4096
    }
    fn truncate(&mut self, _: usize) {}
    fn forward(&mut self, t: &[u32]) -> Vec<f32> {
        std::thread::sleep(Duration::from_millis(20));
        vec![0.0; t.len()]
    }
}

fn post(port: u16, body: &str) -> (u16, Value) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(s, "POST /v1/chat/completions HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).ok();
    let mut r = String::new();
    s.read_to_string(&mut r).unwrap();
    let (head, body) = r.split_once("\r\n\r\n").unwrap();
    (head[9..12].parse().unwrap(), serde_json::from_str(body).unwrap_or(Value::String(body.into())))
}

fn chat(max_tokens: usize) -> String {
    json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": max_tokens}).to_string()
}

#[test]
#[ignore = "needs model file"]
fn request_limits() {
    let g = Gguf::open(model_path()).unwrap();
    let (tok, tpl) = (Tokenizer::from_gguf(&g).unwrap(), ChatTemplate::from_gguf(&g).unwrap());
    let limits = server::Limits { max_body: 1 << 20, queue: Some(0), timeout: Some(Duration::from_secs(1)) };
    let app = server::App::new(tok, tpl, "local-code".into(), Value::Null, 0, limits, || Ok(Slow)).unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            listener.set_nonblocking(true).unwrap();
            let l = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(l, server::router(app, vec![])).await.unwrap();
        })
    });

    // Body limit.
    // The bulk is in a field the server ignores, so without the limit this is a quick 200.
    let big = json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "pad": "x".repeat(3 << 19)}).to_string();
    let (status, body) = post(port, &big);
    assert_eq!(status, 413, "1.5 MiB body under a 1 MiB limit (axum's default is 2 MB): {body}");
    assert_eq!(post(port, &chat(1)).0, 200, "small body");

    // Timeout: 1000 tokens at 20 ms would take 20 s.
    let t0 = Instant::now();
    let (status, r) = post(port, &chat(1000));
    let took = t0.elapsed();
    assert_eq!(status, 200, "{r}");
    assert_eq!(r["choices"][0]["finish_reason"], "length", "{r}");
    let n = r["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(took < Duration::from_secs(3) && n > 10 && n < 1000, "stopped after {took:?}, {n} tokens");

    // Queue: --parallel 1 (the stub's one sequence) + --queue-limit 0 = one request in flight.
    let a = std::thread::spawn(move || post(port, &chat(1000)));
    std::thread::sleep(Duration::from_millis(300));
    let (status, r) = post(port, &chat(1));
    assert_eq!(status, 503, "second request while one runs: {r}");
    assert!(r["error"]["message"].as_str().is_some_and(|m| m.contains("busy")), "{r}");
    assert_eq!(a.join().unwrap().0, 200);
    assert_eq!(post(port, &chat(1)).0, 200, "slot freed after the first finished");
}
