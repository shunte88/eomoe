//! Phase 8 gate: `--api-key` behaves as llama-server's (tests/golden/auth.json, from
//! tools/capture_auth_golden.py): same status, content type and byte-identical 401 body for
//! every case, `/health` open. The engine is a stub: auth never reaches it.
//! `cargo test --release --test phase8_golden -- --include-ignored`

mod common;
use common::{golden, model_path};
use eomoe::qwen3moe::Engine;
use eomoe::{gguf::Gguf, server, template::ChatTemplate, tokenizer::Tokenizer};
use std::io::{Read, Write};
use std::net::TcpStream;

struct Stub;
impl Engine for Stub {
    fn n_vocab(&self) -> usize {
        1
    }
    fn n_ctx(&self) -> usize {
        512
    }
    fn truncate(&mut self, _: usize) {}
    fn forward(&mut self, t: &[u32]) -> Vec<f32> {
        vec![0.0; t.len()]
    }
}

/// (status, content type, body) of one request over HTTP/1.0.
fn http(port: u16, method: &str, path: &str, headers: &serde_json::Map<String, serde_json::Value>) -> (u16, String, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    let body = if method == "POST" { r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}"# } else { "" };
    let mut req = format!("{method} {path} HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: {}\r\n", body.len());
    for (k, v) in headers {
        req += &format!("{k}: {}\r\n", v.as_str().unwrap());
    }
    write!(s, "{req}\r\n{body}").unwrap();
    let mut r = String::new();
    s.read_to_string(&mut r).unwrap();
    let (head, body) = r.split_once("\r\n\r\n").unwrap();
    let ctype = head.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-type: ").map(str::to_owned)).unwrap_or_default();
    (head[9..12].parse().unwrap(), ctype, body.to_owned())
}

#[test]
#[ignore = "needs model file"]
fn api_key_matches_oracle() {
    let g = Gguf::open(model_path()).unwrap();
    let (tok, tpl) = (Tokenizer::from_gguf(&g).unwrap(), ChatTemplate::from_gguf(&g).unwrap());
    let app = server::App::new(tok, tpl, "local-code".into(), serde_json::Value::Null, 0, || Ok(Stub)).unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let keys = vec!["k1".to_string(), "k2".to_string()];
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            listener.set_nonblocking(true).unwrap();
            let l = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(l, server::router(app, keys)).await.unwrap();
        })
    });
    for (name, c) in golden("auth.json").as_object().unwrap() {
        let (status, ctype, body) = http(port, c["method"].as_str().unwrap(), c["path"].as_str().unwrap(), c["headers"].as_object().unwrap());
        assert_eq!(status as u64, c["status"].as_u64().unwrap(), "{name}: status ({body})");
        if status != 200 {
            assert_eq!(ctype, c["content_type"].as_str().unwrap(), "{name}: content type");
            assert_eq!(body, c["body"].as_str().unwrap(), "{name}: body");
        }
    }
}
