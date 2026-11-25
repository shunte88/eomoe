//! Phase 5 gate: `eomoe serve` answers the captured oracle requests (tests/golden/api.json,
//! from tools/capture_api_golden.py) with the same API shape: status, prompt tokens (request
//! conversion + chat template), stop reason, tool calls, and streamed event sequence.
//! Generated text is only required to be non-empty where the oracle's is: greedy text differs
//! between backends (see STATUS.md, Phase 3 precision floor).
//!
//! Needs the GPU and the model: cargo test --release --test phase5_golden -- --ignored

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};

const PORT: u16 = 8097;

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        self.0.kill().ok();
        self.0.wait().ok();
    }
}

fn start() -> Server {
    let mut c = Command::new(env!("CARGO_BIN_EXE_eomoe"))
        .args(["serve", "--port", &PORT.to_string()])
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
fn http(method: &str, path: &str, body: Option<&Value>) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", PORT)).unwrap();
    let b = body.map(Value::to_string).unwrap_or_default();
    write!(s, "{method} {path} HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len()).unwrap();
    let mut r = String::new();
    s.read_to_string(&mut r).unwrap();
    let (head, body) = r.split_once("\r\n\r\n").unwrap();
    (head[9..12].parse().unwrap(), body.to_owned())
}

/// SSE events as (event line, data JSON).
fn events(body: &str) -> Vec<(Option<String>, Value)> {
    let mut out = vec![];
    let mut name = None;
    for l in body.lines() {
        if let Some(n) = l.strip_prefix("event: ") {
            name = Some(n.to_owned());
        } else if let Some(d) = l.strip_prefix("data: ") {
            out.push((name.take(), serde_json::from_str(d).unwrap()));
        }
    }
    out
}

/// Event types, with runs of the same delta type collapsed (delta counts follow the text).
fn shape(evs: &[(Option<String>, Value)]) -> Vec<String> {
    let mut v: Vec<String> = vec![];
    for (name, d) in evs {
        let t = d["type"].as_str().unwrap_or("?").to_owned();
        assert_eq!(name.as_deref(), Some(t.as_str()), "event line matches data type");
        let t = match t.as_str() {
            "content_block_delta" => format!("{t}:{}", d["delta"]["type"].as_str().unwrap()),
            _ => t,
        };
        if t.contains("delta") && v.last() == Some(&t) {
            continue;
        }
        v.push(t);
    }
    v
}

/// The facts compared between oracle and eomoe for a non-streamed body.
fn summary(path: &str, b: &Value) -> Value {
    if path == "/v1/messages" {
        let u = &b["usage"];
        let calls: Vec<Value> = b["content"].as_array().unwrap().iter().filter(|c| c["type"] == "tool_use").map(|c| json!([c["name"], c["input"]])).collect();
        let has_text = b["content"].as_array().unwrap().iter().any(|c| c["type"] == "text" && c["text"].as_str().is_some_and(|t| !t.trim().is_empty()));
        json!({"prompt": u["input_tokens"].as_u64().unwrap() + u["cache_read_input_tokens"].as_u64().unwrap(), "stop": b["stop_reason"], "calls": calls, "text": has_text,
               "keys": keys(b), "model": b["model"]})
    } else {
        let out = b["output"].as_array().unwrap();
        let calls: Vec<Value> = out.iter().filter(|o| o["type"] == "function_call").map(|o| json!([o["name"], serde_json::from_str::<Value>(o["arguments"].as_str().unwrap()).unwrap()])).collect();
        let has_text = out.iter().any(|o| o["type"] == "message" && o["content"][0]["text"].as_str().is_some_and(|t| !t.trim().is_empty()));
        json!({"prompt": b["usage"]["input_tokens"], "status": b["status"], "calls": calls, "text": has_text, "keys": keys(b), "model": b["model"]})
    }
}

fn keys(v: &Value) -> Vec<String> {
    let mut k: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
    k.sort();
    k
}

#[test]
#[ignore]
fn api_matches_oracle() {
    let g: Value = serde_json::from_str(&std::fs::read_to_string("tests/golden/api.json").unwrap()).unwrap();
    let _s = start();
    let mut fails = vec![];
    for (name, c) in g["cases"].as_object().unwrap() {
        let path = c["path"].as_str().unwrap();
        let req = &c["request"];

        let (st, body) = http("POST", path, Some(req));
        assert_eq!(st, 200, "{name}: {body}");
        let ours = summary(path, &serde_json::from_str(&body).unwrap());
        let want = summary(path, &c["full"]["body"]);
        eprintln!("{name} full: ours {ours}\n{name} full: want {want}");
        if ours != want {
            fails.push(format!("{name} full"));
        }

        let mut sreq = req.clone();
        sreq["stream"] = true.into();
        let (st, body) = http("POST", path, Some(&sreq));
        assert_eq!(st, 200, "{name} stream: {body}");
        let ours = events(&body);
        let want: Vec<(Option<String>, Value)> = c["stream"]["events"].as_array().unwrap().iter().map(|e| (e["event"].as_str().map(str::to_owned), e["data"].clone())).collect();
        let (so, sw) = (shape(&ours), shape(&want));
        eprintln!("{name} stream: ours {so:?}\n{name} stream: want {sw:?}");
        // The last event carries the complete response: compare it like the full body.
        let last = |e: &[(Option<String>, Value)]| {
            if path == "/v1/messages" {
                e.iter().rev().find(|x| x.1["type"] == "message_delta").map(|x| x.1["delta"]["stop_reason"].clone())
            } else {
                e.last().map(|x| summary(path, &x.1["response"]))
            }
        };
        if so != sw || last(&ours) != last(&want) {
            fails.push(format!("{name} stream"));
        }
    }

    let ct = &g["count_tokens"];
    let (st, body) = http("POST", "/v1/messages/count_tokens", Some(&{
        let mut r = g["cases"]["messages_tool"]["request"].clone();
        r.as_object_mut().unwrap().remove("max_tokens");
        r.as_object_mut().unwrap().remove("temperature");
        r
    }));
    let ours: Value = serde_json::from_str(&body).unwrap();
    eprintln!("count_tokens: ours {st} {ours} want {}", ct["body"]);
    if st != 200 || ours != ct["body"] {
        fails.push("count_tokens".into());
    }

    let (st, body) = http("GET", "/health", None);
    if st != 200 || serde_json::from_str::<Value>(&body).unwrap() != g["health"]["body"] {
        fails.push("health".into());
    }
    let (st, body) = http("GET", "/v1/models", None);
    let m: Value = serde_json::from_str(&body).unwrap();
    let w = &g["models"]["body"];
    eprintln!("models: ours {m}");
    if st != 200 || keys(&m) != keys(w) || m["data"][0]["id"] != w["data"][0]["id"] || m["models"][0]["name"] != w["models"][0]["name"] {
        fails.push("models".into());
    }
    assert!(fails.is_empty(), "mismatches: {fails:?}");
}
