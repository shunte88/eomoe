//! Phase 11 gate (d): the Ollama API (/api/chat, /api/generate, /api/tags, /api/show, /api/ps,
//! /api/version) answers the captured Ollama requests (tests/golden/ollama.json, from
//! tools/capture_ollama_golden.py against Ollama 0.33.1 with qwen3:0.6b) with the same wire
//! shapes: status, media type, NDJSON line structure, key sets (top level, message, tool call,
//! details), done/done_reason where the model cannot change it, error bodies.
//! The oracle model differs from ours, so text is not compared; instead tool cases must call
//! the tool, json/schema cases must parse, and the tool-result case must answer.
//!
//! Needs the GPU and the model: cargo test --release --test phase11d_ollama -- --ignored

mod common;
use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpStream;

const PORT: u16 = 8093;
const MODEL: &str = "local-code";

/// (status, media type, body) over HTTP/1.0.
fn call(method: &str, path: &str, body: &Value) -> (u16, String, String) {
    let mut s = TcpStream::connect(("127.0.0.1", PORT)).unwrap();
    let b = match body {
        Value::Null => String::new(),
        Value::String(raw) => raw.clone(),
        v => v.to_string(),
    };
    write!(s, "{method} {path} HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len()).unwrap();
    let mut r = String::new();
    s.read_to_string(&mut r).unwrap();
    let (head, body) = r.split_once("\r\n\r\n").unwrap();
    let status = head.split(' ').nth(1).unwrap().parse().unwrap();
    let ctype = head.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-type:").map(|v| v.trim().split(';').next().unwrap().to_owned())).unwrap_or_default();
    (status, ctype, body.to_owned())
}

fn keys(v: &Value) -> Vec<String> {
    let mut k: Vec<String> = v.as_object().into_iter().flat_map(|o| o.keys().cloned()).collect();
    k.sort();
    k
}

/// Key sets at the levels clients read.
fn shape(v: &Value) -> Value {
    let tc = &v["message"]["tool_calls"][0];
    serde_json::json!({"top": keys(v), "message": keys(&v["message"]), "tool_call": keys(tc), "function": keys(&tc["function"]),
                       "details": keys(&v["details"]), "model0": keys(&v["models"][0]), "model0_details": keys(&v["models"][0]["details"])})
}

fn lines(body: &str) -> Vec<Value> {
    body.lines().filter(|l| !l.is_empty()).map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad NDJSON line ({e}): {l}"))).collect()
}

/// Swap the oracle's model name for ours.
fn retarget(body: &Value, oracle_model: &str) -> Value {
    let mut b = body.clone();
    for k in ["model", "name"] {
        if b[k] == oracle_model {
            b[k] = MODEL.into();
        }
    }
    b
}

#[test]
#[ignore]
fn ollama_api_matches_oracle() {
    let g = common::golden("ollama.json");
    let om = g["model"].as_str().unwrap();
    let _s = common::start(PORT);
    let mut fails = vec![];
    for c in g["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        if name == "root" {
            continue; // GET / stays eomoe's chat page (gate revision).
        }
        let (status, ctype, body) = call(c["method"].as_str().unwrap(), c["path"].as_str().unwrap(), &retarget(&c["body"], om));
        let want: Vec<Value> = c["lines"].as_array().unwrap().clone();
        let got = lines(&body);
        let mut check = |ok: bool, what: String| {
            if !ok {
                fails.push(format!("{name}: {what}"));
            }
        };
        check(status == c["status"].as_u64().unwrap() as u16, format!("status {status} vs {}", c["status"]));
        check(ctype == c["content_type"].as_str().unwrap().split(';').next().unwrap(), format!("content type {ctype:?} vs {}", c["content_type"]));
        if got.is_empty() {
            check(false, format!("empty body {body:?}"));
            continue;
        }
        let (wl, gl) = (want.last().unwrap(), got.last().unwrap());
        // Streams: every line but the last has the oracle's first-line shape.
        if want.len() > 1 {
            check(got.len() > 1, format!("{} lines, oracle streamed {}", got.len(), want.len()));
            for (i, l) in got[..got.len() - 1].iter().enumerate() {
                let w = if l["message"]["tool_calls"].is_array() { want.iter().find(|w| w["message"]["tool_calls"].is_array()).unwrap_or(&want[0]) } else { &want[0] };
                check(shape(l) == shape(w), format!("line {i} shape {} vs {}", shape(l), shape(w)));
                check(l["done"] == false, format!("line {i} done"));
            }
        } else {
            check(got.len() == 1, format!("{} lines, oracle 1", got.len()));
        }
        // show: compare the top level and details only (model_info/tensors differ per model).
        check(shape(gl) == shape(wl), format!("final shape\n  got  {}\n  want {}", shape(gl), shape(wl)));
        if wl.get("done").is_some() {
            check(gl["done"] == true, "final done".into());
            let dr = gl["done_reason"].as_str().unwrap_or("");
            if ["chat_length", "chat_load", "generate_load"].contains(&name) {
                check(gl["done_reason"] == wl["done_reason"], format!("done_reason {dr} vs {}", wl["done_reason"]));
            } else {
                check(dr == "stop" || dr == "length", format!("done_reason {dr}"));
            }
            check(gl["model"] == MODEL, format!("model {}", gl["model"]));
        }
        if name.starts_with("err_") {
            check(gl["error"].is_string(), format!("error body {gl}"));
            if name != "err_bad_json" {
                let w = wl["error"].as_str().unwrap().replace(om, MODEL);
                check(gl["error"] == w.as_str(), format!("error {} vs {w}", gl["error"]));
            }
        }
        let content: String = got.iter().filter_map(|l| l["message"]["content"].as_str()).collect();
        match name {
            "chat_tools" | "chat_tools_stream" => {
                let tc = got.iter().find_map(|l| l["message"]["tool_calls"].as_array().cloned()).unwrap_or_default();
                check(tc.len() == 1 && tc[0]["function"]["name"] == "get_weather" && tc[0]["function"]["arguments"]["city"].is_string(), format!("tool call {tc:?}"));
            }
            "chat_tool_result" => check(content.contains("18"), format!("tool result answer {content:?}")),
            "chat_json" => check(serde_json::from_str::<Value>(&content).is_ok_and(|v| v.is_object()), format!("json {content:?}")),
            "chat_schema" => {
                let v: Value = serde_json::from_str(&content).unwrap_or_default();
                check(keys(&v) == ["age", "name"] && v["age"].is_i64() && v["name"].is_string(), format!("schema {content:?}"));
            }
            "generate" | "generate_stream" | "generate_system" => {
                check(gl["context"].as_array().is_some_and(|c| !c.is_empty() && c.iter().all(Value::is_u64)), "context ids".into())
            }
            "tags" => check(gl["models"][0]["name"] == format!("{MODEL}:latest").as_str(), format!("tags name {}", gl["models"][0]["name"])),
            _ => {}
        }
        eprintln!("{name}: {status} {} lines", got.len());
    }
    assert!(fails.is_empty(), "{} mismatches:\n{}", fails.len(), fails.join("\n"));
}
