//! Phase 11 gate (a): raw completions answer the captured oracle requests
//! (tests/golden/completions.json, from tools/capture_completions_golden.py) with the same
//! shape: /v1/completions (OpenAI text_completion), /completion[s] (llama.cpp native) and
//! /infill (FIM). Prompt tokens and the infill prompt text must match exactly (the FIM prompt
//! construction); stop type, stopping word, key sets and the streamed event sequence must
//! match. Generated text is only required to be non-empty (backend precision floor, Phase 3).
//! Native bodies leave out `generation_settings` (llama.cpp's sampler dump).
//!
//! Needs the GPU and the model: cargo test --release --test phase11_golden -- --ignored

mod common;
use common::events;
use serde_json::{json, Value};

const PORT: u16 = 8096;

fn keys(v: &Value) -> Vec<String> {
    let mut k: Vec<String> = v.as_object().into_iter().flat_map(|o| o.keys().cloned()).filter(|k| k != "generation_settings").collect();
    k.sort();
    k
}

/// The facts compared for a final body (or the final stream event).
fn summary(path: &str, b: &Value) -> Value {
    if path == "/v1/completions" {
        let c = &b["choices"][0];
        json!({"prompt": b["usage"]["prompt_tokens"], "finish": c["finish_reason"], "keys": keys(b), "choice_keys": keys(c), "usage_keys": keys(&b["usage"]),
               "timings_keys": keys(&b["timings"]), "object": b["object"], "model": b["model"]})
    } else {
        json!({"prompt": b["tokens_evaluated"], "prompt_text": b["prompt"], "stop": b["stop"], "stop_type": b["stop_type"], "stopping_word": b["stopping_word"],
               "keys": keys(b), "timings_keys": keys(&b["timings"]), "model": b["model"]})
    }
}

fn text(path: &str, b: &Value) -> String {
    let t = if path == "/v1/completions" { &b["choices"][0]["text"] } else { &b["content"] };
    t.as_str().unwrap().to_owned()
}

/// Streamed events: token chunks (runs collapsed, with their key set), then the final event.
fn shape(path: &str, evs: &[Value]) -> Vec<String> {
    let mut v: Vec<String> = vec![];
    for e in evs {
        let t = if e == "[DONE]" {
            "[DONE]".into()
        } else if path == "/v1/completions" {
            match e["choices"][0]["finish_reason"].as_str() {
                Some(f) => format!("finish:{f}"),
                None => format!("text:{:?}", keys(e)),
            }
        } else if e["stop"] == true {
            "final".into()
        } else {
            format!("token:{:?}", keys(e))
        };
        if v.last() != Some(&t) {
            v.push(t);
        }
    }
    v
}

#[test]
#[ignore]
fn completions_match_oracle() {
    let g = common::golden("completions.json");
    let _s = common::start(PORT);
    let mut fails = vec![];
    for (name, c) in g["cases"].as_object().unwrap() {
        let (path, req) = (c["path"].as_str().unwrap(), &c["request"]);
        let (st, body) = common::http(PORT, "POST", path, Some(req));
        assert_eq!(st, 200, "{name}: {body}");
        let b: Value = serde_json::from_str(&body).unwrap();
        let (ours, want) = (summary(path, &b), summary(path, &c["full"]["body"]));
        eprintln!("{name} full: ours {ours}\n{name} full: want {want}\n{name} text: {:?}", text(path, &b));
        if ours != want || text(path, &b).trim().is_empty() {
            fails.push(format!("{name} full"));
        }

        let mut sreq = req.clone();
        sreq["stream"] = true.into();
        let (st, body) = common::http(PORT, "POST", path, Some(&sreq));
        assert_eq!(st, 200, "{name} stream: {body}");
        let ours: Vec<Value> = events(&body).into_iter().map(|e| e.1).collect();
        let want = c["stream"]["events"].as_array().unwrap();
        let (so, sw) = (shape(path, &ours), shape(path, want));
        let last = |e: &[Value]| e.iter().rev().find(|x| *x != "[DONE]").map(|x| summary(path, x));
        eprintln!("{name} stream: ours {so:?} {:?}\n{name} stream: want {sw:?} {:?}", last(&ours), last(want));
        if so != sw || last(&ours) != last(want) {
            fails.push(format!("{name} stream"));
        }
    }
    for (name, c) in g["errors"].as_object().unwrap() {
        let (st, body) = common::http(PORT, "POST", c["path"].as_str().unwrap(), Some(&c["request"]));
        let b: Value = serde_json::from_str(&body).unwrap();
        eprintln!("{name}: ours {st} {b}");
        if st as u64 != c["status"].as_u64().unwrap() || b["error"]["type"] != c["body"]["error"]["type"] {
            fails.push(name.clone());
        }
    }
    assert!(fails.is_empty(), "mismatches: {fails:?}");
}

/// Token ids from the client are checked against the vocabulary: a bad id is a 400, not a
/// panic that kills the handler or the model thread (later requests must still work).
#[test]
#[ignore]
fn bad_token_ids_are_rejected() {
    let _s = common::start(PORT);
    let big = serde_json::json!(4294967296u64 + 13); // wraps to 13 if cast blindly
    for (path, body) in [
        ("/v1/completions", serde_json::json!({"prompt": [9_999_999], "max_tokens": 1})),
        ("/completion", serde_json::json!({"prompt": [9_999_999], "n_predict": 1})),
        ("/v1/completions", serde_json::json!({"prompt": [big], "max_tokens": 1})),
        ("/api/generate", serde_json::json!({"model": "local-code", "prompt": "hi", "context": [9_999_999], "stream": false, "options": {"num_predict": 1}})),
    ] {
        let (st, b) = common::http(PORT, "POST", path, Some(&body));
        assert_eq!(st, 400, "{path} {body}: {b}");
    }
    let (st, b) = common::http(PORT, "POST", "/v1/completions", Some(&serde_json::json!({"prompt": "fn main", "max_tokens": 2})));
    assert_eq!(st, 200, "server still up: {b}");
}
