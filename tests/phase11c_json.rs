//! Phase 11 gate (c): constrained output through the server. With `response_format`
//! json_object the reply is one JSON value (as llama-server: no schema means schema {}, any
//! value, server-common.cpp; not OpenAI's object-only rule); with json_schema it also matches the schema
//! (keys, required, types, enum). Both end with finish_reason "stop", not "length".
//! Gate revision: no oracle capture (CPU-oracle runs locked the machine, see STATUS); the
//! check is that the output parses and validates, which is what llama-server guarantees.
//!
//! Also `logit_bias`: banning the greedy token changes the reply, a +100 bias forces a token.
//!
//! Needs the GPU and the model: cargo test --release --test phase11c_json -- --ignored

mod common;
use serde_json::{json, Value};

const PORT: u16 = 8094;

fn ask(content: &str, rf: Value) -> (Value, String) {
    let body = json!({"messages": [{"role": "user", "content": content}], "max_tokens": 300, "temperature": 0, "response_format": rf});
    let (st, b) = common::http(PORT, "POST", "/v1/chat/completions", Some(&body));
    assert_eq!(st, 200, "{b}");
    let c = serde_json::from_str::<Value>(&b).unwrap()["choices"][0].clone();
    assert_eq!(c["finish_reason"], "stop", "{b}");
    let text = c["message"]["content"].as_str().unwrap().to_owned();
    (serde_json::from_str(&text).unwrap_or_else(|e| panic!("not JSON ({e}): {text:?}")), text)
}

#[test]
#[ignore]
fn response_format_yields_valid_json() {
    let _s = common::start(PORT);

    let (_, text) = ask("Name three primary colors.", json!({"type": "json_object"}));
    eprintln!("json_object: {text}");

    let schema = json!({"type": "object", "properties": {
        "name": {"type": "string"}, "age": {"type": "integer"},
        "lang": {"type": "string", "enum": ["rust", "python", "go"]}}, "required": ["name", "age", "lang"]});
    let (v, text) = ask("Invent a programmer: their name, age and favourite language.",
                        json!({"type": "json_schema", "json_schema": {"name": "dev", "schema": schema}}));
    eprintln!("json_schema: {text}");
    let o = v.as_object().unwrap();
    let mut keys: Vec<&str> = o.keys().map(String::as_str).collect();
    keys.sort();
    assert_eq!(keys, ["age", "lang", "name"], "{text}");
    assert!(o["name"].is_string() && o["age"].is_i64(), "{text}");
    assert!(["rust", "python", "go"].contains(&o["lang"].as_str().unwrap()), "{text}");
}

fn complete(path: &str, prompt: &str, max: u32, bias: Value) -> String {
    let mut body = json!({"max_tokens": max, "temperature": 0, "logit_bias": bias});
    if path == "/v1/completions" {
        body["prompt"] = prompt.into();
    } else {
        body["messages"] = json!([{"role": "user", "content": prompt}]);
    }
    let (st, b) = common::http(PORT, "POST", path, Some(&body));
    assert_eq!(st, 200, "{b}");
    let c = &serde_json::from_str::<Value>(&b).unwrap()["choices"][0];
    c["text"].as_str().or(c["message"]["content"].as_str()).unwrap().to_owned()
}

#[test]
#[ignore]
fn logit_bias_bans_and_forces() {
    let _s = common::start(PORT);
    let (p, chat) = ("What is the capital of France?", "/v1/chat/completions");
    // The greedy first reply token's id: the last id of Ollama's `context` (prompt + output).
    let (_, b) = common::http(PORT, "POST", "/api/generate", Some(&json!({"model": "local-code", "prompt": p, "stream": false, "options": {"temperature": 0, "num_predict": 1}})));
    let id = serde_json::from_str::<Value>(&b).unwrap()["context"].as_array().unwrap().last().unwrap().as_u64().unwrap();
    let greedy = complete(chat, p, 1, Value::Null);
    let banned = complete(chat, p, 1, json!({id.to_string(): -100}));
    let banned_llama = complete(chat, p, 1, json!([[id, false]]));
    let forced = complete("/v1/completions", "Hello", 3, json!({id.to_string(): 100}));
    eprintln!("id {id}: greedy {greedy:?} banned {banned:?} {banned_llama:?} forced {forced:?}");
    assert_ne!(banned, greedy);
    assert_ne!(banned_llama, greedy);
    assert_eq!(forced, greedy.repeat(3));
}
