//! Ollama API adapter: /api/chat, /api/generate, /api/tags, /api/show, /api/ps, /api/version,
//! for tools that only speak Ollama. Shapes follow Ollama 0.33.1 (tests/golden/ollama.json):
//! NDJSON streams (stream defaults to true), `{"error": "..."}` bodies, durations in ns.
//! Requests run through the same pipeline as the OpenAI endpoints.

use super::*;
use crate::gguf::{Gguf, Value as G};
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The Ollama version whose API this matches; clients gate features on /api/version.
const VERSION: &str = "0.33.1";
const CAPABILITIES: [&str; 3] = ["completion", "tools", "insert"];

/// RFC 3339 UTC with nanoseconds, as Go's time.Time marshals.
pub fn rfc3339(t: SystemTime) -> String {
    let d = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    let (s, ns) = (d.as_secs() as i64, d.subsec_nanos());
    // Days to civil date (H. Hinnant's algorithm).
    let z = s.div_euclid(86400) + 719468;
    let (era, sod) = (z.div_euclid(146097), s.rem_euclid(86400));
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let (day, m) = (doy - (153 * mp + 2) / 5 + 1, if mp < 10 { mp + 3 } else { mp - 9 });
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{day:02}T{:02}:{:02}:{:02}.{ns:09}Z", sod / 3600, sod / 60 % 60, sod % 60)
}

/// Ollama's format.HumanNumber: 30.5B, 751.63M.
fn human(n: u64) -> String {
    let f = n as f64;
    match n {
        1_000_000_000.. if f % 1e9 == 0.0 => format!("{:.0}B", f / 1e9),
        1_000_000_000.. => format!("{:.1}B", f / 1e9),
        1_000_000.. => format!("{:.2}M", f / 1e6),
        1_000.. => format!("{:.2}K", f / 1e3),
        _ => n.to_string(),
    }
}

/// The model as Ollama describes it (tags, show, ps), from the GGUF at `path`.
pub fn describe(g: &Gguf, path: &Path) -> Value {
    let meta = std::fs::metadata(path).ok();
    let size = meta.as_ref().map_or(0, |m| m.len());
    let mtime = meta.and_then(|m| m.modified().ok()).unwrap_or(UNIX_EPOCH);
    let arch = g.str("general.architecture").unwrap_or("");
    let params = g.u64("general.parameter_count").unwrap_or_else(|| g.tensors.iter().map(|t| t.dims.iter().product::<u64>()).sum());
    let quant = match g.u64("general.file_type") {
        Some(0) => "F32",
        Some(1) => "F16",
        Some(7) => "Q8_0",
        Some(14) => "Q4_K_S",
        Some(15) => "Q4_K_M",
        Some(16) => "Q5_K_S",
        Some(17) => "Q5_K_M",
        Some(18) => "Q6_K",
        Some(32) => "BF16",
        _ => "unknown",
    };
    // ponytail: a hash of path, size and mtime, not the file's sha256 (an 18 GB read at startup).
    let digest: String = (0..4u64)
        .map(|i| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (i, path, size, rfc3339(mtime)).hash(&mut h);
            format!("{:016x}", h.finish())
        })
        .collect();
    let mut keys: Vec<&String> = g.meta.keys().filter(|k| *k != "tokenizer.chat_template").collect();
    keys.sort();
    // Arrays (vocab, merges) are null, as Ollama shows them without `verbose`.
    let model_info: serde_json::Map<String, Value> = keys
        .into_iter()
        .map(|k| {
            let v = match &g.meta[k] {
                G::U64(v) => json!(v),
                G::I64(v) => json!(v),
                G::F64(v) => json!(v),
                G::Bool(v) => json!(v),
                G::Str(v) => json!(v),
                G::Arr(_) => Value::Null,
            };
            (k.clone(), v)
        })
        .collect();
    let tensors: Vec<Value> = g.tensors.iter().map(|t| json!({"name": t.name, "type": format!("{:?}", t.ty), "shape": t.dims})).collect();
    json!({
        "size": size, "digest": digest, "modified_at": rfc3339(mtime), "path": path.display().to_string(),
        "details": {"parent_model": "", "format": "gguf", "family": arch, "families": [arch], "parameter_size": human(params), "quantization_level": quant},
        "context_length": g.u64(&format!("{arch}.context_length")), "embedding_length": g.u64(&format!("{arch}.embedding_length")),
        "model_info": model_info, "tensors": tensors,
        "license": g.str("general.license").unwrap_or(""), "template": g.str("tokenizer.chat_template").unwrap_or(""),
    })
}

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({ "error": msg.into() }))).into_response()
}

/// The request body, after checking its model names ours. `show` reports the name with its tag.
#[allow(clippy::result_large_err)] // axum's Response, built once per bad request
fn body(app: &App, raw: &[u8], show: bool) -> Result<Value, Response> {
    let b: Value = serde_json::from_slice(raw).map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;
    // `name` is the older field; the ollama CLI sends both, with `model` empty.
    let m = [&b["model"], &b["name"]].iter().filter_map(|v| v.as_str()).find(|m| !m.is_empty()).unwrap_or("");
    let tagged = if m.contains(':') { m.to_owned() } else { format!("{m}:latest") };
    if m.is_empty() {
        Err(err(StatusCode::BAD_REQUEST, "model is required"))
    } else if tagged != format!("{}:latest", app.alias) {
        Err(err(StatusCode::NOT_FOUND, format!("model '{}' not found", if show { &tagged } else { m })))
    } else {
        Ok(b)
    }
}

/// Limits, sampling, stop strings and `format` of an Ollama request.
fn options(b: &Value) -> Chat {
    let o = &b["options"];
    let mut sampling = sampling(o);
    // Ollama's own defaults where they differ from llama.cpp's (api/types.go DefaultOptions).
    // ponytail: repeat_penalty and num_ctx are ignored (no repetition penalty; context is the server's).
    if o["top_p"].is_null() {
        sampling.top_p = 0.9;
    }
    if o["min_p"].is_null() {
        sampling.min_p = 0.0;
    }
    sampling.schema = match &b["format"] {
        Value::String(s) if s == "json" => Some(json!({"type": "object"})), // Ollama's JSON grammar: root ::= object
        s @ Value::Object(_) => Some(s.clone()),
        _ => None,
    };
    let stop = match &o["stop"] {
        Value::String(s) => vec![s.clone()],
        v => v.as_array().into_iter().flatten().filter_map(|s| s.as_str().map(str::to_owned)).collect(),
    };
    // num_predict -1 (default) and -2 (fill the context) both mean no limit here.
    let max_tokens = o["num_predict"].as_i64().filter(|&n| n >= 0).map_or(usize::MAX, |n| n as usize);
    Chat { messages: Value::Null, tools: json!([]), max_tokens, sampling, stop }
}

/// Ollama messages -> OpenAI chat messages (tool-call arguments stay objects; images dropped).
fn messages(ms: &[Value]) -> Value {
    ms.iter()
        .map(|m| {
            let mut o = json!({"role": m["role"], "content": m["content"].as_str().unwrap_or("")});
            if let Some(tcs) = m["tool_calls"].as_array() {
                o["tool_calls"] = tcs.iter().map(|t| json!({"type": "function", "id": t["id"].as_str().map_or_else(call_id, str::to_owned),
                                                            "function": {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}})).collect();
            }
            if let Some(n) = m["tool_name"].as_str() {
                o["name"] = n.into();
            }
            o
        })
        .collect()
}

fn call_id() -> String {
    format!("call_{}", rand_id().to_ascii_lowercase()[..8].to_owned())
}

/// The reply to a request with nothing to generate: Ollama loads the model and says so.
fn loaded(app: &App, chat: bool) -> Response {
    let mut v = json!({"model": app.alias, "created_at": rfc3339(SystemTime::now())});
    if chat {
        v["message"] = json!({"role": "assistant", "content": ""});
    } else {
        v["response"] = "".into();
    }
    (v["done"], v["done_reason"]) = (true.into(), "load".into());
    Json(v).into_response()
}

pub async fn chat(State(app): State<Arc<App>>, raw: Bytes) -> Response {
    let t0 = Instant::now();
    let b = match body(&app, &raw, false) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let Some(ms) = b["messages"].as_array().filter(|m| !m.is_empty()) else {
        return loaded(&app, true);
    };
    let tools: Vec<Value> = b["tools"].as_array().into_iter().flatten().filter(|t| t["type"] == "function").cloned().collect();
    let chat = Chat { messages: messages(ms), tools: tools.into(), ..options(&b) };
    let f = Ollama { app: app.clone(), t0, chat: true, context: None, a: Acc::default() };
    respond(app.run(chat), f, b["stream"].as_bool().unwrap_or(true)).await
}

pub async fn generate(State(app): State<Arc<App>>, raw: Bytes) -> Response {
    let t0 = Instant::now();
    let b = match body(&app, &raw, false) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let (prompt, suffix) = (b["prompt"].as_str().unwrap_or(""), b["suffix"].as_str().unwrap_or(""));
    if prompt.is_empty() && suffix.is_empty() {
        return loaded(&app, false);
    }
    let raw_mode = b["raw"].as_bool().unwrap_or(false);
    let tokens = if raw_mode {
        Ok(app.tok.encode(prompt, true))
    } else if !suffix.is_empty() {
        // The FIM form of Qwen coder templates: <|fim_prefix|>{prompt}<|fim_suffix|>{suffix}<|fim_middle|>.
        Ok(app.tok.encode(&format!("<|fim_prefix|>{prompt}<|fim_suffix|>{suffix}<|fim_middle|>"), true))
    } else {
        let mut ms = vec![];
        if let Some(s) = b["system"].as_str() {
            ms.push(json!({"role": "system", "content": s}));
        }
        ms.push(json!({"role": "user", "content": prompt}));
        app.prompt(&Chat { messages: ms.into(), ..options(&b) })
    };
    // Templated output is chat output (gpt-oss: Harmony channels); raw and FIM output is not.
    let messages = if raw_mode || !suffix.is_empty() { Value::Null } else { json!([]) };
    // A previous response's `context` continues that conversation.
    let tokens = tokens.map(|t| b["context"].as_array().into_iter().flatten().filter_map(|c| c.as_u64().map(super::tok_id)).chain(t).collect::<Vec<u32>>());
    let context = (!raw_mode).then(|| tokens.as_ref().map(Vec::clone).unwrap_or_default());
    let f = Ollama { app: app.clone(), t0, chat: false, context, a: Acc::default() };
    let rx = tokens.and_then(|t| app.generate(t, Chat { messages, ..options(&b) }));
    respond(rx, f, b["stream"].as_bool().unwrap_or(true)).await
}

fn entry(app: &App) -> Value {
    let d = &app.ollama;
    json!({"name": format!("{}:latest", app.alias), "model": format!("{}:latest", app.alias), "size": d["size"], "digest": d["digest"], "details": d["details"]})
}

pub async fn tags(State(app): State<Arc<App>>) -> Json<Value> {
    let (d, mut m) = (&app.ollama, entry(&app));
    m["modified_at"] = d["modified_at"].clone();
    (m["details"]["context_length"], m["details"]["embedding_length"]) = (d["context_length"].clone(), d["embedding_length"].clone());
    m["capabilities"] = json!(CAPABILITIES);
    Json(json!({ "models": [m] }))
}

pub async fn show(State(app): State<Arc<App>>, raw: Bytes) -> Response {
    if let Err(r) = body(&app, &raw, true) {
        return r;
    }
    let d = &app.ollama;
    Json(json!({"license": d["license"], "modelfile": format!("# Modelfile generated by eomoe\nFROM {}\n", d["path"].as_str().unwrap_or("")),
                "parameters": "", "template": d["template"], "details": d["details"], "model_info": d["model_info"], "tensors": d["tensors"],
                "capabilities": CAPABILITIES, "modified_at": d["modified_at"]}))
    .into_response()
}

pub async fn ps(State(app): State<Arc<App>>) -> Json<Value> {
    let mut m = entry(&app);
    // Always loaded: no keep_alive expiry.
    (m["expires_at"], m["size_vram"], m["context_length"]) = ("2318-01-01T00:00:00.000000000Z".into(), 0.into(), app.n_ctx.into());
    Json(json!({ "models": [m] }))
}

pub async fn version() -> Json<Value> {
    Json(json!({ "version": VERSION }))
}

/// Ollama /api/chat and /api/generate lines.
struct Ollama {
    app: Arc<App>,
    t0: Instant,
    chat: bool,
    /// Prompt tokens for /api/generate's `context` (not in raw mode).
    context: Option<Vec<u32>>,
    a: Acc,
}

impl Ollama {
    fn line(&self, text: &str, done: bool) -> Value {
        self.line_thinking(text, "", done)
    }

    /// A line with `thinking` (gpt-oss reasoning) when non-empty.
    fn line_thinking(&self, text: &str, thinking: &str, done: bool) -> Value {
        let mut v = json!({"model": self.app.alias, "created_at": rfc3339(SystemTime::now())});
        let o = if self.chat {
            v["message"] = json!({"role": "assistant", "content": text});
            &mut v["message"]
        } else {
            v["response"] = text.into();
            &mut v
        };
        if !thinking.is_empty() {
            o["thinking"] = thinking.into();
        }
        v["done"] = done.into();
        v
    }

    fn calls(&self) -> Value {
        self.a.calls.iter().enumerate().map(|(k, (_, c))| {
            let args: Value = serde_json::from_str(&c.arguments).unwrap_or_else(|_| json!({}));
            json!({"id": call_id(), "function": {"index": k, "name": c.name, "arguments": args}})
        }).collect()
    }

    fn last(&self, text: &str, thinking: &str) -> Value {
        let a = &self.a;
        let mut v = self.line_thinking(text, thinking, true);
        v["done_reason"] = if a.finish == Some(Finish::Length) { "length" } else { "stop" }.into();
        if let Some(c) = &self.context {
            // ponytail: the response is re-encoded, not the sampled ids; same text either way.
            v["context"] = c.iter().copied().chain(self.app.tok.encode(&a.text, false)).collect();
        }
        let ns = |ms: f64| (ms * 1e6) as u64;
        v["total_duration"] = (self.t0.elapsed().as_nanos() as u64).into();
        v["load_duration"] = 0.into();
        v["prompt_eval_count"] = a.n_prompt.into();
        v["prompt_eval_duration"] = ns(a.timings.prompt_ms).into();
        v["eval_count"] = a.n_out.into();
        v["eval_duration"] = ns(a.timings.predicted_ms).into();
        v
    }
}

impl Format for Ollama {
    fn on(&mut self, out: &Out) -> Vec<String> {
        self.a.on(out);
        match out {
            Out::Start { .. } => vec![],
            Out::Text(t) => vec![format!("{}\n", self.line(t, false))],
            Out::Reasoning(t) => vec![format!("{}\n", self.line_thinking("", t, false))],
            Out::End { .. } => {
                let mut v = vec![];
                if !self.a.calls.is_empty() {
                    let mut l = self.line("", false);
                    l["message"]["tool_calls"] = self.calls();
                    v.push(format!("{l}\n"));
                }
                v.push(format!("{}\n", self.last("", "")));
                v
            }
            Out::Error(e) => vec![format!("{}\n", json!({ "error": self.err(e.clone()) }))],
        }
    }

    fn body(&self) -> Value {
        let mut v = self.last(&self.a.text, &self.a.reasoning);
        if !self.a.calls.is_empty() {
            v["message"]["tool_calls"] = self.calls();
        }
        v
    }

    fn ndjson(&self) -> bool {
        true
    }

    fn err(&self, e: Value) -> Value {
        e["message"].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn go_time_and_human_numbers() {
        assert_eq!(rfc3339(UNIX_EPOCH + std::time::Duration::new(1_700_000_000, 5)), "2023-11-14T22:13:20.000000005Z");
        assert_eq!(rfc3339(UNIX_EPOCH + std::time::Duration::from_secs(951_782_400)), "2000-02-29T00:00:00.000000000Z");
        assert_eq!([human(30_532_122_624), human(751_632_384), human(27_000_000_000)], ["30.5B", "751.63M", "27B"]);
    }
}
