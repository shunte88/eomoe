//! HTTP API, wire-compatible with llama-server for the endpoints Claude Code, Codex and OpenAI
//! clients use: POST /v1/messages (Anthropic), /v1/responses (OpenAI Responses),
//! /v1/chat/completions (OpenAI chat), /v1/messages/count_tokens; raw prompts on
//! /v1/completions (OpenAI text) and /completion, /infill (llama.cpp native, FIM);
//! GET /v1/models, /health. Event sequences and JSON shapes follow llama-server
//! (tools/server/server-task.cpp); tests/golden/api.json holds captured oracle traffic.
//!
//! Every API is converted to OpenAI chat messages, rendered with the GGUF chat template and run
//! on one model thread, which owns the engine (CUDA work stays on the thread that created it).

use crate::generate::{Batcher, Event, Finish, Request, Sampling, Utf8Stream};
use crate::qwen3moe::Engine;
use crate::template::ChatTemplate;
use crate::tokenizer::Tokenizer;
use crate::toolcall::{Splitter, ToolCall};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use tokio::sync::mpsc as tmpsc;

pub struct App {
    tok: Tokenizer,
    tpl: ChatTemplate,
    jobs: mpsc::Sender<Request>,
    alias: String,
    n_ctx: usize,
}

impl App {
    /// Start the model thread; `load` builds the engine on it.
    pub fn new<E: Engine + 'static>(
        tok: Tokenizer,
        tpl: ChatTemplate,
        alias: String,
        ram_cache: usize,
        load: impl FnOnce() -> Result<E, String> + Send + 'static,
    ) -> Result<Arc<Self>, String> {
        // llama.cpp's EOG set for the Qwen vocab: EOS/EOT plus the FIM end tokens.
        let mut eog: Vec<u32> = ["<|im_end|>", "<|endoftext|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>"].iter().filter_map(|t| tok.token_id(t)).collect();
        eog.push(tok.eos);
        let (jobs, rx) = mpsc::channel::<Request>();
        let (ready_tx, ready) = mpsc::channel();
        let eog_t = eog;
        // For constrained output; control tokens never appear inside JSON.
        let token_bytes: Vec<Vec<u8>> = (0..tok.n_vocab() as u32).map(|t| if tok.is_control(t) { vec![] } else { tok.decode_bytes(&[t]) }).collect();
        std::thread::Builder::new()
            .name("eomoe-model".into())
            .spawn(move || {
                let mut b = match load() {
                    Ok(e) => {
                        ready_tx.send(Ok(e.n_ctx())).ok();
                        let mut b = Batcher::new(e, eog_t);
                        b.ram_cache = ram_cache;
                        b.token_bytes = token_bytes;
                        b
                    }
                    Err(e) => return drop(ready_tx.send(Err(e))),
                };
                // A request that panics the engine kills this thread; open streams report an
                // error and later requests get "model thread stopped".
                loop {
                    if b.idle() {
                        match rx.recv() {
                            Ok(r) => b.submit(r),
                            Err(_) => return,
                        }
                    }
                    while let Ok(r) = rx.try_recv() {
                        b.submit(r);
                    }
                    b.step();
                }
            })
            .map_err(|e| e.to_string())?;
        let n_ctx = ready.recv().map_err(|_| "model thread died while loading".to_string())??;
        Ok(Arc::new(Self { tok, tpl, jobs, alias, n_ctx }))
    }
}

/// A request for the model thread whose events go to `tx`, logging its timings when done.
fn request(prompt: Vec<u32>, max_tokens: usize, sampling: Sampling, tx: tmpsc::UnboundedSender<Event>) -> Request {
    let (n_prompt, t0) = (prompt.len(), std::time::Instant::now());
    let (mut n_cached, mut t1, mut n) = (0, None, 0usize);
    let sink = Box::new(move |ev: Event| {
        match ev {
            Event::Started { n_cached: c } => n_cached = c,
            Event::Token(_) => {
                t1.get_or_insert_with(std::time::Instant::now);
                n += 1;
            }
            Event::Done(f) => {
                let t1 = t1.unwrap_or_else(std::time::Instant::now);
                let (pre, dec) = (t1 - t0, t1.elapsed());
                log::info!(
                    "prompt {n_prompt} tok ({n_cached} cached), queue+prefill {:.1}s ({:.0} tok/s), {n} out ({:.1} tok/s), {f:?}",
                    pre.as_secs_f32(), (n_prompt - n_cached) as f32 / pre.as_secs_f32(),
                    n.saturating_sub(1) as f32 / dec.as_secs_f32().max(1e-6)
                );
            }
        }
        tx.send(ev).is_ok()
    });
    Request { prompt, max_tokens, sampling, sink }
}

/// One chat request in the internal (OpenAI chat) form.
struct Chat {
    messages: Value,
    tools: Value,
    max_tokens: usize,
    sampling: Sampling,
    stop: Vec<String>,
}

/// Wall time of a request's prefill (start to first token) and decode (first token to end), ms.
#[derive(Clone, Copy, Default)]
struct Timings {
    prompt_ms: f64,
    predicted_ms: f64,
}

/// What a request's stream carries to the API formatter.
enum Out {
    Start { n_prompt: usize, n_cached: usize },
    Text(String),
    End { calls: Vec<ToolCall>, finish: Finish, n_out: usize, stop_word: Option<String>, timings: Timings },
    /// Generation failed after the response started.
    Error(Value),
}

#[derive(Debug)]
struct ApiError(StatusCode, Value);

impl ApiError {
    fn new(code: StatusCode, ty: &str, msg: impl Into<String>) -> Self {
        Self(code, json!({"code": code.as_u16(), "message": msg.into(), "type": ty}))
    }
    fn bad(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

fn rand_id() -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;
    let mut x = t ^ N.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9E3779B97F4A7C15) | 1;
    const A: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    (0..32)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            A[(x % 62) as usize] as char
        })
        .collect()
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}

/// Content-part arrays become one string (parts joined by "\n", as llama.cpp does for
/// string-only templates); non-text parts are dropped.
fn flatten_content(messages: &mut Value) {
    for m in messages.as_array_mut().into_iter().flatten() {
        if let Some(parts) = m.get("content").and_then(Value::as_array) {
            let text: Vec<&str> = parts.iter().filter(|p| p["type"] == "text").filter_map(|p| p["text"].as_str()).collect();
            m["content"] = text.join("\n").into();
        }
    }
}

impl App {
    fn prompt(&self, chat: &Chat) -> Result<Vec<u32>, ApiError> {
        let mut messages = chat.messages.clone();
        flatten_content(&mut messages);
        let mut req = json!({ "messages": messages, "add_generation_prompt": true });
        if chat.tools.as_array().is_some_and(|t| !t.is_empty()) {
            req["tools"] = chat.tools.clone();
        }
        let text = self.tpl.render(&req).map_err(|e| ApiError::bad(format!("chat template: {e}")))?;
        Ok(self.tok.encode(&text, true))
    }

    /// Queue `chat` on the model thread; the receiver yields Start, Text..., End.
    fn run(self: &Arc<Self>, chat: Chat) -> Result<tmpsc::UnboundedReceiver<Out>, ApiError> {
        self.generate(self.prompt(&chat)?, chat)
    }

    /// Generate from `prompt` with `chat`'s tools, limits and sampling (its messages unused).
    fn generate(self: &Arc<Self>, prompt: Vec<u32>, chat: Chat) -> Result<tmpsc::UnboundedReceiver<Out>, ApiError> {
        if prompt.len() >= self.n_ctx {
            let mut e = ApiError::new(
                StatusCode::BAD_REQUEST,
                "exceed_context_size_error",
                format!("request ({} tokens) exceeds the available context size ({} tokens), try increasing it", prompt.len(), self.n_ctx),
            );
            e.1["n_prompt_tokens"] = prompt.len().into();
            e.1["n_ctx"] = self.n_ctx.into();
            return Err(e);
        }
        let (gtx, mut grx) = tmpsc::unbounded_channel();
        let (otx, orx) = tmpsc::unbounded_channel();
        let n_prompt = prompt.len();
        self.jobs.send(request(prompt, chat.max_tokens, chat.sampling, gtx)).map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "model thread stopped"))?;
        let app = self.clone();
        let has_tools = chat.tools.as_array().is_some_and(|t| !t.is_empty());
        tokio::spawn(async move {
            let (mut utf8, mut split, mut n_out) = (Utf8Stream::default(), Splitter::new(has_tools), 0);
            let mut stops = StopMatcher::new(chat.stop);
            let mut finish = None;
            let (mut t0, mut t1) = (std::time::Instant::now(), None);
            while let Some(g) = grx.recv().await {
                match g {
                    Event::Started { n_cached } => {
                        t0 = std::time::Instant::now();
                        otx.send(Out::Start { n_prompt, n_cached }).ok();
                    }
                    Event::Token(t) => {
                        t1.get_or_insert_with(std::time::Instant::now);
                        if otx.is_closed() {
                            return; // client gone: dropping grx stops the model
                        }
                        n_out += 1;
                        let (text, hit) = stops.push(&utf8.push(&app.tok.decode_bytes(&[t])));
                        let text = split.push(&text);
                        if !text.is_empty() {
                            otx.send(Out::Text(text)).ok();
                        }
                        if hit {
                            finish = Some(Finish::Stop);
                            break;
                        }
                    }
                    Event::Done(f) => finish = Some(f),
                }
            }
            let Some(finish) = finish else {
                let e = ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "generation failed (see server log)");
                otx.send(Out::Error(e.1)).ok();
                return;
            };
            let (rest, calls) = split.finish(&chat.tools);
            let tail = stops.flush();
            let text = rest + &if calls.is_empty() { tail } else { String::new() };
            if !text.is_empty() {
                otx.send(Out::Text(text)).ok();
            }
            let t1 = t1.unwrap_or_else(std::time::Instant::now);
            let timings = Timings { prompt_ms: (t1 - t0).as_secs_f64() * 1e3, predicted_ms: t1.elapsed().as_secs_f64() * 1e3 };
            otx.send(Out::End { calls, finish, n_out, stop_word: stops.hit, timings }).ok();
        });
        Ok(orx)
    }
}

/// Cuts generated text at the first stop string, holding back partial matches.
struct StopMatcher {
    stops: Vec<String>,
    held: String,
    hit: Option<String>,
}

impl StopMatcher {
    fn new(stops: Vec<String>) -> Self {
        Self { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), held: String::new(), hit: None }
    }

    /// (text safe to emit, whether a stop string was found).
    fn push(&mut self, delta: &str) -> (String, bool) {
        self.held.push_str(delta);
        if let Some((p, s)) = self.stops.iter().filter_map(|s| self.held.find(s.as_str()).map(|p| (p, s))).min_by_key(|x| x.0) {
            self.hit = Some(s.clone());
            let out = self.held[..p].to_owned();
            self.held.clear();
            return (out, true);
        }
        let held = &self.held;
        let keep = self.stops.iter().flat_map(|s| (1..s.len()).filter(move |&n| s.is_char_boundary(n) && held.ends_with(&s[..n]))).max().unwrap_or(0);
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_owned();
        self.held.drain(..cut);
        (out, false)
    }

    fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

fn sampling(body: &Value) -> Sampling {
    let d = Sampling::default();
    let f = |k: &str, v: f32| body[k].as_f64().map_or(v, |x| x as f32);
    Sampling {
        temperature: f("temperature", d.temperature).max(0.0),
        top_p: f("top_p", d.top_p).clamp(0.0, 1.0),
        min_p: f("min_p", d.min_p).clamp(0.0, 1.0),
        top_k: body["top_k"].as_u64().map_or(d.top_k, |k| k as usize),
        seed: body["seed"].as_u64().unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos() as u64 | 1),
        schema: schema(body),
    }
}

/// The JSON schema a request constrains its output to: OpenAI `response_format` (json_object,
/// json_schema), Responses `text.format`, or llama.cpp's native `json_schema`. `{}`: any JSON.
fn schema(body: &Value) -> Option<Value> {
    let rf = if body["response_format"].is_object() { &body["response_format"] } else { &body["text"]["format"] };
    let s = match rf["type"].as_str() {
        Some("json_object") => rf.get("schema").cloned().unwrap_or(json!({})),
        Some("json_schema") => rf["json_schema"].get("schema").or(rf.get("schema")).cloned().unwrap_or(json!({})),
        _ => body.get("json_schema").filter(|s| s.is_object()).cloned()?,
    };
    Some(s)
}

fn text_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter(|p| p["type"] == "text").filter_map(|p| p["text"].as_str()).collect(),
        _ => String::new(),
    }
}

/// Anthropic Messages -> chat (llama.cpp server-chat.cpp convert_anthropic_to_oai).
fn anthropic_to_chat(b: &Value) -> Result<Chat, ApiError> {
    let mut msgs = vec![];
    if !b["system"].is_null() {
        msgs.push(json!({"role": "system", "content": text_of(&b["system"])}));
    }
    for m in b["messages"].as_array().ok_or_else(|| ApiError::bad("'messages' is required"))? {
        let role = m["role"].as_str().unwrap_or("user");
        let Some(blocks) = m["content"].as_array() else {
            if !m["content"].is_null() || role != "assistant" {
                msgs.push(m.clone());
            }
            continue;
        };
        let (mut parts, mut calls, mut tools) = (vec![], vec![], vec![]);
        for bl in blocks {
            match bl["type"].as_str() {
                Some("text") => parts.push(json!({"type": "text", "text": bl["text"]})),
                Some("tool_use") => calls.push(json!({"id": bl["id"], "type": "function", "function": {"name": bl["name"], "arguments": bl["input"].to_string()}})),
                Some("tool_result") => tools.push(json!({"role": "tool", "tool_call_id": bl["tool_use_id"], "content": text_of(&bl["content"])})),
                _ => {}
            }
        }
        if !parts.is_empty() || !calls.is_empty() {
            let mut msg = json!({"role": role, "content": if parts.is_empty() { json!("") } else { json!(parts) }});
            if !calls.is_empty() {
                msg["tool_calls"] = calls.into();
            }
            msgs.push(msg);
        }
        msgs.extend(tools);
    }
    let tools: Vec<Value> = b["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| t["name"].is_string())
        .map(|t| json!({"type": "function", "function": {"name": t["name"], "description": t["description"].as_str().unwrap_or(""), "parameters": if t["input_schema"].is_null() { json!({}) } else { t["input_schema"].clone() }}}))
        .collect();
    Ok(Chat {
        messages: msgs.into(),
        tools: tools.into(),
        max_tokens: b["max_tokens"].as_u64().unwrap_or(4096) as usize,
        sampling: sampling(b),
        stop: b["stop_sequences"].as_array().into_iter().flatten().filter_map(|s| s.as_str().map(str::to_owned)).collect(),
    })
}

/// OpenAI Responses -> chat (llama.cpp server-chat.cpp convert_responses_to_chatcmpl).
fn responses_to_chat(b: &Value) -> Result<Chat, ApiError> {
    if b["input"].is_null() {
        return Err(ApiError::bad("'input' is required"));
    }
    if b["previous_response_id"].as_str().is_some_and(|s| !s.is_empty()) {
        return Err(ApiError::bad("eomoe does not support 'previous_response_id'."));
    }
    let mut msgs: Vec<Value> = vec![];
    if let Some(i) = b["instructions"].as_str() {
        msgs.push(json!({"role": "system", "content": i}));
    }
    match &b["input"] {
        Value::String(s) => msgs.push(json!({"role": "user", "content": s})),
        Value::Array(items) => {
            for it in items {
                let merge = msgs.last().is_some_and(|m| m["role"] == "assistant");
                let content = match &it["content"] {
                    Value::String(s) => json!([{"type": "input_text", "text": s}]),
                    c => c.clone(),
                };
                let parts = |c: &Value| -> Vec<Value> {
                    c.as_array().into_iter().flatten().filter_map(|p| p["text"].as_str()).map(|t| json!({"type": "text", "text": t})).collect()
                };
                let role = it["role"].as_str().unwrap_or("");
                let ty = it["type"].as_str().unwrap_or("");
                if matches!(role, "user" | "system" | "developer") && content.is_array() {
                    msgs.push(json!({"role": role, "content": parts(&content)}));
                } else if role == "assistant" && ty == "message" {
                    if merge {
                        let last = msgs.last_mut().unwrap();
                        let mut c = last["content"].as_array().cloned().unwrap_or_default();
                        c.extend(parts(&content));
                        last["content"] = c.into();
                    } else {
                        msgs.push(json!({"role": "assistant", "content": parts(&content)}));
                    }
                } else if ty == "function_call" {
                    let tc = json!({"id": it["call_id"], "type": "function", "function": {"name": it["name"], "arguments": it["arguments"]}});
                    if merge {
                        let last = msgs.last_mut().unwrap();
                        let mut c = last["tool_calls"].as_array().cloned().unwrap_or_default();
                        c.push(tc);
                        last["tool_calls"] = c.into();
                    } else {
                        msgs.push(json!({"role": "assistant", "tool_calls": [tc]}));
                    }
                } else if ty == "function_call_output" {
                    msgs.push(json!({"role": "tool", "tool_call_id": it["call_id"], "content": text_of_output(&it["output"])}));
                } else if ty == "reasoning" {
                    // No reasoning in Qwen3-Coder's template.
                } else {
                    return Err(ApiError::bad("Cannot determine type of 'item'"));
                }
            }
        }
        _ => return Err(ApiError::bad("'input' must be a string or an array")),
    }
    let tools: Vec<Value> = b["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| t["type"] == "function")
        .map(|t| json!({"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"], "strict": t.get("strict").cloned().unwrap_or(json!(true))}}))
        .collect();
    Ok(Chat {
        messages: msgs.into(),
        tools: tools.into(),
        // No limit by default, as llama-server: generation stops at EOG or a full context.
        max_tokens: b["max_output_tokens"].as_u64().map_or(usize::MAX, |n| n as usize),
        sampling: sampling(b),
        stop: vec![],
    })
}

/// OpenAI chat completions -> chat: the internal form already; content parts are flattened at
/// render time.
fn chatcmpl_to_chat(b: &Value) -> Result<Chat, ApiError> {
    let messages = b["messages"].as_array().filter(|m| !m.is_empty()).ok_or_else(|| ApiError::bad("'messages' is required"))?;
    if b["n"].as_u64().is_some_and(|n| n != 1) {
        return Err(ApiError::bad("Only one completion choice is allowed"));
    }
    let tools: Vec<Value> = b["tools"].as_array().into_iter().flatten().filter(|t| t["type"] == "function").cloned().collect();
    Ok(Chat { messages: messages.clone().into(), tools: tools.into(), ..raw_chat(b) })
}

/// Limits, sampling and stop strings of any OpenAI or llama.cpp request; no messages or tools.
fn raw_chat(b: &Value) -> Chat {
    let stop = match &b["stop"] {
        Value::String(s) => vec![s.clone()],
        v => v.as_array().into_iter().flatten().filter_map(|s| s.as_str().map(str::to_owned)).collect(),
    };
    let max = ["max_completion_tokens", "max_tokens", "n_predict"].iter().find_map(|k| b[*k].as_i64());
    Chat {
        messages: Value::Null,
        tools: json!([]),
        // No limit by default (or -1), as llama-server: generation stops at EOG or a full context.
        max_tokens: max.filter(|&n| n >= 0).map_or(usize::MAX, |n| n as usize),
        sampling: sampling(b),
        stop,
    }
}

fn text_of_output(v: &Value) -> Value {
    match v {
        Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<String>().into(),
        v => v.clone(),
    }
}

/// One SSE event: `event: <type>` + data, as llama-server writes them.
fn sse(data: Value) -> String {
    format!("event: {}\ndata: {data}\n\n", data["type"].as_str().unwrap_or(""))
}

/// Formats a request's `Out` stream for one API, both streamed and as the final body.
trait Format: Send + 'static {
    fn on(&mut self, out: &Out) -> Vec<String>;
    fn body(&self) -> Value;
}

#[derive(Default)]
struct Acc {
    text: String,
    calls: Vec<(String, ToolCall)>,
    n_prompt: usize,
    n_cached: usize,
    n_out: usize,
    finish: Option<Finish>,
    stop_word: Option<String>,
    timings: Timings,
}

impl Acc {
    fn on(&mut self, out: &Out) {
        match out {
            Out::Start { n_prompt, n_cached } => (self.n_prompt, self.n_cached) = (*n_prompt, *n_cached),
            Out::Text(t) => self.text += t,
            Out::End { calls, finish, n_out, stop_word, timings } => {
                self.calls = calls.iter().map(|c| (rand_id(), c.clone())).collect();
                (self.finish, self.n_out, self.stop_word, self.timings) = (Some(*finish), *n_out, stop_word.clone(), *timings);
            }
            Out::Error(_) => {}
        }
    }

    // OpenAI finish_reason, usage and llama-server timings (chat and text completions).
    fn oai_finish(&self) -> &'static str {
        match self.finish {
            Some(Finish::Length) => "length",
            _ if !self.calls.is_empty() => "tool_calls",
            _ => "stop",
        }
    }
    fn oai_usage(&self) -> Value {
        json!({"completion_tokens": self.n_out, "prompt_tokens": self.n_prompt, "total_tokens": self.n_prompt + self.n_out, "prompt_tokens_details": {"cached_tokens": self.n_cached}})
    }
    fn oai_timings(&self) -> Value {
        let (t, pn, dn) = (&self.timings, (self.n_prompt - self.n_cached).max(1), self.n_out.max(1));
        json!({"cache_n": self.n_cached, "prompt_n": self.n_prompt - self.n_cached, "prompt_ms": t.prompt_ms, "prompt_per_token_ms": t.prompt_ms / pn as f64,
               "prompt_per_second": pn as f64 * 1e3 / t.prompt_ms.max(1e-3), "predicted_n": self.n_out, "predicted_ms": t.predicted_ms,
               "predicted_per_token_ms": t.predicted_ms / dn as f64, "predicted_per_second": dn as f64 * 1e3 / t.predicted_ms.max(1e-3)})
    }
}

struct Anthropic {
    id: String,
    model: String,
    a: Acc,
    text_open: bool,
}

impl Anthropic {
    fn stop_reason(&self) -> &'static str {
        match self.a.finish {
            Some(Finish::Length) => "max_tokens",
            _ if !self.a.calls.is_empty() => "tool_use",
            _ => "end_turn",
        }
    }
    fn usage(&self) -> Value {
        json!({"cache_read_input_tokens": self.a.n_cached, "input_tokens": self.a.n_prompt - self.a.n_cached, "output_tokens": self.a.n_out})
    }
}

impl Format for Anthropic {
    fn on(&mut self, out: &Out) -> Vec<String> {
        self.a.on(out);
        match out {
            Out::Start { .. } => {
                let mut u = self.usage();
                u["output_tokens"] = 0.into();
                vec![sse(json!({"type": "message_start", "message": {"id": self.id, "type": "message", "role": "assistant", "content": [], "model": self.model, "stop_reason": null, "stop_sequence": null, "usage": u}}))]
            }
            Out::Text(t) => {
                let mut v = vec![];
                if !self.text_open {
                    self.text_open = true;
                    v.push(sse(json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})));
                }
                v.push(sse(json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}})));
                v
            }
            Out::End { .. } => {
                let base = self.text_open as usize;
                let mut v = vec![];
                for (k, (id, c)) in self.a.calls.iter().enumerate() {
                    v.push(sse(json!({"type": "content_block_start", "index": base + k, "content_block": {"type": "tool_use", "id": id, "name": c.name}})));
                    v.push(sse(json!({"type": "content_block_delta", "index": base + k, "delta": {"type": "input_json_delta", "partial_json": c.arguments}})));
                }
                for i in 0..base + self.a.calls.len() {
                    v.push(sse(json!({"type": "content_block_stop", "index": i})));
                }
                v.push(sse(json!({"type": "message_delta", "delta": {"stop_reason": self.stop_reason(), "stop_sequence": self.a.stop_word}, "usage": {"output_tokens": self.a.n_out}})));
                v.push(sse(json!({"type": "message_stop"})));
                v
            }
            Out::Error(e) => vec![format!("event: error\ndata: {e}\n\n")],
        }
    }

    fn body(&self) -> Value {
        let mut content = vec![];
        if !self.a.text.is_empty() {
            content.push(json!({"type": "text", "text": self.a.text}));
        }
        for (id, c) in &self.a.calls {
            let input: Value = serde_json::from_str(&c.arguments).unwrap_or(json!({}));
            content.push(json!({"type": "tool_use", "id": id, "name": c.name, "input": input}));
        }
        json!({"id": self.id, "type": "message", "role": "assistant", "content": content, "model": self.model, "stop_reason": self.stop_reason(), "stop_sequence": self.a.stop_word, "usage": self.usage()})
    }
}

struct Responses {
    id: String,
    msg_id: String,
    model: String,
    created: u64,
    a: Acc,
    text_open: bool,
}

impl Responses {
    fn msg_item(&self) -> Value {
        json!({"type": "message", "status": "completed", "id": self.msg_id, "content": [self.part()], "role": "assistant"})
    }
    fn part(&self) -> Value {
        json!({"type": "output_text", "annotations": [], "logprobs": [], "text": self.a.text})
    }
    fn fc_item(id: &str, c: &ToolCall) -> Value {
        json!({"id": format!("fc_{id}"), "type": "function_call", "status": "completed", "arguments": c.arguments, "call_id": format!("call_{id}"), "name": c.name})
    }
    fn output(&self) -> Vec<Value> {
        let mut o = vec![];
        if self.text_open {
            o.push(self.msg_item());
        }
        o.extend(self.a.calls.iter().map(|(id, c)| Self::fc_item(id, c)));
        o
    }
    fn usage(&self) -> Value {
        json!({"input_tokens": self.a.n_prompt, "output_tokens": self.a.n_out, "total_tokens": self.a.n_prompt + self.a.n_out, "input_tokens_details": {"cached_tokens": self.a.n_cached}})
    }
}

impl Format for Responses {
    fn on(&mut self, out: &Out) -> Vec<String> {
        self.a.on(out);
        let ix = |s: &Self| json!({"item_id": s.msg_id, "output_index": 0, "content_index": 0});
        let with = |mut base: Value, ty: &str, extra: Value| {
            base["type"] = ty.into();
            for (k, v) in extra.as_object().unwrap() {
                base[k] = v.clone();
            }
            sse(base)
        };
        match out {
            Out::Start { .. } => {
                let r = json!({"id": self.id, "object": "response", "status": "in_progress"});
                vec![sse(json!({"type": "response.created", "response": r})), sse(json!({"type": "response.in_progress", "response": r}))]
            }
            Out::Text(t) => {
                let mut v = vec![];
                if !self.text_open {
                    self.text_open = true;
                    v.push(sse(json!({"type": "response.output_item.added", "output_index": 0, "item": {"content": [], "id": self.msg_id, "role": "assistant", "status": "in_progress", "type": "message"}})));
                    v.push(with(ix(self), "response.content_part.added", json!({"part": {"type": "output_text", "text": ""}})));
                }
                v.push(with(ix(self), "response.output_text.delta", json!({"delta": t})));
                v
            }
            Out::End { .. } => {
                let base = self.text_open as usize;
                let mut v = vec![];
                for (k, (id, c)) in self.a.calls.iter().enumerate() {
                    let fc = format!("fc_{id}");
                    v.push(sse(json!({"type": "response.output_item.added", "output_index": base + k, "item": {"id": fc, "arguments": "", "call_id": format!("call_{id}"), "name": c.name, "type": "function_call", "status": "in_progress"}})));
                    v.push(sse(json!({"type": "response.function_call_arguments.delta", "delta": c.arguments, "item_id": fc, "output_index": base + k})));
                }
                if self.text_open {
                    v.push(with(ix(self), "response.output_text.done", json!({"text": self.a.text})));
                    v.push(with(ix(self), "response.content_part.done", json!({"part": self.part()})));
                    v.push(sse(json!({"type": "response.output_item.done", "output_index": 0, "item": self.msg_item()})));
                }
                for (k, (id, c)) in self.a.calls.iter().enumerate() {
                    v.push(sse(json!({"type": "response.output_item.done", "output_index": base + k, "item": Self::fc_item(id, c)})));
                }
                v.push(sse(json!({"type": "response.completed", "response": {"id": self.id, "object": "response", "created_at": self.created, "status": "completed", "model": self.model, "output": self.output(), "usage": self.usage()}})));
                v
            }
            Out::Error(e) => vec![format!("data: {}\n\n", json!({ "error": e }))],
        }
    }

    fn body(&self) -> Value {
        json!({"completed_at": now(), "created_at": self.created, "id": self.id, "model": self.model, "object": "response", "output": self.output(), "status": "completed", "usage": self.usage()})
    }
}

/// OpenAI chat completions (llama-server to_json_oaicompat_chat[_stream]).
struct ChatCmpl {
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
    a: Acc,
}

impl ChatCmpl {
    fn chunk(&self, choices: Value) -> Value {
        json!({"choices": choices, "created": self.created, "id": self.id, "model": self.model, "system_fingerprint": FINGERPRINT, "object": "chat.completion.chunk"})
    }
    fn delta(&self, delta: Value) -> String {
        format!("data: {}\n\n", self.chunk(json!([{"finish_reason": null, "index": 0, "delta": delta}])))
    }
}

/// llama-server's build tag in `system_fingerprint` (the oracle is b1-e6443bd).
const FINGERPRINT: &str = concat!("eomoe-", env!("CARGO_PKG_VERSION"));

impl Format for ChatCmpl {
    fn on(&mut self, out: &Out) -> Vec<String> {
        self.a.on(out);
        match out {
            Out::Start { .. } => vec![self.delta(json!({"role": "assistant", "content": null}))],
            Out::Text(t) => vec![self.delta(json!({"content": t}))],
            Out::End { .. } => {
                let mut v: Vec<String> = self.a.calls.iter().enumerate().map(|(k, (id, c))| {
                    self.delta(json!({"tool_calls": [{"index": k, "id": id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}]}))
                }).collect();
                let mut tail = vec![self.chunk(json!([{"finish_reason": self.a.oai_finish(), "index": 0, "delta": {}}]))];
                if self.include_usage {
                    let mut u = self.chunk(json!([]));
                    u["usage"] = self.a.oai_usage();
                    tail.push(u);
                }
                // llama-server puts timings on the last chunk: the usage chunk when there is one.
                tail.last_mut().unwrap()["timings"] = self.a.oai_timings();
                v.extend(tail.iter().map(|c| format!("data: {c}\n\n")));
                v.push("data: [DONE]\n\n".into());
                v
            }
            Out::Error(e) => vec![format!("data: {}\n\n", json!({ "error": e }))],
        }
    }

    fn body(&self) -> Value {
        let mut msg = json!({"role": "assistant", "content": self.a.text});
        if !self.a.calls.is_empty() {
            msg["tool_calls"] = self.a.calls.iter().map(|(id, c)| json!({"type": "function", "function": {"name": c.name, "arguments": c.arguments}, "id": id})).collect();
        }
        json!({"choices": [{"finish_reason": self.a.oai_finish(), "index": 0, "message": msg}], "created": self.created, "model": self.model,
               "system_fingerprint": FINGERPRINT, "object": "chat.completion", "usage": self.a.oai_usage(), "id": self.id, "timings": self.a.oai_timings()})
    }
}

impl App {
    /// A raw prompt: a string (special tokens parsed) or token ids, as llama-server takes them.
    fn raw_prompt(&self, p: &Value) -> Result<Vec<u32>, ApiError> {
        match p {
            Value::String(s) => Ok(self.tok.encode(s, true)),
            Value::Array(a) if !a.is_empty() && a.iter().all(Value::is_u64) => Ok(a.iter().map(|t| t.as_u64().unwrap() as u32).collect()),
            Value::Array(a) if a.len() == 1 => self.raw_prompt(&a[0]),
            Value::Null => Err(ApiError::bad("\"prompt\" is required")),
            _ => Err(ApiError::bad("\"prompt\" must be a string or an array of token ids (one prompt per request)")),
        }
    }

    /// The repo-level FIM prompt of an /infill request (llama-server format_prompt_infill, with
    /// its defaults: n_batch 2048, n_predict -1, no SPM order, no BOS for this vocab).
    fn infill_prompt(&self, b: &Value) -> Result<Vec<u32>, ApiError> {
        if !b["prompt"].is_null() && !b["prompt"].is_string() {
            return Err(ApiError::bad("\"prompt\" must be a string"));
        }
        let (Some(pre), Some(suf)) = (b["input_prefix"].as_str(), b["input_suffix"].as_str()) else {
            return Err(ApiError::bad("\"input_prefix\" and \"input_suffix\" are required strings"));
        };
        let extra = match &b["input_extra"] {
            Value::Null => &vec![],
            Value::Array(a) => a,
            _ => return Err(ApiError::bad("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}")),
        };
        let special = |t: &str| self.tok.token_id(t).ok_or_else(|| ApiError::new(StatusCode::NOT_IMPLEMENTED, "not_supported_error", format!("Infill is not supported by this model: {t} is missing")));
        let (rep, sep, fpre, fsuf, fmid) = (special("<|repo_name|>")?, special("<|file_sep|>")?, special("<|fim_prefix|>")?, special("<|fim_suffix|>")?, special("<|fim_middle|>")?);
        let enc = |s: &str| self.tok.encode(s, false);
        let mut ext = vec![rep];
        ext.extend(enc("myproject\n"));
        for c in extra {
            let Some(text) = c["text"].as_str() else {
                return Err(ApiError::bad("extra_context chunk must contain a \"text\" field with a string value"));
            };
            if !c["filename"].is_null() && !c["filename"].is_string() {
                return Err(ApiError::bad("extra_context chunk's \"filename\" field must be a string"));
            }
            ext.push(sep);
            ext.extend(enc(&format!("{}\n", c["filename"].as_str().unwrap_or("tmp"))));
            ext.extend(enc(text));
        }
        ext.push(sep);
        ext.extend(enc("filename\n"));
        const N_BATCH: usize = 2048;
        let prompt = self.tok.encode(b["prompt"].as_str().unwrap_or(""), true);
        let (pre, suf) = (enc(pre), enc(suf));
        let n_pre = pre.len().min(3 * (N_BATCH / 4));
        let n_suf = suf.len().min((N_BATCH / 4).saturating_sub(2 + prompt.len()));
        let n_ext = ext.len().min((self.n_ctx + 2).saturating_sub(N_BATCH));
        let mut out = ext[ext.len() - n_ext..].to_vec();
        out.push(fpre);
        out.extend(&pre[pre.len() - n_pre..]);
        out.extend(prompt);
        out.push(fsuf);
        out.extend(&suf[..n_suf]);
        out.push(fmid);
        Ok(out)
    }
}

/// OpenAI text completions (llama-server to_json_oaicompat[_stream]).
struct TextCmpl {
    id: String,
    model: String,
    created: u64,
    a: Acc,
}

impl TextCmpl {
    fn chunk(&self, text: &str, finish: Value) -> Value {
        json!({"choices": [{"text": text, "index": 0, "logprobs": null, "finish_reason": finish}], "created": self.created, "model": self.model,
               "system_fingerprint": FINGERPRINT, "object": "text_completion", "id": self.id})
    }
}

impl Format for TextCmpl {
    fn on(&mut self, out: &Out) -> Vec<String> {
        self.a.on(out);
        match out {
            Out::Start { .. } => vec![],
            Out::Text(t) => vec![format!("data: {}\n\n", self.chunk(t, Value::Null))],
            Out::End { .. } => {
                let mut v = vec![];
                // llama-server: the partial result that hit a stop word carries the timings.
                if self.a.stop_word.is_some() {
                    let mut p = self.chunk("", Value::Null);
                    p["timings"] = self.a.oai_timings();
                    v.push(format!("data: {p}\n\n"));
                }
                let mut c = self.chunk("", self.a.oai_finish().into());
                (c["usage"], c["timings"]) = (self.a.oai_usage(), self.a.oai_timings());
                v.extend([format!("data: {c}\n\n"), "data: [DONE]\n\n".into()]);
                v
            }
            Out::Error(e) => vec![format!("data: {}\n\n", json!({ "error": e }))],
        }
    }

    fn body(&self) -> Value {
        let mut b = self.chunk(&self.a.text, self.a.oai_finish().into());
        (b["usage"], b["timings"]) = (self.a.oai_usage(), self.a.oai_timings());
        b
    }
}

/// llama.cpp native /completion and /infill (llama-server server_task_result_cmpl_*::to_json_non_oaicompat).
struct Native {
    model: String,
    prompt: String,
    n_text: usize,
    a: Acc,
}

impl Format for Native {
    fn on(&mut self, out: &Out) -> Vec<String> {
        self.a.on(out);
        match out {
            Out::Start { .. } => vec![],
            Out::Text(t) => {
                self.n_text += 1;
                let c = json!({"index": 0, "content": t, "tokens": [], "stop": false, "id_slot": -1, "tokens_predicted": self.n_text, "tokens_evaluated": self.a.n_prompt});
                vec![format!("data: {c}\n\n")]
            }
            Out::End { .. } => {
                let mut b = self.body();
                b["content"] = "".into();
                vec![format!("data: {b}\n\n")]
            }
            Out::Error(e) => vec![format!("data: {}\n\n", json!({ "error": e }))],
        }
    }

    fn body(&self) -> Value {
        let a = &self.a;
        let stop_type = match (a.finish, &a.stop_word) {
            (_, Some(_)) => "word",
            (Some(Finish::Length), _) => "limit",
            _ => "eos",
        };
        json!({"index": 0, "content": a.text, "tokens": [], "id_slot": 0, "stop": true, "model": self.model, "tokens_predicted": a.n_out, "tokens_evaluated": a.n_prompt,
               "prompt": self.prompt, "has_new_line": a.text.contains('\n'), "truncated": false, "stop_type": stop_type, "stopping_word": a.stop_word.as_deref().unwrap_or(""),
               "tokens_cached": (a.n_prompt + a.n_out).saturating_sub(1), "timings": a.oai_timings()})
    }
}

/// Stream the formatted events, or collect them into the final JSON body.
async fn respond(rx: Result<tmpsc::UnboundedReceiver<Out>, ApiError>, mut f: impl Format, stream: bool) -> Response {
    let mut rx = match rx {
        Ok(rx) => rx,
        Err(e) => return e.into_response(),
    };
    if !stream {
        while let Some(o) = rx.recv().await {
            if let Out::Error(e) = o {
                return ApiError(StatusCode::INTERNAL_SERVER_ERROR, e).into_response();
            }
            f.on(&o);
        }
        return Json(f.body()).into_response();
    }
    let (tx, srx) = tmpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        while let Some(o) = rx.recv().await {
            for e in f.on(&o) {
                if tx.send(e).is_err() {
                    return;
                }
            }
        }
    });
    // llama-server sends an SSE comment after 30 s without data, so a long prefill does not trip
    // client idle timeouts.
    let body = futures_util::stream::unfold(srx, |mut r| async move {
        let s = match tokio::time::timeout(std::time::Duration::from_secs(30), r.recv()).await {
            Ok(s) => s?,
            Err(_) => ":\n\n".to_owned(),
        };
        Some((Ok::<_, std::convert::Infallible>(s), r))
    });
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body))
        .unwrap()
}

/// The request JSON; a malformed body gets llama-server's error shape, not axum's plain text.
fn parse(body: &[u8]) -> Result<Value, ApiError> {
    serde_json::from_slice(body).map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))
}

async fn messages(State(app): State<Arc<App>>, body: Bytes) -> Response {
    let b = match parse(&body) {
        Ok(b) => b,
        Err(e) => return e.into_response(),
    };
    let f = Anthropic { id: format!("chatcmpl-{}", rand_id()), model: app.alias.clone(), a: Acc::default(), text_open: false };
    let rx = anthropic_to_chat(&b).and_then(|c| app.run(c));
    respond(rx, f, b["stream"].as_bool().unwrap_or(false)).await
}

async fn responses(State(app): State<Arc<App>>, body: Bytes) -> Response {
    let b = match parse(&body) {
        Ok(b) => b,
        Err(e) => return e.into_response(),
    };
    let f = Responses { id: format!("resp_{}", rand_id()), msg_id: format!("msg_{}", rand_id()), model: app.alias.clone(), created: now(), a: Acc::default(), text_open: false };
    let rx = responses_to_chat(&b).and_then(|c| app.run(c));
    respond(rx, f, b["stream"].as_bool().unwrap_or(false)).await
}

async fn chat_completions(State(app): State<Arc<App>>, body: Bytes) -> Response {
    let b = match parse(&body) {
        Ok(b) => b,
        Err(e) => return e.into_response(),
    };
    let include_usage = b["stream_options"]["include_usage"].as_bool().unwrap_or(false);
    let f = ChatCmpl { id: format!("chatcmpl-{}", rand_id()), model: app.alias.clone(), created: now(), include_usage, a: Acc::default() };
    let rx = chatcmpl_to_chat(&b).and_then(|c| app.run(c));
    respond(rx, f, b["stream"].as_bool().unwrap_or(false)).await
}

async fn completions_oai(State(app): State<Arc<App>>, body: Bytes) -> Response {
    let b = match parse(&body) {
        Ok(b) => b,
        Err(e) => return e.into_response(),
    };
    let f = TextCmpl { id: format!("chatcmpl-{}", rand_id()), model: app.alias.clone(), created: now(), a: Acc::default() };
    let rx = app.raw_prompt(&b["prompt"]).and_then(|p| app.generate(p, raw_chat(&b)));
    respond(rx, f, b["stream"].as_bool().unwrap_or(false)).await
}

/// /completion (raw prompt) and /infill (FIM prompt) in llama.cpp's native format.
async fn native(app: Arc<App>, body: Bytes, infill: bool) -> Response {
    let b = match parse(&body) {
        Ok(b) => b,
        Err(e) => return e.into_response(),
    };
    let p = match if infill { app.infill_prompt(&b) } else { app.raw_prompt(&b["prompt"]) } {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };
    let f = Native { model: app.alias.clone(), prompt: String::from_utf8_lossy(&app.tok.decode_bytes(&p)).into_owned(), n_text: 0, a: Acc::default() };
    respond(app.generate(p, raw_chat(&b)), f, b["stream"].as_bool().unwrap_or(false)).await
}

async fn count_tokens(State(app): State<Arc<App>>, body: Bytes) -> Response {
    match parse(&body).and_then(|b| anthropic_to_chat(&b)).and_then(|c| app.prompt(&c)) {
        Ok(p) => Json(json!({"input_tokens": p.len()})).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn input_tokens(State(app): State<Arc<App>>, body: Bytes) -> Response {
    match parse(&body).and_then(|b| responses_to_chat(&b)).and_then(|c| app.prompt(&c)) {
        Ok(p) => Json(json!({"input_tokens": p.len(), "object": "response.input_tokens"})).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    let n = &app.alias;
    Json(json!({
        "models": [{"name": n, "model": n, "modified_at": "", "size": "", "digest": "", "type": "model", "description": "", "tags": [""], "capabilities": ["completion"], "parameters": "",
                    "details": {"parent_model": "", "format": "gguf", "family": "", "families": [""], "parameter_size": "", "quantization_level": ""}}],
        "object": "list",
        "data": [{"id": n, "aliases": [n], "tags": [], "object": "model", "created": now(), "owned_by": "eomoe", "meta": {"n_vocab": app.tok.n_vocab(), "n_ctx": app.n_ctx}}],
    }))
}

/// llama-server's `middleware_validate_api_key`: with keys set, every path but the health checks
/// and the page at `/` needs one, as `Authorization: [Bearer ]KEY` or else `X-Api-Key: KEY`.
async fn check_key(State(keys): State<Arc<Vec<String>>>, req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    let ok = {
        let open = keys.is_empty() || matches!(req.uri().path(), "/health" | "/v1/health" | "/") || req.method() == axum::http::Method::OPTIONS;
        let hv = |h: &str| req.headers().get(h).and_then(|v| v.to_str().ok()).unwrap_or("");
        let key = Some(hv("authorization")).filter(|k| !k.is_empty()).unwrap_or_else(|| hv("x-api-key"));
        open || keys.iter().any(|k| k == key.strip_prefix("Bearer ").unwrap_or(key))
    };
    if ok {
        return next.run(req).await;
    }
    log::warn!("unauthorized: Invalid API Key");
    let body = r#"{"error":{"message":"Invalid API Key","type":"authentication_error","code":401}}"#;
    (StatusCode::UNAUTHORIZED, [(header::CONTENT_TYPE, "application/json; charset=utf-8")], body).into_response()
}

/// The HTTP API; `keys` non-empty turns on API-key checks (`--api-key`).
pub fn router(app: Arc<App>, keys: Vec<String>) -> Router {
    let keys = Arc::new(keys);
    Router::new()
        .route("/", get(|| async { axum::response::Html(include_str!("chat.html")) }))
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/responses", post(responses))
        .route("/responses", post(responses))
        .route("/v1/responses/input_tokens", post(input_tokens))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions_oai))
        .route("/completion", post(|State(a): State<Arc<App>>, b: Bytes| native(a, b, false)))
        .route("/completions", post(|State(a): State<Arc<App>>, b: Bytes| native(a, b, false)))
        .route("/infill", post(|State(a): State<Arc<App>>, b: Bytes| native(a, b, true)))
        .fallback(|| async { ApiError::new(StatusCode::NOT_FOUND, "not_found_error", "File Not Found") })
        .with_state(app)
        .layer(axum::middleware::from_fn_with_state(keys, check_key))
}

/// Serve the API on host:port, over HTTPS when `tls` = (PEM key, PEM cert chain) is set.
pub async fn serve(app: Arc<App>, host: &str, port: u16, keys: Vec<String>, tls: Option<(std::path::PathBuf, std::path::PathBuf)>) -> Result<(), String> {
    let listener = std::net::TcpListener::bind((host, port)).map_err(|e| format!("{host}:{port}: {e}"))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let r = router(app, keys);
    match tls {
        None => {
            log::info!("listening on http://{addr}");
            listener.set_nonblocking(true).map_err(|e| e.to_string())?;
            let l = tokio::net::TcpListener::from_std(listener).map_err(|e| e.to_string())?;
            axum::serve(l, r).await.map_err(|e| e.to_string())
        }
        Some((key, cert)) => {
            rustls::crypto::ring::default_provider().install_default().ok();
            let cfg = axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert, &key).await.map_err(|e| format!("TLS {} / {}: {e}", cert.display(), key.display()))?;
            log::info!("listening on https://{addr}");
            axum_server::from_tcp_rustls(listener, cfg).serve(r.into_make_service()).await.map_err(|e| e.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_matcher_cuts_at_stop_and_holds_partials() {
        let mut m = StopMatcher::new(vec!["END".into(), "".into()]);
        assert_eq!(m.push("abc E"), ("abc ".into(), false));
        assert_eq!(m.push("Nx"), ("ENx".into(), false)); // partial match released
        assert_eq!(m.push("yy EN"), ("yy ".into(), false));
        assert_eq!(m.push("D tail"), ("".into(), true));
        assert_eq!(m.hit.as_deref(), Some("END"));
        let mut m = StopMatcher::new(vec![]);
        assert_eq!(m.push("é"), ("é".into(), false));
    }

    #[test]
    fn anthropic_tool_results_follow_the_user_text() {
        let b = json!({"system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "x"}, {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"c": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "o1"}, {"type": "text", "text": "o2"}]}, {"type": "text", "text": "q"}]}]});
        let c = anthropic_to_chat(&b).unwrap();
        let m = c.messages.as_array().unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "ab"}));
        assert_eq!(m[1]["tool_calls"][0]["function"]["arguments"], r#"{"c":1}"#);
        assert_eq!((m[2]["role"].as_str(), m[3]["role"].as_str(), m[3]["content"].as_str()), (Some("user"), Some("tool"), Some("o1o2")));
        assert_eq!(c.max_tokens, 4096);
    }

    #[test]
    fn responses_function_calls_merge_into_the_assistant_turn() {
        let b = json!({"instructions": "sys", "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
            {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "done"}]});
        let m = responses_to_chat(&b).unwrap().messages;
        assert_eq!(m.as_array().unwrap().len(), 4);
        assert_eq!(m[2]["tool_calls"][0]["id"], "c1");
        assert_eq!(m[3], json!({"role": "tool", "tool_call_id": "c1", "content": "done"}));
        assert!(responses_to_chat(&json!({"input": [{"type": "bogus"}]})).is_err());
    }
}
