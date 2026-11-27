//! Chat-template rendering with minijinja, tuned to match llama.cpp's own jinja engine
//! (common/jinja): Python-style str()/repr, llama.cpp tojson spacing/escaping, and
//! trim_blocks + lstrip_blocks.

use crate::gguf::Gguf;
use minijinja::value::{Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind};

pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    pub fn from_gguf(g: &Gguf) -> Result<Self, String> {
        let src = g.str("tokenizer.chat_template").ok_or("model has no tokenizer.chat_template")?;
        Self::new(src.to_owned())
    }

    pub fn new(src: String) -> Result<Self, String> {
        let mut env = Environment::new();
        env.set_syntax(
            minijinja::syntax::SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|e| e.to_string())?,
        );
        env.set_formatter(|out, _state, v| {
            out.write_str(&py_str(v)).map_err(|_| Error::new(ErrorKind::WriteFailure, "write failed"))
        });
        env.add_filter("string", |v: Value| py_str(&v));
        // ponytail: ensure_ascii (GLM passes False, the output we already make) is read and ignored.
        env.add_filter("tojson", |v: Value, kw: minijinja::value::Kwargs| -> Result<Value, Error> {
            let _: Option<bool> = kw.get("ensure_ascii")?;
            kw.assert_all_used()?;
            Ok(Value::from_safe_string(to_json(&v)))
        });
        env.add_function("strftime_now", |f: String| strftime_now(&f));
        // Python's dict.items() (gpt-oss) and str strip/lstrip/rstrip/split (GLM); other unknown
        // methods stay errors.
        env.set_unknown_method_callback(|_state, v, method, args| match (method, v.kind(), args) {
            ("items", ValueKind::Map, []) => {
                let pairs: Result<Vec<Value>, Error> = v.try_iter()?.map(|k| Ok(Value::from(vec![k.clone(), v.get_item(&k)?]))).collect();
                Ok(Value::from(pairs?))
            }
            ("strip" | "lstrip" | "rstrip", ValueKind::String, [] | [_]) => {
                let (s, chars) = (v.as_str().unwrap_or(""), args.first().and_then(|c| c.as_str()));
                let p = |c: char| chars.map_or(c.is_whitespace(), |cs| cs.contains(c));
                Ok(Value::from(match method {
                    "strip" => s.trim_matches(p),
                    "lstrip" => s.trim_start_matches(p),
                    _ => s.trim_end_matches(p),
                }))
            }
            ("split", ValueKind::String, [sep]) if sep.as_str().is_some_and(|x| !x.is_empty()) => {
                Ok(Value::from(v.as_str().unwrap_or("").split(sep.as_str().unwrap()).map(Value::from).collect::<Vec<_>>()))
            }
            ("split", ValueKind::String, []) => Ok(Value::from(v.as_str().unwrap_or("").split_whitespace().map(Value::from).collect::<Vec<_>>())),
            _ => Err(Error::new(ErrorKind::UnknownMethod, format!("{} has no method named {method}", v.kind()))),
        });
        env.add_function("raise_exception", |m: String| -> Result<String, Error> { Err(Error::new(ErrorKind::InvalidOperation, m)) });
        env.add_template_owned("chat", src).map_err(|e| e.to_string())?;
        Ok(Self { env })
    }

    /// Render an OpenAI-style chat request body (`messages`, optional `tools`,
    /// `add_generation_prompt` default true), like llama-server's /apply-template.
    pub fn render(&self, req: &serde_json::Value) -> Result<String, String> {
        let mut messages = req.get("messages").cloned().ok_or("request has no messages")?;
        // llama.cpp hands templates parsed tool-call arguments, not the OpenAI JSON string.
        for m in messages.as_array_mut().into_iter().flatten() {
            for tc in m.get_mut("tool_calls").and_then(|t| t.as_array_mut()).into_iter().flatten() {
                if let Some(args) = tc.pointer_mut("/function/arguments") {
                    if let Some(s) = args.as_str() {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                            *args = parsed;
                        }
                    }
                }
            }
        }
        let mut ctx = serde_json::json!({
            "messages": messages,
            "add_generation_prompt": req.get("add_generation_prompt").and_then(|v| v.as_bool()).unwrap_or(true),
            "bos_token": "",
            "eos_token": "<|im_end|>",
        });
        if let Some(tools) = req.get("tools").filter(|t| !t.is_null()) {
            ctx["tools"] = tools.clone();
        }
        // llama-server's chat_template_kwargs (e.g. GLM's enable_thinking) become template variables.
        for (k, v) in req.get("chat_template_kwargs").and_then(|o| o.as_object()).into_iter().flatten() {
            ctx[k] = v.clone();
        }
        let tpl = self.env.get_template("chat").map_err(|e| e.to_string())?;
        tpl.render(Value::from(minijinja::value::Serde(&ctx))).map_err(|e| format!("{e:#}"))
    }
}

/// llama.cpp's `strftime_now`: the local time through C strftime (gpt-oss's "Current date").
pub fn strftime_now(fmt: &str) -> String {
    let Ok(f) = std::ffi::CString::new(fmt) else { return String::new() };
    let mut buf = [0u8; 256];
    // SAFETY: tm is filled by localtime_r before use; strftime writes at most buf.len() bytes.
    let n = unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        #[cfg(unix)]
        libc::localtime_r(&t, &mut tm);
        #[cfg(windows)]
        libc::localtime_s(&mut tm, &t);
        strftime(buf.as_mut_ptr().cast(), buf.len(), f.as_ptr(), &tm)
    };
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

#[cfg(unix)]
use libc::strftime;
#[cfg(windows)]
extern "C" {
    // The UCRT has it; the libc crate does not bind it on Windows.
    fn strftime(s: *mut libc::c_char, max: usize, format: *const libc::c_char, tm: *const libc::tm) -> usize;
}

/// llama.cpp `as_string()`: Python str() semantics.
fn py_str(v: &Value) -> String {
    match v.kind() {
        ValueKind::String => v.as_str().unwrap_or_default().to_owned(),
        _ => py_repr(v),
    }
}

/// llama.cpp `value_to_string_repr()`: Python repr() for containers' items.
fn py_repr(v: &Value) -> String {
    match v.kind() {
        ValueKind::Undefined => String::new(),
        ValueKind::None => "None".into(),
        ValueKind::Bool => if v.is_true() { "True" } else { "False" }.into(),
        ValueKind::Number => match i64::try_from(v.clone()) {
            Ok(i) if v.is_integer() => i.to_string(),
            _ => {
                // std::to_string(double) is "%f", then trailing zeros stripped, keeping one.
                let f = f64::try_from(v.clone()).unwrap_or(f64::NAN);
                let s = format!("{f:.6}");
                let s = s.trim_end_matches('0');
                if s.ends_with('.') { format!("{s}0") } else { s.to_owned() }
            }
        },
        ValueKind::String => {
            let s = v.as_str().unwrap_or_default();
            if s.contains('\'') { to_json(v) } else { format!("'{s}'") }
        }
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<String> = v.try_iter().into_iter().flatten().map(|x| py_repr(&x)).collect();
            format!("[{}]", items.join(", "))
        }
        ValueKind::Map => {
            let items: Vec<String> = v
                .try_iter()
                .into_iter()
                .flatten()
                .map(|k| {
                    let val = v.get_item(&k).unwrap_or_default();
                    format!("{}: {}", py_repr(&k), py_repr(&val))
                })
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        _ => v.to_string(),
    }
}

/// llama.cpp `tojson` with defaults: ", " / ": " separators, no HTML escaping, raw UTF-8.
fn to_json(v: &Value) -> String {
    let mut out = String::new();
    json_into(&mut out, v);
    out
}

fn json_into(out: &mut String, v: &Value) {
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
        ValueKind::Number => match i64::try_from(v.clone()) {
            Ok(i) if v.is_integer() => out.push_str(&i.to_string()),
            _ => {
                let f = f64::try_from(v.clone()).unwrap_or(f64::NAN);
                // ponytail: Python float repr; exponent form (1e-09 vs Rust 1e-9) not matched.
                let s = if f.is_nan() {
                    "NaN".into()
                } else if f.is_infinite() {
                    if f < 0.0 { "-Infinity".into() } else { "Infinity".into() }
                } else {
                    let s = f.to_string();
                    if s.contains(['.', 'e']) { s } else { format!("{s}.0") }
                };
                out.push_str(&s);
            }
        },
        ValueKind::String => {
            out.push('"');
            for c in v.as_str().unwrap_or_default().chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\u{8}' => out.push_str("\\b"),
                    '\u{c}' => out.push_str("\\f"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            for (i, x) in v.try_iter().into_iter().flatten().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                json_into(out, &x);
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            for (i, k) in v.try_iter().into_iter().flatten().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                json_into(out, &Value::from(k.to_string()));
                out.push_str(": ");
                json_into(out, &v.get_item(&k).unwrap_or_default());
            }
            out.push('}');
        }
        _ => json_into(out, &Value::from(v.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected strings are what Python str()/json.dumps produce (llama.cpp follows them).
    #[test]
    fn strftime_now_is_local_date() {
        let out = std::process::Command::new("date").arg("+%Y-%m-%d %a").output().unwrap();
        assert_eq!(strftime_now("%Y-%m-%d %a"), String::from_utf8(out.stdout).unwrap().trim());
    }

    #[test]
    fn python_style_conversions() {
        let v = Value::from(minijinja::value::Serde(serde_json::json!({"a": [1, true, null, "x", "it's", 2.5], "b": {"<": "é"}})));
        assert_eq!(py_str(&v), r#"{'a': [1, True, None, 'x', "it's", 2.5], 'b': {'<': 'é'}}"#);
        assert_eq!(to_json(&v), r#"{"a": [1, true, null, "x", "it's", 2.5], "b": {"<": "é"}}"#);
        assert_eq!(py_str(&Value::from(3.0)), "3.0");
    }
}
