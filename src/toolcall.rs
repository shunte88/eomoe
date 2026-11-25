//! Qwen3-Coder tool calls in model output:
//! `<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>\n</tool_call>`.
//!
//! `Splitter` streams the text before the first `<tool_call>` and keeps the rest, which `parse`
//! turns into calls with JSON arguments typed by the tool's schema.

/// A call starts at `<tool_call>`, or at a bare `<function=` (Qwen3-Coder sometimes omits the
/// wrapper); up to two newlines before it are not content (llama.cpp qwen3-coder parser).
const STARTS: [&str; 2] = ["<tool_call>", "<function="];

/// A parsed call: arguments is a JSON object as text.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
}

/// Separates streamed content from the tool-call block at its end. Disabled (all content)
/// when the request has no tools.
pub struct Splitter {
    held: String,
    in_calls: bool,
    enabled: bool,
}

impl Splitter {
    pub fn new(enabled: bool) -> Self {
        Self { held: String::new(), in_calls: false, enabled }
    }

    /// Feed generated text; returns content that is safe to stream now.
    pub fn push(&mut self, delta: &str) -> String {
        self.held.push_str(delta);
        if !self.enabled {
            return std::mem::take(&mut self.held);
        }
        if self.in_calls {
            return String::new();
        }
        if let Some(p) = STARTS.iter().filter_map(|m| self.held.find(m)).min() {
            self.in_calls = true;
            let text = strip_nl(&self.held[..p]).to_owned();
            self.held.drain(..p);
            return text;
        }
        // Hold back a suffix that could still grow into a call start, and the newlines before it.
        let held = &self.held;
        let keep = STARTS
            .iter()
            .flat_map(|m| (1..m.len()).filter(move |&n| held.ends_with(&m[..n])))
            .max()
            .unwrap_or(0);
        let cut = self.held.len() - keep;
        let cut = cut - (self.held[..cut].len() - strip_nl(&self.held[..cut]).len());
        let text = self.held[..cut].to_owned();
        self.held.drain(..cut);
        text
    }

    /// End of generation: (remaining content, tool calls). Text after the calls is dropped.
    pub fn finish(self, tools: &serde_json::Value) -> (String, Vec<ToolCall>) {
        if self.in_calls {
            let calls = parse(&self.held, tools);
            if !calls.is_empty() {
                return (String::new(), calls);
            }
        }
        (self.held, vec![])
    }
}

/// Without up to two trailing newlines.
fn strip_nl(s: &str) -> &str {
    let s = s.strip_suffix('\n').unwrap_or(s);
    s.strip_suffix('\n').unwrap_or(s)
}

/// Every complete `<function=…>` block, in order.
pub fn parse(s: &str, tools: &serde_json::Value) -> Vec<ToolCall> {
    let mut calls = vec![];
    let mut rest = s;
    while let Some(p) = rest.find("<function=") {
        rest = &rest[p + "<function=".len()..];
        let Some(gt) = rest.find('>') else { break };
        let name = rest[..gt].trim().to_owned();
        rest = &rest[gt + 1..];
        let end = rest.find("</function>").unwrap_or(rest.len());
        let body = &rest[..end];
        rest = &rest[end..];
        let schema = tools
            .as_array()
            .into_iter()
            .flatten()
            .map(|t| t.get("function").unwrap_or(t))
            .find(|f| f["name"] == name.as_str())
            .map(|f| &f["parameters"]["properties"]);
        let mut args = vec![];
        let mut b = body;
        while let Some(p) = b.find("<parameter=") {
            b = &b[p + "<parameter=".len()..];
            let Some(gt) = b.find('>') else { break };
            let key = b[..gt].trim();
            b = &b[gt + 1..];
            let end = b.find("</parameter>").or_else(|| b.find("<parameter=")).unwrap_or(b.len());
            let v = b[..end].strip_prefix('\n').unwrap_or(&b[..end]);
            let v = v.strip_suffix('\n').unwrap_or(v);
            b = &b[end..];
            let ty = schema.and_then(|s| s[key]["type"].as_str());
            args.push(format!("{}:{}", json_str(key), typed(v, ty)));
        }
        calls.push(ToolCall { name, arguments: format!("{{{}}}", args.join(",")) });
    }
    calls
}

/// A parameter's JSON: a string for string params, else the model's text if it is valid JSON.
fn typed(v: &str, ty: Option<&str>) -> String {
    if ty != Some("string") {
        let t = v.trim();
        if serde_json::from_str::<serde_json::Value>(t).is_ok() {
            return t.to_owned();
        }
    }
    json_str(v)
}

fn json_str(s: &str) -> String {
    serde_json::Value::from(s).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> serde_json::Value {
        json!([{"type": "function", "function": {"name": "shell", "parameters": {"type": "object", "properties": {
            "command": {"type": "array"}, "note": {"type": "string"}, "n": {"type": "integer"}}}}}])
    }

    #[test]
    fn parses_typed_arguments_like_the_oracle() {
        // Oracle (tests/golden/api.json responses_tool): {"command":["ls", "-l", "/tmp/demo"]}
        let out = "<tool_call>\n<function=shell>\n<parameter=command>\n[\"ls\", \"-l\", \"/tmp/demo\"]\n</parameter>\n<parameter=note>\n42\n</parameter>\n<parameter=n>\n3\n</parameter>\n</function>\n</tool_call>";
        let calls = parse(out, &tools());
        assert_eq!(calls, vec![ToolCall { name: "shell".into(), arguments: r#"{"command":["ls", "-l", "/tmp/demo"],"note":"42","n":3}"#.into() }]);
    }

    #[test]
    fn splitter_streams_text_and_keeps_calls() {
        let mut s = Splitter::new(true);
        let mut text = String::new();
        for d in ["Let me ", "look.\n\n<tool", "_call>\n<function=shell>\n<parameter=note>\na\nb\n</parameter>\n</function>\n</tool_call>"] {
            text += &s.push(d);
        }
        let (rest, calls) = s.finish(&tools());
        assert_eq!((text.as_str(), rest.as_str()), ("Let me look.", ""));
        assert_eq!(calls[0].arguments, r#"{"note":"a\nb"}"#);
        // A '<' that is not a tool call is released.
        let mut s = Splitter::new(true);
        assert_eq!(s.push("a\n<"), "a");
        assert_eq!(s.push("b"), "\n<b");
        // A bare <function= starts a call too.
        let mut s = Splitter::new(true);
        assert_eq!(s.push("ok\n<function=shell>\n</function>"), "ok");
        assert_eq!(s.finish(&tools()).1, vec![ToolCall { name: "shell".into(), arguments: "{}".into() }]);
    }
}
