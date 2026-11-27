//! Qwen3-Coder tool calls in model output:
//! `<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>\n</tool_call>`,
//! and GLM's (`GlmSplit`): `<tool_call>NAME<arg_key>P</arg_key><arg_value>VALUE</arg_value></tool_call>`.
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
    glm: bool,
}

impl Splitter {
    pub fn new(enabled: bool) -> Self {
        Self { held: String::new(), in_calls: false, enabled, glm: false }
    }

    fn starts(&self) -> &'static [&'static str] {
        if self.glm { &STARTS[..1] } else { &STARTS }
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
        if let Some(p) = self.starts().iter().filter_map(|m| self.held.find(m)).min() {
            self.in_calls = true;
            let text = strip_nl(&self.held[..p]).to_owned();
            self.held.drain(..p);
            return text;
        }
        // Hold back a suffix that could still grow into a call start, and the newlines before it.
        let held = &self.held;
        let keep = self
            .starts()
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
            let calls = if self.glm { parse_glm(&self.held, tools) } else { parse(&self.held, tools) };
            if !calls.is_empty() {
                return (String::new(), calls);
            }
        }
        (self.held, vec![])
    }
}

/// GLM output: reasoning up to `</think>` when the prompt opened `<think>`, then content and
/// `<tool_call>` blocks (llama.cpp's auto parser for the GLM-4.7 template).
pub struct GlmSplit {
    /// Some: still in reasoning, holding a possible partial `</think>`.
    think: Option<String>,
    calls: Splitter,
}

const THINK_END: &str = "</think>";

impl GlmSplit {
    pub fn new(thinking: bool, tools: bool) -> Self {
        Self { think: thinking.then(String::new), calls: Splitter { glm: true, ..Splitter::new(tools) } }
    }

    /// Feed generated text; returns (reasoning, content) safe to stream now.
    pub fn push(&mut self, delta: &str) -> (String, String) {
        let Some(h) = &mut self.think else { return (String::new(), self.calls.push(delta)) };
        h.push_str(delta);
        if let Some(p) = h.find(THINK_END) {
            let (r, rest) = (h[..p].to_owned(), h[p + THINK_END.len()..].to_owned());
            self.think = None;
            return (r, self.calls.push(&rest));
        }
        let keep = (1..THINK_END.len()).filter(|&n| h.ends_with(&THINK_END[..n])).max().unwrap_or(0);
        let r: String = h.drain(..h.len() - keep).collect();
        (r, String::new())
    }

    /// End of generation: (reasoning, content) still held, and the tool calls.
    pub fn finish(self, tools: &serde_json::Value) -> (String, String, Vec<ToolCall>) {
        match self.think {
            Some(h) => (h, String::new(), vec![]),
            None => {
                let (c, calls) = self.calls.finish(tools);
                (String::new(), c, calls)
            }
        }
    }
}

/// Every complete GLM `<tool_call>` block, in order.
fn parse_glm(s: &str, tools: &serde_json::Value) -> Vec<ToolCall> {
    let tag = |b: &str, open: &str, close: &str| -> Option<(String, usize)> {
        let p = b.find(open)? + open.len();
        let e = p + b[p..].find(close)?;
        Some((b[p..e].to_owned(), e + close.len()))
    };
    let mut calls = vec![];
    let mut rest = s;
    while let Some(p) = rest.find("<tool_call>") {
        rest = &rest[p + "<tool_call>".len()..];
        let end = rest.find("</tool_call>").unwrap_or(rest.len());
        let body = &rest[..end];
        rest = &rest[end..];
        let name = body[..body.find("<arg_key>").unwrap_or(body.len())].trim().to_owned();
        let props = schema_props(tools, &name);
        let (mut args, mut b) = (vec![], body);
        while let Some((key, n)) = tag(b, "<arg_key>", "</arg_key>") {
            let Some((v, m)) = tag(&b[n..], "<arg_value>", "</arg_value>") else { break };
            b = &b[n + m..];
            let ty = props.and_then(|s| s[key.trim()]["type"].as_str());
            args.push(format!("{}:{}", json_str(key.trim()), typed(&v, ty)));
        }
        calls.push(ToolCall { name, arguments: format!("{{{}}}", args.join(",")) });
    }
    calls
}

/// The `properties` of tool `name`'s parameters.
fn schema_props<'a>(tools: &'a serde_json::Value, name: &str) -> Option<&'a serde_json::Value> {
    tools
        .as_array()
        .into_iter()
        .flatten()
        .map(|t| t.get("function").unwrap_or(t))
        .find(|f| f["name"] == name)
        .map(|f| &f["parameters"]["properties"])
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
        let schema = schema_props(tools, &name);
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
    #[test]
    fn glm_streams_reasoning_then_content_and_calls() {
        let mut g = GlmSplit::new(true, true);
        let (mut r, mut c) = (String::new(), String::new());
        for d in ["Need ", "the dir.</th", "ink>Listing.<tool_", "call>shell<arg_key>command</arg_key><arg_value>[\"ls\"]</arg_value>",
                  "<arg_key>note</arg_key><arg_value>{\"a\": 1}\nx</arg_value><arg_key>n</arg_key><arg_value> 3 </arg_value></tool_call>",
                  "<tool_call>shell</tool_call>"] {
            let (a, b) = g.push(d);
            (r, c) = (r + &a, c + &b);
        }
        let (r2, rest, calls) = g.finish(&tools());
        assert_eq!((r.as_str(), c.as_str(), r2.as_str(), rest.as_str()), ("Need the dir.", "Listing.", "", ""));
        assert_eq!(calls, vec![
            ToolCall { name: "shell".into(), arguments: r#"{"command":["ls"],"note":"{\"a\": 1}\nx","n":3}"#.into() },
            ToolCall { name: "shell".into(), arguments: "{}".into() },
        ]);
        // Thinking off: all content; unfinished reasoning stays reasoning.
        let mut g = GlmSplit::new(false, false);
        assert_eq!(g.push("4</think>"), (String::new(), "4</think>".into()));
        let mut g = GlmSplit::new(true, false);
        assert_eq!(g.push("hmm</"), ("hmm".into(), String::new()));
        assert_eq!(g.finish(&tools()), ("</".into(), String::new(), vec![]));
    }
}
