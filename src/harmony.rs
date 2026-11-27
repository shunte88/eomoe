//! gpt-oss Harmony output (llama.cpp common/parsers/gpt-oss.cpp): messages after the generation
//! prompt's `<|start|>assistant`, each `HEADER<|message|>BODY<|end|><|start|>assistant`. The
//! analysis channel is reasoning, final and commentary are content, and a `to=functions.NAME`
//! recipient makes the body a tool call's JSON arguments (ended by `<|call|>`, an EOG token).

use crate::toolcall::ToolCall;

#[derive(Debug, PartialEq)]
enum Kind {
    Reasoning,
    Content,
    Call(String),
    Drop,
}

/// Streams reasoning and content as they are generated, keeping a tool call for `finish`.
pub struct Harmony {
    held: String,
    /// None: in a header.
    kind: Option<Kind>,
    tools: bool,
}

const END: &str = "<|end|>";

impl Harmony {
    pub fn new(tools: bool) -> Self {
        Self { held: String::new(), kind: None, tools }
    }

    /// Feed generated text; returns (reasoning, content) safe to stream now.
    pub fn push(&mut self, delta: &str) -> (String, String) {
        self.held.push_str(delta);
        let (mut r, mut c) = (String::new(), String::new());
        loop {
            match &self.kind {
                None => {
                    let Some(p) = self.held.find("<|message|>") else { break };
                    self.kind = Some(self.header(&self.held[..p]));
                    self.held.drain(..p + "<|message|>".len());
                }
                Some(Kind::Call(_)) => break,
                Some(k) => {
                    let (end, cut) = match self.held.find(END) {
                        Some(p) => (true, p),
                        // Hold back a suffix that could still grow into <|end|>.
                        None => (false, self.held.len() - (1..END.len()).filter(|&n| self.held.ends_with(&END[..n])).max().unwrap_or(0)),
                    };
                    match k {
                        Kind::Reasoning => r.push_str(&self.held[..cut]),
                        Kind::Content => c.push_str(&self.held[..cut]),
                        _ => {}
                    }
                    self.held.drain(..cut);
                    if !end {
                        break;
                    }
                    self.held.drain(..END.len());
                    self.kind = None;
                }
            }
        }
        (r, c)
    }

    /// End of generation: (reasoning, content) still held, and the tool call if any.
    pub fn finish(self) -> (String, String, Vec<ToolCall>) {
        match self.kind {
            Some(Kind::Reasoning) => (self.held, String::new(), vec![]),
            Some(Kind::Content) => (String::new(), self.held, vec![]),
            Some(Kind::Call(name)) => (String::new(), String::new(), vec![ToolCall { name, arguments: self.held.trim_end().to_owned() }]),
            _ => (String::new(), String::new(), vec![]),
        }
    }

    /// `[<|start|>assistant][ to=R]<|channel|>CHANNEL[ to=R][ <|constrain|>json]`; a stray
    /// `<|channel|>commentary` before the real channel is skipped (the last channel wins).
    fn header(&self, h: &str) -> Kind {
        let word = |s: &str| s.split(|c: char| c.is_whitespace() || c == '<').next().unwrap_or("").to_owned();
        if let Some(p) = h.find("to=") {
            return match word(&h[p + 3..]).strip_prefix("functions.") {
                Some(name) if self.tools => Kind::Call(name.to_owned()),
                _ => Kind::Drop,
            };
        }
        match h.rfind("<|channel|>").map(|p| word(&h[p + "<|channel|>".len()..])).as_deref() {
            Some("analysis") => Kind::Reasoning,
            Some("final" | "commentary") => Kind::Content,
            _ => Kind::Drop,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (reasoning, content, calls) from `out` streamed a char at a time.
    fn run(out: &str, tools: bool) -> (String, String, Vec<ToolCall>) {
        let mut h = Harmony::new(tools);
        let (mut r, mut c) = (String::new(), String::new());
        for ch in out.chars() {
            let (a, b) = h.push(&ch.to_string());
            r += &a;
            c += &b;
        }
        let (a, b, calls) = h.finish();
        (r + &a, c + &b, calls)
    }

    #[test]
    fn splits_analysis_from_final() {
        // gpt-oss-20b's greedy answer (tests/phase13_gptoss.rs cpu_reference_answers).
        let out = "<|channel|>analysis<|message|>The answer is \"Paris\". Just output that.<|end|><|start|>assistant<|channel|>final<|message|>Paris";
        assert_eq!(run(out, false), ("The answer is \"Paris\". Just output that.".into(), "Paris".into(), vec![]));
    }

    #[test]
    fn tool_call_in_channel_and_in_role() {
        let want = |n: &str| ("Need the weather.".to_owned(), String::new(), vec![ToolCall { name: n.into(), arguments: "{\"city\":\"Paris\"}".into() }]);
        let out = "<|channel|>analysis<|message|>Need the weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}";
        assert_eq!(run(out, true), want("get_weather"));
        let out = "<|channel|>analysis<|message|>Need the weather.<|end|><|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{\"city\":\"Paris\"}";
        assert_eq!(run(out, true), want("get_weather"));
        // Without tools a call is unsolicited and dropped.
        let out = "<|channel|>analysis<|message|>Need the weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}";
        assert_eq!(run(out, false), ("Need the weather.".into(), String::new(), vec![]));
    }

    #[test]
    fn preamble_and_stray_commentary_are_content() {
        let out = "<|channel|>commentary<|message|>Checking.<|end|><|start|>assistant<|channel|>commentary<|channel|>analysis<|message|>Hmm.<|end|><|start|>assistant<|channel|>final<|message|>Done.";
        assert_eq!(run(out, false), ("Hmm.".into(), "Checking.Done.".into(), vec![]));
    }
}
