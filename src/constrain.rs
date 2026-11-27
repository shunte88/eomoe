//! Constrained output (`response_format` json_object / json_schema): is a byte string a prefix
//! of JSON that matches a schema? The sampler keeps only tokens that leave the output a valid
//! prefix, and ends generation once the value is complete.
//!
//! Schema subset: `type` (one or a list), `properties` + `required` (keys in any order; a
//! required key not in `properties` takes any value), `additionalProperties` (false, true or a
//! schema; absent means false when `properties` is given, as llama.cpp), `items`, `enum`,
//! `const`, `anyOf` / `oneOf` (oneOf checked as anyOf), `$ref` to `#/...` in the same schema.
//! Anything else (`allOf`, `pattern`, `minItems`, formats ...) is accepted unchecked.
//! ponytail: the output is re-parsed from the start for every candidate token (O(len) each);
//! an incremental parser state per stream if long structured outputs get slow.

use serde_json::Value;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum State {
    /// Can never become valid.
    Invalid,
    /// Valid so far, not finished.
    Partial,
    /// One complete value (trailing whitespace allowed).
    Done,
}

/// The input ran out (Partial) or broke the grammar (Invalid).
type R<T = ()> = Result<T, State>;

/// Whitespace runs are capped so the model cannot pad forever.
const MAX_WS: usize = 16;
/// Digits per number part, as llama.cpp's JSON grammar ([1-9][0-9]{0,15}, fraction {1,16}).
const MAX_DIGITS: usize = 16;
/// Nesting depth of objects and arrays.
pub const MAX_DEPTH: usize = 64;

pub fn check(schema: &Value, text: &[u8]) -> State {
    let mut p = P { s: text, i: 0, root: schema, depth: 0 };
    // No whitespace before the value: the model cannot pad its way to the length limit.
    let r = p.value(schema).and_then(|_| p.ws());
    match r {
        Ok(()) if p.i == text.len() => State::Done,
        Ok(()) => State::Invalid,
        // A top-level number at the end of the text may still grow: Partial.
        Err(e) => e,
    }
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
    /// For `$ref`.
    root: &'a Value,
    depth: usize,
}

impl<'a> P<'a> {
    fn peek(&self) -> R<u8> {
        self.s.get(self.i).copied().ok_or(State::Partial)
    }

    fn eat(&mut self, c: u8) -> R {
        if self.peek()? != c {
            return Err(State::Invalid);
        }
        self.i += 1;
        Ok(())
    }

    fn ws(&mut self) -> R {
        let n = self.s[self.i..].iter().take_while(|c| b" \t\n\r".contains(c)).count();
        self.i += n;
        if n > MAX_WS { Err(State::Invalid) } else { Ok(()) }
    }

    /// `lit` exactly, or the input ends inside it.
    fn lit(&mut self, lit: &[u8]) -> R {
        let rest = &self.s[self.i..];
        if rest.starts_with(lit) {
            self.i += lit.len();
            Ok(())
        } else if lit.starts_with(rest) {
            Err(State::Partial)
        } else {
            Err(State::Invalid)
        }
    }

    fn value(&mut self, sch: &'a Value) -> R {
        if let Some(r) = sch.get("$ref").and_then(Value::as_str) {
            // Unresolvable: unchecked, like the other unsupported keywords.
            let to = r.strip_prefix('#').and_then(|ptr| self.root.pointer(ptr)).unwrap_or(&Value::Null);
            return self.value(to);
        }
        if let Some(alts) = sch.get("anyOf").or(sch.get("oneOf")).and_then(Value::as_array) {
            // JSON values are self-delimiting, so the first alternative that parses wins.
            let mut partial = false;
            for a in alts {
                let mut q = P { s: self.s, i: self.i, root: self.root, depth: self.depth };
                match q.value(a) {
                    Ok(()) => {
                        self.i = q.i;
                        return Ok(());
                    }
                    Err(State::Partial) => partial = true,
                    Err(_) => {}
                }
            }
            return Err(if partial { State::Partial } else { State::Invalid });
        }
        let options: Vec<&Value> = match sch.get("enum").and_then(Value::as_array) {
            Some(e) => e.iter().collect(),
            None => sch.get("const").into_iter().collect(),
        };
        if !options.is_empty() {
            return self.one_of(&options);
        }
        let types: Vec<&str> = match sch.get("type") {
            Some(Value::String(t)) => vec![t],
            Some(Value::Array(ts)) => ts.iter().filter_map(Value::as_str).collect(),
            _ => vec![],
        };
        let ok = |t: &str| types.is_empty() || types.contains(&t) || (t == "number" && types.contains(&"integer"));
        let c = self.peek()?;
        if matches!(c, b'{' | b'[') && self.depth >= MAX_DEPTH {
            return Err(State::Invalid);
        }
        match c {
            b'{' if ok("object") => self.object(sch),
            b'[' if ok("array") => self.array(sch),
            b'"' if ok("string") => self.string().map(drop),
            b't' if ok("boolean") => self.lit(b"true"),
            b'f' if ok("boolean") => self.lit(b"false"),
            b'n' if ok("null") => self.lit(b"null"),
            b'-' | b'0'..=b'9' if ok("number") => self.number(!types.is_empty() && !types.contains(&"number")),
            _ => Err(State::Invalid),
        }
    }

    /// One of the literal values (enum / const), as compact JSON.
    fn one_of(&mut self, options: &[&Value]) -> R {
        let mut partial = false;
        for o in options {
            let mut q = P { s: self.s, i: self.i, root: self.root, depth: self.depth };
            match q.lit(o.to_string().as_bytes()) {
                Ok(()) => {
                    self.i = q.i;
                    return Ok(());
                }
                Err(State::Partial) => partial = true,
                Err(_) => {}
            }
        }
        Err(if partial { State::Partial } else { State::Invalid })
    }

    fn object(&mut self, sch: &'a Value) -> R {
        let props = sch.get("properties").and_then(Value::as_object);
        let required: Vec<&str> = sch.get("required").and_then(Value::as_array).map_or(vec![], |r| r.iter().filter_map(Value::as_str).collect());
        // The schema for keys outside `properties`/`required`; None: not allowed.
        let extra: Option<&Value> = match sch.get("additionalProperties") {
            Some(Value::Bool(false)) => None,
            Some(v @ Value::Object(_)) => Some(v),
            Some(_) => Some(&Value::Null),
            None => props.is_none().then_some(&Value::Null),
        };
        let named = |k: &str| props.is_some_and(|p| p.contains_key(k)) || required.contains(&k);
        let n_named = props.map_or(0, |p| p.len()) + required.iter().filter(|r| !props.is_some_and(|p| p.contains_key(**r))).count();
        let mut seen: Vec<String> = vec![];
        self.depth += 1;
        self.eat(b'{')?;
        self.ws()?;
        if self.peek()? != b'}' {
            loop {
                let start = self.i;
                let key = match self.string() {
                    Ok(k) => k,
                    Err(State::Partial) => {
                        // A partial key must start one still allowed.
                        let part = String::from_utf8_lossy(self.s.get(start + 1..).unwrap_or_default()).into_owned();
                        let names = props.into_iter().flat_map(|p| p.keys().map(String::as_str)).chain(required.iter().copied());
                        let fits = extra.is_some() || names.into_iter().any(|k| k.starts_with(part.as_str()) && !seen.iter().any(|s| s == k));
                        return Err(if fits { State::Partial } else { State::Invalid });
                    }
                    Err(e) => return Err(e),
                };
                if seen.contains(&key) {
                    return Err(State::Invalid);
                }
                let vs = match props.and_then(|p| p.get(&key)) {
                    Some(v) => v,
                    None if named(&key) => &Value::Null,
                    None => extra.ok_or(State::Invalid)?,
                };
                seen.push(key);
                self.ws()?;
                self.eat(b':')?;
                self.ws()?;
                self.value(vs)?;
                self.ws()?;
                match self.peek()? {
                    // Another key only if one is left.
                    b',' if extra.is_some() || n_named > seen.len() => {
                        self.i += 1;
                        self.ws()?;
                    }
                    b'}' => break,
                    _ => return Err(State::Invalid),
                }
            }
        }
        if required.iter().any(|r| !seen.iter().any(|s| s == r)) {
            return Err(State::Invalid);
        }
        self.depth -= 1;
        self.eat(b'}')
    }

    fn array(&mut self, sch: &'a Value) -> R {
        let items = sch.get("items").unwrap_or(&Value::Null);
        self.depth += 1;
        self.eat(b'[')?;
        self.ws()?;
        if self.peek()? == b']' {
            self.i += 1;
            self.depth -= 1;
            return Ok(());
        }
        loop {
            self.value(items)?;
            self.ws()?;
            match self.peek()? {
                b',' => {
                    self.i += 1;
                    self.ws()?;
                }
                b']' => {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(());
                }
                _ => return Err(State::Invalid),
            }
        }
    }

    /// A string; returns its raw contents (escapes kept as written).
    fn string(&mut self) -> R<String> {
        self.eat(b'"')?;
        let start = self.i;
        loop {
            match self.peek()? {
                b'"' => {
                    let s = String::from_utf8_lossy(&self.s[start..self.i]).into_owned();
                    self.i += 1;
                    return Ok(s);
                }
                b'\\' => {
                    self.i += 1;
                    match self.peek()? {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => self.i += 1,
                        b'u' => {
                            self.i += 1;
                            for _ in 0..4 {
                                if !self.peek()?.is_ascii_hexdigit() {
                                    return Err(State::Invalid);
                                }
                                self.i += 1;
                            }
                        }
                        _ => return Err(State::Invalid),
                    }
                }
                c if c < 0x20 => return Err(State::Invalid),
                _ => self.i += 1,
            }
        }
    }

    /// -?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?, without fraction or exponent for integers.
    /// The end of the input inside a number is Partial: more digits may follow.
    fn number(&mut self, integer: bool) -> R {
        let digits = |p: &mut Self| -> R<usize> {
            let n = p.s[p.i..].iter().take_while(|c| c.is_ascii_digit()).count();
            p.i += n;
            if n > MAX_DIGITS {
                Err(State::Invalid)
            } else if p.i == p.s.len() {
                Err(State::Partial)
            } else {
                Ok(n)
            }
        };
        if self.peek()? == b'-' {
            self.i += 1;
        }
        if self.peek()? == b'0' {
            self.i += 1;
            self.peek()?;
        } else if digits(self)? == 0 {
            return Err(State::Invalid);
        }
        if !integer && self.peek()? == b'.' {
            self.i += 1;
            if digits(self)? == 0 {
                return Err(State::Invalid);
            }
        }
        if !integer && matches!(self.peek()?, b'e' | b'E') {
            self.i += 1;
            if matches!(self.peek()?, b'+' | b'-') {
                self.i += 1;
            }
            if digits(self)? == 0 {
                return Err(State::Invalid);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use State::*;

    fn c(s: &Value, t: &str) -> State {
        check(s, t.as_bytes())
    }

    #[test]
    fn any_json_prefixes() {
        let any = json!({});
        for (t, want) in [
            ("", Partial), (" {}", Invalid), ("{", Partial), ("{\"a\": [1, 2.5e3, true, nu", Partial), ("{\"a\": [1, 2.5e3, true, null]}", Done),
            ("{\"a\": [1, 2.5e3, true, null]}  \n", Done), ("{\"a\" 1}", Invalid), ("{]", Invalid), ("hello", Invalid),
            ("\"x\\q\"", Invalid), ("\"x\\u00e9\"", Done), ("{\"a\":1}{", Invalid), ("01", Invalid), ("[1,]", Invalid), (&"9".repeat(17), Invalid),
        ] {
            assert_eq!(c(&any, t), want, "{t:?}");
        }
        assert_eq!(c(&any, &" ".repeat(MAX_WS + 1)), Invalid, "endless whitespace");
    }

    #[test]
    fn schema_restricts_keys_types_and_values() {
        let s = json!({"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                       "tags": {"type": "array", "items": {"enum": ["a", "b"]}}}, "required": ["name", "age"]});
        for (t, want) in [
            ("{", Partial), ("{\"name\": \"x\", ", Partial), ("{\"na", Partial), ("{\"nb", Invalid), ("{\"name\": 3", Invalid), ("{\"name\": \"x\", \"age\": 4.5", Invalid),
            ("{\"name\": \"x\"}", Invalid), ("{\"age\": 4, \"name\": \"x\"}", Done), ("{\"name\": \"x\", \"age\": 4, \"tags\": [\"a\", \"c", Invalid),
            ("{\"name\": \"x\", \"age\": 4, \"tags\": [\"b\"]}", Done), ("{\"name\": \"x\", \"name\": \"y\"", Invalid),
            ("{\"name\": \"x\", \"age\": 4, \"zip\": 1}", Invalid), ("[", Invalid),
            ("{\"name\": \"x\", \"age\": 4, \"tags\": [],", Invalid),
        ] {
            assert_eq!(c(&s, t), want, "{t:?}");
        }
        assert_eq!(c(&json!({"const": 42}), "4"), Partial);
    }

    #[test]
    fn any_of_refs_and_additional_properties() {
        // What Pydantic emits: $defs + $ref, Optional as anyOf [T, null].
        let s = json!({"$defs": {"Pet": {"type": "object", "properties": {"n": {"type": "string"}}, "required": ["n"]}},
                       "type": "object", "properties": {"pet": {"$ref": "#/$defs/Pet"}, "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
                       "required": ["pet", "age"]});
        for (t, want) in [
            ("{\"pet\": {\"n\": \"x\"}, \"age\": null}", Done), ("{\"pet\": {\"n\": \"x\"}, \"age\": 3}", Done),
            ("{\"pet\": {\"n\": 1", Invalid), ("{\"pet\": {\"n\": \"x\"}, \"age\": \"3", Invalid), ("{\"pet\": {\"n\": \"x\"}, \"age\": nu", Partial),
        ] {
            assert_eq!(c(&s, t), want, "{t:?}");
        }
        let open = json!({"type": "object", "properties": {"a": {"type": "integer"}}, "additionalProperties": {"type": "string"}});
        assert_eq!(c(&open, "{\"a\": 1, \"zz\": \"x\"}"), Done);
        assert_eq!(c(&open, "{\"zz\": 2"), Invalid);
        assert_eq!(c(&json!({"type": "object", "properties": {"a": {}}, "additionalProperties": true}), "{\"b\": [1]}"), Done);
        assert_eq!(c(&json!({"type": "object", "additionalProperties": false}), "{\"b"), Invalid);
        assert_eq!(c(&json!({"type": "object", "additionalProperties": false}), "{}"), Done);
        // A required key missing from `properties` is still allowed (any value), not a dead end.
        assert_eq!(c(&json!({"type": "object", "properties": {"a": {}}, "required": ["b"]}), "{\"b\": 1}"), Done);
        // Nesting is capped: the model cannot recurse the parser's stack away.
        assert_eq!(c(&json!({}), &"[".repeat(MAX_DEPTH + 1)), Invalid);
        assert_eq!(c(&json!({}), &"[".repeat(MAX_DEPTH)), Partial);
        assert_eq!(c(&json!({"type": ["string", "null"]}), "null"), Done);
    }
}
