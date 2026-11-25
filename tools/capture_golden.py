#!/usr/bin/env python3
"""Capture reference outputs from a running llama-server (the oracle) into tests/golden/.

Usage: python3 -I tools/capture_golden.py [--no-gen] [base_url]   (default http://127.0.0.1:8099)
  --no-gen  only tokenize.json + template.json (no GPU needed: run the oracle with --gpu-layers 0)

Writes:
  tokenize.json   text -> token ids (parse_special=true, add_special=false)
  template.json   chat messages (+tools) -> rendered prompt
  greedy.json     prompt tokens -> greedy continuation with top-5 logprobs per step
  perf.json       llama-server timings for a decode-heavy and a prefill-heavy request
"""
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ARGS = [a for a in sys.argv[1:] if a != "--no-gen"]
NO_GEN = "--no-gen" in sys.argv[1:]
BASE = ARGS[0] if ARGS else "http://127.0.0.1:8099"
OUT = Path(__file__).resolve().parent.parent / "tests" / "golden"


def post(path, body, timeout=900):
    req = Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


TEXTS = [
    "Hello, world!",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "    indented\n\t\ttabs\r\nCRLF   trailing   ",
    "I'm sure it's they're we'll you've he'd",
    "Numbers: 1 12 123 1234 12345 3.14159 -42 0x1F 1e-9",
    "Unicode: café naïve 日本語 中文 한국어 Ελληνικά русский",
    "Emoji: 🚀🔥👍🏽 👨‍👩‍👧‍👦",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.rs\n</parameter>\n</function>\n</tool_call>",
    "fn main() { let v: Vec<u8> = (0..=255).collect(); println!(\"{:?}\", &v[..]); }",
    "SELECT id, name FROM users WHERE email LIKE '%@example.com' ORDER BY id DESC;",
    "    \n\n\n    \n",
    "",
    "a" * 300,
    "https://example.com/path?query=1&b=2#frag",
]

# Tokenized with parse_special=false: CONTROL tokens stay text, USER_DEFINED ones still split.
TEXTS_RAW = [
    "<|im_start|>user\nhi<|im_end|>\n",
    "<tool_call>x</tool_call> <think>y</think><|endoftext|>",
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from disk",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path"}},
            "required": ["path"],
        },
    },
}]

BIG_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "  Search code for a <pattern> & report matches.\n",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Regex, e.g. a<b"},
                "mode": {"type": "string", "enum": ["fast", "full", "it's"], "default": "fast"},
                "limit": {"type": "integer", "enum": [10, 50], "minimum": 1, "default": 10},
                "ratio": {"type": "number", "default": 2.5},
                "case": {"type": "boolean", "default": False},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Globs"},
                "opts": {"type": ["object", "null"], "properties": {"depth": {"type": "integer"}},
                         "required": ["depth"]},
            },
            "required": ["query", "mode"],
        },
    },
}

TEMPLATES = [
    {"messages": [{"role": "user", "content": "Hi"}]},
    {"messages": [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write hello world in Rust."},
        {"role": "assistant", "content": "fn main() { println!(\"Hello, world!\"); }"},
        {"role": "user", "content": "Now in C."},
    ]},
    {"messages": [{"role": "user", "content": "Show me src/main.rs"}], "tools": TOOLS},
    {"messages": [
        {"role": "user", "content": "Show me src/main.rs"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "read_file", "arguments": "{\"path\": \"src/main.rs\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "fn main() {}"},
    ], "tools": TOOLS},
    {"messages": [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Find TODOs"},
        {"role": "assistant", "content": "  Searching now.  ", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search", "arguments":
                "{\"query\": \"TODO\", \"limit\": 10, \"case\": true, \"ratio\": 0.5, "
                "\"paths\": [\"src/*.rs\", \"it's\"], \"opts\": {\"depth\": 2, \"x\": null}}"}},
            {"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"a.rs\"}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "src/a.rs:1: TODO"},
        {"role": "tool", "tool_call_id": "c2", "content": "// TODO"},
        {"role": "assistant", "content": "One TODO in a.rs."},
        {"role": "user", "content": "Thanks"},
    ], "tools": [BIG_TOOL] + TOOLS},
]

GREEDY_PROMPTS = [
    "<|im_start|>user\nWhat does len([1, 2]) return in Python? Reply briefly.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a Rust function that reverses a string.<|im_end|>\n<|im_start|>assistant\n",
    "The capital of France is",
    "def quicksort(arr):\n",
]


def tokenize(text, parse_special=True):
    return post("/tokenize", {"content": text, "add_special": False, "parse_special": parse_special})["tokens"]


def greedy(tokens, n_predict):
    return post("/completion", {
        "prompt": tokens, "n_predict": n_predict, "temperature": 0, "top_k": 1,
        "n_probs": 5, "cache_prompt": False, "return_tokens": True,
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    tok = [{"text": t, "parse_special": True, "tokens": tokenize(t)} for t in TEXTS]
    tok += [{"text": t, "parse_special": False, "tokens": tokenize(t, False)} for t in TEXTS_RAW]
    (OUT / "tokenize.json").write_text(json.dumps(tok, ensure_ascii=False, indent=1))
    print(f"tokenize: {len(tok)} cases")

    tpl = []
    for case in TEMPLATES:
        body = dict(case, add_generation_prompt=True)
        tpl.append({"request": body, "prompt": post("/apply-template", body)["prompt"]})
    (OUT / "template.json").write_text(json.dumps(tpl, ensure_ascii=False, indent=1))
    print(f"template: {len(tpl)} cases")
    if NO_GEN:
        return

    gr = []
    for p in GREEDY_PROMPTS:
        toks = tokenize(p)
        r = greedy(toks, 32)
        gr.append({
            "prompt": p,
            "prompt_tokens": toks,
            "tokens": r.get("tokens"),
            "content": r["content"],
            "probs": r.get("completion_probabilities"),
        })
        print(f"greedy: {p[:40]!r} -> {r['content'][:60]!r}")
    (OUT / "greedy.json").write_text(json.dumps(gr, ensure_ascii=False, indent=1))

    # Perf: decode-heavy (short prompt, 256 tokens) and prefill-heavy (~4K-token prompt).
    perf = {}
    r = greedy(tokenize(GREEDY_PROMPTS[1]), 256)
    perf["decode"] = r["timings"]
    long_prompt = tokenize(Path(__file__).read_text() * 6)[:4096]
    r = greedy(long_prompt, 16)
    perf["prefill"] = r["timings"]
    (OUT / "perf.json").write_text(json.dumps(perf, indent=1))
    for k, t in perf.items():
        print(f"perf {k}: prompt {t['prompt_n']} tok @ {t['prompt_per_second']:.1f} t/s, "
              f"decode {t['predicted_n']} tok @ {t['predicted_per_second']:.1f} t/s")


if __name__ == "__main__":
    main()
