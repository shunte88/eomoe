#!/usr/bin/env python3
"""Capture Ollama wire shapes for the Phase 11d gate -> tests/golden/ollama.json.

Runs against an Ollama server (default http://127.0.0.1:11434) with a small model pulled
(default qwen3:0.6b). Records status, content type and every NDJSON line per case. The gate
compares shapes (key sets, types, done/done_reason), not generated text.
"""
import json, sys, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:11434"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3:0.6b"
OPT = {"temperature": 0, "num_predict": 24}
USER = [{"role": "user", "content": "Say hello in five words."}]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
WEATHER = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}

CASES = [
    ("chat", "POST", "/api/chat", {"model": MODEL, "messages": USER, "stream": False, "think": False, "options": OPT}),
    ("chat_stream", "POST", "/api/chat", {"model": MODEL, "messages": USER, "think": False, "options": OPT}),
    ("chat_length", "POST", "/api/chat", {"model": MODEL, "messages": USER, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": 3}}),
    ("chat_tools", "POST", "/api/chat", {"model": MODEL, "messages": WEATHER, "tools": TOOLS, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": 64}}),
    ("chat_tools_stream", "POST", "/api/chat", {"model": MODEL, "messages": WEATHER, "tools": TOOLS, "think": False, "options": {"temperature": 0, "num_predict": 64}}),
    ("chat_tool_result", "POST", "/api/chat", {"model": MODEL, "stream": False, "think": False, "tools": TOOLS, "options": OPT, "messages": WEATHER + [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "content": "18C and sunny", "tool_name": "get_weather"}]}),
    ("chat_json", "POST", "/api/chat", {"model": MODEL, "messages": [{"role": "user", "content": "Give a JSON object with a name and an age."}], "format": "json", "stream": False, "think": False, "options": {"temperature": 0, "num_predict": 64}}),
    ("chat_schema", "POST", "/api/chat", {"model": MODEL, "messages": [{"role": "user", "content": "Invent a person."}], "format": SCHEMA, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": 64}}),
    ("chat_load", "POST", "/api/chat", {"model": MODEL, "messages": []}),
    ("generate", "POST", "/api/generate", {"model": MODEL, "prompt": "Why is the sky blue?", "stream": False, "think": False, "options": OPT}),
    ("generate_stream", "POST", "/api/generate", {"model": MODEL, "prompt": "Why is the sky blue?", "think": False, "options": OPT}),
    ("generate_raw", "POST", "/api/generate", {"model": MODEL, "prompt": "fn main() {", "raw": True, "stream": False, "options": OPT}),
    ("generate_system", "POST", "/api/generate", {"model": MODEL, "prompt": "Hi", "system": "Answer in French.", "stream": False, "think": False, "options": OPT}),
    ("generate_load", "POST", "/api/generate", {"model": MODEL}),
    ("err_no_model", "POST", "/api/chat", {"messages": USER}),
    ("err_unknown_model", "POST", "/api/generate", {"model": "nope", "prompt": "hi"}),
    ("err_bad_json", "POST", "/api/chat", "{bad"),
    ("tags", "GET", "/api/tags", None),
    ("show", "POST", "/api/show", {"model": MODEL}),
    ("show_by_name", "POST", "/api/show", {"model": "", "name": MODEL}),  # what the ollama CLI sends
    ("ps", "GET", "/api/ps", None),
    ("version", "GET", "/api/version", None),
    ("root", "GET", "/", None),
]


def call(method, path, body):
    data = None if body is None else (body if isinstance(body, str) else json.dumps(body)).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=300)
    except urllib.error.HTTPError as e:
        r = e
    raw = r.read().decode()
    ctype = r.headers.get("Content-Type", "")
    lines = []
    for l in raw.splitlines():
        try:
            lines.append(json.loads(l))
        except ValueError:
            lines.append(l)
    return {"status": r.status if hasattr(r, "status") else r.code, "content_type": ctype, "lines": lines}


out = []
for name, method, path, body in CASES:
    res = call(method, path, body)
    print(name, res["status"], len(res["lines"]), file=sys.stderr)
    out.append({"name": name, "method": method, "path": path, "body": body, **res})
# Trim what is per-model bulk (license text, tensor list) and other installed models.
for c in out:
    if c["path"] == "/api/show" and c["status"] == 200:
        l = c["lines"][0]
        l["license"], l["tensors"], l["modelfile"], l["template"] = l["license"][:40], l["tensors"][:2], l["modelfile"][:300], l["template"][:200]
    if c["name"] == "tags":
        c["lines"][0]["models"] = [m for m in c["lines"][0]["models"] if m["name"] == MODEL]
json.dump({"ollama": BASE, "model": MODEL, "cases": out}, open("tests/golden/ollama.json", "w"), indent=1)
