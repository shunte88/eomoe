#!/usr/bin/env python3
"""Capture llama-server's gpt-oss chat API behaviour (Harmony reasoning, tool calls, reasoning in
the history) into tests/golden/api_gpt-oss.json, in tools/capture_api_golden.py's format.

  systemd-run --user --scope -p MemoryMax=18G -p MemorySwapMax=0 llama-server \\
      -m gpt-oss-20b-MXFP4.gguf -ngl 99 -ot exps=CPU -c 4096 --port 8099 --jinja --alias local-code
  python3 -I tools/capture_api_gptoss.py [base_url] [name]

name (default gpt-oss) picks tests/golden/api_<name>.json; "glm" (GLM-4.7-Flash: <think> reasoning, its
own tool-call tags) adds a chat_template_kwargs enable_thinking=false case.
"""
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
NAME = sys.argv[2] if len(sys.argv) > 2 else "gpt-oss"
OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / f"api_{NAME}.json"
PROMPT = "What is the capital of France? Answer in one word."
ASK = "What's the weather in Paris right now?"
THOUGHT = "The user wants the current weather in Paris. Use get_weather."
ARGS = {"city": "Paris"}
RESULT = "18 C, light rain"

SCHEMA = {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}
DESC = "Get the current weather for a city."
C_TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": DESC, "parameters": SCHEMA}}]
A_TOOLS = [{"name": "get_weather", "description": DESC, "input_schema": SCHEMA}]
R_TOOLS = [{"type": "function", "name": "get_weather", "description": DESC, "parameters": SCHEMA, "strict": False}]

CASES = {
    "chat_text": ("/v1/chat/completions", {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 200}),
    "chat_tool": ("/v1/chat/completions", {"tools": C_TOOLS, "max_tokens": 200, "messages": [{"role": "user", "content": ASK}]}),
    "chat_tool_result": ("/v1/chat/completions", {"tools": C_TOOLS, "max_tokens": 200, "messages": [
        {"role": "user", "content": ASK},
        {"role": "assistant", "content": None, "reasoning_content": THOUGHT,
         "tool_calls": [{"id": "call_01", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps(ARGS)}}]},
        {"role": "tool", "tool_call_id": "call_01", "content": RESULT}]}),
    "messages_text": ("/v1/messages", {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 200}),
    "messages_tool": ("/v1/messages", {"tools": A_TOOLS, "max_tokens": 200, "messages": [{"role": "user", "content": ASK}]}),
    "messages_tool_result": ("/v1/messages", {"tools": A_TOOLS, "max_tokens": 200, "messages": [
        {"role": "user", "content": ASK},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": THOUGHT, "signature": ""},
                                          {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": ARGS}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": RESULT}]}]}),
    "responses_text": ("/v1/responses", {"input": PROMPT, "max_output_tokens": 200, "store": False}),
    "responses_tool": ("/v1/responses", {"tools": R_TOOLS, "max_output_tokens": 200, "store": False,
                                         "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": ASK}]}]}),
    "responses_tool_result": ("/v1/responses", {"tools": R_TOOLS, "max_output_tokens": 200, "store": False, "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": ASK}]},
        {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": THOUGHT}]},
        {"type": "function_call", "call_id": "call_01", "name": "get_weather", "arguments": json.dumps(ARGS)},
        {"type": "function_call_output", "call_id": "call_01", "output": RESULT}]}),
}
if NAME == "glm":
    CASES["chat_nothink"] = ("/v1/chat/completions", {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 200,
                                                      "chat_template_kwargs": {"enable_thinking": False}})


def post(path, body, stream):
    req = Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=600) as r:
            if not stream:
                return {"status": r.status, "body": json.loads(r.read())}
            events, name = [], None
            for line in r:
                line = line.decode().rstrip("\n")
                if line.startswith("event: "):
                    name = line[7:]
                elif line.startswith("data: "):
                    data = line[6:]
                    events.append({"event": name, "data": data if data == "[DONE]" else json.loads(data)})
                    name = None
            return {"status": r.status, "events": events}
    except HTTPError as e:
        return {"status": e.code, "body": json.loads(e.read() or b"null")}


def get(path):
    with urlopen(BASE + path, timeout=60) as r:
        return {"status": r.status, "body": json.loads(r.read())}


out = {"cases": {}}
for name, (path, body) in CASES.items():
    body = dict(body, model="local-code", temperature=0)
    out["cases"][name] = {"path": path, "request": body, "full": post(path, body, False), "stream": post(path, dict(body, stream=True), True)}
    print(name, json.dumps(out["cases"][name]["full"]["body"])[:300])
out["count_tokens"] = post("/v1/messages/count_tokens", {k: v for k, v in CASES["messages_tool"][1].items() if k != "max_tokens"} | {"model": "local-code"}, False)
out["models"] = get("/v1/models")
out["health"] = get("/health")
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print("wrote", OUT)
