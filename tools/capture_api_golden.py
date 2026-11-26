#!/usr/bin/env python3
"""Capture the oracle llama-server's HTTP API behaviour into tests/golden/api.json.

Usage: python3 -I tools/capture_api_golden.py [base_url] [case-prefix]   (default http://127.0.0.1:8099)
With a case prefix (e.g. chat_), only those cases are captured and merged into the existing file.
Start the oracle like `local-code-llm serve` (`--alias local-code --jinja`), on that port.

Every case is greedy (temperature 0), streamed and not, so eomoe can be diffed on: event sequence,
usage (prompt tokens = request conversion + chat template), text, and parsed tool calls.
"""
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / "api.json"
PROMPT = "What does len([1, 2]) return in Python? Reply briefly."

A_TOOLS = [
    {"name": "Bash", "description": "Run a shell command and return its output.",
     "input_schema": {"type": "object", "properties": {
         "command": {"type": "string", "description": "The command to run"},
         "timeout": {"type": "number", "description": "Timeout in ms"}},
         "required": ["command"]}},
    {"name": "Read", "description": "Read a file from the local filesystem.",
     "input_schema": {"type": "object", "properties": {
         "file_path": {"type": "string"}, "limit": {"type": "integer"}},
         "required": ["file_path"]}},
]
A_SYSTEM = [{"type": "text", "text": "You are a coding agent. Use the tools to act."},
            {"type": "text", "text": "Working directory: /tmp/demo", "cache_control": {"type": "ephemeral"}}]
A_USE = {"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {"command": "ls /tmp/demo"}}

R_TOOLS = [
    {"type": "function", "name": "shell", "description": "Runs a shell command and returns its output.",
     "strict": False,
     "parameters": {"type": "object", "properties": {
         "command": {"type": "array", "items": {"type": "string"}},
         "workdir": {"type": "string"}}, "required": ["command"], "additionalProperties": False}},
]
C_TOOLS = [{"type": "function", "function": t} for t in [
    {"name": "Bash", "description": "Run a shell command and return its output.",
     "parameters": A_TOOLS[0]["input_schema"]},
    {"name": "Read", "description": "Read a file from the local filesystem.",
     "parameters": A_TOOLS[1]["input_schema"]}]]
C_CALL = {"id": "call_01", "type": "function", "function": {"name": "Bash", "arguments": json.dumps({"command": "ls /tmp/demo"})}}

R_CALL = {"type": "function_call", "call_id": "call_01", "name": "shell",
          "arguments": json.dumps({"command": ["ls", "/tmp/demo"]})}

CASES = {
    "messages_text": ("/v1/messages", {
        "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 64}),
    "messages_tool": ("/v1/messages", {
        "system": A_SYSTEM, "tools": A_TOOLS, "max_tokens": 200,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "List the files in the working directory."}]}]}),
    "messages_tool_result": ("/v1/messages", {
        "system": A_SYSTEM, "tools": A_TOOLS, "max_tokens": 120,
        "messages": [
            {"role": "user", "content": "List the files in the working directory."},
            {"role": "assistant", "content": [{"type": "text", "text": "I'll list them."}, A_USE]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01", "content": [{"type": "text", "text": "a.py\nb.rs"}]},
                {"type": "text", "text": "How many are there?"}]},
        ]}),
    "responses_text": ("/v1/responses", {
        "input": PROMPT, "instructions": "Be brief.", "max_output_tokens": 64, "store": False}),
    "responses_tool": ("/v1/responses", {
        "instructions": "You are a coding agent. Use the tools to act.", "tools": R_TOOLS,
        "tool_choice": "auto", "parallel_tool_calls": False, "store": False, "max_output_tokens": 200,
        "input": [{"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": "List the files in /tmp/demo."}]}]}),
    "responses_tool_result": ("/v1/responses", {
        "instructions": "You are a coding agent. Use the tools to act.", "tools": R_TOOLS,
        "store": False, "max_output_tokens": 120,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List the files in /tmp/demo."}]},
            R_CALL,
            {"type": "function_call_output", "call_id": "call_01", "output": "a.py\nb.rs"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "How many are there?"}]},
        ]}),
    "chat_text": ("/v1/chat/completions", {
        "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 64}),
    "chat_tool": ("/v1/chat/completions", {
        "tools": C_TOOLS, "max_tokens": 200,
        "messages": [{"role": "system", "content": "You are a coding agent. Use the tools to act."},
                     {"role": "user", "content": [{"type": "text", "text": "List the files in /tmp/demo."}]}]}),
    "chat_tool_result": ("/v1/chat/completions", {
        "tools": C_TOOLS, "max_tokens": 120, "stop": ["\n\n\n"],
        "messages": [
            {"role": "system", "content": "You are a coding agent. Use the tools to act."},
            {"role": "user", "content": "List the files in /tmp/demo."},
            {"role": "assistant", "content": None, "tool_calls": [C_CALL]},
            {"role": "tool", "tool_call_id": "call_01", "content": "a.py\nb.rs"},
            {"role": "user", "content": "How many are there?"}]}),
}


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


out = json.loads(OUT.read_text()) if ONLY else {"cases": {}}
for name, (path, body) in CASES.items():
    if ONLY and not name.startswith(ONLY):
        continue
    body = dict(body, model="local-code", temperature=0)
    out["cases"][name] = {"path": path, "request": body,
                          "full": post(path, body, False),
                          "stream": post(path, dict(body, stream=True), True)}
    print(name, "ok")
if ONLY:
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    sys.exit(print("merged", ONLY, "into", OUT))
out["count_tokens"] = post("/v1/messages/count_tokens",
                           {k: v for k, v in CASES["messages_tool"][1].items() if k != "max_tokens"} | {"model": "local-code"}, False)
out["models"] = get("/v1/models")
out["health"] = get("/health")
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print("wrote", OUT)
