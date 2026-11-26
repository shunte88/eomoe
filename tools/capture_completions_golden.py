#!/usr/bin/env python3
"""Capture the oracle llama-server's raw-completion endpoints into tests/golden/completions.json.

Usage: python3 -I tools/capture_completions_golden.py [base_url]   (default http://127.0.0.1:8099)
Start the oracle as for capture_api_golden.py (`--gpu-layers 0 --alias local-code --jinja`).

Each case is posted greedy, non-streamed and streamed: /v1/completions (OpenAI text_completion),
/completion (llama.cpp native) and /infill (FIM with repo-level extra context), plus the
/infill and /v1/completions validation errors.
"""
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / "completions.json"

PRE = "import math\n\ndef circle_area(r):\n    "
SUF = "\n\nprint(circle_area(2.0))\n"
CASES = {
    "oai_text": ("/v1/completions", {"prompt": "def fib(n):", "max_tokens": 24}),
    "oai_stop": ("/v1/completions", {"prompt": "# three fruits\n1. apple\n2.", "max_tokens": 40, "stop": ["\n3."]}),
    "oai_special": ("/v1/completions", {"prompt": "<|im_start|>user\nSay hi<|im_end|>\n<|im_start|>assistant\n", "max_tokens": 16}),
    "native_text": ("/completion", {"prompt": "def fib(n):", "n_predict": 24}),
    "native_tokens": ("/completions", {"prompt": [750, 15801, 1445, 1648], "n_predict": 8}),
    "infill": ("/infill", {"input_prefix": PRE, "input_suffix": SUF, "n_predict": 24}),
    "infill_extra": ("/infill", {"input_prefix": PRE, "input_suffix": SUF, "prompt": "return",
                                 "input_extra": [{"filename": "geometry/consts.py", "text": "PI = 3.14159\n"},
                                                 {"text": "# helpers\n"}],
                                 "n_predict": 24}),
}
ERRORS = {
    "infill_no_prefix": ("/infill", {"input_suffix": SUF}),
    "infill_bad_extra": ("/infill", {"input_prefix": PRE, "input_suffix": SUF, "input_extra": [{"filename": "a"}]}),
    "oai_no_prompt": ("/v1/completions", {"max_tokens": 4}),
}


def post(path, body, stream):
    req = Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=600) as r:
            if not stream:
                return {"status": r.status, "body": json.loads(r.read())}
            events = []
            for line in r:
                line = line.decode().rstrip("\n")
                if line.startswith("data: "):
                    d = line[6:]
                    events.append(d if d == "[DONE]" else json.loads(d))
            return {"status": r.status, "events": events}
    except HTTPError as e:
        return {"status": e.code, "body": json.loads(e.read() or b"null")}


out = {"cases": {}, "errors": {}}
for name, (path, body) in CASES.items():
    body = dict(body, model="local-code", temperature=0)
    out["cases"][name] = {"path": path, "request": body, "full": post(path, body, False),
                          "stream": post(path, dict(body, stream=True), True)}
    print(name, out["cases"][name]["full"]["status"])
for name, (path, body) in ERRORS.items():
    out["errors"][name] = {"path": path, "request": body, **post(path, body, False)}
    print(name, out["errors"][name]["status"], out["errors"][name]["body"])
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print("wrote", OUT)
