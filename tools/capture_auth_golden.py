#!/usr/bin/env python3
"""Capture the oracle llama-server's --api-key behaviour into tests/golden/auth.json.

Usage: python3 -I tools/capture_auth_golden.py [base_url]   (default http://127.0.0.1:8099)
Start the oracle with `--api-key k1,k2` (any model, `--gpu-layers 0` is enough).

Each case records the status, content type and exact body bytes for a request without a key,
with a wrong key, with either accepted key in either header, and on the open paths.
"""
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / "auth.json"
BODY = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}

# name: (method, path, headers)
CASES = {
    "no_key": ("POST", "/v1/messages", {}),
    "wrong_bearer": ("POST", "/v1/messages", {"Authorization": "Bearer nope"}),
    "wrong_x_api_key": ("POST", "/v1/chat/completions", {"x-api-key": "nope"}),
    "bare_authorization": ("GET", "/v1/models", {"Authorization": "k1"}),
    "bearer_k2": ("GET", "/v1/models", {"Authorization": "Bearer k2"}),
    "x_api_key_k1": ("GET", "/v1/models", {"x-api-key": "k1"}),
    # Authorization wins over x-api-key when both are sent.
    "bad_auth_good_x": ("GET", "/v1/models", {"Authorization": "Bearer nope", "x-api-key": "k1"}),
    "health_open": ("GET", "/health", {}),
    "v1_health_open": ("GET", "/v1/health", {}),
    "models_no_key": ("GET", "/models", {}),
    # The web UI at / stays open (llama-server: frontend paths are public).
    "page_open": ("GET", "/", {"Accept-Encoding": "gzip"}),
}


def call(method, path, headers):
    data = json.dumps(BODY).encode() if method == "POST" else None
    h = {"Content-Type": "application/json", **headers} if data else headers
    try:
        with urlopen(Request(BASE + path, data=data, headers=h, method=method), timeout=600) as r:
            return r.status, r.headers.get("Content-Type"), r.read().decode(errors="replace")
    except HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read().decode()


out = {}
for name, (method, path, headers) in CASES.items():
    status, ctype, body = call(method, path, headers)
    # 200 bodies (model lists, generations) are covered by api.json; keep only their status.
    out[name] = {"method": method, "path": path, "headers": headers, "status": status,
                 "content_type": ctype, "body": body if status != 200 else None}
    print(name, status, ctype, (body if status != 200 else "")[:100])
OUT.write_text(json.dumps(out, indent=1) + "\n")
print("wrote", OUT)
