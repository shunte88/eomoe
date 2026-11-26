"""Aggregate decode throughput with N concurrent greedy chat requests, against any
llama-server-compatible endpoint (the oracle or eomoe), so both are measured the same way.

    python3 -I tools/bench_concurrent.py http://127.0.0.1:8099 4 [max_tokens]

Prompts are fixed coding asks that produce long answers. Prints per-request tokens and the
aggregate: sum(completion_tokens) / wall time (prefill of the short prompts included).
Prints one JSON line at the end for scripts.
"""
import json
import sys
import threading
import time
from urllib.request import Request, urlopen

BASE = sys.argv[1]
N = int(sys.argv[2])
MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 128
ASKS = [
    "Write a Python module implementing a thread-safe LRU cache with TTL expiry, with docstrings.",
    "Write a Rust function that parses an INI file into a HashMap, with error handling and tests.",
    "Write a bash script that backs up a directory with rotation of the last 7 archives.",
    "Write a TypeScript class for a debounced event emitter, with JSDoc comments and examples.",
    "Write a Go HTTP server with graceful shutdown and a /health endpoint, fully commented.",
    "Write a C function that reverses a singly linked list in place, plus a test main().",
    "Write a SQL schema for a blog (users, posts, comments, tags) with indexes and comments.",
    "Write a Python asyncio crawler limited to 5 concurrent requests, with retries.",
]


def ask(i, out):
    body = {"model": "local-code", "temperature": 0, "max_tokens": MAX,
            "messages": [{"role": "user", "content": ASKS[i % len(ASKS)]}]}
    t0 = time.time()
    with urlopen(Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                         {"Content-Type": "application/json"}), timeout=900) as r:
        b = json.loads(r.read())
    out[i] = (b["usage"]["completion_tokens"], time.time() - t0, b["choices"][0]["message"]["content"])


out = [None] * N
ths = [threading.Thread(target=ask, args=(i, out)) for i in range(N)]
t0 = time.time()
for t in ths:
    t.start()
for t in ths:
    t.join()
wall = time.time() - t0
toks = sum(o[0] for o in out)
for i, (n, dt, _) in enumerate(out):
    print(f"req {i}: {n} tok in {dt:.1f} s ({n / dt:.1f} tok/s)")
print(f"N={N}: {toks} tok in {wall:.1f} s = {toks / wall:.1f} tok/s aggregate")
print(json.dumps({"n": N, "tokens": toks, "wall": wall, "agg": toks / wall, "texts": [o[2] for o in out]}))
