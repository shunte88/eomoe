#!/usr/bin/env python3
"""llama.cpp's own backend floor on gpt-oss: replay tests/golden/greedy_gpt-oss.json teacher-forced
against a llama-server in a different backend configuration and compare it with the golden, the
same way tests/common::greedy_floor_gate compares eomoe (top-1 vs oracle, mean |dlogprob| over the
oracle's top 5).

  python3 -I tools/capture_floor_gptoss.py [base_url]
"""
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden" / "greedy_gpt-oss.json"


def post(path, body):
    req = Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=900) as r:
        return json.loads(r.read())


fails, d, worst, steps = [], [], 0.0, 0
for ci, case in enumerate(json.loads(GOLDEN.read_text())):
    for k, step in enumerate(case["probs"]):
        r = post("/completion", {"prompt": case["prompt_tokens"] + case["tokens"][:k], "n_predict": 1,
                                 "temperature": 0, "top_k": 1, "n_probs": 20, "cache_prompt": True})
        mine = {e["id"]: e["logprob"] for e in r["completion_probabilities"][0]["top_logprobs"]}
        top = step["top_logprobs"]
        steps += 1
        best = max(mine, key=mine.get)
        if best != top[0]["id"]:
            fails.append(f"case {ci} step {k}: {best} vs golden {top[0]['id']} (golden margin {top[0]['logprob'] - top[1]['logprob']:.3f})")
        for e in top:
            x = abs(mine.get(e["id"], min(mine.values())) - e["logprob"])
            d.append(x)
            worst = max(worst, x)
print(f"top1 differs {len(fails)}/{steps}, mean |dlogprob| {sum(d) / len(d):.4f}, max {worst:.4f}")
print("\n".join(fails))
