#!/usr/bin/env python3
"""Greedy oracle for gpt-oss-20b: prompt tokens -> 32 greedy tokens with top-5 logprobs, from a
running llama-server (GPU, experts on CPU), into tests/golden/greedy_gpt-oss.json.

  llama-server -m gpt-oss-20b-MXFP4.gguf -ngl 99 -ot exps=CPU -c 4096 --port 8099 --jinja
  python3 -I tools/capture_greedy_gptoss.py [base_url]

Chat prompts go through the server's own template; two are longer than the 128-token window.
"""
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / "greedy_gpt-oss.json"

FILLER = ("The weather that spring was mild, with long grey mornings that burned off into bright afternoons. "
          "Farmers in the valley planted barley and beans, and the river ran high with snowmelt from the hills. "
          "In the town, the bakery opened an hour earlier than usual, and the library extended its hours on Thursdays. "
          "Children walked to school along the canal path, counting herons and arguing about football. "
          "By May the orchards were in blossom and the market square filled with stalls selling cheese, honey and wool.")
CHATS = [
    "What is the capital of France? Answer in one word.",
    "Write a Python function that reverses a linked list.",
    f"Remember this: the vault code is 4817. {FILLER} {FILLER}\n\nWhat is the vault code?",
]
RAW = ["def fib(n):\n", f"{FILLER} {FILLER} {FILLER}"]


def post(path, body):
    req = Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def main():
    prompts = [post("/apply-template", {"messages": [{"role": "user", "content": c}]})["prompt"] for c in CHATS] + RAW
    out = []
    for p in prompts:
        toks = post("/tokenize", {"content": p, "add_special": False, "parse_special": True})["tokens"]
        r = post("/completion", {"prompt": toks, "n_predict": 32, "temperature": 0, "top_k": 1,
                                 "n_probs": 5, "cache_prompt": False, "return_tokens": True})
        out.append({"prompt": p, "prompt_tokens": toks, "tokens": r["tokens"], "content": r["content"],
                    "probs": r["completion_probabilities"]})
        print(f"{len(toks)} tokens: {p[-40:]!r} -> {r['content'][:60]!r}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))


main()
