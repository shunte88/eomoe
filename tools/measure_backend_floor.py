"""Measure llama.cpp's own CPU-vs-CUDA disagreement on tests/golden/greedy.json: the
noise floor behind FLOOR_MEAN in tests/phase2_golden.rs.
Start a CPU-only oracle first: llama-server --model <gguf> --port 8099 --gpu-layers 0
Usage: python3 tools/measure_backend_floor.py tests/golden/greedy.json [out.json]"""
import json, sys
from urllib.request import Request, urlopen
g = json.load(open(sys.argv[1]))
def top(toks):
    b = {"prompt": toks, "n_predict": 1, "temperature": 0, "top_k": 1, "n_probs": 10, "cache_prompt": False}
    r = json.loads(urlopen(Request("http://127.0.0.1:8099/completion", json.dumps(b).encode(), {"Content-Type": "application/json"})).read())
    return {e["id"]: e["logprob"] for e in r["completion_probabilities"][0]["top_logprobs"]}
tot = dict(top1=0, top5=0, lp=0, steps=0); worst = 0; out = []; ds = []
for ci, c in enumerate(g):
    cw = 0
    for k, st in enumerate(c["probs"]):
        cpu = top(c["prompt_tokens"] + c["tokens"][:k])
        o = [(e["id"], e["logprob"]) for e in st["top_logprobs"]]
        ids = sorted(cpu, key=cpu.get, reverse=True)
        tot["steps"] += 1
        if ids[0] != o[0][0]: tot["top1"] += 1
        if sorted(ids[:5]) != sorted(i for i, _ in o): tot["top5"] += 1
        for i, l in o:
            d = abs(cpu.get(i, -99) - l) if i in cpu else None
            if d is None or d > 0.05: tot["lp"] += 1
            if d is not None: cw = max(cw, d); ds.append(d)
        out.append({"case": ci, "step": k, "cpu": cpu})
    worst = max(worst, cw); print(f"case {ci}: max |dlogprob| {cw:.4f}", flush=True)
print(tot, f"worst {worst:.4f}, mean |dlogprob| {sum(ds) / len(ds):.4f}  <- FLOOR_MEAN")
if len(sys.argv) > 2: json.dump(out, open(sys.argv[2], "w"))
