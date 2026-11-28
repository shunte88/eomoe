#!/usr/bin/env python3
"""Same-machine engine shoot-out: eomoe vs llama.cpp (tuned) vs RustingStrata, each serving the
same GGUF, the same prompt and the same client requests (OpenAI chat, streamed, greedy).

    python3 -I tools/bench.py --model M.gguf [--engines eomoe,llama,strata] [--loads 2] [--runs 5]
        [--llama-flags "-ncmoe 30" --llama-flags "--cpu-moe --moe-cache-mib 4000"] [--out bench.json]

Method (after RustingStrata's docs/BENCHMARKS.md rules):
- One engine at a time, fenced: systemd-run --scope MemoryMax (llama-server 18G, others 26G), swap off.
  llama-server always gets -ngl 99 (never a CPU-only run).
- Loads alternate between engines (A B C A B C ...). Each load: a warm-up request (discarded), then
  --runs requests with a ~4K-token prompt made unique by a leading nonce, so no prompt cache helps.
- Prefill tok/s = prompt_tokens / time to first token. Decode tok/s = (completion_tokens - 1) /
  (last token - first token). Thinking off, temperature 0, max_tokens 128.
- Reports mean and range per engine (and per llama flag set), and writes every run to --out.
"""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--engines", default="eomoe,llama,strata")
ap.add_argument("--loads", type=int, default=2)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--ctx", type=int, default=8192)
ap.add_argument("--prompt-chars", type=int, default=16000, help="of tests/golden/long_prompt.txt (~4K tokens)")
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--llama-flags", action="append", help="one llama-server config per flag; default: -ot exps=CPU")
ap.add_argument("--eomoe", default=str(ROOT / "target/release/eomoe"))
ap.add_argument("--llama", default=str(HOME / ".local/share/local-code-llm/build/bin/llama-server"))
ap.add_argument("--strata", default=str(HOME / ".local/share/eomoe/bench/rustingstrata/rustingstrata"))
ap.add_argument("--port", type=int, default=8197)
ap.add_argument("--out", default="bench.json")
a = ap.parse_args()

PORT, M = a.port, os.path.expanduser(a.model)
TEXT = (ROOT / "tests/golden/long_prompt.txt").read_text()[: a.prompt_chars]


def configs():
    """(label, process name, MemoryMax, argv) per engine config."""
    out = []
    for e in a.engines.split(","):
        if e == "eomoe":
            out.append(("eomoe", "eomoe", "26G", [a.eomoe, "serve", "--model", M, "--ctx-size", str(a.ctx), "--port", str(PORT)]))
        elif e == "llama":
            for f in a.llama_flags or ["-ot exps=CPU"]:
                out.append((f"llama.cpp {f}", "llama-server", "18G", [a.llama, "-m", M, "-ngl", "99", "-fa", "on", "-c", str(a.ctx),
                            "--port", str(PORT), "--jinja", "--alias", "local-code", *f.split()]))
        elif e == "strata":
            out.append(("rustingstrata", "rustingstrata", "26G", [a.strata, "serve", "--model", M, "--gpu", "--ctx", str(a.ctx),
                        "--mtp", "0", "--addr", f"127.0.0.1:{PORT}"]))
        else:
            sys.exit(f"unknown engine {e}")
    return out


def gone(name):
    return subprocess.run(["pgrep", "-x", name], capture_output=True).returncode != 0


def start(name, mem, argv, log):
    if not gone(name):
        sys.exit(f"{name} already running; stop it first")
    p = subprocess.Popen(["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem}", "-p", "MemorySwapMax=0", *argv],
                         stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        if p.poll() is not None:
            print(f"{name} exited at start, skipped (see {log.name})", flush=True)
            return None
        try:
            urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2).read()
            return p
        except OSError:
            time.sleep(1)
    stop(p, name)
    sys.exit(f"{name} not ready after 10 min")


def stop(p, name):
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-x", name])
    while not gone(name):
        time.sleep(1)


def ask(prompt):
    body = {"model": "local-code", "temperature": 0, "max_tokens": a.max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    t0, first, last, n, usage = time.time(), None, None, 0, None
    req = Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            d = (ev.get("choices") or [{}])[0].get("delta", {})
            if d.get("content") or d.get("reasoning_content"):
                last = time.time()
                first = first or last
                n += 1
    if not usage or first is None:
        raise RuntimeError(f"no tokens or usage in the stream: {usage}")
    nc = usage.get("completion_tokens") or n
    return {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": nc,
            "prefill": usage["prompt_tokens"] / (first - t0),
            "decode": (nc - 1) / (last - first) if last > first else 0.0}


def summary(rows, key):
    v = [r[key] for r in rows] or [0.0]
    return f"{sum(v) / len(v):7.1f} ({min(v):.1f}-{max(v):.1f})"


cfgs = configs()
runs = {c[0]: [] for c in cfgs}
logdir = Path(a.out).resolve().parent
for load in range(a.loads):
    for label, name, mem, argv in cfgs:
        with open(logdir / f"bench-{name}.log", "a") as log:
            p = start(name, mem, argv, log)
            if p is None:
                continue
            try:
                ask("Say hi.")
                for i in range(a.runs):
                    r = ask(f"[run {load}.{i} {time.time()}] Summarize this code in two sentences.\n\n{TEXT}")
                    r["load"] = load
                    runs[label].append(r)
                    print(f"{label:28} load {load} run {i}: prompt {r['prompt_tokens']}, prefill {r['prefill']:.1f}, "
                          f"decode {r['decode']:.1f} tok/s ({r['completion_tokens']} tok)", flush=True)
            except (OSError, RuntimeError) as e:
                print(f"{label}: load {load} failed ({e}), see {log.name}", flush=True)
            finally:
                stop(p, name)

print(f"\n{Path(M).name}, {a.loads} loads x {a.runs} runs, mean (range), tok/s")
print(f"{'engine':28} {'prefill':>22} {'decode':>22}")
for label, rows in runs.items():
    print(f"{label:28} {summary(rows, 'prefill'):>22} {summary(rows, 'decode'):>22}")
Path(a.out).write_text(json.dumps({"model": M, "args": vars(a), "runs": runs}, indent=1))
