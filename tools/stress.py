#!/usr/bin/env python3
"""Serving stress for tools/soak.sh: concurrent clients replay the captured oracle request shapes
(tests/golden/api.json: chat/messages/responses, text and tools, streamed and not) against a
running `eomoe serve` for N minutes. Fails on any non-200, unreadable or empty response, a failed
health check, or server RSS growing more than 1 GiB (a leak).

  python3 -I tools/stress.py --port 8095 --minutes 30 --pid <eomoe pid> [--clients 4]
"""
import argparse, json, random, threading, time
from pathlib import Path
from urllib.request import Request, urlopen

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8095)
ap.add_argument("--minutes", type=float, default=30)
ap.add_argument("--pid", type=int, required=True)
ap.add_argument("--clients", type=int, default=4)
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"
CASES = [(c["path"], c["request"]) for c in json.load(open(Path(__file__).resolve().parent.parent / "tests/golden/api.json"))["cases"].values()]
end = time.time() + a.minutes * 60
lock, stats, errors, lat = threading.Lock(), {"ok": 0}, [], []


def rss():
    for line in open(f"/proc/{a.pid}/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024


def one(rng):
    path, body = rng.choice(CASES)
    body = dict(body, stream=rng.random() < 0.5)
    t = time.time()
    with urlopen(Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900) as r:
        data = r.read()
        assert r.status == 200, r.status
    assert data.strip(), "empty body"
    if not body["stream"]:
        json.loads(data)
    return time.time() - t


def worker(i):
    rng = random.Random(i)
    while time.time() < end:
        try:
            dt = one(rng)
            with lock:
                stats["ok"] += 1
                lat.append(dt)
        except Exception as e:
            with lock:
                errors.append(repr(e))


rss0 = rss()
ts = [threading.Thread(target=worker, args=(i,)) for i in range(a.clients)]
for t in ts:
    t.start()
while any(t.is_alive() for t in ts):
    time.sleep(30)
    try:
        assert urlopen(BASE + "/health", timeout=60).status == 200
    except Exception as e:
        errors.append(f"health: {e!r}")
    print(f"{time.strftime('%H:%M:%S')} ok {stats['ok']} errors {len(errors)} rss {rss() / 2**30:.2f} GiB", flush=True)
growth = rss() - rss0
lat.sort()
print(f"requests {stats['ok']}, errors {len(errors)}, p50 {lat[len(lat) // 2] if lat else 0:.1f}s, "
      f"max {lat[-1] if lat else 0:.1f}s, rss growth {growth / 2**20:.0f} MiB")
for e in errors[:10]:
    print("error:", e)
raise SystemExit(1 if errors or growth > 2**30 or not lat else 0)
