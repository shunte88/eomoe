#!/usr/bin/env bash
# Weekly GPU soak on media2026 (.github/workflows/soak.yml, or by hand): every long-form gate,
# then a serving stress run. Each step is memory-fenced so an overrun kills the step, not the box.
# NEVER runs tests/phase11b_kvcache.rs: it hard-locked this machine twice.
# Env: STRESS_MINUTES (30), SOAK_OUT (soak-logs), EOMOE_TEST_* model overrides (tests/common).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${SOAK_OUT:-soak-logs}
mkdir -p "$OUT"
for p in eomoe llama-server; do
    if pgrep -x "$p" >/dev/null; then echo "::error::$p is running: the soak needs the GPU and RAM to itself"; exit 1; fi
done
fence() { systemd-run --user --scope -q -p MemoryMax=26G -p MemorySwapMax=0 "$@"; }
fence cargo build --release --locked || exit 1

fails=()
gate() {
    echo "== $1"
    if fence cargo test --release --locked "${@:2}" -- --include-ignored --test-threads=1 >"$OUT/$1.log" 2>&1; then
        grep "^test result" "$OUT/$1.log" | tail -1
    else
        fails+=("$1"); tail -30 "$OUT/$1.log"
    fi
}
gate lib --lib
for t in tests/*.rs; do
    n=$(basename "$t" .rs)
    [[ $n == *11b* ]] && continue
    gate "$n" --test "$n"
done

echo "== stress (${STRESS_MINUTES:-30} min)"
fence target/release/eomoe serve --port 8095 >"$OUT/stress-server.log" 2>&1 &
for _ in $(seq 180); do curl -sf localhost:8095/health >/dev/null && break; sleep 2; done
if python3 -I tools/stress.py --port 8095 --minutes "${STRESS_MINUTES:-30}" --pid "$(pgrep -x eomoe)" >"$OUT/stress.log" 2>&1; then
    tail -5 "$OUT/stress.log"
else
    fails+=(stress); tail -30 "$OUT/stress.log"
fi
pkill -x eomoe
for _ in $(seq 60); do pgrep -x eomoe >/dev/null || break; sleep 1; done

if [ ${#fails[@]} -gt 0 ]; then echo "::error::soak failures: ${fails[*]}"; exit 1; fi
echo "soak: all green"
