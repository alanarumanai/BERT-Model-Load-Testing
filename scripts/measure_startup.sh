#!/usr/bin/env bash
# Start the server, measure how long it takes to become ready and how much memory
# each process uses when idle, then stop it. Results are printed for the report.
#
# Usage:  scripts/measure_startup.sh [CPU_LIST]     e.g. scripts/measure_startup.sh 0-3
set -euo pipefail
cd "$(dirname "$0")/.."

HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"

# A server that is already running would answer the health check instantly and
# make the measurement meaningless (and the new server couldn't bind the port).
if curl -s -o /dev/null --max-time 2 "$HEALTH_URL"; then
  echo "ERROR: something is already answering at $HEALTH_URL."
  echo "Stop the running server (Ctrl+C in its terminal) and run this script again."
  exit 1
fi

LOG="$(mktemp)"

start=$(date +%s.%N)
bash scripts/run_server.sh "${1:-}" >"$LOG" 2>&1 &
MASTER=$!    # run_server.sh uses exec, so this is the Gunicorn master's PID
trap 'kill "$MASTER" 2>/dev/null || true; wait "$MASTER" 2>/dev/null || true; rm -f "$LOG"' EXIT

fail() { echo "ERROR: $1"; echo "--- server log ---"; cat "$LOG"; exit 1; }

# 1. Wait until /health answers 200 (at least one worker has loaded the model).
until curl -sf "$HEALTH_URL" >/dev/null; do
  kill -0 "$MASTER" 2>/dev/null || fail "server exited during startup"
  sleep 0.2
done
first_ready=$(date +%s.%N)

# 2. Wait until every worker has finished starting up ("Application startup complete",
#    logged by Gunicorn/Uvicorn whatever APP_LOG_LEVEL is).
expected=$(grep -oE 'Starting [0-9]+ worker' "$LOG" | grep -oE '[0-9]+' | head -1 || true)
[[ -n "$expected" ]] || fail "could not find the 'Starting N worker(s)' line in the server log"
deadline=$((SECONDS + 300))
while (( $(grep -c 'Application startup complete' "$LOG" || true) < expected )); do
  kill -0 "$MASTER" 2>/dev/null || fail "server exited during startup"
  (( SECONDS < deadline )) || fail "not all workers loaded within 300s"
  sleep 0.2
done
all_ready=$(date +%s.%N)
sleep 2    # let memory settle

echo "=== Startup ==="
grep -m1 'Starting [0-9]* worker' "$LOG" | sed 's/.*\] //'
awk -v s="$start" -v a="$first_ready" -v b="$all_ready" 'BEGIN {
  printf "First worker ready (health 200): %.1f s\nAll workers ready:               %.1f s\n", a-s, b-s }'
echo "Per-worker model load (from the app's INFO logs; empty if APP_LOG_LEVEL hides them):"
grep 'Loaded ' "$LOG" | sed -E 's/.*\[([0-9]+)\].*Loaded .* in ([0-9.]+s).*torch threads=([0-9]+).*/  worker \1: \2, torch threads=\3/'

echo
echo "=== Idle memory (RSS) ==="
ps -o pid=,rss= -p "$MASTER" | awk '{ printf "  master %s: %6.1f MB\n", $1, $2/1024 }'
ps -o pid=,rss= --ppid "$MASTER" | awk '{ printf "  worker %s: %6.1f MB\n", $1, $2/1024; t += $2; n++ }
  END { if (n) printf "  workers total: %.1f MB (%d workers, avg %.1f MB)\n", t/1024, n, t/1024/n }'