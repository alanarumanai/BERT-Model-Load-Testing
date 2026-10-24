#!/usr/bin/env bash
# Record a py-spy flame graph of the Gunicorn workers while Locust keeps them busy.
#
# Usage:  bash profiling/profile_server.sh [LABEL]
#   e.g.  bash profiling/profile_server.sh baseline
#         GUNICORN_WORKERS=2 APP_TORCH_NUM_THREADS=8 bash profiling/profile_server.sh oversubscribed
#
# Settings (environment variables, all optional):
#   SERVER_CPUS  CPUs for the service      (default: 0-7)
#   LOCUST_CPUS  CPUs for Locust           (default: 8-9)
#   USERS        concurrent users          (default: 16)
#   DURATION     seconds to record         (default: 60)
#   RATE         samples per second        (default: 50)
#   NATIVE       1 to include C/C++ frames inside PyTorch, 0 for Python only (default: 1)
#   TARGET       one-worker (default) or all (the master and every worker)
#   Workers all do the same work, so one is representative and halves py-spy's work.
#   If py-spy reports it is "behind in sampling", lower RATE or set NATIVE=0.
#   Any GUNICORN_* or APP_* variable is passed to the server.
#
# Output: profiling/results/<date-time>_flame_<label>/ with flamegraph.svg (open it in a
# browser), the Locust statistics for the profiled period, the server log and settings.
# The Locust numbers here are slightly slowed by profiling: use them for context only,
# never for before/after comparisons.
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:-baseline}"
SERVER_CPUS="${SERVER_CPUS:-0-7}"
LOCUST_CPUS="${LOCUST_CPUS:-8-9}"
USERS="${USERS:-16}"
DURATION="${DURATION:-60}"
RATE="${RATE:-50}"
NATIVE="${NATIVE:-1}"
TARGET="${TARGET:-one-worker}"
HOST="http://127.0.0.1:8000"

command -v py-spy >/dev/null || { echo "py-spy not found. Run: bash scripts/install_deps.sh --dev"; exit 1; }
if curl -s -o /dev/null --max-time 2 "$HOST/health"; then
  echo "ERROR: something is already answering at $HOST. Stop it first."; exit 1
fi

# Attaching to another process needs permission, controlled by kernel.yama.ptrace_scope:
#   0  any process of the same user may attach: nothing to do
#   1  only a parent may attach, unless the target allows it: the server is started
#      with ALLOW_PROFILER_ATTACH=1, so each Gunicorn process allows it (see
#      gunicorn.conf.py) and py-spy runs as you, without sudo. This also works in
#      containers, where even root is often not allowed to attach.
#   2+ only administrators may attach: py-spy is run with sudo.
PYSPY=(py-spy)
scope=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo 0)
export ALLOW_PROFILER_ATTACH=1
if [[ $EUID -ne 0 && "$scope" -ge 2 ]]; then
  echo "ptrace_scope=$scope: py-spy needs sudo to attach (you may be asked for your password)."
  PYSPY=(sudo "$(command -v py-spy)")
  sudo -v
fi

OUT="profiling/results/$(date +%Y%m%d-%H%M%S)_flame_${LABEL}"
mkdir -p "$OUT"
{
  echo "label=$LABEL SERVER_CPUS=$SERVER_CPUS LOCUST_CPUS=$LOCUST_CPUS USERS=$USERS"
  echo "DURATION=$DURATION RATE=$RATE NATIVE=$NATIVE TARGET=$TARGET"
  env | grep -E '^(GUNICORN_|APP_|TOKENIZERS_|OMP_|KMP_|GOMP_|MKL_)' | sort || true
} > "$OUT/settings.txt"

bash scripts/run_server.sh "$SERVER_CPUS" > "$OUT/server.log" 2>&1 &
MASTER=$!
LOCUST_PID=""
cleanup() {
  [[ -n "$LOCUST_PID" ]] && kill "$LOCUST_PID" 2>/dev/null || true
  kill "$MASTER" 2>/dev/null || true
  wait "$MASTER" 2>/dev/null || true
}
trap cleanup EXIT

deadline=$((SECONDS + 300))
until curl -sf "$HOST/health" >/dev/null; do
  kill -0 "$MASTER" 2>/dev/null || { echo "Server exited; see $OUT/server.log"; exit 1; }
  (( SECONDS < deadline )) || { echo "Server not healthy in time"; exit 1; }
  sleep 1
done
expected=$(grep -oE 'Starting [0-9]+ worker' "$OUT/server.log" | grep -oE '[0-9]+' | head -1 || true)
while (( $(grep -c 'Application startup complete' "$OUT/server.log" || true) < ${expected:-1} )); do sleep 1; done
grep -m1 'Starting [0-9]* worker' "$OUT/server.log" | sed 's/.*\] /Server: /' || true

# Load for longer than the recording, so the service is busy for the whole of it.
echo "Starting Locust with $USERS users"
taskset -c "$LOCUST_CPUS" locust -f loadtests/locustfile.py --headless --host "$HOST" \
  -u "$USERS" -r "$USERS" -t "$((DURATION + 25))s" --only-summary \
  --csv "$OUT/locust" > "$OUT/locust.log" 2>&1 &
LOCUST_PID=$!
sleep 15   # let the load settle before recording

NATIVE_FLAG=()
[[ "$NATIVE" == "1" ]] && NATIVE_FLAG=(--native)
# Check py-spy can read one worker before recording, and explain if it can't.
WORKER=$(pgrep -P "$MASTER" | head -1 || true)
if [[ -n "$WORKER" ]] && ! "${PYSPY[@]}" dump --pid "$WORKER" > "$OUT/dump_check.txt" 2>&1; then
  echo "ERROR: py-spy cannot read worker $WORKER:"
  sed 's/^/  /' "$OUT/dump_check.txt"
  echo "If it mentions permissions, see 'Permissions' in the Phase 7 notes, or run"
  echo "  sudo sysctl kernel.yama.ptrace_scope=0   (until the next reboot)"
  exit 1
fi
if [[ "$TARGET" == "all" ]]; then
  TARGET_ARGS=(--pid "$MASTER" --subprocesses)  # follow the master into every worker
else
  TARGET_ARGS=(--pid "$WORKER")
fi
echo "Recording $TARGET for ${DURATION}s at ${RATE} samples/s ..."
# py-spy runs on Locust's CPUs, so its own work doesn't slow the service it measures.
taskset -c "$LOCUST_CPUS" "${PYSPY[@]}" record "${TARGET_ARGS[@]}" --rate "$RATE" \
  --duration "$DURATION" "${NATIVE_FLAG[@]}" --output "$OUT/flamegraph.svg" 2>&1 \
  | tee "$OUT/py-spy.log"
behind=$(grep -c "behind in sampling" "$OUT/py-spy.log" || true)
if (( behind > 0 )); then
  echo "WARNING: py-spy fell behind $behind time(s); the graph may be inaccurate."
  echo "         Rerun with a lower RATE (e.g. RATE=20) or NATIVE=0."
fi

wait "$LOCUST_PID" || true
LOCUST_PID=""
[[ "${PYSPY[0]}" == sudo && -f "$OUT/flamegraph.svg" ]] && sudo chown "$(id -u):$(id -g)" "$OUT/flamegraph.svg" || true
echo "Flame graph: $OUT/flamegraph.svg"