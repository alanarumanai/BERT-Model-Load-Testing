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
#   TARGET       one-worker (default): the busiest worker
#                each: every worker, recorded separately at the same time (one graph each)
#                all:  the master and every worker in one graph (py-spy --subprocesses;
#                      with native frames this can crash py-spy, so prefer "each")
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
# The busiest worker, by CPU used so far (all workers started together and have
# been under load for the settling period). With few users, some workers may get
# no connections at all, and profiling an idle worker would show nothing useful.
WORKER=$(ps -o pid=,pcpu= --ppid "$MASTER" | sort -k2 -nr | awk 'NR==1 {print $1}')
WORKERS=$(ps -o pid=,pcpu= --ppid "$MASTER" | sort -k2 -nr | awk '{print $1}')
ps -o pid=,pcpu= --ppid "$MASTER" | sort -k2 -nr \
  | awk '{printf "  worker %s: %s%% CPU so far\n", $1, $2}' | tee "$OUT/workers_cpu.txt"
# Check py-spy can read a worker before recording, and explain if it can't.
if [[ -n "$WORKER" ]] && ! "${PYSPY[@]}" dump --pid "$WORKER" > "$OUT/dump_check.txt" 2>&1; then
  echo "ERROR: py-spy cannot read worker $WORKER:"
  sed 's/^/  /' "$OUT/dump_check.txt"
  echo "If it mentions permissions, see 'Permissions' in the Phase 7 notes, or run"
  echo "  sudo sysctl kernel.yama.ptrace_scope=0   (until the next reboot)"
  exit 1
fi
echo "Recording $TARGET for ${DURATION}s at ${RATE} samples/s ..."
# py-spy runs on Locust's CPUs, so its own work doesn't slow the service it measures.
record() {  # record <output.svg> <py-spy target arguments...>
  local output=$1
  shift
  taskset -c "$LOCUST_CPUS" "${PYSPY[@]}" record "$@" --rate "$RATE" \
    --duration "$DURATION" "${NATIVE_FLAG[@]}" --output "$output"
}
case "$TARGET" in
  all)
    record "$OUT/flamegraph.svg" --pid "$MASTER" --subprocesses 2>&1 | tee "$OUT/py-spy.log"
    ;;
  each)
    # One py-spy per worker, all at the same time, each without --subprocesses.
    pids=()
    for w in $WORKERS; do
      record "$OUT/flamegraph_worker_$w.svg" --pid "$w" > "$OUT/py-spy_$w.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    cat "$OUT"/py-spy_*.log > "$OUT/py-spy.log"
    grep -h "Wrote flamegraph" "$OUT"/py-spy_*.log || true
    ;;
  *)
    record "$OUT/flamegraph.svg" --pid "$WORKER" 2>&1 | tee "$OUT/py-spy.log"
    ;;
esac
if grep -q -i "segmentation fault\|panicked" "$OUT/py-spy.log" || ! ls "$OUT"/flamegraph*.svg >/dev/null 2>&1; then
  echo "WARNING: py-spy did not produce a flame graph (it may have crashed)."
  echo "         Try TARGET=each (or the default one-worker), or NATIVE=0."
fi
behind=$(grep -c "behind in sampling" "$OUT/py-spy.log" || true)
if (( behind > 0 )); then
  echo "WARNING: py-spy fell behind $behind time(s); the graph may be inaccurate."
  echo "         Rerun with a lower RATE (e.g. RATE=20) or NATIVE=0."
fi

wait "$LOCUST_PID" || true
LOCUST_PID=""
[[ "${PYSPY[0]}" == sudo ]] && sudo chown "$(id -u):$(id -g)" "$OUT"/flamegraph*.svg 2>/dev/null || true
echo "Flame graph(s):"
ls "$OUT"/flamegraph*.svg 2>/dev/null | sed 's/^/  /' || true