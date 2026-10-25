#!/usr/bin/env bash
# Run the load-test scenarios against a freshly started server and save everything
# (Locust results, resource samples, server log, startup time and idle memory,
# environment details, summary) into
# loadtests/results/<date-time>_<label>/.
#
# Usage:  bash loadtests/run_load_tests.sh [SCENARIO ...]
#   Scenarios: step, length, spike, soak. Default: all four, in that order.
#   Example:   SERVER_CPUS=0-3 LOCUST_CPUS=4-5 bash loadtests/run_load_tests.sh step
#
# Settings (environment variables, all optional):
#   LABEL             added to the results folder name         (default: run)
#   SERVER_CPUS       CPUs for the service, e.g. 0-3            (default: all CPUs)
#   LOCUST_CPUS       CPUs for Locust and the monitor, e.g. 4-5 (default: all CPUs)
#   USER_LEVELS       step scenario concurrency levels          (default: "1 2 4 8 16 32 64")
#   STEP_DURATION     length of each step run                   (default: 2m)
#   REPEATS           how many times to run the step scenario   (default: 1)
#   LENGTH_USERS      users for the short/long text runs        (default: 8)
#   LENGTH_DURATION   length of each short/long text run        (default: 2m)
#   SPIKE_BASE_USERS, SPIKE_PEAK_USERS, SPIKE_STAGE_SECONDS     (default: 4, 64, 60)
#   SOAK_USERS        users for the soak run                    (default: 8)
#   SOAK_DURATION     length of the soak run                    (default: 15m)
#   COOLDOWN          idle seconds between runs                 (default: 10)
#   RESULTS_ROOT      folder to create the session in           (default: loadtests/results)
#   Any GUNICORN_* or APP_* variable is passed to the server, e.g. GUNICORN_WORKERS=4.
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${LABEL:-run}"
SERVER_CPUS="${SERVER_CPUS:-}"
LOCUST_CPUS="${LOCUST_CPUS:-}"
USER_LEVELS="${USER_LEVELS:-1 2 4 8 16 32 64}"
STEP_DURATION="${STEP_DURATION:-2m}"
REPEATS="${REPEATS:-1}"
LENGTH_USERS="${LENGTH_USERS:-8}"
LENGTH_DURATION="${LENGTH_DURATION:-2m}"
SPIKE_BASE_USERS="${SPIKE_BASE_USERS:-4}"
SPIKE_PEAK_USERS="${SPIKE_PEAK_USERS:-64}"
SPIKE_STAGE_SECONDS="${SPIKE_STAGE_SECONDS:-60}"
SOAK_USERS="${SOAK_USERS:-8}"
SOAK_DURATION="${SOAK_DURATION:-15m}"
COOLDOWN="${COOLDOWN:-10}"
RESULTS_ROOT="${RESULTS_ROOT:-loadtests/results}"
HOST="${HOST:-http://127.0.0.1:8000}"
SCENARIOS=("$@")
[[ ${#SCENARIOS[@]} -gt 0 ]] || SCENARIOS=(step length spike soak)

for scenario in "${SCENARIOS[@]}"; do
  [[ "$scenario" =~ ^(step|length|spike|soak)$ ]] || { echo "Unknown scenario: $scenario"; exit 1; }
done
command -v locust >/dev/null || { echo "Locust not found. Run: bash scripts/install_deps.sh --dev"; exit 1; }
if curl -s -o /dev/null --max-time 2 "$HOST/health"; then
  echo "ERROR: something is already answering at $HOST. Stop it first, so every run"
  echo "uses a fresh server started with the recorded configuration."
  exit 1
fi

# Locust and the monitor are pinned to LOCUST_CPUS so they don't compete with the service.
PIN_LOCUST=()
[[ -n "$LOCUST_CPUS" ]] && PIN_LOCUST=(taskset -c "$LOCUST_CPUS")

SESSION="$RESULTS_ROOT/$(date +%Y%m%d-%H%M%S)_${LABEL}"
mkdir -p "$SESSION/environment" "$SESSION/runs"
echo "Results folder: $SESSION"

# --- 1. Record the environment, so the results can be reproduced ---------------------
{
  echo "date: $(date -Iseconds)"
  echo "scenarios: ${SCENARIOS[*]}"
  for var in LABEL SERVER_CPUS LOCUST_CPUS USER_LEVELS STEP_DURATION REPEATS LENGTH_USERS \
             LENGTH_DURATION SPIKE_BASE_USERS SPIKE_PEAK_USERS SPIKE_STAGE_SECONDS \
             SOAK_USERS SOAK_DURATION COOLDOWN HOST RESULTS_ROOT; do
    echo "$var=${!var}"
  done
  echo "--- server and model settings from the environment ---"
  env | grep -E '^(GUNICORN_|APP_|HF_|TOKENIZERS_|OMP_|KMP_|GOMP_|MKL_)' | sort || true
} > "$SESSION/environment/settings.txt"
[[ -f .env ]] && grep -E '^[[:space:]]*APP_' .env > "$SESSION/environment/dotenv_app_settings.txt" || true
lscpu > "$SESSION/environment/lscpu.txt" 2>&1 || true
free -h > "$SESSION/environment/memory.txt" 2>&1 || true
cat /etc/os-release > "$SESSION/environment/os-release.txt" 2>&1 || true
uname -a > "$SESSION/environment/kernel.txt"
python --version > "$SESSION/environment/python.txt" 2>&1
python -m pip freeze > "$SESSION/environment/pip-freeze.txt"
{
  git rev-parse HEAD 2>/dev/null || echo "not a git repository"
  git status --short 2>/dev/null || true
} > "$SESSION/environment/git.txt"

# --- 2. Start the server and wait until every worker has loaded the model -------------
# Each worker logs "Application startup complete" once its lifespan (model loading) is
# done. That line comes from Gunicorn/Uvicorn, so it appears whatever APP_LOG_LEVEL is.
server_start=$(date +%s.%N)
bash scripts/run_server.sh "$SERVER_CPUS" > "$SESSION/server.log" 2>&1 &
MASTER=$!   # run_server.sh uses exec, so this is the Gunicorn master's PID
MONITOR=""
cleanup() {
  [[ -n "$MONITOR" ]] && kill "$MONITOR" 2>/dev/null || true
  kill "$MASTER" 2>/dev/null || true
  wait "$MASTER" 2>/dev/null || true
}
trap cleanup EXIT
fail() { echo "ERROR: $1 (see $SESSION/server.log)"; exit 1; }

deadline=$((SECONDS + 300))
until curl -sf "$HOST/health" >/dev/null; do
  kill -0 "$MASTER" 2>/dev/null || fail "server exited during startup"
  (( SECONDS < deadline )) || fail "server not healthy within 300s"
  sleep 0.2
done
first_ready=$(date +%s.%N)
expected=$(grep -oE 'Starting [0-9]+ worker' "$SESSION/server.log" | grep -oE '[0-9]+' | head -1 || true)
while (( $(grep -c 'Application startup complete' "$SESSION/server.log" || true) < ${expected:-1} )); do
  (( SECONDS < deadline )) || fail "not all workers loaded within 300s"
  sleep 0.2
done
all_ready=$(date +%s.%N)
grep -m1 'Starting [0-9]* worker' "$SESSION/server.log" | sed 's/.*\] /Server: /' || true

# Record startup time and idle memory (the same figures as scripts/measure_startup.sh).
sleep 2   # let memory settle
{
  echo "=== Startup ==="
  grep -m1 'Starting [0-9]* worker' "$SESSION/server.log" | sed 's/.*\] //' || true
  awk -v s="$server_start" -v a="$first_ready" -v b="$all_ready" 'BEGIN {
    printf "First worker ready (health 200): %.1f s\nAll workers ready:               %.1f s\n", a-s, b-s }'
  echo "Per-worker model load (from the app's INFO logs; empty if APP_LOG_LEVEL hides them):"
  grep 'Loaded ' "$SESSION/server.log" \
    | sed -E 's/.*\[([0-9]+)\].*Loaded .* in ([0-9.]+s).*torch threads=([0-9]+).*/  worker \1: \2, torch threads=\3/' || true
  echo
  echo "=== Idle memory (RSS) before any load ==="
  ps -o pid=,rss= -p "$MASTER" | awk '{ printf "  master %s: %6.1f MB\n", $1, $2/1024 }'
  ps -o pid=,rss= --ppid "$MASTER" | awk '{ printf "  worker %s: %6.1f MB\n", $1, $2/1024; t += $2; n++ }
    END { if (n) printf "  workers total: %.1f MB (%d workers, avg %.1f MB)\n", t/1024, n, t/1024/n }'
} > "$SESSION/startup.txt"
sed -n 's/^All workers ready: */Startup: all workers ready in /p' "$SESSION/startup.txt"

# --- 3. Warm up: a short untimed run so caches and connections are ready -------------
echo "[$(date +%T)] warm-up: 4 users for 20s (not recorded)"
"${PIN_LOCUST[@]}" locust -f loadtests/locustfile.py --headless --host "$HOST" \
  -u 4 -r 4 -t 20s --only-summary > "$SESSION/warmup.log" 2>&1 || true
sleep "$COOLDOWN"

# --- 4. The runs ----------------------------------------------------------------------
RUN_NUMBER=0

# run <name> <users|shape> <duration> [VAR=value ...]
run() {
  local name=$1 users=$2 duration=$3
  shift 3
  RUN_NUMBER=$((RUN_NUMBER + 1))
  local dir
  dir="$SESSION/runs/$(printf '%02d' "$RUN_NUMBER")_$name"
  mkdir -p "$dir"
  kill -0 "$MASTER" 2>/dev/null || fail "server is no longer running"

  local load_args=(-u "$users" -r "$users" -t "$duration" --reset-stats)
  if [[ "$users" == "shape" ]]; then
    load_args=()   # the load shape in the locustfile controls users and duration
  fi
  local text_mix="mixed"
  for setting in "$@"; do [[ "$setting" == LOAD_TEXT_MIX=* ]] && text_mix="${setting#*=}"; done

  echo "[$(date +%T)] $(basename "$dir"): users=$users duration=$duration texts=$text_mix"
  local started
  started=$(date -Iseconds)

  # Locust runs in the background so the monitor can record its CPU use too.
  # -r equal to -u starts all users within a second; --reset-stats then discards
  # the statistics from that start-up second.
  env "$@" "${PIN_LOCUST[@]}" locust -f loadtests/locustfile.py --headless --host "$HOST" \
    "${load_args[@]}" --stop-timeout 10 \
    --csv "$dir/locust" --csv-full-history --html "$dir/report.html" \
    --only-summary > "$dir/locust.log" 2>&1 &
  local locust_pid=$!
  "${PIN_LOCUST[@]}" python loadtests/monitor.py --server-pid "$MASTER" \
    --locust-pid "$locust_pid" --output "$dir/resources.csv" &
  MONITOR=$!

  local exit_code=0
  wait "$locust_pid" || exit_code=$?   # Locust exits with 1 if any request failed
  kill "$MONITOR" 2>/dev/null || true
  wait "$MONITOR" 2>/dev/null || true
  MONITOR=""

  cat > "$dir/run.json" <<JSON
{
  "name": "$name",
  "users": "$users",
  "duration": "$duration",
  "text_mix": "$text_mix",
  "settings": "$*",
  "started": "$started",
  "finished": "$(date -Iseconds)",
  "locust_exit_code": $exit_code
}
JSON
  sleep "$COOLDOWN"
}

for scenario in "${SCENARIOS[@]}"; do
  case "$scenario" in
    step)
      for repeat in $(seq 1 "$REPEATS"); do
        suffix=""
        (( REPEATS > 1 )) && suffix="_r$repeat"
        for users in $USER_LEVELS; do
          run "step${suffix}_${users}u" "$users" "$STEP_DURATION" LOAD_TEXT_MIX=mixed
        done
      done
      ;;
    length)
      for mix in short long; do
        run "length_${mix}_${LENGTH_USERS}u" "$LENGTH_USERS" "$LENGTH_DURATION" LOAD_TEXT_MIX="$mix"
      done
      ;;
    spike)
      run "spike_${SPIKE_BASE_USERS}-${SPIKE_PEAK_USERS}u" shape "$((3 * SPIKE_STAGE_SECONDS))s" \
        LOAD_SHAPE=spike LOAD_TEXT_MIX=mixed SPIKE_BASE_USERS="$SPIKE_BASE_USERS" \
        SPIKE_PEAK_USERS="$SPIKE_PEAK_USERS" SPIKE_STAGE_SECONDS="$SPIKE_STAGE_SECONDS"
      ;;
    soak)
      run "soak_${SOAK_USERS}u" "$SOAK_USERS" "$SOAK_DURATION" LOAD_TEXT_MIX=mixed
      ;;
  esac
done

# --- 5. Summarize ---------------------------------------------------------------------
echo
python loadtests/summarize.py "$SESSION"
echo "All results saved in $SESSION"