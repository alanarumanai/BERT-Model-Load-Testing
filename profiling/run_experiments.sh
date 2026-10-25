#!/usr/bin/env bash
# Run the Phase 7 experiments back to back, each as its own load-test session with
# exactly one setting changed from the baseline, then compare each with the baseline.
#
# Usage:  bash profiling/run_experiments.sh [EXPERIMENT ...]
#   Default: all experiments below, starting with a fresh baseline.
#   Custom experiments are written as "name|VAR=value VAR=value", for example:
#     bash profiling/run_experiments.sh "quantized-8x1|APP_QUANTIZE=true GUNICORN_WORKERS=8 APP_TORCH_NUM_THREADS=1"
#   A baseline is always included, so every comparison is made in the same sitting.
#
# Settings (environment variables, all optional):
#   SERVER_CPUS    CPUs for the service      (default: 0-7)
#   LOCUST_CPUS    CPUs for Locust           (default: 8-9)
#   USER_LEVELS    step-load levels          (default: "1 8 32")
#   STEP_DURATION  length of each level      (default: 90s)
#   SCENARIOS      load-test scenarios       (default: step)
#
# Output: profiling/results/experiments_<date-time>/ with one session per experiment
# and comparison.md (every experiment against the baseline).
set -euo pipefail
cd "$(dirname "$0")/.."

export SERVER_CPUS="${SERVER_CPUS:-0-7}"
export LOCUST_CPUS="${LOCUST_CPUS:-8-9}"
export USER_LEVELS="${USER_LEVELS:-1 8 32}"
export STEP_DURATION="${STEP_DURATION:-90s}"
SCENARIOS="${SCENARIOS:-step}"

# Settings left over in the shell would silently change every experiment.
leftover=$(env | grep -E '^(GUNICORN_WORKERS|APP_TORCH_NUM_THREADS|APP_QUANTIZE|APP_MAX_LENGTH|APP_MAX_CONCURRENT_PREDICTIONS|APP_LOG_LEVEL|APP_PRECISION|OMP_NUM_THREADS|OMP_WAIT_POLICY)=' || true)
if [[ -n "$leftover" ]]; then
  echo "ERROR: these variables are set in your shell and would affect every experiment:"
  echo "$leftover"
  echo "Run 'unset <NAME>' for each, then try again."
  exit 1
fi

# Number of CPUs in a list like "0-7" or "0-3,8-11".
count_cpus() {
  local total=0 part
  IFS=',' read -ra parts <<< "$1"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then total=$((total + ${part#*-} - ${part%-*} + 1))
    else total=$((total + 1)); fi
  done
  echo "$total"
}
N=$(count_cpus "$SERVER_CPUS")
(( N >= 2 )) || { echo "Experiments need at least 2 server CPUs (SERVER_CPUS=$SERVER_CPUS)"; exit 1; }
HALF=$((N / 2)); QUARTER=$(( N / 4 > 0 ? N / 4 : 1 ))

# name | settings (one change from the baseline: 2 workers, CPUs/2 threads each)
EXPERIMENTS=(
  "baseline|"
  "oversubscribed|GUNICORN_WORKERS=2 APP_TORCH_NUM_THREADS=$N"
  "workers-1x$N|GUNICORN_WORKERS=1 APP_TORCH_NUM_THREADS=$N"
  "workers-4x$QUARTER|GUNICORN_WORKERS=4 APP_TORCH_NUM_THREADS=$QUARTER"
  "workers-${N}x1|GUNICORN_WORKERS=$N APP_TORCH_NUM_THREADS=1"
  "one-at-a-time|APP_MAX_CONCURRENT_PREDICTIONS=1"
  "quantized|APP_QUANTIZE=true"
  "bf16|APP_PRECISION=bf16"
  "fp16|APP_PRECISION=fp16"
  "max-length-64|APP_MAX_LENGTH=64"
  "quiet-logs|APP_LOG_LEVEL=WARNING"
)

SELECTED=()
for arg in "$@"; do
  if [[ "$arg" == *"|"* ]]; then
    # A custom experiment: "name|settings". If it has the same name as a built-in
    # one, it replaces it, so each name runs exactly once.
    kept=()
    for entry in "${EXPERIMENTS[@]}"; do
      [[ "${entry%%|*}" == "${arg%%|*}" ]] || kept+=("$entry")
    done
    EXPERIMENTS=("${kept[@]}" "$arg")
    SELECTED+=("${arg%%|*}")
  else
    SELECTED+=("$arg")
  fi
done
# Drop repeated names, keeping the first occurrence.
unique=()
for name in "${SELECTED[@]}"; do
  [[ " ${unique[*]} " == *" $name "* ]] || unique+=("$name")
done
SELECTED=("${unique[@]}")
names=()
for entry in "${EXPERIMENTS[@]}"; do names+=("${entry%%|*}"); done
if [[ ${#SELECTED[@]} -eq 0 ]]; then
  SELECTED=("${names[@]}")
else
  for name in "${SELECTED[@]}"; do
    [[ " ${names[*]} " == *" $name "* ]] || { echo "Unknown experiment: $name"; echo "Known: ${names[*]}"; exit 1; }
  done
  [[ " ${SELECTED[*]} " == *" baseline "* ]] || SELECTED=(baseline "${SELECTED[@]}")
fi

OUT="profiling/results/experiments_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
echo "Server CPUs: $SERVER_CPUS ($N), baseline: 2 workers x $HALF threads"
echo "Experiments: ${SELECTED[*]}"
echo "Results: $OUT"

failed=()
for name in "${SELECTED[@]}"; do
  for entry in "${EXPERIMENTS[@]}"; do
    [[ "${entry%%|*}" == "$name" ]] || continue
    read -ra settings <<< "${entry#*|}"
    echo
    echo "=== $name: ${settings[*]:-default settings} ==="
    # One failed experiment shouldn't throw away the others: note it and carry on.
    # shellcheck disable=SC2086
    if ! env "${settings[@]}" LABEL="$name" RESULTS_ROOT="$OUT" \
      bash loadtests/run_load_tests.sh $SCENARIOS; then
      echo "WARNING: experiment '$name' failed; continuing with the rest."
      failed+=("$name")
    fi
    break   # each name runs once
  done
done

# Compare every experiment that produced a summary with the baseline.
baseline_session=$(ls -d "$OUT"/*_baseline 2>/dev/null | head -1 || true)
if [[ -z "$baseline_session" || ! -f "$baseline_session/summary.csv" ]]; then
  echo "ERROR: the baseline did not complete, so nothing can be compared."
  exit 1
fi
others=()
for name in "${SELECTED[@]}"; do
  [[ "$name" == baseline ]] && continue
  session=$(ls -d "$OUT"/*_"$name" 2>/dev/null | head -1 || true)
  [[ -n "$session" && -f "$session/summary.csv" ]] && others+=("$session")
done
if [[ ${#others[@]} -gt 0 ]]; then
  python profiling/compare_sessions.py "$baseline_session" "${others[@]}" --output "$OUT/comparison.md"
fi
[[ ${#failed[@]} -eq 0 ]] || echo "Failed experiments (see their server.log): ${failed[*]}"
echo "All experiment results are in $OUT"