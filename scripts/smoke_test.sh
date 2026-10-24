#!/usr/bin/env bash
# Smoke test: start the real service under Gunicorn, check that it becomes
# healthy, answers a prediction correctly and rejects bad input, then stop it.
#
# Usage:  scripts/smoke_test.sh
# Exits with status 0 on success and 1 on failure (printing the server log).
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"

if curl -s -o /dev/null --max-time 2 "$BASE_URL/health"; then
  echo "ERROR: something is already answering at $BASE_URL. Stop it and try again."
  exit 1
fi

LOG="$(mktemp)"
bash scripts/run_server.sh >"$LOG" 2>&1 &
MASTER=$!    # run_server.sh uses exec, so this is the Gunicorn master's PID
trap 'kill "$MASTER" 2>/dev/null || true; wait "$MASTER" 2>/dev/null || true; rm -f "$LOG"' EXIT

fail() { echo "SMOKE TEST FAILED: $1"; echo "--- server log ---"; cat "$LOG"; exit 1; }

# 1. The service must become healthy within the timeout.
deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl -sf "$BASE_URL/health" >/dev/null; do
  kill -0 "$MASTER" 2>/dev/null || fail "server exited during startup"
  (( SECONDS < deadline )) || fail "not healthy within ${STARTUP_TIMEOUT}s"
  sleep 1
done
grep -m1 'Starting [0-9]* worker' "$LOG" | sed 's/.*\] //' || true
echo "Health:     $(curl -s "$BASE_URL/health")"

# 2. A clearly positive sentence must come back POSITIVE with HTTP 200.
response=$(curl -s -w '\n%{http_code}' -X POST "$BASE_URL/predict" \
  -H 'Content-Type: application/json' \
  -d '{"text": "This movie was absolutely wonderful, I loved it."}')
status="${response##*$'\n'}"
body="${response%$'\n'*}"
[[ "$status" == "200" ]] || fail "/predict returned HTTP $status: $body"
[[ "$body" == *'"label":"POSITIVE"'* ]] || fail "unexpected prediction: $body"
echo "Prediction: $body"

# 3. Invalid input must be rejected with HTTP 422.
status=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/predict" \
  -H 'Content-Type: application/json' -d '{"text": ""}')
[[ "$status" == "422" ]] || fail "empty text returned HTTP $status, expected 422"
echo "Validation: empty text rejected with 422"

echo "Smoke test passed"
