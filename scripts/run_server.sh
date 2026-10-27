#!/usr/bin/env bash
# Start the service under Gunicorn.
#
# Usage:  [PROFILE=throughput|accuracy] scripts/run_server.sh [CPU_LIST]
#   scripts/run_server.sh          use all CPUs
#   scripts/run_server.sh 0-3      pin the service to CPUs 0-3 (via taskset)
#
# Run `python scripts/download_model.py` once first. Activate the virtual
# environment before running. Any GUNICORN_* or APP_* variable can be set in
# front of the command, e.g.  GUNICORN_WORKERS=4 scripts/run_server.sh 0-3
set -euo pipefail
cd "$(dirname "$0")/.."    # project root, so .env and gunicorn.conf.py are found

# Optional settings profile, e.g. PROFILE=throughput loads profiles/throughput.env.
# Variables already set in the environment take precedence over the profile.
if [[ -n "${PROFILE:-}" ]]; then
  profile_file="profiles/${PROFILE}.env"
  [[ -f "$profile_file" ]] || { echo "Unknown profile: $PROFILE (no $profile_file)"; exit 1; }
  while IFS='=' read -r key value; do
    [[ -z "$key" || "$key" == \#* ]] && continue
    [[ -n "${!key:-}" ]] || export "$key=$value"
  done < "$profile_file"
  echo "Using profile $PROFILE ($profile_file)"
fi

# Load the model from the local cache only; never contact the Hub at runtime.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
APP_MODULE="${APP_MODULE:-app.main:app}"

if [[ -n "${1:-}" ]]; then
  exec taskset -c "$1" gunicorn -c gunicorn.conf.py "$APP_MODULE"
else
  exec gunicorn -c gunicorn.conf.py "$APP_MODULE"
fi