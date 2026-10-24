#!/usr/bin/env bash
# Start the service under Gunicorn.
#
# Usage:  scripts/run_server.sh [CPU_LIST]
#   scripts/run_server.sh          use all CPUs
#   scripts/run_server.sh 0-3      pin the service to CPUs 0-3 (via taskset)
#
# Run `python scripts/download_model.py` once first. Activate the virtual
# environment before running. Any GUNICORN_* or APP_* variable can be set in
# front of the command, e.g.  GUNICORN_WORKERS=4 scripts/run_server.sh 0-3
set -euo pipefail
cd "$(dirname "$0")/.."    # project root, so .env and gunicorn.conf.py are found

# Load the model from the local cache only; never contact the Hub at runtime.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
APP_MODULE="${APP_MODULE:-app.main:app}"

if [[ -n "${1:-}" ]]; then
  exec taskset -c "$1" gunicorn -c gunicorn.conf.py "$APP_MODULE"
else
  exec gunicorn -c gunicorn.conf.py "$APP_MODULE"
fi
