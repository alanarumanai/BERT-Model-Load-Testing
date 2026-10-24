#!/usr/bin/env bash
# Install the project's dependencies into the active Python environment.
#
# Usage:  scripts/install_deps.sh         runtime dependencies only
#         scripts/install_deps.sh --dev   runtime + test and lint tools
#
# Used both locally and in CI, so both install exactly the same way.
set -euo pipefail
cd "$(dirname "$0")/.."

REQUIREMENTS=requirements.txt
if [[ "${1:-}" == "--dev" ]]; then
  REQUIREMENTS=requirements-dev.txt
fi

# Outside CI, warn if no virtual environment is active (packages would go into
# the system Python instead).
if [[ -z "${CI:-}" ]] && python -c 'import sys; sys.exit(sys.prefix != sys.base_prefix)'; then
  echo "WARNING: no virtual environment is active. Run 'source .venv/bin/activate' first."
fi

# Install torch first from PyTorch's CPU-only index, at the version pinned in
# requirements.txt (e.g. "torch==2.x.y"), or the latest if it isn't pinned.
# The default PyPI build includes large GPU libraries this CPU service never uses.
# The later `pip install -r` then finds torch already satisfied and keeps it.
TORCH_SPEC=$(grep -iE '^[[:space:]]*torch[[:space:]]*($|[=<>!~])' requirements.txt | tr -d '[:space:]' || true)
python -m pip install "${TORCH_SPEC:-torch}" --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r "$REQUIREMENTS"
