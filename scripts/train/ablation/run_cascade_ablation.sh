#!/usr/bin/env bash
set -euo pipefail

# Non-invasive HMRLBA PLM-stream cascade ablation launcher.
# Extra arguments are passed through to run_cascade_ablation.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROT_ROOT="${PROT:-/root/HMRLBA_V1.0}"

export PROT="$PROT_ROOT"
export PYTHONPATH="${PROT_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_NAME="${ENV_NAME:-pyg}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n "$ENV_NAME" "$PYTHON_BIN")
else
  echo "Please activate the $ENV_NAME environment first, or set PYTHON_BIN to its python executable." >&2
  exit 1
fi

"${PYTHON_CMD[@]}" "$SCRIPT_DIR/run_cascade_ablation.py" "$@"
