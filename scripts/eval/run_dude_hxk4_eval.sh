#!/usr/bin/env bash
set -euo pipefail

# Evaluate a trained HMRLBA checkpoint on DUD-E HXK4.
# Override EXP_NAME, EXP_DIR, DATA_DIR, or OUT_DIR from the shell when needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROT_ROOT="${PROT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PROT_ROOT}/Datasets}"
EXP_DIR="${EXP_DIR:-${PROT_ROOT}/Experiments}"
EXP_NAME="${EXP_NAME:-run-20241124_204606-r94ymd7y}"
OUT_DIR="${OUT_DIR:-${PROT_ROOT}/results/dude_hxk4}"
ENV_NAME="${ENV_NAME:-pyg}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PROT="$PROT_ROOT"
export PYTHONPATH="${PROT_ROOT}:${PYTHONPATH:-}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n "$ENV_NAME" "$PYTHON_BIN")
else
  echo "Please activate the $ENV_NAME environment first, or set PYTHON_BIN to its python executable." >&2
  exit 1
fi

"${PYTHON_CMD[@]}" scripts/train/ablation/run_cascade_ablation.py \
  --mode eval \
  --data-dir "$DATA_DIR" \
  --exp-dir "$EXP_DIR" \
  --exp-name "$EXP_NAME" \
  --eval-dataset dude_hxk4 \
  --eval-mode test \
  --virtual-screening \
  --variants full \
  --out-dir "$OUT_DIR"
