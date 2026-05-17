#!/usr/bin/env bash
set -euo pipefail

# Prepare graph caches for the DUD-E HXK4 virtual-screening dataset.
# Run this from the HMRLBA repository root after setting PROT if needed.

PROT_ROOT="${PROT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROT_ROOT}/Datasets}"
DATASET="${DATASET:-dude_hxk4}"
PROT_MODE="${PROT_MODE:-surface2backbone}"
PLMS="${PLMS:-ankh esm1b prottrans}"
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

for plm in $PLMS; do
  "${PYTHON_CMD[@]}" scripts/preprocess/prepare_graphs.py \
    --dataset "$DATASET" \
    --data_dir "$DATA_DIR" \
    --prot_mode "$PROT_MODE" \
    --plm "$plm"
done
