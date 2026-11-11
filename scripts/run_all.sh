#!/usr/bin/env bash
# Reproduce every result and figure: download -> tests -> cohort report -> E1 -> E2 -> E4 -> E5 -> E3 -> E6 -> figures.
# Ensembles whose checkpoints already exist in checkpoints/ are loaded instead of retrained.
# Usage: bash scripts/run_all.sh   (set DEVICE to choose the GPU; default cuda)
set -euo pipefail

cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-.venv/bin/python}"

mkdir -p results/preds
bash scripts/download_data.sh
"$PYTHON" -m pytest -q
"$PYTHON" scripts/cohort_report.py
"$PYTHON" scripts/run_e1.py --device "$DEVICE"
"$PYTHON" scripts/run_e2.py --device "$DEVICE"
"$PYTHON" scripts/run_e4.py
"$PYTHON" scripts/run_e5.py
"$PYTHON" scripts/run_e3.py --device "$DEVICE"
"$PYTHON" scripts/run_e6.py
"$PYTHON" scripts/make_figures.py
