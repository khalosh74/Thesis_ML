#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install uv

python -m uv sync --frozen --extra dev
python -m uv run python -m pytest -q
python -m uv run thesisml-run-decision-support --help

mkdir -p outputs/workbooks
python -m uv run thesisml-workbook --output outputs/workbooks/bootstrap_thesis_experiment_program.xlsx

echo "Bootstrap complete. Workbook written to outputs/workbooks/bootstrap_thesis_experiment_program.xlsx"
