# Operator Runbook

This runbook covers normal operation on a clean machine and day-to-day rerun handling.

## 1) Install

Canonical:

```bash
python -m pip install --upgrade pip
python -m pip install uv
python -m uv sync --frozen --extra dev
```

Compatibility:

```bash
python -m pip install -e ".[dev]"
```

## 2) Paths and defaults

Configured in `src/Thesis_ML/config/paths.py`.

- Registry default: `configs/decision_support_registry.json`
- Workbook template default: `templates/thesis_experiment_program.xlsx`
- Output root: `outputs/`
  - experiment reports: `outputs/reports/experiments/`
  - decision-support campaign artifacts: `outputs/artifacts/decision_support/`
  - workbook write-back files: `outputs/workbooks/`

## 3) Standard experiment run

```bash
thesisml-run-experiment \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --target coarse_affect \
  --model ridge \
  --cv within_subject_loso_session \
  --subject sub-001 \
  --run-id within_sub001_ridge
```

## 4) Rerun / resume behavior

`run_experiment` is intentionally strict.

- Existing completed run + same `run_id`: fails unless `--force`
- Existing partial/unknown run dir: fails unless `--resume` or `--force`
- `--force`: clears run directory, then reruns
- `--resume`: resumes from existing run directory
- `--reuse-completed-artifacts`: allows same-run section artifact reuse
- `--force` and `--resume` are mutually exclusive

Run state file:
- `outputs/reports/experiments/<run_id>/run_status.json`

## 5) Decision-support campaign (registry)

```bash
thesisml-run-decision-support \
  --registry configs/decision_support_registry.json \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --output-root outputs/artifacts/decision_support \
  --all
```

## 6) Decision-support campaign (workbook-driven)

```bash
thesisml-run-decision-support \
  --workbook templates/thesis_experiment_program.xlsx \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --output-root outputs/artifacts/decision_support \
  --all \
  --write-back-workbook
```

## 7) Workbook generation

```bash
thesisml-workbook --output templates/thesis_experiment_program.xlsx
```

## 8) Health checks

```bash
python -m mypy
python -m ruff check src/Thesis_ML/artifacts src/Thesis_ML/orchestration src/Thesis_ML/workbook \
  src/Thesis_ML/experiments/segment_execution.py src/Thesis_ML/experiments/sections.py \
  src/Thesis_ML/experiments/run_experiment.py
python -m ruff format --check src/Thesis_ML/artifacts src/Thesis_ML/orchestration src/Thesis_ML/workbook \
  src/Thesis_ML/experiments/segment_execution.py src/Thesis_ML/experiments/sections.py \
  src/Thesis_ML/experiments/run_experiment.py
python -m pytest -q
```

