# Operator Guide (Canonical)

Use these commands as the supported operator path.

## 1) Environment

Canonical operator path is source checkout + `uv.lock`.

```bash
python -m pip install --upgrade pip
python -m pip install uv
python -m uv sync --frozen --extra dev
```

Optional Optuna support:

```bash
python -m uv sync --frozen --extra dev --extra optuna
```

Installed wheel path is also supported; default decision-support registry is packaged in the wheel.

## 2) Canonical CLIs

- `thesisml-run-experiment`
- `thesisml-run-decision-support`
- `thesisml-workbook`
- `thesisml-run-baseline`

Compatibility wrappers are kept only for migration and are deprecated.

## 3) First-use checks

Generate workbook template:

```bash
thesisml-workbook --output templates/thesis_experiment_program.xlsx
```

Template note:
- `templates/thesis_experiment_program.xlsx` is non-runnable by default
- enable/populate executable rows in `Experiment_Definitions` before using `--workbook`

Registry dry-run:

```bash
thesisml-run-decision-support \
  --registry configs/decision_support_registry.json \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --output-root outputs/artifacts/decision_support \
  --all \
  --dry-run
```

Installed-wheel note:
- the default registry path is packaged in the wheel, so `--registry` is optional
- passing explicit `--registry` remains supported and recommended for controlled campaigns

## 4) Gold acceptance command

Run this before releases or high-value campaigns:

```bash
python scripts/acceptance_smoke.py
```

This command validates:
- shipped workbook template/schema compatibility
- workbook generation through canonical CLI
- shipped registry compilation/dry-run through canonical CLI
