# Operator Runbook

This runbook covers normal operation on a clean machine and day-to-day rerun handling.

## 1) Install

Canonical:

```bash
python -m pip install --upgrade pip
python -m pip install uv
python -m uv sync --frozen --extra dev
```

Optional Optuna support:

```bash
python -m uv sync --frozen --extra dev --extra optuna
```

Compatibility:

```bash
python -m pip install -e ".[dev]"
```

For Optuna mode with editable installs:

```bash
python -m pip install -e ".[dev,optuna]"
```

## 2) Paths and defaults

Configured in `src/Thesis_ML/config/paths.py`.

- Decision-support registry default:
  - source checkout: `configs/decision_support_registry.json`
  - installed wheel: packaged asset under `Thesis_ML/assets/configs/decision_support_registry.json`
- Workbook generation default output: `templates/thesis_experiment_program.xlsx` under current project/cwd
- Shipped workbook template asset:
  - source checkout: `templates/thesis_experiment_program.xlsx`
  - installed wheel: packaged asset under `Thesis_ML/assets/templates/thesis_experiment_program.xlsx`
- Output root: `outputs/`
  - exploratory reports: `outputs/reports/exploratory/`
  - locked comparison reports: `outputs/reports/comparisons/`
  - confirmatory reports: `outputs/reports/confirmatory/`
  - decision-support campaign artifacts: `outputs/artifacts/decision_support/`
  - workbook write-back files: `outputs/workbooks/`

## 3) Framework mode commands

Exploratory run:

```bash
thesisml-run-experiment \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --target coarse_affect \
  --model ridge \
  --cv within_subject_loso_session \
  --subject sub-001 \
  --run-id exploratory_sub001_ridge
```

Locked comparison dry-run:

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/model_family_comparison_v1.json \
  --all-variants \
  --reports-root outputs/reports/comparisons \
  --dry-run
```

Locked comparison execution:

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/model_family_comparison_v1.json \
  --all-variants \
  --reports-root outputs/reports/comparisons
```

Grouped nested comparison execution:

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/model_family_grouped_nested_comparison_v1.json \
  --variant ridge \
  --reports-root outputs/reports/comparisons
```

Confirmatory canonical protocol run:

```bash
thesisml-run-protocol \
  --protocol configs/protocols/thesis_canonical_v1.json \
  --all-suites \
  --reports-root outputs/reports/confirmatory
```

Confirmatory dry-run validation/compilation:

```bash
thesisml-run-protocol \
  --protocol configs/protocols/thesis_canonical_v1.json \
  --all-suites \
  --reports-root outputs/reports/confirmatory \
  --dry-run
```

Policy note:
- exploratory mode is flexible and not confirmatory evidence.
- locked comparison mode allows only declared variants from comparison specs.
- confirmatory mode must load thesis-critical settings from protocol JSON, not ad hoc CLI flags.
- comparison/protocol specs must declare one methodology policy:
  `fixed_baselines_only` or `grouped_nested_tuning`.
- locked comparison outputs include `comparison_decision.json` for machine-readable winner/inconclusive/invalid decisions.

Metric policy note (official runs):
- one declared `primary_metric` governs tuning, decision selection, permutation testing, and headline reporting
- `secondary_metrics` are descriptive-only
- official permutation metric must equal the primary metric
- `config.json`, `metrics.json`, and mode-level manifests include `metric_policy_effective`
- implicit metric fallbacks are disabled; workbook/registry/search-space inputs must explicitly declare required metric fields

## 4) Rerun / resume behavior

`run_experiment` is intentionally strict.

- Existing completed run + same `run_id`: fails unless `--force`
- Existing partial/unknown run dir: fails unless `--resume` or `--force`
- `--force`: clears run directory, then reruns
- `--resume`: resumes from existing run directory
- `--resume` automatically enables same-run section artifact reuse
- `--reuse-completed-artifacts`: explicitly enables same-run section artifact reuse
- `--reuse-completed-artifacts` never reuses artifacts from a different `run_id`
- `--force` and `--resume` are mutually exclusive

`thesisml-run-protocol` forwards `--force`/`--resume` to underlying concrete runs.
`thesisml-run-comparison` forwards `--force`/`--resume` to underlying concrete runs.

Run state file:
- `outputs/reports/<mode>/<run_id>/run_status.json`

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

Optuna mode requires optional dependency installation (`--extra optuna` or `.[optuna]`):

```bash
thesisml-run-decision-support \
  --registry configs/decision_support_registry.json \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --output-root outputs/artifacts/decision_support \
  --all \
  --search-mode optuna \
  --optuna-trials 25
```

## 6) Decision-support campaign (workbook-driven)

`templates/thesis_experiment_program.xlsx` is a **planning template** by default.
It validates structurally but is intentionally non-runnable until at least one row in
`Experiment_Definitions` is enabled (`enabled=Yes`) and populated with required fields
(`target`, `cv`, `model`).

Prepare a runnable workbook by generating a copy and editing executable rows first:

```bash
thesisml-workbook --output outputs/workbooks/my_campaign.xlsx
# edit outputs/workbooks/my_campaign.xlsx: set enabled=Yes and fill required columns
```

```bash
thesisml-run-decision-support \
  --workbook outputs/workbooks/my_campaign.xlsx \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --output-root outputs/artifacts/decision_support \
  --all \
  --write-back-workbook
```

Write-back safety:
- write-back target name is versioned (`<source>__results_<campaign_id>.xlsx`)
- existing targets are never overwritten unless explicit API-level `overwrite_existing=True` is used

## 7) Workbook generation

```bash
thesisml-workbook --output templates/thesis_experiment_program.xlsx
```

Template policy:
- generated/shipped template is governance-first and non-runnable by default
- workbook compiler will fail until executable rows are explicitly enabled/populated

## 8) Health checks

```bash
python -m mypy
python -m ruff check src/Thesis_ML/artifacts src/Thesis_ML/orchestration src/Thesis_ML/workbook \
  src/Thesis_ML/experiments/segment_execution.py src/Thesis_ML/experiments/sections.py \
  src/Thesis_ML/experiments/run_experiment.py src/Thesis_ML/protocols src/Thesis_ML/comparisons \
  src/Thesis_ML/cli/protocol_runner.py src/Thesis_ML/cli/comparison_runner.py
python -m ruff format --check src/Thesis_ML/artifacts src/Thesis_ML/orchestration src/Thesis_ML/workbook \
  src/Thesis_ML/experiments/segment_execution.py src/Thesis_ML/experiments/sections.py \
  src/Thesis_ML/experiments/run_experiment.py src/Thesis_ML/protocols src/Thesis_ML/comparisons \
  src/Thesis_ML/cli/protocol_runner.py src/Thesis_ML/cli/comparison_runner.py
python -m pytest -q
```

`mypy` is a required CI gate and includes nibabel boundary modules:
- `src/Thesis_ML/spm/extract_glm.py`
- `src/Thesis_ML/features/nifti_features.py`

## 9) Gold acceptance check

```bash
python scripts/acceptance_smoke.py
```

This is the canonical pre-release/pre-campaign acceptance path and validates:
- shipped workbook asset compatibility
- canonical workbook CLI generation
- shipped registry dry-run through canonical decision-support CLI
