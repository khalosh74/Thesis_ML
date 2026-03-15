# Thesis_ML

Leakage-safe, experiment-ready fMRI ML framework for session-level GLM extraction, dataset indexing,
feature caching, and reproducible multi-model evaluation.

## Data policy

- Keep real data local and untracked.
- Keep reusable logic in `src/Thesis_ML` (not notebooks).
- Do not commit dataset files, model binaries, or generated report artifacts.

## Expected data layout

Use a BIDS-like hierarchy under a local data root (`Data/` or `data/`):

```text
Data/
  sub-001/
    ses-04/
      BAS2/
        beta_0001.nii
        beta_0002.nii
        ...
        mask.nii
        regressor_labels.csv
        SPM.mat            # optional for extraction (if missing, extraction still works)
```

`regressor_labels.csv` must map 1:1 by row index (1-based) to `beta_0001.nii ... beta_NNNN.nii`.

## Supported Python

- Canonical and CI-tested: `Python 3.13`
- Package constraint: `>=3.11,<3.14` (from `pyproject.toml`)

The repository includes `.python-version` pinned to `3.13`.

## Reproducible setup (canonical: uv lockfile)

`uv.lock` is the primary reproducible dependency path for this project.

### PowerShell

```powershell
python -m pip install --upgrade pip
python -m pip install uv
python -m uv sync --frozen --extra dev
```

### Bash

```bash
python -m pip install --upgrade pip
python -m pip install uv
python -m uv sync --frozen --extra dev
```

Optional extra for Optuna-backed search mode:

```powershell
python -m uv sync --frozen --extra dev --extra optuna
```

## Bootstrap commands (clean machine)

You can run the scripts:
- PowerShell: `./scripts/bootstrap_env.ps1`
- Bash: `./scripts/bootstrap_env.sh`

Or run the commands directly:

```powershell
python -m uv sync --frozen --extra dev
python -m uv run python -m pytest -q
python -m uv run thesisml-run-comparison --help
python -m uv run thesisml-run-protocol --help
python -m uv run thesisml-run-decision-support --help
python -m uv run thesisml-workbook --output outputs/workbooks/bootstrap_thesis_experiment_program.xlsx
```

Official confirmatory thesis run path (canonical protocol):

```powershell
python -m uv run thesisml-run-protocol `
  --protocol configs/protocols/thesis_canonical_v1.json `
  --all-suites `
  --reports-root outputs/reports/confirmatory
```

Framework mode lifecycle:
- `thesisml-run-experiment` -> exploratory mode (`framework_mode=exploratory`, `canonical_run=false`), default reports root `outputs/reports/exploratory/`
- `thesisml-run-comparison` -> locked comparison mode (`framework_mode=locked_comparison`, `canonical_run=false`), default reports root `outputs/reports/comparisons/`
- `thesisml-run-protocol` -> confirmatory mode (`framework_mode=confirmatory`, `canonical_run=true`), default reports root `outputs/reports/confirmatory/`
- official comparison/protocol contracts must pick exactly one methodology policy:
  - `fixed_baselines_only`
  - `grouped_nested_tuning`

Locked comparison example:

```powershell
python -m uv run thesisml-run-comparison `
  --comparison configs/comparisons/model_family_comparison_v1.json `
  --all-variants `
  --reports-root outputs/reports/comparisons
```

Grouped nested comparison example:

```powershell
python -m uv run thesisml-run-comparison `
  --comparison configs/comparisons/model_family_grouped_nested_comparison_v1.json `
  --variant ridge `
  --reports-root outputs/reports/comparisons
```

Decision-support campaign command (requires index/cache/data paths to exist):

```powershell
python -m uv run thesisml-run-decision-support `
  --registry configs/decision_support_registry.json `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --output-root outputs/artifacts/decision_support `
  --all
```

Registry default behavior:
- source checkout default resolves to `configs/decision_support_registry.json`
- installed wheel default resolves to packaged asset `Thesis_ML/assets/configs/decision_support_registry.json`
- explicit `--registry` remains supported and is recommended for controlled runs

Optional Optuna-backed variant search:

```powershell
python -m uv run thesisml-run-decision-support `
  --registry configs/decision_support_registry.json `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --output-root outputs/artifacts/decision_support `
  --all `
  --search-mode optuna `
  --optuna-trials 25
```

## Workbook template policy

`templates/thesis_experiment_program.xlsx` is a **planning template** by default.
It is structurally valid but intentionally non-runnable until at least one row in
`Experiment_Definitions` is enabled (`enabled=Yes`) and includes required values
(`target`, `cv`, `model`).

Safe first-use pattern:

```powershell
python -m uv run thesisml-workbook --output outputs/workbooks/my_campaign.xlsx
# edit outputs/workbooks/my_campaign.xlsx and enable/populate executable rows
python -m uv run thesisml-run-decision-support `
  --workbook outputs/workbooks/my_campaign.xlsx `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --output-root outputs/artifacts/decision_support `
  --all `
  --write-back-workbook
```

## Compatibility install path (kept)

The editable `pip` flow remains supported:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev,spm]"
```

If you do not need `SPM.mat` parsing, `.[dev]` is sufficient.
If you need Optuna search mode, install `.[optuna]` (or `.[dev,optuna]`).

## Docker / devcontainer

- Docker image: `Dockerfile` (Python `3.13`, `uv sync --frozen --extra dev`)
- VS Code devcontainer: `.devcontainer/devcontainer.json`

Build and run tests in Docker:

```bash
docker build -t thesis-ml:dev .
docker run --rm thesis-ml:dev
```

## CI and release gate

- CI workflow: `.github/workflows/ci.yml` (push/PR)
- Release validation workflow: `.github/workflows/release_gate.yml` (tags `v*`)
- Release details: `docs/RELEASE.md`
- Gold acceptance path used by CI/release: `python scripts/acceptance_smoke.py`

## Operator documentation

- Architecture: `docs/ARCHITECTURE.md`
- Operator quick path: `docs/OPERATOR_GUIDE.md`
- Runbook: `docs/RUNBOOK.md`
- Workbook workflow: `docs/WORKBOOK_WORKFLOW.md`
- Segment execution: `docs/SEGMENT_EXECUTION.md`
- Maintainer quick path: `docs/MAINTAINER_GUIDE.md`
- Extension guide: `docs/EXTENDING.md`
- Schema/version migration notes: `docs/SCHEMA_MIGRATIONS.md`
- Decision-support specifics: `docs/DECISION_SUPPORT_AUTOMATION.md`
- Experiment semantics: `docs/EXPERIMENTS.md`

## Quickstart (end-to-end)

### 1) Extract one session mapping + summary

```powershell
thesisml-extract-glm `
  --glm-dir Data/sub-001/ses-04/BAS2 `
  --out-dir Data/processed/extractions/sub-001/ses-04/BAS2
```

Outputs:
- `regressor_beta_mapping.csv`
- `session_summary.json`

### 2) Build dataset index across many sessions

```powershell
thesisml-build-index `
  --data-root Data `
  --out-csv Data/processed/dataset_index.csv `
  --pattern "sub-*/ses-*/BAS2"
```

The output index includes at least:
- `subject`, `session`, `bas`, `run`, `task`, `emotion`, `coarse_affect`, `modality`
- `beta_path`, `mask_path`, `regressor_label`

Path convention:
- `beta_path` and `mask_path` are stored relative to `--data-root` when possible for portability.

### 3) Cache masked beta features

```powershell
thesisml-cache-features `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache
```

Cache behavior:
- One compressed `.npz` per `subject_session_bas` group.
- Contains `X` (`n_samples x n_voxels`), `y`, row metadata, and spatial-signature metadata.
- Each beta is validated against its mask for shape + affine compatibility before vectorization.
- Existing cache files are skipped unless `--force` is passed.

### 4) Run exploratory within-person session-held-out experiment (`within_subject_loso_session`)

```powershell
thesisml-run-experiment `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --target coarse_affect `
  --model ridge `
  --cv within_subject_loso_session `
  --subject sub-001 `
  --seed 42
```

### 5) Run exploratory frozen cross-person transfer (`frozen_cross_person_transfer`)

```powershell
thesisml-run-experiment `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --target coarse_affect `
  --model ridge `
  --cv frozen_cross_person_transfer `
  --train-subject sub-001 `
  --test-subject sub-002 `
  --seed 42
```

Auxiliary grouped comparison mode (non-primary thesis evidence path):

```powershell
thesisml-run-experiment `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --target coarse_affect `
  --filter-task passive `
  --filter-modality audio `
  --model logreg `
  --cv loso_session `
  --seed 42 `
  --n-permutations 1000
```

## Experiment artifacts

Each run writes:

```text
outputs/reports/<mode>/<run_id>/
  config.json
  metrics.json
  fold_metrics.csv
  fold_splits.csv
  predictions.csv
  subgroup_metrics.json
  subgroup_metrics.csv
  tuning_summary.json
  best_params_per_fold.csv
  spatial_compatibility_report.json
  interpretability_summary.json
```

Default `<mode>` by command:
- `thesisml-run-experiment` -> `exploratory`
- `thesisml-run-comparison` -> `comparisons`
- `thesisml-run-protocol` -> `confirmatory`

Within-subject linear runs (`logreg`, `linearsvc`, `ridge`) additionally write:
- `interpretability_fold_explanations.csv`
- `interpretability/fold_###_coefficients.npz`

All models use the same split/metrics core:
- `accuracy`
- `balanced_accuracy`
- `macro_f1`
- confusion matrix + fold-level metrics

Interpretability outputs are supporting model-behavior robustness evidence only; they are not
direct neural localization claims.

Mode-level manifests:
- locked comparison executions write `comparison_runs/<comparison_id>__<comparison_version>/...`
- confirmatory protocol executions write `protocol_runs/<protocol_id>__<protocol_version>/...`
- locked comparison manifests include `comparison_decision.json` with
  `winner_selected` / `inconclusive` / `invalid_comparison`.

## Adding a new model to the registry

Edit `src/Thesis_ML/experiments/model_factory.py`:
- Add a new case in `make_model`.
- Keep preprocessing/model composition inside `build_pipeline` so transforms are fit only on
  training folds.
- Add the model name to `MODEL_NAMES`.
- Keep output schema unchanged (`metrics.json`, `fold_metrics.csv`, `predictions.csv`) for
  comparability.

## Quality checks

```powershell
python -m uv run python -m mypy
python -m uv run python -m ruff check src/Thesis_ML/artifacts src/Thesis_ML/comparisons src/Thesis_ML/protocols src/Thesis_ML/orchestration src/Thesis_ML/workbook src/Thesis_ML/experiments/segment_execution.py src/Thesis_ML/experiments/sections.py src/Thesis_ML/experiments/run_experiment.py src/Thesis_ML/cli/comparison_runner.py src/Thesis_ML/cli/protocol_runner.py --exclude src/Thesis_ML/workbook/template_builder.py
python -m uv run python -m ruff format --check src/Thesis_ML/artifacts src/Thesis_ML/comparisons src/Thesis_ML/protocols src/Thesis_ML/orchestration src/Thesis_ML/workbook src/Thesis_ML/experiments/segment_execution.py src/Thesis_ML/experiments/sections.py src/Thesis_ML/experiments/run_experiment.py src/Thesis_ML/cli/comparison_runner.py src/Thesis_ML/cli/protocol_runner.py --exclude src/Thesis_ML/workbook/template_builder.py
python -m uv run python -m pytest -q
python -m uv run python scripts/acceptance_smoke.py
```

`mypy` in CI is required and checks:
- `src/Thesis_ML/artifacts`
- `src/Thesis_ML/orchestration`
- `src/Thesis_ML/workbook`
- `src/Thesis_ML/spm/extract_glm.py`
- `src/Thesis_ML/features/nifti_features.py`
- `src/Thesis_ML/experiments/run_experiment.py`
- `src/Thesis_ML/comparisons`
- `src/Thesis_ML/protocols`
- `src/Thesis_ML/cli/comparison_runner.py`
- `src/Thesis_ML/cli/protocol_runner.py`
- `src/Thesis_ML/experiments/segment_execution.py`
- `src/Thesis_ML/experiments/sections.py`

## Compatibility wrappers (deprecated)

The following script paths are still supported but deprecated:

- `run_decision_support_experiments.py` -> use `thesisml-run-decision-support`
- `create_thesis_experiment_workbook.py` -> use `thesisml-workbook`
- `scripts/create_thesis_experiment_workbook.py` -> use `thesisml-workbook`
- `scripts/run_baseline.py` -> use `thesisml-run-baseline`

## Existing synthetic baseline

The original synthetic baseline remains available and unchanged:

```powershell
thesisml-run-baseline
```

Outputs:
- `outputs/reports/metrics.json`
- `outputs/models/baseline_model.npz`
