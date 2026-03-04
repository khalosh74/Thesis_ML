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
      BAS1/
        beta_0001.nii
        beta_0002.nii
        ...
        mask.nii
        regressor_labels.csv
        SPM.mat            # optional for extraction (if missing, extraction still works)
```

`regressor_labels.csv` must map 1:1 by row index (1-based) to `beta_0001.nii ... beta_NNNN.nii`.

## Install (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,spm]"
```

If you do not need `SPM.mat` parsing, `.[dev]` is enough.

## Quickstart (end-to-end)

### 1) Extract one session mapping + summary

```powershell
thesisml-extract-glm `
  --glm-dir Data/sub-001/ses-04/BAS1 `
  --out-dir Data/processed/extractions/sub-001/ses-04/BAS1
```

Outputs:
- `regressor_beta_mapping.csv`
- `session_summary.json`

### 2) Build dataset index across many sessions

```powershell
thesisml-build-index `
  --data-root Data `
  --out-csv Data/processed/dataset_index.csv `
  --pattern "sub-*/ses-*/BAS*"
```

The output index includes at least:
- `subject`, `session`, `bas`, `run`, `task`, `emotion`, `modality`
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
- Contains `X` (`n_samples x n_voxels`), `y`, and row metadata.
- Existing cache files are skipped unless `--force` is passed.

### 4) Run leakage-safe grouped-CV experiment

```powershell
thesisml-run-experiment `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --target emotion `
  --model ridge `
  --cv loso_session `
  --seed 42
```

Optional filters and permutation test:

```powershell
thesisml-run-experiment `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --target emotion `
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
reports/experiments/<run_id>/
  config.json
  metrics.json
  fold_metrics.csv
  predictions.csv
```

All models use the same split schema and metric set:
- `accuracy`
- `balanced_accuracy`
- `macro_f1`
- confusion matrix + fold-level metrics

## Adding a new model to the registry

Edit `src/Thesis_ML/experiments/run_experiment.py`:
- Add a new case in `_make_model`.
- Keep preprocessing/model composition inside `_build_pipeline` so transforms are fit only on
  training folds.
- Add the model name to CLI choices (`_MODEL_NAMES` + parser choices).
- Keep output schema unchanged (`metrics.json`, `fold_metrics.csv`, `predictions.csv`) for
  comparability.

## Quality checks

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

## Existing synthetic baseline

The original synthetic baseline remains available and unchanged:

```powershell
python scripts/run_baseline.py
```

Outputs:
- `reports/metrics.json`
- `models/baseline_model.npz`
