# Experiments Guide (Current Thesis Stage)

This guide documents the current implemented experiment modes in
`src/Thesis_ML/experiments/run_experiment.py`.

Method priority for this thesis stage:
- Primary evidence path: within-subject held-out-session decoding
- Secondary evidence path: frozen cross-person transfer
- Auxiliary mode: grouped `loso_session` (non-primary)

## Leakage rules

Leakage-safe behavior is enforced by the runner:
- Preprocessing and model fitting are inside a sklearn `Pipeline`.
- The scaler (`StandardScaler`) is fit only on training data.
- Test data is never used during fold fitting.
- In permutation mode (`--n-permutations N`), labels are shuffled only in training data before fit.
- Within-subject and frozen-transfer runs are intentionally separate modes answering different questions.

## Spatial safeguards

Spatial checks are enforced in two stages:
- During feature extraction/cache build, each beta image must match its mask in shape and affine.
  - On mismatch, extraction fails immediately (no resampling or auto-repair).
- Before experiment stacking/fitting, cache groups selected for a run must share compatible spatial signatures.
  - A machine-readable `spatial_compatibility_report.json` is written for auditability.

## Primary Thesis Experiment

Primary settings:
- `--target coarse_affect`
- `--cv within_subject_loso_session`
- one subject per run (`--subject`)
- training and testing are from the same subject, with held-out session folds

Example:

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

## Secondary Thesis Experiment

Frozen cross-person transfer settings:
- `--cv frozen_cross_person_transfer`
- explicit `--train-subject` and `--test-subject`
- `train_subject` and `test_subject` must differ
- model is fit once on train subject only, then applied to test subject without re-fit/re-tuning

Example:

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

Reverse direction should be run separately when needed.

## Interpretability (Within-Subject Linear Runs)

For `within_subject_loso_session`, fold-level explanation artifacts are exported for supported linear baselines (`logreg`, `linearsvc`, `ridge`):
- per-fold coefficient files: `interpretability/fold_###_coefficients.npz`
- fold explanation index: `interpretability_fold_explanations.csv`
- stability summary: `interpretability_summary.json`

Stability summary currently reports simple fold-level coefficient stability measures:
- pairwise coefficient correlation
- sign consistency
- top-k overlap

Interpretability outputs are for model-behavior robustness evidence only. They are not direct neural localization claims.

## Optional Filters

Task and modality filters are available in all modes:
- `--filter-task <task>`
- `--filter-modality <modality>`

Example (primary mode + task filter):

```powershell
thesisml-run-experiment `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --target coarse_affect `
  --model linearsvc `
  --cv within_subject_loso_session `
  --subject sub-001 `
  --filter-task passive `
  --seed 42
```

## Auxiliary Grouped Mode (Non-Primary)

`loso_session` remains available as a grouped auxiliary analysis mode and is not the primary thesis evaluation path.

Example:

```powershell
thesisml-run-experiment `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --target coarse_affect `
  --model logreg `
  --cv loso_session `
  --seed 42
```

## Artifacts Reference

Each run writes to `reports/experiments/<run_id>/`:
- `config.json`
- `metrics.json`
- `fold_metrics.csv`
- `fold_splits.csv`
- `predictions.csv`
- `spatial_compatibility_report.json`
- `interpretability_summary.json`

For within-subject linear runs, the run directory also includes:
- `interpretability_fold_explanations.csv`
- `interpretability/fold_###_coefficients.npz`

Audit notes:
- `config.json` records mode, subjects, target, model, seed, and interpretability status/paths.
- `config.json` also records spatial compatibility status and report path.
- `fold_splits.csv` records train/test split membership and sample counts.
- `predictions.csv` includes row-level metadata (`subject`, `session`, labels) and prediction outputs.
- `metrics.json` contains aggregate metrics plus interpretability and spatial-compatibility linkage.
- `spatial_compatibility_report.json` records checked groups, reference signature, tolerance, and pass/fail details.
