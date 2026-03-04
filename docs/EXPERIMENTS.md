# Experiments Guide

## Leakage rules

The experiment runner enforces leakage-safe evaluation by design:

- Splits are grouped with `LeaveOneGroupOut` using `subject_session` groups.
- Any transformation is fit only on training folds through sklearn `Pipeline`.
- Current default pipeline:
  - `StandardScaler(with_mean=True, with_std=True)`
  - selected linear classifier (`logreg`, `linearsvc`, `ridge`)
- No scaler/model fitting happens on test fold data.

Permutation mode (`--n-permutations N`) is also leakage-safe:
- Labels are shuffled only inside each training fold before fitting.
- Test labels are never shuffled.

## Recommended templates

### Template 1: within-subject/session-grouped LOSO (default)

Use all available sessions and evaluate generalization to unseen sessions:

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

### Template 2: task-restricted comparison

Focus only one task to reduce heterogeneity:

```powershell
thesisml-run-experiment `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --target emotion `
  --filter-task passive `
  --model linearsvc `
  --cv loso_session `
  --seed 42
```

### Template 3: modality-restricted comparison

Compare models on single-modality inputs:

```powershell
thesisml-run-experiment `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --target emotion `
  --filter-modality audio `
  --model logreg `
  --cv loso_session `
  --seed 42
```

## Artifacts and interpretation

Each run writes to `reports/experiments/<run_id>/`:

- `config.json`
  - full CLI args, seed, package versions, and git commit (if available)
- `metrics.json`
  - overall metrics: `accuracy`, `balanced_accuracy`, `macro_f1`
  - confusion matrix and class labels
  - optional permutation stats and p-value
- `fold_metrics.csv`
  - one row per CV fold with fold-level metrics and held-out groups
- `predictions.csv`
  - per-sample predictions:
    - `y_true`, `y_pred`
    - decision/probability outputs when available
    - metadata (`subject`, `session`, `task`, `modality`, `bas`)

## Notes for thesis reporting

- Report both `accuracy` and `balanced_accuracy`, especially with class imbalance.
- Use `macro_f1` for class-balanced summary independent of support.
- Include fold-level variance from `fold_metrics.csv` to discuss stability.
- Keep model comparisons on identical splits and identical filters for fairness.
