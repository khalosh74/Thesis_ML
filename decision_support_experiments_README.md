# Decision-Support Experiment Automation

`run_decision_support_experiments.py` automates thesis decision-support experiments E01-E11
from `configs/decision_support_registry.json` using the existing `run_experiment` pipeline.

## Typical commands

Run Stage 1 (target lock):

```powershell
python run_decision_support_experiments.py `
  --stage "Stage 1 - Target lock" `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache
```

Run one experiment:

```powershell
python run_decision_support_experiments.py --experiment-id E06
```

Run all decision-support experiments:

```powershell
python run_decision_support_experiments.py --all
```

Dry-run expansion only:

```powershell
python run_decision_support_experiments.py --all --dry-run
```

## Output layout

Per-experiment execution roots:

- `outputs/artifacts/decision_support/E01/<campaign_timestamp>/`
- `.../outputs/reports/experiments/<run_id>/...` for run artifacts from `run_experiment`
- `.../run_manifests/*.json` for per-variant audit manifests

Campaign summary root:

- `outputs/artifacts/decision_support/campaigns/<campaign_timestamp>/`
- `run_log_export.csv`
- `decision_support_summary.csv`
- `decision_recommendations.md`
- `stage1_target_lock_summary.csv`, `stage2_split_lock_summary.csv`, etc.

## Notes

- Unsupported variants are marked as `blocked` with explicit reasons.
- Single blocked experiment requests fail clearly.
- This system is decision-support only; confirmatory runs are out of scope.
