# Decision-Support Automation (E01-E11)

This repo now includes a registry-driven orchestrator to run thesis decision-support experiments
(E01-E11) using the existing `Thesis_ML.experiments.run_experiment` path.

## Files

- `configs/decision_support_registry.json`: machine-readable experiment definitions for E01-E11.
- `src/Thesis_ML/orchestration/campaign_engine.py`: campaign execution core.
- `src/Thesis_ML/orchestration/campaign_cli.py`: CLI parsing and terminal output layer.
- `src/Thesis_ML/orchestration/decision_support.py`: compatibility facade used by legacy imports.
- `run_decision_support_experiments.py`: deprecated compatibility CLI shim.
- Output root (default): `outputs/artifacts/decision_support/`.

## What it does

For each selected experiment/stage:

- expands registry templates into concrete run variants (subject/task/modality/pair scoped),
- runs supported variants through `run_experiment` (or marks unsupported variants as blocked),
- writes per-variant run manifests,
- exports workbook-friendly summaries:
  - `run_log_export.csv`
  - `decision_support_summary.csv`
  - `decision_recommendations.md`
- writes stage summary artifacts:
  - `stage1_target_lock_summary.csv`
  - `stage2_split_lock_summary.csv`
  - `stage3_model_lock_summary.csv`
  - `stage4_feature_lock_summary.csv`

## Usage

Run one stage (recommended first run):

```powershell
thesisml-run-decision-support `
  --stage "Stage 1 - Target lock" `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache
```

Compatibility shim (deprecated, still supported):

```powershell
python run_decision_support_experiments.py `
  --stage "Stage 1 - Target lock" `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache
```

Run one experiment:

```powershell
thesisml-run-decision-support --experiment-id E06
```

Run all registry experiments:

```powershell
thesisml-run-decision-support --all
```

Dry-run (expand + manifest only, no model fitting):

```powershell
thesisml-run-decision-support --all --dry-run
```

## Notes

- Unsupported variants are not silently skipped; they are recorded as `blocked` with reasons.
- Stage ordering follows the thesis governance sequence in the registry.
- This automation is decision-support only; it does not run confirmatory experiments.
- Campaign IDs include microseconds to avoid output-path collisions on rapid reruns.
