# Experiments Guide (Canonical Thesis Protocol)

## Official thesis command

Official thesis runs must be executed through:

```powershell
thesisml-run-protocol `
  --protocol configs/protocols/thesis_canonical_v1.json `
  --all-suites `
  --reports-root outputs/reports/experiments
```

Dry-run (compile + validation + protocol artifacts only):

```powershell
thesisml-run-protocol `
  --protocol configs/protocols/thesis_canonical_v1.json `
  --all-suites `
  --reports-root outputs/reports/experiments `
  --dry-run
```

For official thesis runs, science-affecting settings are loaded from the protocol and not from CLI flags.

## Canonical protocol source of truth

Canonical protocol file:
- `configs/protocols/thesis_canonical_v1.json`

Protocol includes and locks:
- scientific contract (`target`, primary/secondary metric policy, seed policy)
- split policy (`within_subject_loso_session` primary, `frozen_cross_person_transfer` secondary)
- model policy (fixed baselines)
- control policy (dummy baseline + permutation controls)
- interpretability policy (explicitly allowed suites/modes/models only)
- sensitivity policy role
- artifact contract
- official run suites (`primary_within_subject`, `secondary_cross_person_transfer`, `primary_controls`)

## Leakage and spatial safeguards

Leakage-safe behavior remains enforced by `run_experiment`:
- preprocessing/model fit inside sklearn `Pipeline`
- scaler fit on train fold only
- no test-fold fit leakage
- permutation labels shuffled in train folds only

Spatial safeguards remain mandatory:
- beta/mask affine + shape validation during cache build
- cross-group spatial-signature compatibility validation before stacking

## Metric and control policy

For canonical protocol runs:
- primary metric is `balanced_accuracy`
- permutation testing uses the protocol metric (no silent fallback to plain accuracy)
- control runs are protocol-declared (`dummy` baseline and permutation policy)

## Interpretability policy

Interpretability is protocol-controlled:
- enabled only for suites/modes/models allowed in protocol
- output is supporting model-behavior evidence only
- disallowed suites/modes/models do not emit interpretability artifacts

## Run and protocol artifacts

Each concrete run still writes under `outputs/reports/experiments/<run_id>/`:
- `config.json`
- `metrics.json`
- `fold_metrics.csv`
- `fold_splits.csv`
- `predictions.csv`
- `spatial_compatibility_report.json`
- `interpretability_summary.json`

Canonical-run metadata is written to run artifacts (`config.json` and `metrics.json`):
- `canonical_run`
- `protocol_id`
- `protocol_version`
- `protocol_schema_version`
- `suite_id`
- `claim_ids`

Protocol-level artifacts are written under:
- `outputs/reports/experiments/protocol_runs/<protocol_id>__<protocol_version>/`

Files:
- `protocol.json`
- `compiled_protocol_manifest.json`
- `claim_to_run_map.json`
- `suite_summary.json`
- `execution_status.json`
- `report_index.csv`

## Low-level ad hoc runner

`thesisml-run-experiment` is still supported for exploratory/ad hoc execution, including section-level resume controls.
It is no longer the official thesis-facing command for confirmatory runs.

See `docs/SEGMENT_EXECUTION.md` for low-level segmented execution behavior.
