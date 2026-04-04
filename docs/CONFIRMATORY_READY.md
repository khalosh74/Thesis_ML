# Confirmatory-Ready Criteria

This document defines the release-facing criteria for labeling an official output as
**confirmatory-ready**.

## Required criteria

A confirmatory output directory is confirmatory-ready only when all checks pass:

1. Scientific confirmatory scope and thesis runtime registry are aligned (or explicitly deferred via scope exceptions).
2. Official artifact verification passes (`verify_official_artifacts.py`).
3. `confirmatory_status` is `confirmatory` (not downgraded).
4. No science-critical deviations are detected.
5. Confirmatory controls are valid (`controls_valid_for_confirmatory=true`).
6. Required evidence status is valid (`required_evidence_status.valid=true`).
7. Dataset fingerprint evidence is present and complete.
8. All confirmatory runs are completed.
9. If reproducibility summary is provided, it must report `passed=true`.
10. If `--require-control-coverage` is used, every runtime confirmatory anchor must have E12+E13 coverage.

## Verification command

```bash
python scripts/verify_confirmatory_ready.py \
  --output-dir <confirmatory_output_dir> \
  --repro-summary <optional_repro_summary.json> \
  --scope-config configs/confirmatory/confirmatory_scope_v1.json \
  --runtime-registry configs/decision_support_registry_revised_execution.json \
  --scope-exceptions <optional_scope_exceptions.json> \
  --require-control-coverage \
  --summary-out outputs/release/confirmatory_ready_summary.json
```

The optional reproducibility summary comes from:

- `scripts/replay_official_paths.py` (`replay_verification_summary.json`)

The optional scope exceptions file is only for deliberate scientific deviations and must be explicit
(for example, deferred cells by within-subject ID or transfer pair).

The command emits a machine-readable JSON summary with:

- `passed`
- per-criterion pass/fail
- failure issues
- embedded official artifact verification summary

## Release gate integration

Use RC gate integration to include confirmatory-ready checks in release summaries:

```bash
python scripts/rc1_release_gate.py \
  --verify-official-dir <confirmatory_output_dir> \
  --confirmatory-ready-dir <confirmatory_output_dir> \
  --confirmatory-ready-summary-out outputs/release/confirmatory_ready_summary.json
```

## Governance boundary

Confirmatory-ready means the run satisfies contract/reproducibility/governance criteria for
frozen campaign evidence. It does **not** imply clinical readiness, deployment approval,
causal validity, or unrestricted external generalization.

## Chapter 4 Control-Coverage Artifact

When decision-support confirmatory controls are executed, campaign outputs include:

- `special_aggregations/E12/e12_permutation_analysis_summary.csv` and `.json`
- `special_aggregations/E13/e13_dummy_baseline_analysis_summary.csv` and `.json`
- `special_aggregations/confirmatory/confirmatory_anchor_control_coverage.csv` and `.json`

The confirmatory anchor coverage artifact maps each confirmatory analysis label to:

- runtime anchor identity (`experiment_id`, `template_id`)
- E12 coverage status
- E13 coverage status
- control metric/report paths when available
