# Confirmatory-Ready Criteria

This document defines the release-facing criteria for labeling an official output as
**confirmatory-ready**.

## Required criteria

A confirmatory output directory is confirmatory-ready only when all checks pass:

1. Official artifact verification passes (`verify_official_artifacts.py`).
2. `confirmatory_status` is `confirmatory` (not downgraded).
3. No science-critical deviations are detected.
4. Confirmatory controls are valid (`controls_valid_for_confirmatory=true`).
5. Required evidence status is valid (`required_evidence_status.valid=true`).
6. Dataset fingerprint evidence is present and complete.
7. All confirmatory runs are completed.
8. If reproducibility summary is provided, it must report `passed=true`.

## Verification command

```bash
python scripts/verify_confirmatory_ready.py \
  --output-dir <confirmatory_output_dir> \
  --repro-summary <optional_repro_summary.json> \
  --summary-out outputs/release/confirmatory_ready_summary.json
```

The optional reproducibility summary can come from:

- `scripts/verify_official_reproducibility.py`
- `scripts/replay_official_paths.py` (`replay_verification_summary.json`)

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
