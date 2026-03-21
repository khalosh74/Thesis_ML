# Release Gate

This repository uses a tag-triggered release validation workflow:

- Workflow: `.github/workflows/release_gate.yml`
- Trigger: push tag matching `v*` (for example `v0.2.0`)
- Manual trigger: GitHub Actions `workflow_dispatch`

Canonical reproducibility/operator path is documented in `docs/REPRODUCIBILITY.md`
(Python 3.13 + `uv.lock` + `uv sync --frozen --extra dev`).

## What it validates

1. Builds `sdist` and `wheel` with `python -m build`.
2. Validates package metadata with `python -m twine check dist/*`.
3. Installs the built wheel (not editable mode).
4. Runs gold acceptance smoke:
   - `python scripts/acceptance_smoke.py`
5. Verifies installed-wheel default registry behavior with canonical CLI:
   - `thesisml-run-decision-support --all --dry-run ...` (no explicit `--registry`)
6. Runs a lightweight runtime test:
   - `python -m pytest tests/test_import_decoupling.py -q`
7. Uploads `dist/*` as workflow artifacts.

## Local pre-tag check (recommended)

```bash
python -m pip install --upgrade pip
python -m pip install build twine pytest
python -m build --sdist --wheel
python -m twine check dist/*
python -m pip install --force-reinstall dist/*.whl
python scripts/acceptance_smoke.py
python - <<'PY'
from pathlib import Path
import pandas as pd

temp_root = Path("/tmp/thesisml_release_gate")
temp_root.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    [{"subject": "sub-001", "task": "passive", "modality": "audio"}]
).to_csv(temp_root / "dataset_index.csv", index=False)
PY
thesisml-run-decision-support \
  --index-csv /tmp/thesisml_release_gate/dataset_index.csv \
  --data-root /tmp/thesisml_release_gate/Data \
  --cache-dir /tmp/thesisml_release_gate/cache \
  --output-root /tmp/thesisml_release_gate/outputs \
  --all \
  --dry-run
python -m pytest tests/test_import_decoupling.py -q
```

Additional local hygiene/performance checks:

```bash
python scripts/release_hygiene_check.py
python scripts/performance_smoke.py --output outputs/performance/performance_smoke_summary.json
```

## RC-1 hardening gate (official paths)

RC-1 adds explicit official-run release checks for:
- strict official artifact completeness/invariants
- deterministic rerun verification on a small official path
- optional one-command wrapper for hygiene/lint/tests/performance
- structured failure payloads in mode-level execution artifacts (`error_code`, `error_type`, `failure_stage`, `error_details`)
- evidence-layer artifact invariants for official modes:
  - confirmatory: `repeated_run_metrics.csv`, `repeated_run_summary.json`,
    `confidence_intervals.json`, `metric_intervals.csv`
  - comparison: all confirmatory evidence artifacts plus
    `paired_model_comparisons.json` and `paired_model_comparisons.csv`
- data-layer artifact invariants for official modes:
  - `dataset_card.json`, `dataset_card.md`
  - `dataset_summary.json`, `dataset_summary.csv`
  - `data_quality_report.json`, `class_balance_report.csv`, `missingness_report.csv`
  - `leakage_audit.json`
  - `external_dataset_card.json`, `external_dataset_summary.json`,
    `external_validation_compatibility.json` (required when external compatibility checks are configured)
- governance metadata/docs presence checks via `release_hygiene_check.py`:
  - `LICENSE`
  - `CITATION.cff`
  - `docs/PRIVACY_AND_DATA_HANDLING.md`
  - `docs/USE_AND_MISUSE_BOUNDARIES.md`
  - `docs/CONFIRMATORY_READY.md`

### 1) Official artifact verification

Verify one locked-comparison or confirmatory output directory:

```bash
python scripts/verify_official_artifacts.py --output-dir <official_output_dir>
```

Optional mode hint:

```bash
python scripts/verify_official_artifacts.py --output-dir <official_output_dir> --mode confirmatory
python scripts/verify_official_artifacts.py --output-dir <official_output_dir> --mode comparison
```

### 2) Deterministic rerun verification

Run a small official path twice and compare deterministic invariants:

```bash
python scripts/verify_official_reproducibility.py \
  --mode protocol \
  --config configs/protocols/thesis_confirmatory_v1.json \
  --index-csv <dataset_index.csv> \
  --data-root <data_root> \
  --cache-dir <cache_dir> \
  --suite confirmatory_primary_within_subject
```

Comparison example:

```bash
python scripts/verify_official_reproducibility.py \
  --mode comparison \
  --config configs/comparisons/model_family_comparison_v1.json \
  --index-csv <dataset_index.csv> \
  --data-root <data_root> \
  --cache-dir <cache_dir> \
  --variant ridge
```

Canonical one-command replay path (comparison + confirmatory using checked-in demo data):

```bash
python scripts/replay_official_paths.py \
  --mode both \
  --use-demo-dataset \
  --reports-root outputs/reproducibility/official_replay \
  --verify-determinism
```

This command emits:
- `replay_summary.json`
- `replay_verification_summary.json`
- `reproducibility_manifest.json`

### 3) Publishable bundle build + verification

Build canonical directory+manifest bundle:

```bash
python scripts/build_publishable_bundle.py \
  --output-dir outputs/reproducibility/publishable_bundle \
  --comparison-output <comparison_output_dir> \
  --confirmatory-output <confirmatory_output_dir> \
  --replay-summary outputs/reproducibility/official_replay/replay_summary.json \
  --replay-verification-summary outputs/reproducibility/official_replay/replay_verification_summary.json \
  --repro-manifest outputs/reproducibility/official_replay/reproducibility_manifest.json
```

Verify bundle structure/hashes/contracts:

```bash
python scripts/verify_publishable_bundle.py \
  --bundle-dir outputs/reproducibility/publishable_bundle
```

### 4) RC wrapper gate

Run release hygiene plus optional lint/tests/performance and official checks:

```bash
python scripts/rc1_release_gate.py \
  --run-ruff \
  --run-pytest \
  --run-performance-smoke \
  --verify-official-dir <official_output_dir>
```

Optional confirmatory-ready governance check:

```bash
python scripts/rc1_release_gate.py \
  --verify-official-dir <confirmatory_output_dir> \
  --confirmatory-ready-dir <confirmatory_output_dir> \
  --confirmatory-ready-summary-out outputs/release/confirmatory_ready_summary.json
```

Use `--repro-command` to include a reproducibility check in the same gate run:

```bash
python scripts/rc1_release_gate.py \
  --run-ruff \
  --run-pytest \
  --repro-command "python scripts/verify_official_reproducibility.py --mode protocol --config configs/protocols/thesis_confirmatory_v1.json --index-csv <dataset_index.csv> --data-root <data_root> --cache-dir <cache_dir> --suite confirmatory_primary_within_subject"
```

Or use native replay/bundle options:

```bash
python scripts/rc1_release_gate.py \
  --run-official-replay \
  --replay-use-demo-dataset \
  --replay-verify-determinism \
  --verify-bundle-dir outputs/reproducibility/publishable_bundle
```

### 5) Phased frozen campaign orchestration

`scripts/run_frozen_campaign.ps1` now supports explicit phases:

- `precheck`
- `confirmatory`
- `comparison`
- `replay`
- `bundle`
- `all` (ordered execution: `precheck -> confirmatory -> comparison -> replay -> bundle`)

Execution modes (`-ExecutionMode`):

- `fresh`: refuse to run when selected phase root contains prior artifacts.
- `resume` (default): reuse completed artifacts and rerun only missing/failed/`timed_out` units.
- `force`: rerun selected phase work from scratch.

Dependency enforcement:

- `confirmatory` requires `precheck=passed`.
- `comparison` requires `precheck=passed`.
- `replay` requires `precheck=passed`, `confirmatory=passed`, `comparison=passed`.
- `bundle` requires `precheck=passed`, `confirmatory=passed`, `comparison=passed`, `replay=passed`.

Blocking model:

- confirmatory readiness blockers: `precheck`, `confirmatory`
- campaign sign-off blockers: all phases

Timeout watchdog and terminal status semantics:

- official run terminal states are explicit and machine-readable:
  - `success`
  - `failed`
  - `timed_out`
  - `skipped_due_to_policy`
- default timeout policy for official runs:
  - confirmatory: 45 minutes
  - comparison: 90 minutes
  - `logreg` override: 120 minutes
  - shutdown grace: 30 seconds
  - absolute hard ceiling: 180 minutes
- timed-out runs emit `timeout_diagnostics.json` in run directories and are counted in phase summaries (`n_timed_out`).
- confirmatory phase treats timed-out runs as blocking.
- comparison phase can be recorded as `partial` when timed-out/skipped runs are present.
- replay/bundle remain dependency-gated and will not proceed unless required upstream phases are `passed`.

Operational parallel scheduling control:

- protocol/comparison runners accept `--max-parallel-runs <N>` (default `1`).
- frozen campaign orchestration forwards `-MaxParallelRuns <N>` to official protocol/comparison phases.
- ordering of run-level reporting artifacts is deterministic after parallel execution (sorted by compiled plan order).
- this control is operational only; it does not alter splits, metrics, models, or evidence policy.

Machine-readable orchestration artifacts:

- campaign manifest: `outputs/campaign/<CampaignTag>/campaign_manifest.json`
- per-phase status: `<phase_root>/phase_status.json`
- per-phase summary: `<phase_root>/phase_summary.json`
- per-run resumability index (confirmatory/comparison): `<official_output_dir>/run_index.json`
- per-run reconciliation summary (confirmatory/comparison): `<official_output_dir>/resume_reconciliation.json`

Additive timing artifacts (Phase A Slice 1):

- run-level `fold_metrics.csv` includes per-fold fit/search timing fields.
- tuned `best_params_per_fold.csv` includes search timing summary columns.
- each run emits `fit_timing_summary.json`; path is stamped in `config.json` and `metrics.json`.

Additive compute-policy metadata and scheduling (PR 5):

- exploratory, comparison, and protocol CLIs accept operational compute controls:
  `--hardware-mode`, `--gpu-device-id`, `--deterministic-compute`, `--allow-backend-fallback`.
- exploratory CLI also exposes bounded lane-planning controls:
  `--max-parallel-runs`, `--max-parallel-gpu-runs`.
- PR 5 records resolved compute metadata in run artifacts and enables exploratory run-level
  `max_both` CPU/GPU lane assignment while preserving model/scientific semantics.
- PR 5 does not introduce in-fit hybrid execution and does not change official scientific semantics.
- PR 6 introduces locked-comparison GPU admission gates:
  - official comparison `gpu_only` is selectively admitted only for explicit approved combinations (currently `ridge` + `torch_gpu`);
  - `deterministic_compute=true` is required;
  - `allow_backend_fallback=true` remains rejected;
  - official `max_both` remains rejected.
- PR 7 extends the same conservative gate to confirmatory protocol execution:
  - official confirmatory `gpu_only` is selectively admitted only for explicit approved combinations (currently `ridge` + `torch_gpu`);
  - `deterministic_compute=true` is required;
  - `allow_backend_fallback=true` remains rejected;
  - official `max_both` remains rejected.

## Official RC checklist

Before freezing an experiment campaign:

1. Prepare a clean campaign output root and archive legacy output folders:
   `powershell -ExecutionPolicy Bypass -File scripts/prepare_frozen_campaign.ps1 -CampaignTag "campaign-YYYY-MM-DD-rc1"`.
2. Run frozen campaign precheck phase:
   `powershell -ExecutionPolicy Bypass -File scripts/run_frozen_campaign.ps1 -CampaignTag "campaign-YYYY-MM-DD-rc1" -IndexCsv "<index_csv>" -DataRoot "<data_root>" -CacheDir "<cache_dir>" -Phase precheck -ExecutionMode fresh`.
   This phase includes formal model-cost policy validation and runtime profiling precheck, and writes:
   `outputs/campaign/<CampaignTag>/release/precheck/model_cost_policy_precheck_summary.json`.
   `outputs/campaign/<CampaignTag>/release/precheck/campaign_runtime_profile_summary.json`.
   Profiling run artifacts are precheck-only/non-evidentiary and are written under:
   `outputs/campaign/<CampaignTag>/release/precheck/runtime_profile_runs/`.
   Grouped nested tuning cohorts use the smallest valid profiling slice when possible.
   If a cohort has no valid measured profiling slice on the dataset, precheck records an explicit
   conservative fallback estimate (`estimate_source=conservative_fallback`, low confidence).
   If profiling execution fails unexpectedly, precheck fails with explicit `issues`.
3. Run frozen campaign confirmatory phase:
   `powershell -ExecutionPolicy Bypass -File scripts/run_frozen_campaign.ps1 -CampaignTag "campaign-YYYY-MM-DD-rc1" -IndexCsv "<index_csv>" -DataRoot "<data_root>" -CacheDir "<cache_dir>" -Phase confirmatory -ExecutionMode resume`.
4. Run frozen campaign comparison phase:
   `powershell -ExecutionPolicy Bypass -File scripts/run_frozen_campaign.ps1 -CampaignTag "campaign-YYYY-MM-DD-rc1" -IndexCsv "<index_csv>" -DataRoot "<data_root>" -CacheDir "<cache_dir>" -Phase comparison -ExecutionMode resume`.
5. Run frozen campaign replay phase:
   `powershell -ExecutionPolicy Bypass -File scripts/run_frozen_campaign.ps1 -CampaignTag "campaign-YYYY-MM-DD-rc1" -IndexCsv "<index_csv>" -DataRoot "<data_root>" -CacheDir "<cache_dir>" -Phase replay -ExecutionMode resume`.
6. Run frozen campaign bundle phase:
   `powershell -ExecutionPolicy Bypass -File scripts/run_frozen_campaign.ps1 -CampaignTag "campaign-YYYY-MM-DD-rc1" -IndexCsv "<index_csv>" -DataRoot "<data_root>" -CacheDir "<cache_dir>" -Phase bundle -ExecutionMode resume`.
7. Optional one-command equivalent:
   `powershell -ExecutionPolicy Bypass -File scripts/run_frozen_campaign.ps1 -CampaignTag "campaign-YYYY-MM-DD-rc1" -IndexCsv "<index_csv>" -DataRoot "<data_root>" -CacheDir "<cache_dir>" -Phase all -ExecutionMode resume`.

Model-cost policy enforcement in official paths:

- confirmatory protocols must restrict models to `official_fast`/`official_allowed` tiers.
- locked comparison specs must explicitly allow `benchmark_expensive` variants via:
  `cost_policy.explicit_benchmark_expensive_models`.
- projected runtime limits are policy fields in the spec/protocol contracts:
  - `model_cost_policy.max_projected_runtime_seconds_per_run`
  - `cost_policy.max_projected_runtime_seconds_per_run`
- timeout watchdogs remain fallback safety; policy precheck is the primary guardrail.

## Confirmatory-ready boundary

Confirmatory-ready means governance + contract + artifact checks passed for frozen
scientific evidence. It does **not** imply clinical readiness, deployment approval, or
unrestricted external generalization.
