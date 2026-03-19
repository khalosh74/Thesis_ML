# Operator Runbook

This runbook covers normal operation on a clean machine and day-to-day rerun handling.

## 1) Install

Canonical:

```bash
python -m pip install --upgrade pip
python -m pip install uv
python -m uv sync --frozen --extra dev
```

Optional Optuna support:

```bash
python -m uv sync --frozen --extra dev --extra optuna
```

Compatibility:

```bash
python -m pip install -e ".[dev]"
```

For Optuna mode with editable installs:

```bash
python -m pip install -e ".[dev,optuna]"
```

## 2) Paths and defaults

Configured in `src/Thesis_ML/config/paths.py`.

- Decision-support registry default:
  - source checkout: `configs/decision_support_registry.json`
  - installed wheel: packaged asset under `Thesis_ML/assets/configs/decision_support_registry.json`
- Workbook generation default output: `templates/thesis_experiment_program.xlsx` under current project/cwd
- Shipped workbook template asset:
  - source checkout: `templates/thesis_experiment_program.xlsx`
  - installed wheel: packaged asset under `Thesis_ML/assets/templates/thesis_experiment_program.xlsx`
- Output root: `outputs/`
  - exploratory reports: `outputs/reports/exploratory/`
  - locked comparison reports: `outputs/reports/comparisons/`
  - confirmatory reports: `outputs/reports/confirmatory/`
  - decision-support campaign artifacts: `outputs/artifacts/decision_support/`
  - workbook write-back files: `outputs/workbooks/`

## 3) Framework mode commands

Frozen campaign preparation (PowerShell, recommended before running `scripts/run_frozen_campaign.ps1`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_frozen_campaign.ps1 `
  -CampaignTag "campaign-YYYY-MM-DD-rc1"
```

Optional prune after successful archive:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_frozen_campaign.ps1 `
  -CampaignTag "campaign-YYYY-MM-DD-rc1" `
  -PruneAfterArchive
```

This command archives legacy folders that often cause confusion
(`outputs/manual_validation`, `outputs/reproducibility`, `outputs/reports`),
creates a clean `outputs/campaign/<CampaignTag>/...` directory tree, and writes:
`outputs/campaign/<CampaignTag>/release/prep_summary.json`.

Frozen campaign phased execution (`scripts/run_frozen_campaign.ps1`):

- `-Phase precheck`
- `-Phase confirmatory`
- `-Phase comparison`
- `-Phase replay`
- `-Phase bundle`
- `-Phase all` (runs in order: `precheck -> confirmatory -> comparison -> replay -> bundle`)

Execution modes (`-ExecutionMode`, default `resume`):

- `fresh`
  - fails if the selected phase output root already contains artifacts (other than `phase_status.json`)
  - never silently reuses old outputs
- `resume`
  - reuses completed work from disk
  - reruns only missing / failed / `timed_out` runs in confirmatory/comparison phases
  - replay/bundle phases are skipped when existing verification artifacts already report `passed=true`
- `force`
  - reruns selected phase work from scratch, ignoring existing outputs

Phase dependencies and blocking semantics:

- `confirmatory` depends on `precheck`.
- `comparison` depends on `precheck`.
- `replay` depends on `precheck`, `confirmatory`, and `comparison`.
- `bundle` depends on `precheck`, `confirmatory`, `comparison`, and `replay`.
- Confirmatory readiness blockers: `precheck`, `confirmatory`.
- Campaign sign-off blockers: all phases.

Model cost policy (official paths):

- model runtime cost is enforced as policy before frozen execution, not only by timeout fallback.
- cost tiers:
  - `official_fast`
  - `official_allowed`
  - `benchmark_expensive`
  - `exploratory_only`
- confirmatory protocols may only use `official_fast` and `official_allowed`.
- `benchmark_expensive` variants in locked comparisons must be declared in
  `cost_policy.explicit_benchmark_expensive_models`.
- projected runtime per run must be below the declared policy threshold:
  - confirmatory: `model_cost_policy.max_projected_runtime_seconds_per_run`
  - comparison: `cost_policy.max_projected_runtime_seconds_per_run`
- frozen campaign `precheck` runs:
  `scripts/verify_model_cost_policy_precheck.py`
  and writes:
  `outputs/campaign/<CampaignTag>/release/precheck/model_cost_policy_precheck_summary.json`

Runtime profiling precheck (non-evidentiary):

- frozen campaign `precheck` also runs:
  `scripts/verify_campaign_runtime_profile.py`
- it compiles the official confirmatory/comparison plans, profiles one representative run per
  runtime cohort with a profiling-only fold cap (`max_outer_folds=1`), and scales ETA estimates.
- summary artifact:
  `outputs/campaign/<CampaignTag>/release/precheck/campaign_runtime_profile_summary.json`
- profiling run artifacts are isolated under:
  `outputs/campaign/<CampaignTag>/release/precheck/runtime_profile_runs/`
- profiling outputs are precheck-only and non-canonical; do not use them as thesis evidence.
- if any runtime cohort cannot be profiled successfully, precheck fails and records the failing
  cohorts in `issues`.

Timeout watchdog policy (official runs):

- terminal run statuses are explicit:
  - `success`
  - `failed`
  - `timed_out`
  - `skipped_due_to_policy`
- default per-run wall-clock limits:
  - confirmatory mode: 45 minutes
  - locked comparison mode: 90 minutes
  - `logreg` override: 120 minutes
  - shutdown grace period: 30 seconds
  - absolute hard ceiling: 180 minutes
- timed-out runs emit `run_status.json` with `status=timed_out` and `timeout_diagnostics.json` in the run directory.
- phase behavior with timed-out runs:
  - confirmatory phase is blocking and does not pass when timed-out runs are present.
  - comparison phase can complete as `partial` and records timeout counts in phase summary.
  - replay and bundle phases remain dependency-gated and do not run unless upstream phases are `passed`.

Phase output roots:

- `outputs/campaign/<CampaignTag>/release/precheck/`
- `outputs/campaign/<CampaignTag>/confirmatory/`
- `outputs/campaign/<CampaignTag>/comparison/`
- `outputs/campaign/<CampaignTag>/release/replay/`
- `outputs/campaign/<CampaignTag>/bundle/`
- `outputs/campaign/<CampaignTag>/logs/<phase>/`

Campaign manifest and per-phase machine-readable artifacts:

- `outputs/campaign/<CampaignTag>/campaign_manifest.json`
- `<phase_root>/phase_status.json`
- `<phase_root>/phase_summary.json`
- phase summaries include `n_success`, `n_failed`, `n_timed_out`, `n_skipped_due_to_policy`, `n_planned`
- phase summaries also include resumability counters:
  - `n_existing_success`, `n_existing_failed`, `n_existing_timed_out`, `n_existing_skipped_due_to_policy`
  - `n_missing`, `n_rerun`, `n_reused`, `n_skipped_as_already_complete`
- confirmatory/comparison official outputs now include:
  - `run_index.json`
  - `resume_reconciliation.json`

Example phased commands:

```powershell
# precheck only
powershell -ExecutionPolicy Bypass -File scripts/run_frozen_campaign.ps1 `
  -CampaignTag "campaign-YYYY-MM-DD-rc1" `
  -IndexCsv "<index_csv>" `
  -DataRoot "<data_root>" `
  -CacheDir "<cache_dir>" `
  -Phase precheck `
  -ExecutionMode fresh

# confirmatory only
powershell -ExecutionPolicy Bypass -File scripts/run_frozen_campaign.ps1 `
  -CampaignTag "campaign-YYYY-MM-DD-rc1" `
  -IndexCsv "<index_csv>" `
  -DataRoot "<data_root>" `
  -CacheDir "<cache_dir>" `
  -Phase confirmatory `
  -ExecutionMode resume

# comparison only
powershell -ExecutionPolicy Bypass -File scripts/run_frozen_campaign.ps1 `
  -CampaignTag "campaign-YYYY-MM-DD-rc1" `
  -IndexCsv "<index_csv>" `
  -DataRoot "<data_root>" `
  -CacheDir "<cache_dir>" `
  -Phase comparison `
  -ExecutionMode resume

# replay only
powershell -ExecutionPolicy Bypass -File scripts/run_frozen_campaign.ps1 `
  -CampaignTag "campaign-YYYY-MM-DD-rc1" `
  -IndexCsv "<index_csv>" `
  -DataRoot "<data_root>" `
  -CacheDir "<cache_dir>" `
  -Phase replay `
  -ExecutionMode resume

# bundle only
powershell -ExecutionPolicy Bypass -File scripts/run_frozen_campaign.ps1 `
  -CampaignTag "campaign-YYYY-MM-DD-rc1" `
  -IndexCsv "<index_csv>" `
  -DataRoot "<data_root>" `
  -CacheDir "<cache_dir>" `
  -Phase bundle `
  -ExecutionMode resume

# all phases
powershell -ExecutionPolicy Bypass -File scripts/run_frozen_campaign.ps1 `
  -CampaignTag "campaign-YYYY-MM-DD-rc1" `
  -IndexCsv "<index_csv>" `
  -DataRoot "<data_root>" `
  -CacheDir "<cache_dir>" `
  -Phase all `
  -ExecutionMode resume
```

Exploratory run:

```bash
thesisml-run-experiment \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --target coarse_affect \
  --model ridge \
  --cv within_subject_loso_session \
  --subject sub-001 \
  --run-id exploratory_sub001_ridge
```

Locked comparison dry-run:

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/model_family_comparison_v1.json \
  --all-variants \
  --reports-root outputs/reports/comparisons \
  --dry-run
```

Locked comparison execution:

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/model_family_comparison_v1.json \
  --all-variants \
  --reports-root outputs/reports/comparisons
```

Grouped nested comparison execution:

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/model_family_grouped_nested_comparison_v1.json \
  --variant ridge \
  --reports-root outputs/reports/comparisons
```

Confirmatory frozen protocol run:

```bash
thesisml-run-protocol \
  --protocol configs/protocols/thesis_confirmatory_v1.json \
  --all-suites \
  --reports-root outputs/reports/confirmatory
```

Confirmatory dry-run validation/compilation:

```bash
thesisml-run-protocol \
  --protocol configs/protocols/thesis_confirmatory_v1.json \
  --all-suites \
  --reports-root outputs/reports/confirmatory \
  --dry-run
```

Policy note:
- exploratory mode is flexible and not confirmatory evidence.
- locked comparison mode allows only declared variants from comparison specs.
- confirmatory mode must load thesis-critical settings from protocol JSON, not ad hoc CLI flags.
- comparison/protocol specs must declare one methodology policy:
  `fixed_baselines_only` or `grouped_nested_tuning`.
- official checked-in comparison/protocol specs use `repeat_count=3` by default.
- locked comparison specs require significant paired wins by default (`require_significant_win=true`).
- locked comparison outputs include `comparison_decision.json` for machine-readable winner/inconclusive/invalid decisions.
- locked comparison outputs also include evidence artifacts:
  `repeated_run_metrics.csv`, `repeated_run_summary.json`,
  `confidence_intervals.json`, `metric_intervals.csv`,
  `paired_model_comparisons.json`, `paired_model_comparisons.csv`.
- confirmatory protocol outputs include evidence artifacts:
  `repeated_run_metrics.csv`, `repeated_run_summary.json`,
  `confidence_intervals.json`, `metric_intervals.csv`.
- official comparison/confirmatory paths enforce strict preflight contracts and fail fast on violations.
- `run_status.json` now includes structured failure diagnostics (`error_code`, `error_type`, `failure_stage`) and warning/timing/resource summaries.

Metric policy note (official runs):
- one declared `primary_metric` governs tuning, decision selection, permutation testing, and headline reporting
- `secondary_metrics` are descriptive-only
- official permutation metric must equal the primary metric
- `config.json`, `metrics.json`, and mode-level manifests include `metric_policy_effective`
- implicit metric fallbacks are disabled; workbook/registry/search-space inputs must explicitly declare required metric fields
- official run artifacts include deterministic provenance blocks (`git_provenance`, `dataset_fingerprint`)
- official run artifacts include evidence metadata:
  `evidence_policy_effective`, `repeat_id`, `repeat_count`, `base_run_id`, `evidence_run_role`
- calibration status is explicit in official run artifacts:
  - `performed` when probability outputs are available
  - `not_applicable` when a model path does not expose probability outputs
- official comparison/confirmatory contracts declare `data_policy` explicitly:
  - structural integrity/leakage blockers fail official preflight
  - threshold checks (class balance, missingness) warn by default unless configured as blocking
  - external validation support is compatibility-only in this phase
- official run directories now include standardized data-layer artifacts:
  `dataset_card.json`, `dataset_card.md`, `dataset_summary.json`, `dataset_summary.csv`,
  `data_quality_report.json`, `class_balance_report.csv`, `missingness_report.csv`,
  `leakage_audit.json`, `external_dataset_card.json`, `external_dataset_summary.json`,
  `external_validation_compatibility.json`
- governance references:
  - `docs/PRIVACY_AND_DATA_HANDLING.md`
  - `docs/USE_AND_MISUSE_BOUNDARIES.md`
  - `docs/CONFIRMATORY_READY.md`
  - `docs/REPRODUCIBILITY.md`

Implementation ownership notes (for maintainers):
- framework/methodology/metric runtime resolution: `src/Thesis_ML/experiments/runtime_policies.py`
- run artifact payload stamping/building: `src/Thesis_ML/experiments/run_artifacts.py`
- comparison decision logic: `src/Thesis_ML/comparisons/decision.py`
- workbook study-review guardrail helpers: `src/Thesis_ML/orchestration/study_review.py`

## 4) Rerun / resume behavior

`run_experiment` is intentionally strict.

- Existing completed run + same `run_id`: fails unless `--force`
- Existing partial/unknown run dir: fails unless `--resume` or `--force`
- `--force`: clears run directory, then reruns
- `--resume`: resumes from existing run directory
- `--resume` automatically enables same-run section artifact reuse
- `--reuse-completed-artifacts`: explicitly enables same-run section artifact reuse
- `--reuse-completed-artifacts` never reuses artifacts from a different `run_id`
- `--force` and `--resume` are mutually exclusive

`thesisml-run-protocol` forwards `--force`/`--resume` to underlying concrete runs.
`thesisml-run-comparison` forwards `--force`/`--resume` to underlying concrete runs.

Run state file:
- `outputs/reports/<mode>/<run_id>/run_status.json`

## 5) Decision-support campaign (registry)

```bash
thesisml-run-decision-support \
  --registry configs/decision_support_registry.json \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --output-root outputs/artifacts/decision_support \
  --all
```

Optuna mode requires optional dependency installation (`--extra optuna` or `.[optuna]`):

```bash
thesisml-run-decision-support \
  --registry configs/decision_support_registry.json \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --output-root outputs/artifacts/decision_support \
  --all \
  --search-mode optuna \
  --optuna-trials 25
```

## 6) Decision-support campaign (workbook-driven)

`templates/thesis_experiment_program.xlsx` is a **planning template** by default.
It validates structurally but is intentionally non-runnable until at least one row in
`Experiment_Definitions` is enabled (`enabled=Yes`) and populated with required fields
(`target`, `cv`, `model`).

Prepare a runnable workbook by generating a copy and editing executable rows first:

```bash
thesisml-workbook --output outputs/workbooks/my_campaign.xlsx
# edit outputs/workbooks/my_campaign.xlsx: set enabled=Yes and fill required columns
```

```bash
thesisml-run-decision-support \
  --workbook outputs/workbooks/my_campaign.xlsx \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --output-root outputs/artifacts/decision_support \
  --all \
  --write-back-workbook
```

Write-back safety:
- write-back target name is versioned (`<source>__results_<campaign_id>.xlsx`)
- existing targets are never overwritten unless explicit API-level `overwrite_existing=True` is used

## 7) Workbook generation

```bash
thesisml-workbook --output templates/thesis_experiment_program.xlsx
```

Template policy:
- generated/shipped template is governance-first and non-runnable by default
- workbook compiler will fail until executable rows are explicitly enabled/populated

## 8) Health checks

```bash
python -m mypy
python -m ruff check src/Thesis_ML/artifacts src/Thesis_ML/orchestration src/Thesis_ML/workbook \
  src/Thesis_ML/experiments/segment_execution.py src/Thesis_ML/experiments/sections.py \
  src/Thesis_ML/experiments/run_experiment.py src/Thesis_ML/protocols src/Thesis_ML/comparisons \
  src/Thesis_ML/cli/protocol_runner.py src/Thesis_ML/cli/comparison_runner.py
python -m ruff format --check src/Thesis_ML/artifacts src/Thesis_ML/orchestration src/Thesis_ML/workbook \
  src/Thesis_ML/experiments/segment_execution.py src/Thesis_ML/experiments/sections.py \
  src/Thesis_ML/experiments/run_experiment.py src/Thesis_ML/protocols src/Thesis_ML/comparisons \
  src/Thesis_ML/cli/protocol_runner.py src/Thesis_ML/cli/comparison_runner.py
python -m pytest -q
```

Release hygiene + performance smoke:

```bash
python scripts/release_hygiene_check.py
python scripts/performance_smoke.py --output outputs/performance/performance_smoke_summary.json
python scripts/verify_official_artifacts.py --output-dir <official_output_dir>
python scripts/verify_confirmatory_ready.py --output-dir <confirmatory_output_dir>
python scripts/verify_official_reproducibility.py --mode protocol --index-csv <...> --data-root <...> --cache-dir <...> --suite confirmatory_primary_within_subject
python scripts/replay_official_paths.py --mode both --use-demo-dataset --verify-determinism
python scripts/build_publishable_bundle.py --output-dir <bundle_dir> --comparison-output <...> --confirmatory-output <...> --replay-summary <...> --replay-verification-summary <...> --repro-manifest <...>
python scripts/verify_publishable_bundle.py --bundle-dir <bundle_dir>
python scripts/rc1_release_gate.py --run-ruff --run-pytest --run-performance-smoke
```

`scripts/performance_smoke.py` records timing for:
- workbook build/save/load/compile
- `thesisml-run-comparison` dry-run
- `thesisml-run-protocol` dry-run

`mypy` is a required CI gate and includes nibabel boundary modules:
- `src/Thesis_ML/spm/extract_glm.py`
- `src/Thesis_ML/features/nifti_features.py`

## 9) Gold acceptance check

```bash
python scripts/acceptance_smoke.py
```

This is the canonical pre-release/pre-campaign acceptance path and validates:
- shipped workbook asset compatibility
- canonical workbook CLI generation
- shipped registry dry-run through canonical decision-support CLI
