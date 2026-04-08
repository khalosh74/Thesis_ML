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

Authority layering (Phase 1):
- scientific authority: `configs/confirmatory/confirmatory_scope_v1.json`, `docs/Science/CLAIM_EVALUATION_RULES.md`, `docs/Science/THESIS_TRACEABILITY_MATRIX.md`
- thesis runtime authority: `configs/decision_support_registry_revised_execution.json`
- package/demo registry: `configs/decision_support_registry.json`
- generation authority: `templates/thesis_experiment_program.xlsx` plus workbook compiler/template constants
- derived mirrors only: `src/Thesis_ML/assets/configs/decision_support_registry.json`, `src/Thesis_ML/assets/templates/thesis_experiment_program.xlsx`
- archive-only backups: `configs/archive/registries/`
- derived/package/archive files are not peer authorities to scientific/runtime/generation truth

- Decision-support registry default:
  - source checkout thesis runtime: `configs/decision_support_registry_revised_execution.json`
  - source checkout package/demo: `configs/decision_support_registry.json`
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
- operational parallel fan-out control:
  - `-MaxParallelRuns <N>` (default `1`, serial-safe)
  - affects scheduling only; scientific methodology/artifact contracts remain unchanged

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
- for grouped nested tuning cohorts, profiler selects the smallest valid representative slice.
  if no valid measured slice exists on the dataset, it uses an explicit conservative fallback
  estimate for that cohort.
- summary artifact:
  `outputs/campaign/<CampaignTag>/release/precheck/campaign_runtime_profile_summary.json`
- profiling run artifacts are isolated under:
  `outputs/campaign/<CampaignTag>/release/precheck/runtime_profile_runs/`
- profiling outputs are precheck-only and non-canonical; do not use them as thesis evidence.
- if a runtime cohort cannot produce either a measured profile estimate or an explicit fallback
  estimate, precheck fails and records the failing cohorts in `issues`.
- per-cohort summary rows include:
  - `estimate_source`: `measured_profile` or `conservative_fallback`
  - `estimate_confidence`: `medium` or `low`
  - `profiling_subset_description`

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

Additive fit/search timing outputs:

- `fold_metrics.csv` now includes per-fold fit/search timing fields:
  - `outer_fold_elapsed_seconds`
  - `estimator_fit_elapsed_seconds`
  - `tuned_search_elapsed_seconds`
  - `tuned_search_candidate_count`
  - `cv_mean_fit_time_seconds`, `cv_std_fit_time_seconds`
  - `cv_mean_score_time_seconds`, `cv_std_score_time_seconds`
- `best_params_per_fold.csv` now includes tuned-search timing summary columns.
- `fit_timing_summary.json` is written per run and referenced from `config.json`, `metrics.json`, and run result payloads.

Stage observability artifacts (additive evidence layer):

- direct runs now emit stage boundary and observed evidence artifacts:
  - `stage_events.jsonl`
  - `stage_observed_evidence.json`
- when process profiling is enabled, per-stage resource attribution is merged and persisted in:
  - `stage_resource_attribution.json`
- campaign-level decision-support exports now include:
  - `stage_execution_summary.json`
  - `stage_resource_summary.csv`
  - `backend_fallback_summary.json`

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
  --comparison configs/comparisons/model_family_grouped_nested_comparison_v2.json \
  --all-variants \
  --reports-root outputs/reports/comparisons \
  --dry-run
```

Locked comparison execution:

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/model_family_grouped_nested_comparison_v2.json \
  --all-variants \
  --reports-root outputs/reports/comparisons
```

Grouped nested comparison execution:

```bash
thesisml-run-comparison \
  --comparison configs/archive/comparisons/model_family_grouped_nested_comparison_v1.json \
  --variant ridge \
  --reports-root outputs/reports/comparisons
```

Confirmatory frozen protocol run:

```bash
thesisml-run-protocol \
  --protocol configs/protocols/thesis_canonical_nested_v2.json \
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
- compute controls are operational:
  `--hardware-mode {cpu_only,gpu_only,max_both}`, `--gpu-device-id`,
  `--deterministic-compute`, `--allow-backend-fallback`,
  `--max-parallel-runs`, `--max-parallel-gpu-runs`.
- GPU execution requires a CUDA-enabled `torch` build in the project environment;
  `uv sync --frozen --extra dev` alone does not provision `torch`.
- PR 5 exploratory behavior:
  `gpu_only` can execute `ridge` and `logreg` through the torch GPU backend when capability is valid.
  `max_both` now performs deterministic run-level CPU/GPU lane assignment (no in-fit hybrid execution).
- PR 8 official behavior:
  locked comparison and confirmatory protocol can admit `--hardware-mode gpu_only` only for
  explicit approved combinations (currently `ridge` on `torch_gpu`).
  locked comparison can now admit `--hardware-mode max_both` conservatively with deterministic
  compute, no fallback, and explicit run-level lane stamping; only approved model/backend
  combinations are GPU-lane eligible (currently `ridge` on `torch_gpu`).
  confirmatory `--hardware-mode max_both` remains rejected.
  unsupported official GPU model/backend requests still fail clearly.
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
- `config.json`, `metrics.json`, and official execution status rows now also stamp additive compute-policy metadata for auditability.

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

## Windows SSH access to Olympus

Use the built-in OpenSSH client on Windows. The reusable config file is `%USERPROFILE%\.ssh\config`.

```sshconfig
Host olympus
  HostName olympus.dsv.su.se
  User khal6952
  Port 22
  ServerAliveInterval 30
  ServerAliveCountMax 3
  IdentitiesOnly yes
  # If you use a key, uncomment and point this at your private key.
  # IdentityFile ~/.ssh/id_ed25519
```

Connect with:

```powershell
ssh olympus
```

If you already have a working password login, you can leave `IdentityFile` out. If you use a key, the private key stays on your Windows machine and the matching public key must be registered with the university account or server-side SSH setup.

VS Code Remote-SSH can reuse the same `olympus` alias.

This only covers the login layer. When we later touch Slurm job scripts, I will wire in your required `--account=user` flag there rather than here.

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
  --registry configs/decision_support_registry_revised_execution.json \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --output-root outputs/artifacts/decision_support \
  --all
```

Freeze workbook to execution JSON registry (recommended for stable reruns):

```bash
python scripts/freeze_workbook_registry.py \
  --workbook workbooks/thesis_program_instances/thesis_experiment_program_revised_v1.xlsx \
  --output configs/decision_support_registry_revised_execution.json
```

Note: `workbooks/thesis_program_instances/thesis_experiment_program_revised_v1.xlsx` is treated as a study workbook instance,
not as a generic reusable template.

Run from the frozen execution registry:

```bash
thesisml-run-decision-support \
  --registry configs/decision_support_registry_revised_execution.json \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --output-root outputs/artifacts/decision_support \
  --all
```

Optuna mode requires optional dependency installation (`--extra optuna` or `.[optuna]`).
Package/demo registry example (non-thesis runtime default):

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
python scripts/profile_runtime_baseline.py --mode ci_synthetic
python scripts/profile_runtime_baseline.py --mode operator_dataset --index-csv <...> --data-root <...> --cache-dir <...>
python scripts/profile_runtime_baseline.py --mode ci_synthetic --compare-against outputs/performance/baselines/<previous_tag>/baseline_bundle.json
python scripts/verify_official_artifacts.py --output-dir <official_output_dir>
python scripts/verify_confirmatory_ready.py --output-dir <confirmatory_output_dir>
python scripts/replay_official_paths.py --mode confirmatory --index-csv <...> --data-root <...> --cache-dir <...> --suite confirmatory_primary_within_subject --verify-determinism --skip-confirmatory-ready
python scripts/replay_official_paths.py --mode both --use-demo-dataset --verify-determinism
python scripts/build_publishable_bundle.py --output-dir <bundle_dir> --comparison-output <...> --confirmatory-output <...> --replay-summary <...> --replay-verification-summary <...> --repro-manifest <...>
python scripts/verify_publishable_bundle.py --bundle-dir <bundle_dir>
python scripts/rc1_release_gate.py --run-ruff --run-pytest --run-performance-smoke
```

`scripts/performance_smoke.py` records timing for:
- workbook build/save/load/compile
- `thesisml-run-comparison` dry-run
- `thesisml-run-protocol` dry-run

`scripts/profile_runtime_baseline.py` is the Phase 0 canonical baseline bundle runner:
- purpose: proof-oriented measurement and parity comparison for future performance work
- `--mode ci_synthetic`: runs the full suite using only shipped assets
- `--mode operator_dataset`: runs the same suite against an operator dataset
- cases:
  - `performance_smoke_existing`
  - `single_run_ridge_direct` (direct run with process profiling artifacts)
  - `runtime_profile_precheck`
  - `decision_support_dry_run`
- `--compare-against <prior_bundle>` produces scientific-parity and observability-parity comparison output

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

## Merged content: Operator Guide (from `OPERATOR_GUIDE.md`)

The following content was merged from `docs/OPERATOR_GUIDE.md` on 2026-04-02. A backup of the original `OPERATOR_GUIDE.md` is available at `docs/archived/OPERATOR_GUIDE_backup_2026-04-02.md`.

---

## Operator Guide (Canonical)

Use these commands as the supported operator path.

### 1) Environment

Canonical operator path is source checkout + `uv.lock`.

```bash
python -m pip install --upgrade pip
python -m pip install uv
python -m uv sync --frozen --extra dev
```

Optional Optuna support:

```bash
python -m uv sync --frozen --extra dev --extra optuna
```

Installed wheel path is also supported; default decision-support registry is packaged in the wheel.

### 2) Canonical CLIs

- `thesisml-run-protocol` (official thesis-facing command)
- `thesisml-run-experiment`
- `thesisml-run-comparison`
- `thesisml-run-decision-support`
- `thesisml-workbook`
- `thesisml-run-baseline`

Compatibility wrappers are kept only for migration and are deprecated.

### 3) First-use checks

Frozen confirmatory protocol dry-run:

```bash
thesisml-run-protocol \
  --protocol configs/protocols/thesis_canonical_nested_v2.json \
  --all-suites \
  --reports-root outputs/reports/confirmatory \
  --dry-run
```

Frozen confirmatory protocol execution:

```bash
thesisml-run-protocol \
  --protocol configs/protocols/thesis_confirmatory_v1.json \
  --all-suites \
  --reports-root outputs/reports/confirmatory
```

Locked comparison dry-run:

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/model_family_grouped_nested_comparison_v2.json \
  --all-variants \
  --reports-root outputs/reports/comparisons \
  --dry-run
```

Locked comparison execution:

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/model_family_grouped_nested_comparison_v2.json \
  --all-variants \
  --reports-root outputs/reports/comparisons
```

Grouped nested comparison execution:

```bash
thesisml-run-comparison \
  --comparison configs/archive/comparisons/model_family_grouped_nested_comparison_v1.json \
  --variant ridge \
  --reports-root outputs/reports/comparisons
```

Official-policy note:
- exploratory mode (`thesisml-run-experiment`) is ad hoc and non-confirmatory.
- locked comparison mode (`thesisml-run-comparison`) is restricted to registered variants.
- confirmatory mode (`thesisml-run-protocol`) is the final thesis evidence path.
- default mode roots are `outputs/reports/exploratory`, `outputs/reports/comparisons`, and `outputs/reports/confirmatory`.
- comparison/protocol contracts must choose exactly one methodology policy:
  `fixed_baselines_only` or `grouped_nested_tuning`.
- official checked-in comparison/protocol specs use `repeat_count=3` by default.
- locked comparison specs require significant paired wins by default (`require_significant_win=true`).
- locked comparison runs emit `comparison_decision.json` for machine-readable selection outcomes.
- locked comparison runs emit evidence artifacts:
  `repeated_run_metrics.csv`, `repeated_run_summary.json`,
  `confidence_intervals.json`, `metric_intervals.csv`,
  `paired_model_comparisons.json`, `paired_model_comparisons.csv`.
- confirmatory runs emit evidence artifacts:
  `repeated_run_metrics.csv`, `repeated_run_summary.json`,
  `confidence_intervals.json`, `metric_intervals.csv`.
- official comparison/confirmatory paths run strict preflight contract validation and fail fast on violations.
- run-level `run_status.json` exposes structured diagnostics (`error_code`, `error_type`, `failure_stage`) and warning/timing/resource summaries.

Official metric policy:
- declare one `primary_metric` per comparison/protocol contract (`balanced_accuracy`, `macro_f1`, or `accuracy`)
- primary metric is decision-driving for tuning, comparison winner selection, permutation testing, and headline summaries
- secondary metrics are reporting-only
- official permutation metric must match primary metric
- artifacts include `metric_policy_effective` so operators can audit effective primary/decision/tuning/permutation metrics
- official run artifacts include deterministic provenance (`git_provenance`, `dataset_fingerprint`) for rerun traceability
- official run artifacts include evidence metadata (`evidence_policy_effective`, repeat metadata, `evidence_run_role`)
- calibration status is explicit in run artifacts:
  - `performed` when probability outputs are available
  - `not_applicable` when a model path has no probability outputs
- official comparison/confirmatory contracts also declare explicit `data_policy`:
  - structural data/leakage blockers fail official preflight
  - class-balance and missingness thresholds warn by default unless set to blocking
  - external validation is compatibility-only in this phase (explicitly labeled external/non-confirmatory)
- official run artifacts include standardized data-layer outputs:
  `dataset_card.json`, `dataset_card.md`, `dataset_summary.json`, `dataset_summary.csv`,
  `data_quality_report.json`, `class_balance_report.csv`, `missingness_report.csv`,
  `leakage_audit.json`, `external_dataset_card.json`, `external_dataset_summary.json`,
  `external_validation_compatibility.json`
- additive stage observability artifacts are emitted for run-level plan-vs-observed evidence:
  `stage_events.jsonl`, `stage_observed_evidence.json`
- with process profiling enabled, stage-scoped resource attribution is available in
  `stage_resource_attribution.json`

Governance references:
- `docs/PRIVACY_AND_DATA_HANDLING.md`
- `docs/USE_AND_MISUSE_BOUNDARIES.md`
- `docs/CONFIRMATORY_READY.md`
- `docs/REPRODUCIBILITY.md`

Architecture ownership (where to extend safely):
- runner policy resolution belongs in `src/Thesis_ML/experiments/runtime_policies.py`
- run artifact payload shaping belongs in `src/Thesis_ML/experiments/run_artifacts.py`
- comparison decision rules belong in `src/Thesis_ML/comparisons/decision.py`
- workbook study-review guardrail helpers belong in `src/Thesis_ML/orchestration/study_review.py`

Generate workbook template:

```bash
thesisml-workbook --output templates/thesis_experiment_program.xlsx
```

Template note:
- `templates/thesis_experiment_program.xlsx` is non-runnable by default
- enable/populate executable rows in `Experiment_Definitions` for single-experiment flow
- or enable/populate `Study_Design` (+ `Factors`/`Fixed_Controls`/`Constraints` as needed)
  for factorial flow
- optionally complete `Study_Rigor_Checklist` and `Analysis_Plan` for explicit
  scientific-rigor metadata and stricter confirmatory validation

## End merged content
