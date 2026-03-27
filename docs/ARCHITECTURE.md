# Architecture Overview

This project is a packaged, reproducible experimentation framework centered on:

- leakage-aware fMRI ML execution
- governed workbook-driven campaign control
- artifact traceability and write-back

## Package structure

Core package root: `src/Thesis_ML/`

- `artifacts/`
  - SQLite artifact registry (`registry.py`)
  - artifact types, compatibility lookup, run-level listing
- `cli/`
  - canonical user entry points (`comparison_runner.py`, `protocol_runner.py`, `decision_support.py`, `workbook.py`, `baseline.py`)
- `comparisons/`
  - strict comparison spec schema (`models.py`)
  - comparison JSON loader/validation (`loader.py`)
  - locked variant compiler (`compiler.py`)
  - comparison runner bridge onto low-level runner (`runner.py`)
  - comparison decision logic (`decision.py`)
  - comparison-level artifact writers (`artifacts.py`)
- `config/`
  - default paths (`paths.py`)
  - framework lifecycle mode enum (`framework_mode.py`)
  - methodology + decision policy contracts (`methodology.py`)
  - centralized metric policy registry/scorers (`metric_policy.py`)
  - evidence policy contracts (`methodology.py`: repeat/CI/paired/permutation/calibration/required package)
  - schema/version constants (`schema_versions.py`)
- `data/`, `features/`, `spm/`
  - indexing, cache extraction, SPM utilities
- `experiments/`
  - full pipeline wrapper (`run_experiment.py`)
  - central model registry + admission policy (`model_registry.py`, `model_admission.py`)
  - fairness/evaluation contract validation (`comparison_contract.py`)
  - backend registry (`backend_registry.py`)
  - backend implementations (`backends/`)
  - compute capability probing (`compute_capabilities.py`)
  - compute policy resolution/stamping (`compute_policy.py`)
  - exploratory lane scheduler (`compute_scheduler.py`)
  - official preflight + artifact contract validation (`official_contracts.py`)
  - official data-policy evaluation + data artifact writer (`data_reporting.py`)
  - structured error taxonomy for machine-readable failures (`errors.py`)
  - deterministic provenance helpers (git + dataset fingerprints) (`provenance.py`)
  - runtime policy/context resolution (`runtime_policies.py`)
  - run artifact payload builders/stamping (`run_artifacts.py`)
  - section contracts/runtime adapters (`sections.py`)
  - shared section I/O models (`section_models.py`)
  - stage-aware execution metadata skeleton (`stage_execution.py`)
  - segment helper layer (`segment_execution_helpers.py`)
  - segment planner/executor (`segment_execution.py`)
  - idempotency/run-state policy (`execution_policy.py`)
  - grouped nested tuning search-space registry (`tuning_search_spaces.py`)
  - shared evidence statistics (`evidence_statistics.py`)
- `protocols/`
  - strict thesis protocol schema (`models.py`)
  - protocol JSON loader/validation (`loader.py`)
  - canonical suite compiler (`compiler.py`)
  - protocol runner bridge onto low-level runner (`runner.py`)
  - protocol-level artifact writers (`artifacts.py`)
- `orchestration/`
  - manifest contracts/compiler (`contracts.py`, `compiler.py`, `workbook_compiler.py`)
  - extracted study-design workbook compilation layer (`workbook_study_design_layer.py`)
  - workbook study-review guardrail helpers (`study_review.py`)
  - campaign execution (`campaign_engine.py`)
  - campaign CLI plumbing (`campaign_cli.py`)
  - result aggregation core + row rendering (`result_aggregation_core.py`, `result_aggregation_rows.py`)
  - compatibility facade (`result_aggregation.py`)
  - reporting/write-back bridge modules
  - backward-compatible facade (`decision_support.py`, `campaign_runner.py`)
- `workbook/`
  - workbook builder facade (`builder.py`)
  - workbook builder/orchestration facade (`template_builder.py`)
  - workbook constants catalog (`template_constants.py`)
  - governance sheet builders (`governance_sheet_builders.py`)
  - structured sheet core + design + operations modules
    (`structured_sheet_core.py`, `structured_sheet_design.py`, `structured_sheet_operations.py`)
  - compatibility structured sheet facade (`structured_execution_sheets.py`)
  - workbook primitives (`template_primitives.py`)
  - workbook validation layer (`template_validation.py`, `validation.py`)
  - metadata helpers (`schema_metadata.py`)
  - compatibility sheet/style helpers (`sheets/`, `styles.py`, `named_ranges.py`)
- `verification/`
  - official artifact invariant verifier (`official_artifacts.py`)
  - confirmatory-ready governance verifier (`confirmatory_ready.py`)
  - deterministic official-output comparator (`reproducibility.py`)
  - centralized reproducibility manifest builder (`repro_manifest.py`)

## Runtime flows

1. Exploratory experiment run (`thesisml-run-experiment`)
- Resolve run mode (`fresh`, `resume`, `forced_rerun`) and status file.
- Resolve framework/methodology/metric runtime policy via `experiments/runtime_policies.py`.
- Resolve operational compute policy/capabilities via `experiments/compute_policy.py` and `experiments/compute_capabilities.py`.
- Resolve run-level compute lane assignment through `experiments/compute_scheduler.py`.
- Resolve estimator construction through `experiments/backend_registry.py`; in PR 5 exploratory `ridge` and `logreg` can run on `torch_gpu` lanes in `gpu_only` or `max_both` scheduling when capability is valid, while other models remain CPU reference.
- Plan section path from `start_section`/`end_section`.
- Execute section adapters in order.
- Build/stamp run artifacts via `experiments/run_artifacts.py`.
- Register section artifacts in SQLite.
- Stamp `framework_mode=exploratory` and `canonical_run=false`.
- Write metrics/report files in `outputs/reports/exploratory/<run_id>/` by default.

2. Locked comparison run (`thesisml-run-comparison`)
- Load and validate registered comparison JSON (`comparison-spec-v1`).
- Compile declared variants into explicit concrete run specs.
- Validate official compute controls before dispatch (centralized in model admission policy):
  - `cpu_only` remains default.
  - `gpu_only` is admitted only for approved locked-comparison model/backend pairs (currently `ridge` on `torch_gpu`) with `deterministic_compute=true`.
  - `max_both` is admitted conservatively in locked comparison with deterministic-only execution, no fallback, explicit run-level lane assignment, and GPU lanes restricted to approved pairs (currently `ridge` on `torch_gpu`).
  - `allow_backend_fallback=true` remains rejected for official runs.
- Execute each run spec through existing `run_experiment(...)`.
- Stamp `framework_mode=locked_comparison` and comparison identity metadata.
- Emit comparison-level manifests/summaries under `outputs/reports/comparisons/`.
- Emit evidence-layer artifacts (`repeated_run_*`, `confidence_intervals.*`, `paired_model_comparisons.*`).
- Emit machine-readable `comparison_decision.json` (`winner_selected`, `inconclusive`, `invalid_comparison`) via `comparisons/decision.py`.
- Validate mode-level + run-level official artifacts via `verification/official_artifacts.py` before success return.

3. Confirmatory canonical thesis protocol run (`thesisml-run-protocol`)
- Load and validate canonical protocol JSON (`thesis-protocol-v1`).
- Compile official suites into explicit concrete run specs.
- Validate official compute controls before dispatch (centralized in model admission policy):
  - `cpu_only` remains default.
  - `gpu_only` is admitted only for approved confirmatory model/backend pairs (currently `ridge` on `torch_gpu`) with `deterministic_compute=true`.
  - `max_both` remains rejected for official runs.
  - `allow_backend_fallback=true` remains rejected for official runs.
- Execute each run spec through existing `run_experiment(...)`.
- Stamp `framework_mode=confirmatory` and `canonical_run=true`.
- Persist protocol metadata in run artifacts and emit protocol-level manifests/summaries under `outputs/reports/confirmatory/`.
- Emit evidence-layer artifacts (`repeated_run_*`, `confidence_intervals.*`) and required-evidence status in suite summaries.
- Validate mode-level + run-level official artifacts via `verification/official_artifacts.py` before success return.

4. Decision-support campaign (`thesisml-run-decision-support`)
- Compile JSON registry or workbook rows to internal manifest.
- Select experiments and expand variants/search spaces.
- Dispatch variants through `run_experiment(...)`.
- Export campaign summaries + manifests + decision notes.
- Optionally write results back to a versioned workbook copy.

5. Workbook lifecycle (`thesisml-workbook`)
- Generate governed template workbook.
- Compile executable rows to internal manifest (`workbook_compiler.py`).
- Run campaign and write machine-owned outputs only.

## Canonical interfaces

Use packaged commands from `pyproject.toml`:

- `thesisml-extract-glm`
- `thesisml-build-index`
- `thesisml-cache-features`
- `thesisml-run-experiment`
- `thesisml-run-comparison`
- `thesisml-run-protocol`
- `thesisml-run-decision-support`
- `thesisml-run-baseline`
- `thesisml-workbook`

Framework guardrails:
- `thesisml-run-experiment` cannot label outputs as confirmatory or accept protocol/comparison contexts.
- `thesisml-run-comparison` can execute only registered variants from a comparison spec.
- `thesisml-run-protocol` can execute only canonical protocol suites and cannot accept ad hoc science overrides.
- comparison/protocol contracts must explicitly declare methodology policy (`fixed_baselines_only` or `grouped_nested_tuning`).
- comparison/protocol contracts must explicitly declare `evidence_policy`.
- primary metric is centralized in `config.metric_policy` and is the single metric authority for official runs.
- decision metric, tuning metric, and permutation metric are resolved from primary metric and must align (drift raises validation/runtime errors).
- secondary metrics are emitted for descriptive reporting only.
- official artifacts (`config.json`, `metrics.json`, comparison/protocol manifests and summaries) persist `metric_policy_effective` for auditability.
- run artifacts also persist additive compute-policy metadata (`hardware_mode_*`, backend selection/fallback metadata, GPU device metadata when available); exploratory `ridge`/`logreg` may execute through `torch_gpu` lanes, PR 6 allows selectively gated locked-comparison `gpu_only` execution (`ridge`/`torch_gpu`), PR 7 allows the same conservative gate in confirmatory (`ridge`/`torch_gpu`), and PR 8 admits locked-comparison `max_both` with run-level lane stamping while confirmatory `max_both` remains disabled.
- run artifacts also persist additive scheduling metadata (`assigned_compute_lane`, `assigned_backend_family`, `lane_assignment_reason`, `scheduler_mode_effective`) for exploratory scheduling auditability and for officially admitted locked-comparison `max_both` runs (PR 8).
- run artifacts also persist additive stage-execution metadata (`stage_execution.policy`, `stage_execution.assignments`, `stage_execution.telemetry`) through the stage planner/registry layer (`experiments/stage_planner.py`, `experiments/stage_registry.py`) and `experiments/stage_execution.py`; Phase 2 routes `model_fit`, `tuning`, and `permutation` through typed stage assignments while preserving scientific behavior and existing workflow contracts, and Phase 3 adds a specialized exact-CPU grouped-nested `logreg` tuning executor with conservative planner-side resource/admissibility downgrades and explicit fallback reasons.
- run artifacts persist model governance metadata from the central registry/admission layer (`logical_model_name`, `model_family`, backend family/id, feature recipe, tuning search-space id/version, official admission summary, deterministic/scheduler metadata, and `model_registry_version`) so official-path model decisions are auditable from `config.json` and `metrics.json`.
- official comparison/confirmatory runs persist `data_policy_effective` and standardized data-layer artifacts
  (`dataset_card.*`, `dataset_summary.*`, `data_quality_report.json`, class/missingness reports, `leakage_audit.json`,
  and external compatibility artifacts when configured).
- run-level status artifacts (`run_status.json`) include structured failure diagnostics (`error_code`, `error_type`, `failure_stage`) plus warning/timing/resource summaries.
- official run artifacts persist deterministic provenance blocks (`git_provenance`, `dataset_fingerprint`) for rerun auditability.
- reproducibility orchestration scripts (`scripts/replay_official_paths.py`, `scripts/build_publishable_bundle.py`, `scripts/verify_publishable_bundle.py`) compose existing runners/verifiers without altering framework-mode contracts.

Compatibility wrappers are still present but deprecated:

- `run_decision_support_experiments.py`
- `create_thesis_experiment_workbook.py`
- `scripts/create_thesis_experiment_workbook.py`
- `scripts/run_baseline.py`
