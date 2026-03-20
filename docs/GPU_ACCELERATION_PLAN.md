# GPU Acceleration Enablement Plan
## CPU-only (default) + GPU-only + GPU+CPU Hybrid

Status: Planning document (no code implementation in this document)  
Audience: maintainers implementing runtime infrastructure and reviewers validating contract safety

---

## 1. Executive Summary

This document defines an implementation-ready plan to add GPU execution capabilities to the Thesis_ML framework while preserving:

1. Official-path boundaries (`exploratory`, `locked_comparison`, `confirmatory`).
2. Scientific contract strictness (no science-affecting drift through hardware flags).
3. Reproducibility and artifact integrity.

The system must support three operational hardware modes:

1. `cpu_only` (default, current baseline behavior).
2. `gpu_only` (GPU required; fail fast if invalid/unavailable).
3. `gpu_cpu_hybrid` (GPU lane + CPU lane concurrently; deterministic scheduling).

`cpu_only` remains the baseline/default mode. `gpu_only` and `gpu_cpu_hybrid` are additional opt-in operational modes.

---

## 2. Hard Requirements and Repository Invariants

The following are non-negotiable:

1. Official mode entrypoints remain unchanged in meaning:
   - `thesisml-run-experiment` = exploratory only
   - `thesisml-run-comparison` = locked comparison only
   - `thesisml-run-protocol` = confirmatory only
2. Hardware flags are operational controls only; they must not alter scientific declarations from comparison/protocol specs.
3. Existing official artifact contracts are preserved:
   - no removal/rename of required artifacts
   - no weakening of required metadata keys currently validated
4. Primary metric governance remains unchanged.
5. Methodology policy governance remains unchanged.
6. Default behavior remains `cpu_only` unless user explicitly opts into GPU modes.

---

## 3. Scope and Non-Goals

### 3.1 In Scope

1. Add runtime compute policy contracts.
2. Add GPU capability detection and validation.
3. Add backend abstraction for model fitting and tuning.
4. Add GPU execution paths for supported models.
5. Add dual-lane CPU/GPU scheduling for official and decision-support orchestration.
6. Add compute observability metadata to run artifacts (additive only).
7. Add deterministic controls and GPU failure taxonomy.
8. Extend runtime profiling/precheck to estimate hybrid execution.

### 3.2 Out of Scope

1. Changing official science contracts in protocol/comparison specs.
2. Changing target definitions, split logic, metric policies, or methodology governance semantics.
3. Changing feature cache schema as part of initial GPU rollout.
4. Multi-node/distributed cluster execution.
5. Rewriting non-model sections (dataset selection, spatial validation) for GPU.

---

## 4. Hardware Operating Modes (Normative)

### 4.1 `cpu_only` (default)

1. All sections run on CPU using current baseline implementation.
2. GPU detection may run for diagnostics only; no GPU compute is used.
3. Any GPU-related error is non-blocking in this mode.

### 4.2 `gpu_only`

1. GPU-capable path is mandatory for GPU-eligible model-fit/tuning operations.
2. If GPU capability checks fail, run fails before execution.
3. No automatic fallback to CPU.

### 4.3 `gpu_cpu_hybrid`

1. Scheduler manages two lanes:
   - GPU lane for GPU-eligible work
   - CPU lane for CPU-only/non-eligible work
2. Lane execution can be concurrent.
3. GPU failures may fallback to CPU only if fallback policy explicitly allows.

### 4.4 Mode Precedence

1. Explicit CLI flag overrides defaults.
2. If no flag is provided: `cpu_only`.
3. For official paths, any mode must remain operational-only and cannot override scientific run-spec values.

---

## 5. Minimum Compatibility Target

1. Baseline GPU target includes GTX 1070-class hardware (Pascal, compute capability 6.1).
2. Initial compatibility target uses CUDA 11.8-compatible package stack.
3. If environment does not satisfy GPU runtime compatibility, behavior follows selected mode rules:
   - `gpu_only`: fail
   - `gpu_cpu_hybrid`: fallback according to policy
   - `cpu_only`: continue

---

## 6. Compute Policy Contract

Introduce a normalized compute-policy object resolved at run start.

Required fields:

1. `hardware_mode`: `cpu_only | gpu_only | gpu_cpu_hybrid`
2. `gpu_device_ids`: list of non-negative ints
3. `max_parallel_gpu_runs`: int >= 0
4. `max_parallel_cpu_runs`: int >= 1
5. `deterministic_compute`: bool
6. `gpu_memory_budget_mb`: int or null
7. `fallback_policy`: `none | gpu_to_cpu_on_oom | gpu_to_cpu_on_runtime_error`

Default values:

1. `hardware_mode=cpu_only`
2. `gpu_device_ids=[0]`
3. `max_parallel_gpu_runs=1`
4. `max_parallel_cpu_runs=1` (official paths continue honoring current serial default unless explicitly increased)
5. `deterministic_compute=true`
6. `gpu_memory_budget_mb=null`
7. `fallback_policy=none` for `cpu_only` and `gpu_only`; explicit-only for `gpu_cpu_hybrid`

Validation rules:

1. `gpu_only` requires non-empty `gpu_device_ids`.
2. `gpu_cpu_hybrid` requires `max_parallel_cpu_runs >= 1`.
3. `max_parallel_gpu_runs` must not exceed `len(gpu_device_ids)` unless a multiplexing policy is explicitly added (not in v1).
4. Invalid combinations fail during argument validation, before run dispatch.

---

## 7. Hardware Capability Resolver

Add a resolver that returns a normalized capability payload.

Payload fields:

1. `gpu_available` (bool)
2. `detected_device_count` (int)
3. `devices` (array with `id`, `name`, `compute_capability`, `total_memory_mb`)
4. `cuda_runtime_version` (str or null)
5. `cuda_driver_version` (str or null)
6. `torch_version` (str or null)
7. `cupy_version` (str or null)
8. `compatibility_status` (`compatible | incompatible | unavailable`)
9. `incompatibility_reasons` (list of strings)

Required checks:

1. CUDA runtime availability.
2. Device visibility and IDs requested by user.
3. Compute capability threshold for configured backend.
4. Basic backend import checks (`torch`, `cupy`) when GPU modes are requested.

---

## 8. Backend Abstraction for Model Fit

Add an execution backend abstraction with at least:

1. `cpu_sklearn_backend` (current baseline behavior).
2. `gpu_torch_backend` (new).

The section contract (`model_fit` and downstream artifacts) must remain stable:

1. `fold_rows`, `split_rows`, `prediction_rows` schemas unchanged.
2. `metrics.json` schema unchanged except additive compute metadata.
3. `tuning_summary.json` and `best_params_per_fold.csv` retained.

---

## 9. GPU Estimator Coverage

Supported in v1:

1. `ridge` via torch implementation.
2. `logreg` via torch implementation.
3. `linearsvc` via torch implementation.
4. `dummy` remains CPU-only.

Implementation expectations:

1. Deterministic initialization and seed control.
2. Chunked operations for high-dimensional voxel data to avoid OOM.
3. Class-weight handling parity with existing CPU semantics (`none` / `balanced`).
4. Scoring outputs compatible with existing evaluation functions.

---

## 10. GPU Nested Tuning Engine

For `grouped_nested_tuning` in GPU modes:

1. Preserve existing search-space source (`tuning_search_spaces.py`).
2. Preserve grouped inner-CV semantics (`grouped_leave_one_group_out`).
3. Preserve artifact schema:
   - `tuning_summary.json`
   - `best_params_per_fold.csv`
4. Preserve metric objective semantics (declared primary metric).

Control behavior:

1. `dummy` still bypasses tuning as today.
2. If GPU tuning cannot run:
   - `gpu_only`: fail
   - `gpu_cpu_hybrid`: fallback only if policy allows

---

## 11. CLI and Interface Changes (Operational Only)

### 11.1 `thesisml-run-experiment`

Add flags:

1. `--hardware-mode {cpu_only,gpu_only,gpu_cpu_hybrid}`
2. `--gpu-device-ids <comma-separated>`
3. `--deterministic-compute {true,false}`
4. `--gpu-memory-budget-mb <int>`

Defaults:

1. `--hardware-mode cpu_only`
2. `--deterministic-compute true`

### 11.2 `thesisml-run-comparison` and `thesisml-run-protocol`

Add flags:

1. `--hardware-mode {cpu_only,gpu_only,gpu_cpu_hybrid}`
2. `--gpu-device-ids <comma-separated>`
3. `--max-parallel-gpu-runs <int>`
4. `--max-parallel-cpu-runs <int>`

Defaults:

1. `--hardware-mode cpu_only`
2. existing official operational defaults remain conservative

### 11.3 `thesisml-run-decision-support`

Add flags:

1. `--hardware-mode {cpu_only,gpu_only,gpu_cpu_hybrid}`
2. `--gpu-device-ids <comma-separated>`
3. `--max-parallel-gpu-runs <int>`
4. `--max-parallel-cpu-runs <int>`

Defaults:

1. `--hardware-mode cpu_only`

---

## 12. Scheduler Design for `gpu_cpu_hybrid`

### 12.1 Goals

1. Deterministic queue ordering.
2. No GPU oversubscription by default.
3. Preserve current CPU thread-cap behavior for CPU workers.

### 12.2 Dispatch Model

1. Build a stable ordered run queue (existing `order_index` semantics).
2. Tag each run as:
   - `gpu_eligible`
   - `cpu_only`
3. Route to lanes:
   - GPU lane for `gpu_eligible`
   - CPU lane for `cpu_only`
4. GPU lane workers bind `CUDA_VISIBLE_DEVICES=<assigned_id>`.
5. CPU lane workers set `CUDA_VISIBLE_DEVICES=""` and preserve existing BLAS/OpenMP caps.

### 12.3 Fallback Handling

When GPU lane run fails:

1. If fallback policy allows reroute, enqueue rerun on CPU lane with same `run_id` policy behavior and explicit fallback metadata.
2. If fallback is not allowed, mark as failed with GPU-specific error code.

---

## 13. Determinism and Reproducibility Controls

Controls to enforce when `deterministic_compute=true`:

1. Set and log all seeds used by numpy/torch/cupy.
2. Enable deterministic algorithms in GPU backend where supported.
3. Record warning metadata when strict determinism cannot be guaranteed by backend/platform.

Reproducibility invariants:

1. Fold split generation remains CPU deterministic and unchanged by hardware mode.
2. Artifact naming and required file set remain stable.
3. Report metadata must capture the effective backend and fallback path.

---

## 14. Memory, OOM, and Failure Taxonomy

Add explicit failure categories and codes:

1. `gpu_dependency_missing`
2. `gpu_unavailable`
3. `gpu_incompatible_capability`
4. `gpu_oom`
5. `gpu_device_assignment_error`
6. `gpu_runtime_error`

Mode behavior:

1. `cpu_only`: GPU failures are diagnostics only.
2. `gpu_only`: fail immediately on GPU failure.
3. `gpu_cpu_hybrid`: optional reroute to CPU per fallback policy.

---

## 15. Artifact Metadata Additions (Additive, Non-Breaking)

Add to both `config.json` and `metrics.json`:

1. `hardware_mode_requested`
2. `hardware_mode_effective`
3. `compute_backend_effective`
4. `gpu_device_ids_requested`
5. `gpu_device_id_effective`
6. `gpu_device_name_effective`
7. `cuda_runtime_version`
8. `torch_version`
9. `cupy_version`
10. `deterministic_compute`
11. `fallback_policy`
12. `fallback_applied`
13. `fallback_reason`
14. `gpu_peak_memory_mb` (nullable)
15. `gpu_memory_budget_mb` (nullable)

Contract note:

1. Existing required metadata keys remain unchanged in v1.
2. New keys are additive and must not break official validators.

---

## 16. Runtime Profiling and Precheck Updates

Enhance runtime profile cohorts to include backend mode:

1. `cpu_only`
2. `gpu_only`
3. `gpu_cpu_hybrid`

Add lane-aware estimator outputs:

1. estimated GPU lane wall-time
2. estimated CPU lane wall-time
3. estimated total wall-time as lane max plus overhead
4. recommended lane caps for single-GPU 8GB-class hardware

---

## 17. Workbook and Reporting Updates

Current workbook bridge writes `gpu=not_recorded`. Update to:

1. Detect and record effective GPU/device string.
2. Record selected hardware mode in notes.
3. Keep worksheet schema unchanged.

---

## 18. File-Level Implementation Workstreams

### 18.1 Compute policy and capability

Primary files:

1. `src/Thesis_ML/experiments/run_experiment.py`
2. `src/Thesis_ML/experiments/section_models.py`
3. new module for capability resolution (recommended: `src/Thesis_ML/experiments/compute_capability.py`)
4. new module for compute policy validation (recommended: `src/Thesis_ML/experiments/compute_policy.py`)

### 18.2 Backend and model-fit integration

Primary files:

1. `src/Thesis_ML/experiments/model_factory.py`
2. `src/Thesis_ML/experiments/sections_impl.py`
3. `src/Thesis_ML/experiments/segment_execution.py`
4. new GPU backend module(s) (recommended: `src/Thesis_ML/experiments/gpu_backend.py`)

### 18.3 Scheduler integration

Primary files:

1. `src/Thesis_ML/experiments/parallel_execution.py`
2. `src/Thesis_ML/protocols/runner.py`
3. `src/Thesis_ML/comparisons/runner.py`
4. `src/Thesis_ML/orchestration/campaign_engine.py`

### 18.4 CLI integration

Primary files:

1. `src/Thesis_ML/experiments/run_experiment.py` (parser)
2. `src/Thesis_ML/cli/protocol_runner.py`
3. `src/Thesis_ML/cli/comparison_runner.py`
4. `src/Thesis_ML/orchestration/campaign_cli.py`

### 18.5 Reporting and workbook

Primary files:

1. `src/Thesis_ML/experiments/run_artifacts.py`
2. `src/Thesis_ML/orchestration/workbook_bridge.py`

### 18.6 Runtime projection

Primary files:

1. `src/Thesis_ML/verification/campaign_runtime_profile.py`
2. `src/Thesis_ML/experiments/model_catalog.py` (if runtime projections are extended by backend mode)

---

## 19. Dependency Matrix (Planned)

Target compatibility baseline:

1. Python: repo-supported version range.
2. CUDA runtime compatibility: 11.8 baseline target.
3. PyTorch: CUDA-enabled build compatible with CUDA 11.8.
4. CuPy: CUDA 11.x compatible build.

Operational installation policy:

1. CPU-only install remains valid and sufficient for default behavior.
2. GPU extras must be optional dependency path.
3. Missing GPU dependencies must produce clear runtime diagnostics.

---

## 20. Test Plan and Acceptance Criteria

### 20.1 Compute-policy validation tests

1. Parse and normalize all hardware modes.
2. Reject invalid mode/flag combinations.
3. Validate fallback policy gating by mode.

### 20.2 Capability resolver tests

1. GPU absent path.
2. GPU present and compatible path.
3. Unsupported capability path.
4. GTX 1070-compatible path acceptance.

### 20.3 Backend parity tests

1. Fixed synthetic data for `ridge`, `logreg`, `linearsvc`.
2. CPU vs GPU metric drift thresholds documented and enforced.
3. Class-weight behavior parity tests.

### 20.4 Determinism tests

1. Same seed, same backend, same mode => stable splits and stable metrics where deterministic backend guarantees apply.
2. Determinism warning path validated when strict determinism is unavailable.

### 20.5 GPU tuning tests

1. Grouped nested tuning artifact schema unchanged.
2. Candidate count and selected-params serialization validated.

### 20.6 Scheduler tests

1. GPU device assignment correctness.
2. CPU thread caps preserved.
3. Hybrid concurrent execution without oversubscription.
4. Stable final output ordering regardless of completion ordering.

### 20.7 Artifact contract regression

1. Existing official contract tests must pass unchanged.
2. Added compute metadata appears in config and metrics artifacts.

### 20.8 Runtime profiling tests

1. Backend-aware cohort generation.
2. Hybrid wall-time estimate logic.
3. Fallback estimate warnings.

### 20.9 End-to-end smoke tests

1. Exploratory one-run smoke per mode.
2. Decision-support mini-campaign smoke in hybrid mode.
3. Official dry-run plus small real run in opt-in `gpu_only` and `gpu_cpu_hybrid`.

---

## 21. Rollout Gates

### Gate 1: Exploratory Enablement

Exit criteria:

1. compute-policy tests passing
2. capability tests passing
3. backend parity and determinism checks passing
4. artifact contract regressions clean

### Gate 2: Decision-Support Hybrid Enablement

Exit criteria:

1. dual-lane scheduler validated in campaign runner
2. dry-run manifests include hardware mode visibility
3. mini hybrid campaign smoke passes

### Gate 3: Official Opt-In Enablement

Exit criteria:

1. comparison/protocol operational flags integrated
2. official regression suites pass
3. timeout and failure handling validated for GPU modes

### Gate 4: Default Policy Review

Exit criteria:

1. repeated-run reproducibility evidence meets acceptance thresholds
2. operational stability evidence collected
3. if criteria not met, retain `cpu_only` as default indefinitely

---

## 22. Operations Runbook (Planned CLI Examples)

### 22.1 Exploratory

CPU default:

```bash
thesisml-run-experiment ... --hardware-mode cpu_only
```

GPU-only:

```bash
thesisml-run-experiment ... --hardware-mode gpu_only --gpu-device-ids 0
```

Hybrid:

```bash
thesisml-run-experiment ... --hardware-mode gpu_cpu_hybrid --gpu-device-ids 0
```

### 22.2 Locked comparison

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/...json \
  --hardware-mode gpu_cpu_hybrid \
  --gpu-device-ids 0 \
  --max-parallel-gpu-runs 1 \
  --max-parallel-cpu-runs 2
```

### 22.3 Confirmatory

```bash
thesisml-run-protocol \
  --protocol configs/protocols/...json \
  --hardware-mode cpu_only
```

---

## 23. Troubleshooting Matrix (Planned)

1. GPU dependency import error -> install GPU extras or run `cpu_only`.
2. Device not visible -> validate `gpu_device_ids` and environment visibility.
3. CUDA mismatch -> use compatible CUDA runtime and package versions.
4. GPU OOM -> reduce lane concurrency, apply chunking, configure memory budget, or fallback if allowed.
5. Determinism mismatch -> enforce deterministic flags and verify backend deterministic support.

---

## 24. What Does Not Change

1. Official workflow boundaries and meanings.
2. Protocol/comparison scientific governance and lock semantics.
3. Required official artifacts and current minimum required metadata keys.
4. Default operational behavior (`cpu_only`).
5. Target/split/model/methodology policy definitions as scientific controls.

---

## 25. Final Implementation Directive

The implementation must be staged and gated exactly as defined above.  
`cpu_only` stays default and baseline.  
`gpu_only` and `gpu_cpu_hybrid` are additional, explicit, opt-in operational modes.
