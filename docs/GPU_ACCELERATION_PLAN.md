# GPU Backend Implementation Plan

## Goal

Add a compute-backend system that supports three operational modes without weakening the repository’s scientific contracts:

1. `cpu_only`
2. `gpu_only`
3. `max_both` (the user-facing name for a bounded mixed CPU+GPU execution mode)

The critical constraint is that these are **operational execution policies**, not scientific method changes. The same protocol, comparison spec, target, metric policy, split policy, tuning policy, artifact contract, and confirmatory governance must remain authoritative.

This plan is intentionally conservative. It is optimized for correctness, artifact integrity, parity validation, and staged rollout.

---

## Executive decision

The implementation should **not** begin with all three modes at once.

The correct order is:

### Stage 1
Build the backend contract, capability resolver, metadata stamping, and `cpu_only` / `gpu_only` support for exploratory runs.

### Stage 2
Extend `gpu_only` into locked comparison and then confirmatory, but only with strict parity and deterministic-compute gates.

### Stage 3
Add `max_both` only after the GPU estimator path is stable, measurable, and artifact-safe.

This is still a three-mode design, but the implementation sequence must be staged.

---

## Non-negotiable repository invariants

These must remain true throughout the redesign.

### Workflow boundaries
- `thesisml-run-experiment` remains exploratory only.
- `thesisml-run-comparison` remains locked comparison only.
- `thesisml-run-protocol` remains confirmatory only.

### Governance boundaries
- Hardware mode must remain an operational control.
- It must never silently change the scientific meaning of a protocol or comparison.
- Primary metric, methodology policy, and split policy remain unchanged.
- Confirmatory evidence must still come only from canonical protocol execution.

### Artifact boundaries
- Existing official artifacts remain required.
- New compute metadata must be additive.
- Existing downstream validators must continue to pass unless intentionally updated in lockstep.

### Safety rule
- `cpu_only` remains the default for all official paths until explicit promotion gates are passed.

---

## Current repository reality

The current codebase is strongly sklearn-shaped and CPU-shaped.

### Existing strengths we can reuse
- `src/Thesis_ML/experiments/run_experiment.py` already centralizes execution control.
- `src/Thesis_ML/experiments/model_factory.py` already centralizes model construction.
- `src/Thesis_ML/experiments/sections_impl.py` centralizes fit/tuning logic.
- `src/Thesis_ML/experiments/segment_execution.py` and `section_models.py` already carry structured run inputs.
- `src/Thesis_ML/protocols/runner.py` and `src/Thesis_ML/comparisons/runner.py` already orchestrate official runs.
- `src/Thesis_ML/experiments/run_artifacts.py` already stamps metadata and writes artifacts.
- Phase A already introduced timing visibility and run-level parallelism.

### Existing constraints we must respect
- sklearn `Pipeline`, `clone`, and `GridSearchCV` shape the fit stack.
- Downstream reporting expects sklearn-like estimator methods and attributes.
- Current performance evidence indicates a large amount of non-fit overhead, so GPU work must not be sold as an automatic near-term speedup.

### Practical implication
The safest design is a **backend abstraction layer with sklearn-compatible estimators first**, not a wholesale replacement of the training engine.

---

## Definitions

### `cpu_only`
All model fitting and scoring happen on CPU using the current sklearn-backed reference path.

### `gpu_only`
The selected estimator backend is GPU-native. The run is still orchestrated by the existing repository machinery, but model fit/predict/decision paths are routed through a GPU estimator wrapper.

### `max_both`
A bounded mixed compute mode where the scheduler is allowed to use both CPU and GPU lanes at the run level. It does **not** mean one estimator fit is simultaneously split across CPU and GPU. It means the execution engine may schedule some runs on GPU and some on CPU according to a policy.

This definition is important. It keeps `max_both` operationally tractable and scientifically auditable.

---

## Core design principle

Separate the system into three layers.

### Layer A: scientific plan layer
This is the existing protocol/comparison/experiment declaration layer.

Examples:
- target
- split policy
- methodology policy
- primary metric
- suite / variant identity
- artifact requirements

This layer must remain backend-agnostic.

### Layer B: compute policy layer
This is a new operational layer that resolves how a run is executed.

Examples:
- `hardware_mode`
- `gpu_device_id`
- deterministic compute
- backend family
- effective device
- mixed-lane scheduling policy

This layer may affect runtime and hardware use, but must not affect scientific declarations.

### Layer C: backend implementation layer
This is where concrete CPU and GPU estimators live.

Examples:
- sklearn CPU estimator implementations
- torch GPU estimator wrappers
- backend capability probes
- scheduler lane selection

---

## Final target architecture

## 1. Compute policy object

Introduce a normalized compute policy object resolved at run start.

### Required fields for v1
- `hardware_mode`: `cpu_only | gpu_only | max_both`
- `requested_backend_family`: `sklearn_cpu | torch_gpu | auto_mixed`
- `effective_backend_family`: resolved backend actually used
- `gpu_device_id`: integer or `null`
- `deterministic_compute`: boolean
- `gpu_memory_soft_limit_mb`: integer or `null`
- `allow_backend_fallback`: boolean
- `fallback_reason`: string or `null`
- `run_parallel_lane`: `cpu | gpu | null`
- `backend_stack_id`: versioned identifier of the tested stack

### Policy rules
- `cpu_only` must never require GPU capability.
- `gpu_only` must fail clearly if a compatible GPU stack is unavailable.
- `max_both` must never silently reroute official runs unless explicitly allowed by policy.
- Official `gpu_only` or `max_both` runs must stamp deterministic compute settings and backend metadata.
- In the first official rollout, `gpu_only` and `max_both` should force conservative scheduler settings.

---

## 2. Capability resolver

Create a hardware capability resolver.

### New module
- `src/Thesis_ML/experiments/compute_capabilities.py`

### Responsibilities
- detect whether torch is installed
- detect whether CUDA is available
- enumerate visible devices
- validate requested device id
- capture device name and memory
- capture torch and CUDA version metadata
- return a structured capability payload

### Output fields
- `gpu_available`
- `gpu_count`
- `requested_device_visible`
- `device_id`
- `device_name`
- `device_total_memory_mb`
- `torch_version`
- `cuda_runtime_version`
- `compatibility_status`
- `incompatibility_reasons`
- `tested_stack_id`

### v1 rule
The resolver should be torch-based only. CuPy is not required for v1.

---

## 3. Backend registry

Introduce a backend registry that maps a model family plus compute policy to a concrete estimator implementation.

### New module
- `src/Thesis_ML/experiments/backend_registry.py`

### Responsibilities
- map `(model_name, hardware_mode, backend_family)` to a constructor
- expose backend support matrix
- refuse unsupported combinations clearly

### Example support matrix at rollout start
- `ridge`:
  - `cpu_only`: supported
  - `gpu_only`: supported
  - `max_both`: supported via lane scheduling
- `logreg`:
  - `cpu_only`: supported
  - `gpu_only`: experimental first, then gated
  - `max_both`: only after `gpu_only` is stable
- `linearsvc`:
  - `cpu_only`: supported
  - `gpu_only`: deferred
  - `max_both`: deferred
- `dummy`:
  - `cpu_only`: supported
  - `gpu_only`: unnecessary, keep CPU
  - `max_both`: CPU lane only

---

## 4. Estimator wrapper design

Do not replace the entire fit stack initially. Instead, add sklearn-shaped wrappers around GPU-native implementations.

### New package
- `src/Thesis_ML/experiments/backends/`

### Submodules
- `cpu_reference.py`
- `torch_ridge.py`
- `torch_logreg.py`
- `common.py`

### Required wrapper behavior
Each wrapper must provide sklearn-like behavior where downstream code expects it.

Required methods/attributes where applicable:
- `fit(X, y)`
- `predict(X)`
- `decision_function(X)`
- `predict_proba(X)` if supported
- `get_params(deep=True)`
- `set_params(**kwargs)`
- `classes_`
- `coef_`
- `intercept_`
- `n_features_in_`

### Important design choice
For v1, wrappers may still accept NumPy arrays on input and handle CPU→GPU tensor conversion internally. This is slower than a full GPU-native pipeline, but far safer for integration.

---

## 5. Scheduler semantics for `max_both`

`max_both` must be defined carefully.

### Recommended meaning
It is a run-level mixed-lane scheduler.

It does not mean:
- the same fit uses both CPU and GPU together
- one fold is partly on CPU and partly on GPU
- automatic per-batch migration inside a run

It means:
- some runs can execute on CPU lanes
- some runs can execute on GPU lanes
- the scheduler chooses the lane using explicit policy

### Why this is the right first implementation
- it matches the current run-level orchestration design
- it keeps official artifacts easy to audit
- it avoids mixed numerical semantics inside one estimator fit
- it is feasible with the current process-based runner design

---

## Detailed staged implementation plan

## Stage 0: branch, scoping, and contract freeze

### Objective
Create an isolated redesign branch and freeze the current CPU path as the scientific reference.

### Actions
1. Create a new branch, for example:
   - `feature/backend-compute-redesign`
2. Do not modify checked-in canonical protocol/comparison semantics at this stage.
3. Create a reference benchmark record for the current CPU path.
4. Freeze a small parity workload for regression testing.

### Deliverables
- branch created
- baseline benchmark artifact committed under `outputs/tmp` or a docs artifact directory if that fits repo policy
- reference parity cases documented

### No-code deliverable
Add a short architecture note explaining that CPU remains the reference backend.

---

## Stage 1: compute policy and capability plumbing

### Objective
Add the compute-policy abstraction without changing model behavior.

### Files to create
- `src/Thesis_ML/experiments/compute_capabilities.py`
- `src/Thesis_ML/experiments/compute_policy.py`

### Files to modify
- `src/Thesis_ML/experiments/run_experiment.py`
- `src/Thesis_ML/experiments/segment_execution.py`
- `src/Thesis_ML/experiments/section_models.py`
- `src/Thesis_ML/experiments/run_artifacts.py`
- `src/Thesis_ML/protocols/runner.py`
- `src/Thesis_ML/comparisons/runner.py`
- `src/Thesis_ML/cli/protocol_runner.py`
- `src/Thesis_ML/cli/comparison_runner.py`

### What to add
#### CLI and runtime inputs
Add operational flags such as:
- `--hardware-mode` with values `cpu_only | gpu_only | max_both`
- `--gpu-device-id`
- `--deterministic-compute`
- `--allow-backend-fallback` for exploratory only in v1

### Important constraints
- exploratory can accept these flags directly
- comparison/protocol may accept them only as operational controls, not spec-level scientific overrides

#### Resolved compute policy
At run start, resolve the requested policy into an effective policy.

#### Artifact metadata
Stamp additive fields into:
- `config.json`
- `metrics.json`
- run result payloads
- execution status summaries where appropriate

### Artifact fields to add
- `hardware_mode_requested`
- `hardware_mode_effective`
- `requested_backend_family`
- `effective_backend_family`
- `gpu_device_id`
- `gpu_device_name`
- `gpu_device_total_memory_mb`
- `deterministic_compute`
- `backend_stack_id`
- `backend_fallback_used`
- `backend_fallback_reason`

### Tests
Create:
- `tests/test_compute_policy.py`
- `tests/test_compute_capabilities.py`

Cover:
- cpu-only resolution with no GPU
- gpu-only failure when torch/CUDA unavailable
- valid device selection
- artifact metadata presence
- official-path restrictions preserved

### Done condition
No GPU estimator exists yet, but the repository can now resolve and record compute policy safely.

---

## Stage 2: backend registry and CPU reference integration

### Objective
Introduce the backend abstraction while keeping behavior exactly identical for CPU.

### Files to create
- `src/Thesis_ML/experiments/backend_registry.py`
- `src/Thesis_ML/experiments/backends/__init__.py`
- `src/Thesis_ML/experiments/backends/common.py`
- `src/Thesis_ML/experiments/backends/cpu_reference.py`

### Files to modify
- `src/Thesis_ML/experiments/model_factory.py`
- `src/Thesis_ML/experiments/sections_impl.py`

### What to do
#### In `model_factory.py`
Split responsibilities:
- keep hyperparameter-space and model-family declarations
- delegate estimator construction to the backend registry when compute policy is present

#### In `backend_registry.py`
Map requested model/backend pairs to concrete constructors.

#### In `cpu_reference.py`
Wrap current sklearn estimators in a stable interface so the GPU path later targets the same contract.

### Tests
- `tests/test_backend_registry.py`
- parity tests proving CPU path is unchanged

### Done condition
The repository still runs only CPU fits, but now through a backend abstraction with no result drift.

---

## Stage 3: GPU exploratory v1 for ridge only

### Objective
Add the first real GPU estimator path in exploratory mode only.

### Files to create
- `src/Thesis_ML/experiments/backends/torch_ridge.py`
- optionally `src/Thesis_ML/experiments/backends/torch_utils.py`

### Files to modify
- `src/Thesis_ML/experiments/backend_registry.py`
- `src/Thesis_ML/experiments/sections_impl.py`
- `src/Thesis_ML/experiments/run_artifacts.py`

### Why ridge first
- lowest implementation risk
- linear, interpretable, stable baseline
- easiest to compare with CPU behavior

### Implementation specifics
#### Wrapper behavior
The torch ridge wrapper must:
- accept NumPy input from the current pipeline
- convert to tensors on the configured device
- fit with deterministic settings where possible
- expose sklearn-shaped outputs for downstream reporting
- return CPU NumPy arrays for artifact writers when needed

#### Determinism controls
If `deterministic_compute=true`:
- enable deterministic torch settings where supported
- record any determinism limitations in metadata

### Artifact additions
Add optional GPU fit diagnostics:
- `gpu_memory_peak_mb` if available
- `device_transfer_seconds` if easy to capture
- `torch_deterministic_enforced`

### Tests
Create:
- `tests/test_torch_ridge_backend.py`

Cover:
- fit/predict shape parity vs CPU
- metric tolerance parity on fixed seed small data
- metadata stamping
- graceful failure when torch/CUDA unavailable

### Rollout gate
Do not expose this in comparison/protocol yet.

### Done condition
Exploratory `ridge` can run in `gpu_only` mode on one device with artifact-compatible outputs.

---

## Stage 4: exploratory logreg GPU path

### Objective
Add GPU `logreg` only after ridge v1 is stable.

### Files to create
- `src/Thesis_ML/experiments/backends/torch_logreg.py`

### Why later than ridge
- greater parity risk
- solver behavior and convergence behavior need closer validation
- current CPU `logreg` is operationally expensive, so this backend matters, but it must be validated carefully

### Implementation specifics
The wrapper must expose:
- `decision_function`
- `predict_proba` if supported by implementation
- `coef_`
- `intercept_`
- classes and thresholds matching repository expectations as closely as possible

### Important caution
Do not claim exact sklearn equivalence. Define tolerance-based parity thresholds instead.

### Tests
Create:
- `tests/test_torch_logreg_backend.py`

Cover:
- convergence behavior on representative small datasets
- metric parity tolerance
- reporting compatibility
- failure paths when GPU unavailable

### Done condition
Exploratory `logreg` can run in `gpu_only`, but remains explicitly experimental.

---

## Stage 5: exploratory `max_both` scheduler

### Objective
Introduce the third requested mode as a safe run-level mixed-lane scheduler.

### Files to create
- `src/Thesis_ML/experiments/compute_scheduler.py`

### Files to modify
- `src/Thesis_ML/experiments/parallel_execution.py`
- `src/Thesis_ML/protocols/runner.py`
- `src/Thesis_ML/comparisons/runner.py`
- `src/Thesis_ML/experiments/supervised_worker.py`
- `src/Thesis_ML/experiments/timeout_watchdog.py`

### Scheduler design
#### Inputs
- run plan
- hardware mode
- backend availability
- supported model/backend matrix
- `max_parallel_runs`
- `max_parallel_gpu_runs`
- device ids
- memory soft limits

#### Scheduling policy for v1
For `max_both`:
- GPU-eligible runs may be assigned to GPU lanes
- unsupported or disallowed runs go to CPU lanes
- scheduling is explicit and recorded
- no automatic in-run migration

### Default rule set
- `ridge`: GPU-eligible
- `logreg`: GPU-eligible only after Stage 4 gate passes
- `linearsvc`: CPU lane only
- `dummy`: CPU lane only

### Metadata
Each run must stamp:
- requested mode: `max_both`
- assigned lane: `cpu` or `gpu`
- assigned backend family
- reason for lane selection

### Tests
Create:
- `tests/test_compute_scheduler.py`

Cover:
- deterministic lane assignment
- mixed plans preserve order and artifacts
- unsupported models remain CPU lane only
- scheduler obeys GPU lane limit

### Done condition
Exploratory runs can use `max_both` safely with recorded lane assignment.

---

## Stage 6: official-path admission gates

### Objective
Promote selected GPU support from exploratory into comparison and confirmatory, but only under strict gates.

### Promotion order
1. locked comparison
2. confirmatory

### Required evidence before comparison rollout
- CPU parity suite passes for `ridge`
- timing and artifact parity validated
- no validator breakage
- explicit docs updated

### Required evidence before confirmatory rollout
- comparison-path stability on representative workloads
- reproducibility checks under deterministic mode
- artifact completeness and official verification intact

### Official policy rules for first official rollout
- `cpu_only` remains default
- `gpu_only` official runs require deterministic compute
- `gpu_only` official runs force conservative scheduler settings
- `max_both` official runs are disallowed until after comparison-stage stability

### Files to modify
- `src/Thesis_ML/experiments/runtime_policies.py`
- `src/Thesis_ML/protocols/models.py`
- `src/Thesis_ML/comparisons/models.py`
- relevant verification modules under `src/Thesis_ML/verification/`

### Tests
Add official-path verification tests proving:
- unsupported GPU official requests fail clearly
- supported official GPU requests stamp required metadata
- CPU default remains unchanged

### Done condition
Official comparison may optionally use `gpu_only` for approved model families. Confirmatory follows only after explicit gate review.

---

## Stage 7: confirmatory GPU eligibility

### Objective
Allow carefully gated confirmatory `gpu_only` for approved backends.

### Critical rule
No automatic fallback in confirmatory mode. If requested GPU capability is unavailable or invalid, fail clearly.

### Why
Silent backend fallback would weaken auditability of thesis evidence.

### Required documentation changes
- `docs/RUNBOOK.md`
- `docs/RELEASE.md`
- confirmatory runbook notes explaining when GPU use is allowed and how it is recorded

### Done condition
Canonical protocol runs may request `gpu_only` only if the backend/model family is approved and all deterministic/audit conditions are satisfied.

---

## Stage 8: official `max_both` consideration

### Objective
Only after GPU-only official support is stable, evaluate whether `max_both` should be admitted to locked comparison, then possibly confirmatory.

### Default recommendation
Keep official `max_both` disabled for a long time.

### Reason
It complicates:
- reproducibility
- device assignment auditability
- timing comparability
- backend drift interpretation

### If eventually allowed
It must stamp lane assignment per run and remain fully deterministic at the run-plan level.

---

## Cross-cutting implementation details

## A. Metadata design

Do not rely on a single coarse field like `compute_backend_family` only.

### Recommended additive metadata
- `hardware_mode_requested`
- `hardware_mode_effective`
- `assigned_compute_lane`
- `feature_backend_effective`
- `preprocessing_backend_effective`
- `estimator_backend_effective`
- `gpu_device_id`
- `gpu_device_name`
- `gpu_device_total_memory_mb`
- `deterministic_compute`
- `torch_version`
- `cuda_runtime_version`
- `backend_stack_id`
- `backend_fallback_used`
- `backend_fallback_reason`

This is more honest and more future-proof than one coarse field.

---

## B. Determinism policy

### Exploratory
- deterministic compute optional
- failures may fall back if explicitly allowed

### Comparison
- deterministic compute strongly recommended
- fallback should be explicit and likely disallowed once backend is considered stable

### Confirmatory
- deterministic compute required
- no silent fallback
- backend stack must be fully stamped in artifacts

---

## C. Scheduler policy for `max_both`

### v1 scheduling rule
Do not schedule more GPU-lane runs than the configured lane limit.

### Required knobs
- `max_parallel_runs`
- `max_parallel_gpu_runs`
- `gpu_device_id` or visible device list
- `gpu_memory_soft_limit_mb`

### Safety rule
If the scheduler cannot assign a GPU lane safely, either:
- fail, or
- assign CPU lane explicitly if policy allows

Never silently drift.

---

## D. Performance-gain gate

A GPU backend should not be promoted just because it works.

Add an explicit benchmark gate requiring:
- artifact parity
- metric parity within tolerance
- meaningful throughput benefit on representative workloads

### Promotion requirement
A backend/model family should not move into official use unless it yields a meaningful operational benefit or a scientifically necessary capability.

---

## E. Relationship to existing Phase A work

This plan must not replace Phase A conclusions.

Current evidence suggests non-fit overhead is still substantial. Therefore:
- GPU backend work should be understood as backend capability work and future architecture work
- it should not be sold internally as the next guaranteed speedup
- worker-local feature reuse may still be the higher-value mainline optimization even if GPU work begins on a separate branch

---

## File-by-file implementation map

## New modules to create
- `src/Thesis_ML/experiments/compute_policy.py`
- `src/Thesis_ML/experiments/compute_capabilities.py`
- `src/Thesis_ML/experiments/backend_registry.py`
- `src/Thesis_ML/experiments/compute_scheduler.py`
- `src/Thesis_ML/experiments/backends/__init__.py`
- `src/Thesis_ML/experiments/backends/common.py`
- `src/Thesis_ML/experiments/backends/cpu_reference.py`
- `src/Thesis_ML/experiments/backends/torch_ridge.py`
- `src/Thesis_ML/experiments/backends/torch_logreg.py`
- `tests/test_compute_policy.py`
- `tests/test_compute_capabilities.py`
- `tests/test_backend_registry.py`
- `tests/test_torch_ridge_backend.py`
- `tests/test_torch_logreg_backend.py`
- `tests/test_compute_scheduler.py`

## Existing files likely to modify
- `src/Thesis_ML/experiments/run_experiment.py`
- `src/Thesis_ML/experiments/model_factory.py`
- `src/Thesis_ML/experiments/sections_impl.py`
- `src/Thesis_ML/experiments/segment_execution.py`
- `src/Thesis_ML/experiments/section_models.py`
- `src/Thesis_ML/experiments/parallel_execution.py`
- `src/Thesis_ML/experiments/supervised_worker.py`
- `src/Thesis_ML/experiments/timeout_watchdog.py`
- `src/Thesis_ML/experiments/runtime_policies.py`
- `src/Thesis_ML/experiments/run_artifacts.py`
- `src/Thesis_ML/protocols/runner.py`
- `src/Thesis_ML/comparisons/runner.py`
- `src/Thesis_ML/cli/protocol_runner.py`
- `src/Thesis_ML/cli/comparison_runner.py`
- relevant verification modules under `src/Thesis_ML/verification/`
- `docs/RUNBOOK.md`
- `docs/RELEASE.md`
- `docs/ARCHITECTURE.md`
- `pyproject.toml`

---

## Validation strategy

## 1. CPU parity regression suite
Before any GPU work is trusted, create a frozen CPU parity suite.

### Cases
- exploratory ridge
- exploratory logreg
- one comparison variant
- one protocol suite
- one tuned path if cheap enough

### Assertions
- same run count
- same artifact set
- same required metadata plus additive compute fields
- same metrics within exact or current tolerance

---

## 2. GPU estimator parity suite

### Ridge parity requirements
- prediction parity tolerance defined and checked
- metric parity tolerance defined and checked
- artifact-writing path identical except additive compute metadata

### Logreg parity requirements
- same as ridge, plus convergence and probability behavior checks where relevant

---

## 3. Scheduler parity suite
For `max_both`, verify:
- deterministic lane assignment
- stable output order
- no artifact drift beyond additive compute metadata
- run results remain attributable to the correct scientific spec and run id

---

## 4. Official verification suite
Before official rollout, extend existing verification to assert:
- backend metadata present when requested
- unsupported official GPU requests fail clearly
- fallback behavior is absent where forbidden
- bundle/release steps still pass

---

## Risk register

## High-risk items
### 1. Overstating performance benefit
Mitigation:
- require benchmark gate before promotion
- document that v1 may remain CPU-heavy outside estimator fit

### 2. Numerical parity drift
Mitigation:
- model-by-model rollout
- tolerance-based parity tests
- do not admit unstable models to official paths

### 3. Artifact contract drift
Mitigation:
- additive fields only
- validator and test updates in lockstep

### 4. Mixed-backend audit ambiguity
Mitigation:
- explicit lane and backend metadata
- conservative `max_both` semantics

### 5. Scheduler complexity explosion
Mitigation:
- exploratory-first rollout
- no multi-GPU v1
- no in-run hybrid v1

---

## Recommended implementation order by pull request

## PR 1
Compute policy, capability resolver, additive metadata, no behavioral change.

## PR 2
Backend registry and CPU reference path through the abstraction, no result drift.

## PR 3
Exploratory `gpu_only` ridge backend.

## PR 4
Exploratory `gpu_only` logreg backend, if parity is acceptable.

## PR 5
Exploratory `max_both` scheduler using run-level CPU/GPU lanes.

## PR 6
Locked comparison GPU admission gates.

## PR 7
Confirmatory GPU admission gates.

## PR 8
Only later, consider official `max_both`.

---

## Recommendation on naming

For internal code and docs, keep the external names the user wants, but internally use precise names.

### External user-facing names
- `cpu_only`
- `gpu_only`
- `max_both`

### Internal descriptive names
- `cpu_only`
- `gpu_only`
- `mixed_run_lanes`

This keeps `max_both` understandable to users while making the internal implementation honest.

---

## Final recommendation

Build this as a **backend redesign branch**, not as a quick optimization patch on mainline.

The right mental model is:
- preserve the current CPU system as the scientific reference
- add backend abstraction first
- admit GPU model families gradually
- define `max_both` as run-level mixed scheduling, not in-fit hybrid compute
- promote GPU into official paths only after parity, artifact, and performance gates pass

That gives you the three modes you want while keeping the thesis framework scientifically defensible and maintainable.

