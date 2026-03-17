# AGENTS.md

## Purpose

This repository is a research-software framework for leakage-aware fMRI decoding with three explicit execution modes:

- **exploratory**
- **locked_comparison**
- **confirmatory**

The repository must be treated as a **policy-driven scientific execution system**, not as an ad hoc ML sandbox.

When working in this repo, prioritize:

1. correctness
2. contract strictness
3. reproducibility
4. artifact integrity
5. behavior preservation during refactors
6. explicitness over convenience

Do not silently weaken official workflows.

---

## Non-negotiable repository truths

### 1. Official paths are not interchangeable

These commands have fixed meanings:

- `thesisml-run-experiment` → **exploratory only**
- `thesisml-run-comparison` → **locked comparison only**
- `thesisml-run-protocol` → **confirmatory only**

Never blur these roles.

### 2. Confirmatory runs must come only from canonical protocol execution

Do not generate final thesis evidence through exploratory flags or ad hoc comparison overrides.

If a task asks for official thesis evidence, use:

- `thesisml-run-protocol`
- a checked-in canonical protocol JSON under `configs/protocols/`

### 3. Locked comparisons must come only from registered comparison specs

Do not treat comparison mode as a free-form experiment matrix.

Use:

- `thesisml-run-comparison`
- checked-in comparison specs under `configs/comparisons/`

### 4. One primary metric must drive official decisions

For official comparison and confirmatory paths:

- the declared **primary metric** must drive:
  - tuning objective
  - winner selection
  - headline reporting
  - permutation/statistical controls

Secondary metrics may be reported, but must not silently decide outcomes.

### 5. Methodology policy must be explicit

Official specs/protocols must explicitly choose one methodology policy:

- `fixed_baselines_only`
- `grouped_nested_tuning`

Do not leave thesis-critical method behavior implicit, “planned”, or manual.

### 6. Official artifact outputs are contracts

For official paths, artifacts are not optional byproducts.
They are part of the contract.

Do not remove, rename, or silently degrade required official artifacts without updating:
- contracts
- validators
- tests
- docs

---

## Working style for Codex

### Default behavior

When asked to work in this repository:

- inspect first
- prefer minimal, explicit changes
- preserve behavior unless change is the explicit goal
- keep official workflow boundaries intact
- update tests/docs when behavior or structure changes
- do not invent unsupported workflow steps
- do not claim something is verified unless it was actually checked

### If asked to audit or verify

Default to **verification mode**, not implementation mode.

That means:
- do not change source files unless explicitly asked
- run checks
- inspect artifacts
- report verified vs partially verified vs not verified vs failed
- be skeptical of assumptions

### If asked to refactor

Refactor incrementally:
- extract cohesive responsibilities
- keep public entrypoints stable where practical
- use façade/re-export shims if needed
- do not mix structural refactor with behavior changes unless explicitly required

### If asked to improve performance

Use measured evidence:
- profile first
- optimize the most expensive real hotspots
- avoid speculative micro-optimizations
- preserve artifact behavior and policy correctness

---

## Repository priorities

The order of importance in this repo is:

1. **official-path correctness**
2. **reproducibility**
3. **artifact integrity**
4. **contract strictness**
5. **diagnosability**
6. **maintainability**
7. **performance**
8. **convenience**

Never trade official-path correctness for convenience.

---

## Mode boundaries

### Exploratory mode

Command:
- `thesisml-run-experiment`

Purpose:
- idea generation
- debugging
- free-form experimentation
- local probing

Allowed:
- flexible flags
- exploratory-only changes
- non-official reporting

Not allowed:
- presenting outputs as confirmatory evidence
- bypassing official protocol/comparison rules while claiming official status

Expected metadata:
- `framework_mode = exploratory`
- `canonical_run = false`

### Locked comparison mode

Command:
- `thesisml-run-comparison`

Purpose:
- predeclared benchmark/method-lock comparisons
- selecting between registered alternatives

Allowed:
- only registered variant space from comparison spec
- only declared methodology/metric/control policies

Not allowed:
- free-form science-affecting overrides
- confirmatory labeling
- changing undeclared factors inside the comparison

Expected metadata:
- `framework_mode = locked_comparison`
- comparison identity/version
- machine-readable comparison decision

### Confirmatory mode

Command:
- `thesisml-run-protocol`

Purpose:
- final official thesis evidence generation

Allowed:
- only protocol-declared settings
- only operational overrides like output location if supported

Not allowed:
- ad hoc target/model/cv/metric/tuning overrides
- using exploratory or comparison commands for confirmatory evidence

Expected metadata:
- `framework_mode = confirmatory`
- `canonical_run = true`
- protocol identity/version
- claim/suite mapping where supported

---

## Current official commands

Use these commands exactly unless the repo changes.

### Exploratory

```bash
thesisml-run-experiment \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --target coarse_affect \
  --model ridge \
  --cv within_subject_loso_session \
  --subject sub-001 \
  --seed 42