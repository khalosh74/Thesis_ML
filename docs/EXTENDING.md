# Extension Guide

This guide covers two common extension tasks:

- adding a new execution section
- adding a new workbook sheet safely
- adding a new model family safely

## Add a model family safely

Treat model additions as policy + contract changes, not just estimator wiring.

1. Add model metadata in the registry
- File: `src/Thesis_ML/experiments/model_registry.py`
- Define logical name, family, cost tier, supported class-weight policies, allowed feature recipes, backend bindings, tuning policy, and official admission metadata.

2. Update centralized admission policy if official behavior changes
- File: `src/Thesis_ML/experiments/model_admission.py`
- Keep locked-comparison and confirmatory admissions explicit for `gpu_only` and `max_both`.

3. Wire backend constructors
- File: `src/Thesis_ML/experiments/backend_registry.py`
- Ensure every declared backend binding resolves to a concrete constructor/support check.

4. Ensure stage routing tokens are covered
- Files:
  - `src/Thesis_ML/experiments/stage_planner.py`
  - `src/Thesis_ML/experiments/stage_registry.py`
- Add executor mapping only where needed; preserve conservative official-path admission.

5. Keep tuning policy explicit
- File: `src/Thesis_ML/experiments/tuning_search_spaces.py`
- Register search-space ids/versions and validate model-specific allowed ids.
- For official grouped-nested updates, add new versioned config files under:
  - `configs/comparisons/`
  - `configs/protocols/`
  without mutating historical files in place.

6. Preserve fairness/contract validation
- File: `src/Thesis_ML/experiments/comparison_contract.py`
- Ensure compared variants in the same evaluation scope share target/split/metric/methodology/tuning/control/class-weight contract semantics.

7. Stamp auditable governance metadata
- File: `src/Thesis_ML/experiments/run_artifacts.py`
- Ensure outputs include logical model identity, backend family/id, feature recipe, tuning search-space id/version, admission summary, deterministic/scheduler metadata, and registry version.

8. Add tests
- Cover registry authority, admission behavior, backend routing, grouped-nested config compatibility, and fairness-contract rejection paths.

## Add a new execution section

Keep changes incremental and contract-first.

1. Add section name enum value
- File: `src/Thesis_ML/orchestration/contracts.py`
- Update `SectionName`

2. Update section planning/execution order
- File: `src/Thesis_ML/experiments/segment_execution.py`
- Update `EXECUTION_SECTION_ORDER`
- Review base artifact requirements if the new section changes prerequisites

3. Add typed section input/output contracts
- File: `src/Thesis_ML/experiments/sections.py`
- Add `...Input` / `...Output` models
- Add wrapper function in `sections.py`

4. Implement execution logic
- File: `src/Thesis_ML/experiments/sections_impl.py` (or new focused module)
- Keep side effects explicit and deterministic

5. Wire section into segment executor
- File: `src/Thesis_ML/experiments/segment_execution.py`
- Add section branch with clear prerequisite checks
- Add artifact registration where appropriate

6. Update compiler/workbook validations if section is user-selectable
- Files:
  - `src/Thesis_ML/orchestration/compiler.py`
  - `src/Thesis_ML/orchestration/workbook_compiler.py`

7. Add tests
- Unit test for section behavior
- Segment-path test including the new section
- Artifact registration test where relevant

## Add a workbook sheet safely

1. Define purpose first
- Governance-only sheet
- Machine-readable planning sheet
- Machine-owned write-back sheet

2. Add sheet creation logic
- Current implementation anchor: `src/Thesis_ML/workbook/template_builder.py`
- Add consistent styles, table definitions, widths, validations, and named ranges

3. Update workbook order/validation
- Ensure sheet appears in `SHEET_ORDER`
- Extend workbook `validate(...)` checks for required columns/formulas if needed

4. If machine-owned, update write-back service
- File: `src/Thesis_ML/orchestration/workbook_writeback.py`
- Add required columns and append logic
- Keep write-back non-destructive to human-authored sheets

5. If executable planning sheet, update compiler
- File: `src/Thesis_ML/orchestration/workbook_compiler.py`
- Add required column checks and parsing rules

6. Update schema metadata if public contract changed
- File: `src/Thesis_ML/config/schema_versions.py`
- Add migration note in docs (`docs/SCHEMA_MIGRATIONS.md`)

7. Add regression tests
- workbook generation validation
- compile path for new sheet semantics
- write-back checks if machine-owned
