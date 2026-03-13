# Extension Guide

This guide covers two common extension tasks:

- adding a new execution section
- adding a new workbook sheet safely

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

