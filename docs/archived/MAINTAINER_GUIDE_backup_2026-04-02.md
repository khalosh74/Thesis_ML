# Backup: MAINTAINER_GUIDE.md

This is an archived backup copy of `docs/MAINTAINER_GUIDE.md` created on 2026-04-02 by the repository agent during a documentation consolidation operation. The original content follows.

----------

# Maintainer Guide (Concise)

This document is the canonical maintainer quick reference.

## Canonical public surface

Packaged entry points from `pyproject.toml` are authoritative:

- `thesisml-extract-glm`
- `thesisml-build-index`
- `thesisml-cache-features`
- `thesisml-run-experiment`
- `thesisml-run-decision-support`
- `thesisml-workbook`
- `thesisml-run-baseline`

Compatibility scripts are transitional only and should not be documented as primary paths.

## Current incremental module split points

- `src/Thesis_ML/experiments/section_models.py`: shared section I/O models.
- `src/Thesis_ML/experiments/segment_execution_helpers.py`: segment planning/reuse/base-artifact helpers.
- `src/Thesis_ML/orchestration/result_aggregation_core.py`: aggregation logic.
- `src/Thesis_ML/orchestration/result_aggregation_rows.py`: summary row rendering.
- `src/Thesis_ML/orchestration/workbook_compiler.py`: workbook-to-manifest compiler
  including factorial design expansion and scientific-rigor metadata validation.
- `src/Thesis_ML/orchestration/workbook_bridge.py`: workbook write-back row builders
  (trial/design/effect rows).
- `src/Thesis_ML/workbook/template_primitives.py`: workbook styling/validation primitives.
- `src/Thesis_ML/workbook/structured_execution_sheets.py`: structured workbook sheet builders.
- `src/Thesis_ML/workbook/template_validation.py`: workbook validation logic.

Facades retained for compatibility:
- `src/Thesis_ML/orchestration/result_aggregation.py`
- `src/Thesis_ML/workbook/template_builder.py`

## Gold acceptance path

```bash
python scripts/acceptance_smoke.py
```

Used in CI/release gate to validate shipped assets and canonical CLI flow.

Shipped-asset drift protection:
- `tests/test_shipped_assets.py` asserts packaged assets match committed
  `configs/` and `templates/` copies
- template metadata and non-runnable template policy are regression-tested

Factorial-design guardrails:
- do not silently expand unsupported study types;
- `fractional_factorial` must fail explicitly unless implemented rigorously;
- keep effect summaries labeled descriptive unless inferential methods are added with tests/docs;
- maintain backward compatibility for existing `Experiment_Definitions` flow.
- keep scientific-rigor layer additive: metadata + validation only, no execution semantics changes.
- maintain explicit study guardrail dispositions (`allowed`, `warning`, `blocked`) and
  keep confirmatory blocking requirements strict/auditable.

## Scientific-rigor extension map

- Schema and contract models:
  - `src/Thesis_ML/orchestration/contracts.py`
  - key models: `StudyDesignSpec`, `StudyRigorChecklistSpec`, `AnalysisPlanSpec`,
    `StudyReviewSummary`
- Workbook schema and sheet columns:
  - `src/Thesis_ML/workbook/template_builder.py`
  - `src/Thesis_ML/workbook/structured_execution_sheets.py`
  - `src/Thesis_ML/workbook/template_validation.py`
- Guardrail validation policy and study review generation:
  - `src/Thesis_ML/orchestration/workbook_compiler.py` (`_build_study_review`)
- Runtime study-review artifact and workbook write-back bridge:
  - `src/Thesis_ML/orchestration/campaign_engine.py`
  - `src/Thesis_ML/orchestration/workbook_bridge.py`
  - `src/Thesis_ML/orchestration/workbook_writeback.py`

Adding a new rigor field safely:
1. Add field to contract model(s) in `contracts.py`.
2. Add workbook column(s) in template builder/sheet builders.
3. Add parsing + validation in `workbook_compiler.py` (including study cross-references).
4. Extend `StudyReviewSummary` if the field affects guardrail visibility/disposition.
5. Add tests for schema presence, validation behavior, and write-back visibility.
6. Update operator/maintainer docs with policy impact and limits.

Extending supported study types honestly:
- Add explicit model + parser support first.
- Keep unsupported types rejected with clear validation errors until fully implemented.
- Do not label partial implementations as complete design support.

## Core verification commands

```bash
python -m mypy
python -m ruff check src/Thesis_ML tests --exclude src/Thesis_ML/workbook/template_builder.py
python -m ruff format --check src/Thesis_ML tests --exclude src/Thesis_ML/workbook/template_builder.py
python -m pytest -q
python scripts/acceptance_smoke.py
```

## Compatibility policy

- Keep wrapper scripts while downstream users migrate.
- Each wrapper must emit a deprecation warning that points to the canonical CLI.
- New docs/examples should only use canonical CLIs.
