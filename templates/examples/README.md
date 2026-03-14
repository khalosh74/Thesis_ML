# Workbook Examples

This directory contains small, auditable workbook examples for study-design workflows.

## Canonical Designed Study

- File: `canonical_designed_study.xlsx`
- Scope: minimal 2x2 `full_factorial` study with explicit rigor metadata
- Study ID: `S_CANONICAL_2X2`

What it demonstrates:

- `Study_Design` setup for an enabled designed study
- `Factors` with two manipulated factors (`model`, `filter_modality`)
- `Fixed_Controls` for `target`, `cv`, `subject`, and `filter_task`
- `Study_Rigor_Checklist` and `Analysis_Plan` rows aligned to the same `study_id`
- pre-execution review/guardrail output in `Study_Review`
- machine-managed outputs in `Generated_Design_Matrix`, `Trial_Results`, and `Effect_Summaries`

Scientific scope note:

- This workbook is an execution example, not a claim of automatic scientific validity.
- The framework executes the design that the user specifies.
- Grouped summaries are descriptive unless inferential methods are explicitly implemented.
