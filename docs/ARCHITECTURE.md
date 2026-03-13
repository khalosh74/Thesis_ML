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
  - canonical user entry points (`decision_support.py`, `workbook.py`, `baseline.py`)
- `config/`
  - default paths (`paths.py`)
  - schema/version constants (`schema_versions.py`)
- `data/`, `features/`, `spm/`
  - indexing, cache extraction, SPM utilities
- `experiments/`
  - full pipeline wrapper (`run_experiment.py`)
  - section contracts/runtime adapters (`sections.py`)
  - shared section I/O models (`section_models.py`)
  - segment helper layer (`segment_execution_helpers.py`)
  - segment planner/executor (`segment_execution.py`)
  - idempotency/run-state policy (`execution_policy.py`)
- `orchestration/`
  - manifest contracts/compiler (`contracts.py`, `compiler.py`, `workbook_compiler.py`)
  - campaign execution (`campaign_engine.py`)
  - campaign CLI plumbing (`campaign_cli.py`)
  - result aggregation core + row rendering (`result_aggregation_core.py`, `result_aggregation_rows.py`)
  - compatibility facade (`result_aggregation.py`)
  - reporting/write-back bridge modules
  - backward-compatible facade (`decision_support.py`, `campaign_runner.py`)
- `workbook/`
  - workbook builder facade (`builder.py`)
  - workbook builder/orchestration facade (`template_builder.py`)
  - workbook primitives and structured sheet helpers (`template_primitives.py`, `structured_execution_sheets.py`)
  - workbook validation layer (`template_validation.py`, `validation.py`)
  - metadata helpers (`schema_metadata.py`)
  - compatibility sheet/style helpers (`sheets/`, `styles.py`, `named_ranges.py`)

## Runtime flows

1. Experiment run (`thesisml-run-experiment`)
- Resolve run mode (`fresh`, `resume`, `forced_rerun`) and status file.
- Plan section path from `start_section`/`end_section`.
- Execute section adapters in order.
- Register section artifacts in SQLite.
- Write metrics/report files in `outputs/reports/experiments/<run_id>/`.

2. Decision-support campaign (`thesisml-run-decision-support`)
- Compile JSON registry or workbook rows to internal manifest.
- Select experiments and expand variants/search spaces.
- Dispatch variants through `run_experiment(...)`.
- Export campaign summaries + manifests + decision notes.
- Optionally write results back to a versioned workbook copy.

3. Workbook lifecycle (`thesisml-workbook`)
- Generate governed template workbook.
- Compile executable rows to internal manifest (`workbook_compiler.py`).
- Run campaign and write machine-owned outputs only.

## Canonical interfaces

Use packaged commands from `pyproject.toml`:

- `thesisml-extract-glm`
- `thesisml-build-index`
- `thesisml-cache-features`
- `thesisml-run-experiment`
- `thesisml-run-decision-support`
- `thesisml-run-baseline`
- `thesisml-workbook`

Compatibility wrappers are still present but deprecated:

- `run_decision_support_experiments.py`
- `create_thesis_experiment_workbook.py`
- `scripts/create_thesis_experiment_workbook.py`
- `scripts/run_baseline.py`
