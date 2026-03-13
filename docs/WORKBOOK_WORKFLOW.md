# Workbook-Driven Workflow

This framework supports a full round trip:

1. workbook template generation
2. workbook compilation to internal manifest
3. execution
4. workbook write-back to a versioned output file

## Authoritative workbook files

- Template: `templates/thesis_experiment_program.xlsx`
- Versioned write-back outputs: `outputs/workbooks/`

## Generate template

```bash
thesisml-workbook --output templates/thesis_experiment_program.xlsx
```

## Sheets used for executable planning

Primary machine-readable execution sheets:

- `Experiment_Definitions`
- `Search_Spaces` (optional)

Governance sheets remain in place and are not removed.

## Compile behavior

Compiler module: `src/Thesis_ML/orchestration/workbook_compiler.py`

Validation includes:

- required sheets and required columns
- valid section names
- valid `reuse_policy`
- start/end section ordering
- base artifact usage rules
- supported workbook schema version in README metadata

## Execute from workbook

```bash
thesisml-run-decision-support \
  --workbook templates/thesis_experiment_program.xlsx \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --output-root outputs/artifacts/decision_support \
  --all \
  --write-back-workbook \
  --workbook-output-dir outputs/workbooks
```

## Write-back safety rules

Write-back module: `src/Thesis_ML/orchestration/workbook_writeback.py`

- Source workbook is never overwritten.
- Output file name is deterministic:
  - `<source_stem>__results_<version_tag>.xlsx`
- Existing target path is refused by default.
- Overwrite requires explicit `overwrite_existing=True` in API calls.

## Machine-owned vs human-owned sheets

Machine-owned write-back targets:

- `Machine_Status`
- `Trial_Results`
- `Summary_Outputs`
- optional append to `Run_Log`

Human-authored governance sheets are preserved.

