# Workbook-Driven Workflow

This framework supports a full round trip:

1. workbook template generation
2. workbook compilation to internal manifest
3. execution
4. workbook write-back to a versioned output file

## Authoritative workbook files

- Template: `templates/thesis_experiment_program.xlsx`
- Versioned write-back outputs: `outputs/workbooks/`

Template policy:
- `templates/thesis_experiment_program.xlsx` is a governed planning template
- it is intentionally non-runnable until executable rows are enabled/populated in
  `Experiment_Definitions`

## Generate template

```bash
thesisml-workbook --output templates/thesis_experiment_program.xlsx
```

## Sheets used for executable planning

Primary machine-readable execution sheets:

- `Experiment_Definitions`
- `Search_Spaces` (optional)
- `Study_Design`
- `Study_Rigor_Checklist`
- `Analysis_Plan`
- `Factors`
- `Fixed_Controls`
- `Constraints`
- `Blocking_and_Replication`
- `Generated_Design_Matrix` (machine-managed; can be authored for `custom_matrix`)
- `Trial_Results` (machine-managed)
- `Effect_Summaries` (machine-managed descriptive grouped summaries)

Governance sheets remain in place and are not removed.

## Single-experiment vs factorial studies

- `Experiment_Definitions` remains the canonical single-experiment entry path.
- `Study_Design` enables design-driven execution:
  - `study_type=full_factorial`: compiler expands all valid factor combinations.
  - `study_type=custom_matrix`: compiler executes rows from `Generated_Design_Matrix`.
  - `study_type=single_experiment`: design-layer metadata path (single cell semantics).
  - `study_type=fractional_factorial`: currently rejected with explicit validation error.

Scientific scope:
- the framework executes the design the user specifies;
- it does not auto-invent a scientifically valid design;
- constraints are explicit and auditable;
- effect summaries are descriptive unless explicitly extended by user methodology.

Scientific-rigor metadata scope:
- `Study_Rigor_Checklist` records leakage/bias/reporting readiness metadata per study.
- `Analysis_Plan` records primary contrast and interpretation policy metadata per study.
- These sheets add metadata and validation only; they do not change execution semantics.

## Compile behavior

Compiler module: `src/Thesis_ML/orchestration/workbook_compiler.py`

Validation includes:

- required sheets and required columns
- valid section names
- valid `reuse_policy`
- start/end section ordering
- base artifact usage rules
- supported workbook schema version in README metadata
- study/factor/control/constraint cross-sheet consistency
- study rigor/analysis-plan cross-sheet consistency and duplicate detection
- supported study/factor types and replication fields
- stricter confirmatory checks (`confirmatory_lock_applied`, `primary_contrast`,
  `multiplicity_handling`, `interpretation_rules`)
- exploratory-mode warnings for missing rigor metadata (auditable in manifest warnings)
- explicit rejection of unsupported design modes (for example `fractional_factorial`)

## Execute from workbook

Prepare a runnable workbook first (recommended):

```bash
thesisml-workbook --output outputs/workbooks/my_campaign.xlsx
# edit outputs/workbooks/my_campaign.xlsx:
# - set enabled=Yes for executable rows
# - fill target/cv/model and any optional segment/search columns
```

Run workbook campaign:

```bash
thesisml-run-decision-support \
  --workbook outputs/workbooks/my_campaign.xlsx \
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
- `Generated_Design_Matrix`
- `Effect_Summaries`
- optional append to `Run_Log`

Human-authored governance sheets are preserved.
