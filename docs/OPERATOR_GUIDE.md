# Operator Guide (Canonical)

Use these commands as the supported operator path.

## 1) Environment

Canonical operator path is source checkout + `uv.lock`.

```bash
python -m pip install --upgrade pip
python -m pip install uv
python -m uv sync --frozen --extra dev
```

Optional Optuna support:

```bash
python -m uv sync --frozen --extra dev --extra optuna
```

Installed wheel path is also supported; default decision-support registry is packaged in the wheel.

## 2) Canonical CLIs

- `thesisml-run-experiment`
- `thesisml-run-decision-support`
- `thesisml-workbook`
- `thesisml-run-baseline`

Compatibility wrappers are kept only for migration and are deprecated.

## 3) First-use checks

Generate workbook template:

```bash
thesisml-workbook --output templates/thesis_experiment_program.xlsx
```

Template note:
- `templates/thesis_experiment_program.xlsx` is non-runnable by default
- enable/populate executable rows in `Experiment_Definitions` for single-experiment flow
- or enable/populate `Study_Design` (+ `Factors`/`Fixed_Controls`/`Constraints` as needed)
  for factorial flow
- optionally complete `Study_Rigor_Checklist` and `Analysis_Plan` for explicit
  scientific-rigor metadata and stricter confirmatory validation

Registry dry-run:

```bash
thesisml-run-decision-support \
  --registry configs/decision_support_registry.json \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --output-root outputs/artifacts/decision_support \
  --all \
  --dry-run
```

Installed-wheel note:
- the default registry path is packaged in the wheel, so `--registry` is optional
- passing explicit `--registry` remains supported and recommended for controlled campaigns

## 4) Gold acceptance command

Run this before releases or high-value campaigns:

```bash
python scripts/acceptance_smoke.py
```

This command validates:
- shipped workbook template/schema compatibility
- workbook generation through canonical CLI
- shipped registry compilation/dry-run through canonical CLI

## 5) Factorial design operator notes

- Define the scientific design explicitly in workbook sheets.
- Required user-owned design decisions:
  - factors and allowed levels
  - fixed controls
  - invalid combinations (constraints)
  - replication and seed policy
- Framework behavior:
  - compiles design into executable trials
  - executes trials through existing engine/artifact lineage
  - writes descriptive grouped summaries to `Effect_Summaries`
- Scientific-rigor metadata behavior:
  - records checklist and analysis-plan metadata per study
  - writes pre-execution `Study_Review` disposition (`allowed` / `warning` / `blocked`)
  - validates confirmatory studies more strictly than exploratory studies
  - does not auto-design science or guarantee scientific validity
- The framework does not auto-design study validity or perform automatic inferential statistics.

Execution policy:
- Exploratory studies:
  - require core fields (`question`, `generalization_claim`, `primary_metric`, `cv_scheme`)
  - can run with warnings when non-core rigor metadata is incomplete
- Confirmatory studies:
  - are blocked when required rigor fields are missing
  - must include lock/analysis-plan completeness before execution

Before trusting results:
- review `Study_Review` in workbook outputs
- check `study_review_summary.json` in campaign exports
- verify why a study was allowed, warned, or blocked
