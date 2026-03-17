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

- `thesisml-run-protocol` (official thesis-facing command)
- `thesisml-run-experiment`
- `thesisml-run-comparison`
- `thesisml-run-decision-support`
- `thesisml-workbook`
- `thesisml-run-baseline`

Compatibility wrappers are kept only for migration and are deprecated.

## 3) First-use checks

Frozen confirmatory protocol dry-run:

```bash
thesisml-run-protocol \
  --protocol configs/protocols/thesis_confirmatory_v1.json \
  --all-suites \
  --reports-root outputs/reports/confirmatory \
  --dry-run
```

Frozen confirmatory protocol execution:

```bash
thesisml-run-protocol \
  --protocol configs/protocols/thesis_confirmatory_v1.json \
  --all-suites \
  --reports-root outputs/reports/confirmatory
```

Locked comparison dry-run:

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/model_family_comparison_v1.json \
  --all-variants \
  --reports-root outputs/reports/comparisons \
  --dry-run
```

Locked comparison execution:

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/model_family_comparison_v1.json \
  --all-variants \
  --reports-root outputs/reports/comparisons
```

Grouped nested comparison execution:

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/model_family_grouped_nested_comparison_v1.json \
  --variant ridge \
  --reports-root outputs/reports/comparisons
```

Official-policy note:
- exploratory mode (`thesisml-run-experiment`) is ad hoc and non-confirmatory.
- locked comparison mode (`thesisml-run-comparison`) is restricted to registered variants.
- confirmatory mode (`thesisml-run-protocol`) is the final thesis evidence path.
- default mode roots are `outputs/reports/exploratory`, `outputs/reports/comparisons`, and `outputs/reports/confirmatory`.
- comparison/protocol contracts must choose exactly one methodology policy:
  `fixed_baselines_only` or `grouped_nested_tuning`.
- locked comparison runs emit `comparison_decision.json` for machine-readable selection outcomes.
- official comparison/confirmatory paths run strict preflight contract validation and fail fast on violations.
- run-level `run_status.json` exposes structured diagnostics (`error_code`, `error_type`, `failure_stage`) and warning/timing/resource summaries.

Official metric policy:
- declare one `primary_metric` per comparison/protocol contract (`balanced_accuracy`, `macro_f1`, or `accuracy`)
- primary metric is decision-driving for tuning, comparison winner selection, permutation testing, and headline summaries
- secondary metrics are reporting-only
- official permutation metric must match primary metric
- artifacts include `metric_policy_effective` so operators can audit effective primary/decision/tuning/permutation metrics
- official run artifacts include deterministic provenance (`git_provenance`, `dataset_fingerprint`) for rerun traceability

Architecture ownership (where to extend safely):
- runner policy resolution belongs in `src/Thesis_ML/experiments/runtime_policies.py`
- run artifact payload shaping belongs in `src/Thesis_ML/experiments/run_artifacts.py`
- comparison decision rules belong in `src/Thesis_ML/comparisons/decision.py`
- workbook study-review guardrails belong in `src/Thesis_ML/orchestration/study_review.py`

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

- Canonical example workbook:
  - `templates/examples/canonical_designed_study.xlsx`
  - includes one small 2x2 designed study with rigor metadata and analysis plan
  - intended as a reference pattern, not a production claim

- Single experiment vs designed study:
  - single experiment: populate `Experiment_Definitions` rows directly
  - designed study: define one study in `Study_Design` and related sheets (`Factors`,
    `Fixed_Controls`, `Constraints`, `Blocking_and_Replication`)

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
- It also does not perform automatic causal inference or automatic significance testing.

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

How to fill scientific-rigor sheets:
- `Study_Rigor_Checklist`: record leakage review, unit-of-analysis, hierarchy assumptions,
  missing-data/class-imbalance/subgroup plans, and lock status.
- `Analysis_Plan`: record primary contrast, aggregation level, uncertainty method,
  multiplicity handling, interaction reporting policy, and interpretation rules.

How to read machine-managed study outputs:
- `Generated_Design_Matrix`: concrete generated design cells/trials from your study definition.
- `Trial_Results`: per-trial execution outcomes and artifact paths.
- `Effect_Summaries`: descriptive grouped summaries by factor levels/combinations.
- `Study_Review`: guardrail disposition (`allowed`, `warning`, `blocked`), missing fields, and
  warning/error counts that explain execution eligibility.

## 6) Release hygiene and performance smoke

Run before cutting a release candidate:

```bash
python scripts/release_hygiene_check.py
python scripts/acceptance_smoke.py
python scripts/verify_official_artifacts.py --output-dir <official_output_dir>
```

Optional lightweight performance snapshot:

```bash
python scripts/performance_smoke.py --output outputs/performance/performance_smoke_summary.json
python scripts/verify_official_reproducibility.py --mode protocol --index-csv <...> --data-root <...> --cache-dir <...> --suite confirmatory_primary_within_subject
python scripts/rc1_release_gate.py --run-ruff --run-pytest --run-performance-smoke
```
