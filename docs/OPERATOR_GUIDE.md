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

Grouped nested comparison execution (archived spec example):

```bash
thesisml-run-comparison \
  --comparison configs/archive/comparisons/model_family_grouped_nested_comparison_v1.json \
  --variant ridge \
  --reports-root outputs/reports/comparisons
```

# Operator Guide (Merged)

This document has been merged into `docs/RUNBOOK.md` (merged on 2026-04-02). The full original content is preserved in `docs/archived/OPERATOR_GUIDE_backup_2026-04-02.md` and the canonical operator guidance now lives at `docs/RUNBOOK.md#merged-content-operator-guide-from-operator_guide.md`.

If you need the original standalone operator guide, find it at `docs/archived/OPERATOR_GUIDE_backup_2026-04-02.md`.

-- Merged on 2026-04-02 by repository agent
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
python scripts/verify_confirmatory_ready.py --output-dir <confirmatory_output_dir>
python scripts/replay_official_paths.py --mode both --use-demo-dataset --verify-determinism
python scripts/build_publishable_bundle.py --output-dir <bundle_dir> --comparison-output <...> --confirmatory-output <...> --replay-summary <...> --replay-verification-summary <...> --repro-manifest <...>
python scripts/verify_publishable_bundle.py --bundle-dir <bundle_dir>
```

Optional lightweight performance snapshot:

```bash
python scripts/performance_smoke.py --output outputs/performance/performance_smoke_summary.json
python scripts/profile_runtime_baseline.py --mode ci_synthetic
python scripts/profile_runtime_baseline.py --mode operator_dataset --index-csv <...> --data-root <...> --cache-dir <...>
python scripts/profile_runtime_baseline.py --mode ci_synthetic --compare-against outputs/performance/baselines/<previous_tag>/baseline_bundle.json
python scripts/replay_official_paths.py --mode confirmatory --index-csv <...> --data-root <...> --cache-dir <...> --suite confirmatory_primary_within_subject --verify-determinism --skip-confirmatory-ready
python scripts/rc1_release_gate.py --run-ruff --run-pytest --run-performance-smoke
```

Phase 0 runtime baselining is advisory proof infrastructure for future performance work:
- `scripts/performance_smoke.py` is a quick smoke timer for workbook build + protocol/comparison dry-run CLI paths.
- `scripts/profile_runtime_baseline.py` is the canonical baseline bundle runner that captures comparable artifacts for:
  - `performance_smoke_existing`
  - `single_run_ridge_direct` (direct `run_experiment()` with process profiling)
  - `runtime_profile_precheck`
  - `decision_support_dry_run`
- `--mode ci_synthetic` runs only from shipped repository assets (no external dataset dependency).
- `--mode operator_dataset` runs the same baseline suite against an operator-provided dataset triple (`index_csv`, `data_root`, `cache_dir`).
- `--compare-against` compares a new bundle to an earlier bundle and reports scientific parity vs observability parity.
