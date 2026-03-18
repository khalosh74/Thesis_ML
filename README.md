# Thesis_ML

Leakage-safe, experiment-ready fMRI ML framework for session-level GLM extraction, dataset indexing,
feature caching, and reproducible multi-model evaluation.

## Data policy

- Keep real data local and untracked.
- Keep reusable logic in `src/Thesis_ML` (not notebooks).
- Do not commit dataset files, model binaries, or generated report artifacts.

## Governance and responsible use

Governance policy and release-facing boundaries are explicit and versioned:

- Privacy/data handling: `docs/PRIVACY_AND_DATA_HANDLING.md`
- Intended/non-intended use and misuse warnings: `docs/USE_AND_MISUSE_BOUNDARIES.md`
- Confirmatory-ready criteria and verification command: `docs/CONFIRMATORY_READY.md`
- Canonical reproducibility workflow: `docs/REPRODUCIBILITY.md`

Confirmatory-ready means contract/governance checks passed for frozen scientific runs.
It does **not** mean deployable clinical decision support.

## Expected data layout

Use a BIDS-like hierarchy under a local data root (`Data/` or `data/`):

```text
Data/
  sub-001/
    ses-04/
      BAS2/
        beta_0001.nii
        beta_0002.nii
        ...
        mask.nii
        regressor_labels.csv
        SPM.mat            # optional for extraction (if missing, extraction still works)
```

`regressor_labels.csv` must map 1:1 by row index (1-based) to `beta_0001.nii ... beta_NNNN.nii`.

## Supported Python

- Canonical and CI-tested: `Python 3.13`
- Package constraint: `>=3.11,<3.14` (from `pyproject.toml`)

The repository includes `.python-version` pinned to `3.13`.

## Reproducible setup (canonical: uv lockfile)

`uv.lock` is the primary reproducible dependency path for this project.

### PowerShell

```powershell
python -m pip install --upgrade pip
python -m pip install uv
python -m uv sync --frozen --extra dev
```

### Bash

```bash
python -m pip install --upgrade pip
python -m pip install uv
python -m uv sync --frozen --extra dev
```

Optional extra for Optuna-backed search mode:

```powershell
python -m uv sync --frozen --extra dev --extra optuna
```

## Bootstrap commands (clean machine)

You can run the scripts:
- PowerShell: `./scripts/bootstrap_env.ps1`
- Bash: `./scripts/bootstrap_env.sh`

Or run the commands directly:

```powershell
python -m uv sync --frozen --extra dev
python -m uv run python -m pytest -q
python -m uv run thesisml-run-comparison --help
python -m uv run thesisml-run-protocol --help
python -m uv run thesisml-run-decision-support --help
python -m uv run thesisml-workbook --output outputs/workbooks/bootstrap_thesis_experiment_program.xlsx
```

Official confirmatory thesis run path (frozen protocol):

```powershell
python -m uv run thesisml-run-protocol `
  --protocol configs/protocols/thesis_confirmatory_v1.json `
  --all-suites `
  --reports-root outputs/reports/confirmatory
```

Framework mode lifecycle:
- `thesisml-run-experiment` -> exploratory mode (`framework_mode=exploratory`, `canonical_run=false`), default reports root `outputs/reports/exploratory/`
- `thesisml-run-comparison` -> locked comparison mode (`framework_mode=locked_comparison`, `canonical_run=false`), default reports root `outputs/reports/comparisons/`
- `thesisml-run-protocol` -> confirmatory mode (`framework_mode=confirmatory`, `canonical_run=true`), default reports root `outputs/reports/confirmatory/`
- official comparison/protocol contracts must pick exactly one methodology policy:
  - `fixed_baselines_only`
  - `grouped_nested_tuning`

Strict metric consistency policy:
- official runs must declare one primary metric (`balanced_accuracy`, `macro_f1`, or `accuracy`)
- the declared primary metric drives tuning objective, comparison winner selection, confirmatory summaries, and permutation testing
- secondary metrics are reporting-only and do not drive selection decisions
- official permutation metric must equal the declared primary metric
- metric drift fallbacks are disallowed; decision-support/workbook/search-space inputs must declare explicit metric fields
- official run artifacts persist effective metric policy in `config.json` / `metrics.json` under `metric_policy_effective`

Evidence layer policy (official comparison + confirmatory paths):
- contracts must declare explicit `evidence_policy`
- repeated-run settings (`repeat_count`, `seed_stride`) are compiled into concrete run specs
- grouped-bootstrap confidence intervals and paired sign-flip comparisons are emitted as machine-readable artifacts
- calibration artifacts are always emitted (`performed` or explicit `not_applicable`)
- required evidence package checks (dummy/permutation/untuned baseline when tuning) are reflected in mode-level summaries/decisions
- official checked-in comparison/protocol specs use `repeat_count=3` by default
- official checked-in locked comparison specs require significant paired wins (`require_significant_win=true`) for winner selection
- calibration policy is explicit: calibration is performed when probability scores exist; otherwise runs emit `status=not_applicable` (no synthetic calibration)

Official data-layer policy (official comparison + confirmatory paths):
- contracts now carry explicit `data_policy` (class balance, missingness, leakage, external-validation compatibility).
- structural integrity failures are blocking on official paths (missing required columns/fields, invalid grouping requirements, empty filtered subsets, blocking leakage findings).
- threshold checks default to warnings unless explicitly configured as blocking in `data_policy`.
- official run artifacts now include:
  - `dataset_card.json`, `dataset_card.md`
  - `dataset_summary.json`, `dataset_summary.csv`
  - `data_quality_report.json`
  - `class_balance_report.csv`, `missingness_report.csv`
  - `leakage_audit.json`
  - `external_dataset_card.json`, `external_dataset_summary.json`, `external_validation_compatibility.json`
- external validation in this phase is compatibility-only (schema/coverage compatibility checks); it is explicitly labeled as external/non-confirmatory evidence.

RC-1 hardening policy (official runs):
- confirmatory and locked-comparison runs enforce strict preflight contract validation before execution
- run artifacts now include deterministic provenance (`git_provenance` and `dataset_fingerprint`)
- run-level `run_status.json` now exposes structured failure diagnostics (`error_code`, `error_type`, `failure_stage`) and warning/timing/resource summaries
- mode-level runners verify official artifact completeness/invariants before returning success

Modular architecture highlights:
- `src/Thesis_ML/experiments/runtime_policies.py` owns framework-context, methodology, and official metric-policy resolution.
- `src/Thesis_ML/experiments/run_artifacts.py` owns run identity extraction and run-level artifact payload stamping/building.
- `src/Thesis_ML/comparisons/decision.py` owns comparison winner/inconclusive/invalid decision logic.
- `src/Thesis_ML/orchestration/study_review.py` owns workbook study-review guardrail evaluation helpers.
- `src/Thesis_ML/workbook/template_constants.py` owns workbook catalog constants and seeded experiment definitions.
- `src/Thesis_ML/workbook/governance_sheet_builders.py` owns governance sheet rendering logic; `template_builder.py` is the coordinator/facade.
- `src/Thesis_ML/workbook/structured_sheet_core.py`, `structured_sheet_design.py`, and `structured_sheet_operations.py` own structured execution-sheet rendering internals behind `structured_execution_sheets.py`.
- `src/Thesis_ML/orchestration/workbook_study_design_layer.py` owns study-design layer compilation; `workbook_compiler.py` coordinates top-level workbook compilation flow.
- top-level runners remain orchestration-first and delegate policy/decision/artifact logic to dedicated modules.

Locked comparison example:

```powershell
python -m uv run thesisml-run-comparison `
  --comparison configs/comparisons/model_family_comparison_v1.json `
  --all-variants `
  --reports-root outputs/reports/comparisons
```

Grouped nested comparison example:

```powershell
python -m uv run thesisml-run-comparison `
  --comparison configs/comparisons/model_family_grouped_nested_comparison_v1.json `
  --variant ridge `
  --reports-root outputs/reports/comparisons
```

Decision-support campaign command (requires index/cache/data paths to exist):

```powershell
python -m uv run thesisml-run-decision-support `
  --registry configs/decision_support_registry.json `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --output-root outputs/artifacts/decision_support `
  --all
```

Registry default behavior:
- source checkout default resolves to `configs/decision_support_registry.json`
- installed wheel default resolves to packaged asset `Thesis_ML/assets/configs/decision_support_registry.json`
- explicit `--registry` remains supported and is recommended for controlled runs

Optional Optuna-backed variant search:

```powershell
python -m uv run thesisml-run-decision-support `
  --registry configs/decision_support_registry.json `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --output-root outputs/artifacts/decision_support `
  --all `
  --search-mode optuna `
  --optuna-trials 25
```

## Workbook template policy

`templates/thesis_experiment_program.xlsx` is a **planning template** by default.
It is structurally valid but intentionally non-runnable until at least one row in
`Experiment_Definitions` is enabled (`enabled=Yes`) and includes required values
(`target`, `cv`, `model`).

Safe first-use pattern:

```powershell
python -m uv run thesisml-workbook --output outputs/workbooks/my_campaign.xlsx
# edit outputs/workbooks/my_campaign.xlsx and enable/populate executable rows
python -m uv run thesisml-run-decision-support `
  --workbook outputs/workbooks/my_campaign.xlsx `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --output-root outputs/artifacts/decision_support `
  --all `
  --write-back-workbook
```

## Compatibility install path (kept)

The editable `pip` flow remains supported:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev,spm]"
```

If you do not need `SPM.mat` parsing, `.[dev]` is sufficient.
If you need Optuna search mode, install `.[optuna]` (or `.[dev,optuna]`).

## Docker / devcontainer

- Docker image: `Dockerfile` (Python `3.13`, `uv sync --frozen --extra dev`)
- VS Code devcontainer: `.devcontainer/devcontainer.json`

Build and run tests in Docker:

```bash
docker build -t thesis-ml:dev .
docker run --rm thesis-ml:dev
```

## CI and release gate

- CI workflow: `.github/workflows/ci.yml` (push/PR)
- Release validation workflow: `.github/workflows/release_gate.yml` (tags `v*`)
- Release details: `docs/RELEASE.md`
- Gold acceptance path used by CI/release: `python scripts/acceptance_smoke.py`
- Local release hygiene check: `python scripts/release_hygiene_check.py`
- Lightweight performance smoke: `python scripts/performance_smoke.py`
- Official artifact invariant check: `python scripts/verify_official_artifacts.py --output-dir <official_output_dir>`
- Confirmatory-ready governance check: `python scripts/verify_confirmatory_ready.py --output-dir <confirmatory_output_dir>`
- Deterministic rerun check: `python scripts/verify_official_reproducibility.py --mode protocol --index-csv <...> --data-root <...> --cache-dir <...> --suite confirmatory_primary_within_subject`
- One-command official replay (comparison + confirmatory): `python scripts/replay_official_paths.py --mode both --use-demo-dataset`
- Publishable bundle tooling:
  - `python scripts/build_publishable_bundle.py --output-dir <bundle_dir> --comparison-output <...> --confirmatory-output <...> --replay-summary <...> --replay-verification-summary <...> --repro-manifest <...>`
  - `python scripts/verify_publishable_bundle.py --bundle-dir <bundle_dir>`
- RC wrapper gate script: `python scripts/rc1_release_gate.py --run-ruff --run-pytest --run-performance-smoke`

## Operator documentation

- Architecture: `docs/ARCHITECTURE.md`
- Operator quick path: `docs/OPERATOR_GUIDE.md`
- Runbook: `docs/RUNBOOK.md`
- Privacy/data handling: `docs/PRIVACY_AND_DATA_HANDLING.md`
- Use/misuse boundaries: `docs/USE_AND_MISUSE_BOUNDARIES.md`
- Confirmatory-ready criteria: `docs/CONFIRMATORY_READY.md`
- Reproducibility workflow: `docs/REPRODUCIBILITY.md`
- Workbook workflow: `docs/WORKBOOK_WORKFLOW.md`
- Segment execution: `docs/SEGMENT_EXECUTION.md`
- Maintainer quick path: `docs/MAINTAINER_GUIDE.md`
- Extension guide: `docs/EXTENDING.md`
- Schema/version migration notes: `docs/SCHEMA_MIGRATIONS.md`
- Decision-support specifics: `docs/DECISION_SUPPORT_AUTOMATION.md`
- Experiment semantics: `docs/EXPERIMENTS.md`

## How to cite

Use the repository citation metadata in `CITATION.cff`.

## Quickstart (end-to-end)

### 1) Extract one session mapping + summary

```powershell
thesisml-extract-glm `
  --glm-dir Data/sub-001/ses-04/BAS2 `
  --out-dir Data/processed/extractions/sub-001/ses-04/BAS2
```

Outputs:
- `regressor_beta_mapping.csv`
- `session_summary.json`

### 2) Build dataset index across many sessions

```powershell
thesisml-build-index `
  --data-root Data `
  --out-csv Data/processed/dataset_index.csv `
  --pattern "sub-*/ses-*/BAS2"
```

The output index includes at least:
- `subject`, `session`, `bas`, `run`, `task`, `emotion`, `coarse_affect`, `modality`
- `beta_path`, `mask_path`, `regressor_label`

Path convention:
- `beta_path` and `mask_path` are stored relative to `--data-root` when possible for portability.

### 3) Cache masked beta features

```powershell
thesisml-cache-features `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache
```

Cache behavior:
- One compressed `.npz` per `subject_session_bas` group.
- Contains `X` (`n_samples x n_voxels`), `y`, row metadata, and spatial-signature metadata.
- Each beta is validated against its mask for shape + affine compatibility before vectorization.
- Existing cache files are skipped unless `--force` is passed.

### 4) Run exploratory within-person session-held-out experiment (`within_subject_loso_session`)

```powershell
thesisml-run-experiment `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --target coarse_affect `
  --model ridge `
  --cv within_subject_loso_session `
  --subject sub-001 `
  --seed 42
```

### 5) Run exploratory frozen cross-person transfer (`frozen_cross_person_transfer`)

```powershell
thesisml-run-experiment `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --target coarse_affect `
  --model ridge `
  --cv frozen_cross_person_transfer `
  --train-subject sub-001 `
  --test-subject sub-002 `
  --seed 42
```

Auxiliary grouped comparison mode (non-primary thesis evidence path):

```powershell
thesisml-run-experiment `
  --index-csv Data/processed/dataset_index.csv `
  --data-root Data `
  --cache-dir Data/processed/feature_cache `
  --target coarse_affect `
  --filter-task passive `
  --filter-modality audio `
  --model logreg `
  --cv loso_session `
  --seed 42 `
  --n-permutations 1000
```

## Experiment artifacts

Each run writes:

```text
outputs/reports/<mode>/<run_id>/
  config.json
  metrics.json
  fold_metrics.csv
  fold_splits.csv
  predictions.csv
  subgroup_metrics.json
  subgroup_metrics.csv
  tuning_summary.json
  best_params_per_fold.csv
  spatial_compatibility_report.json
  interpretability_summary.json
  calibration_summary.json
  calibration_table.csv
```

Default `<mode>` by command:
- `thesisml-run-experiment` -> `exploratory`
- `thesisml-run-comparison` -> `comparisons`
- `thesisml-run-protocol` -> `confirmatory`

Within-subject linear runs (`logreg`, `linearsvc`, `ridge`) additionally write:
- `interpretability_fold_explanations.csv`
- `interpretability/fold_###_coefficients.npz`

All models use the same split/metrics core:
- `accuracy`
- `balanced_accuracy`
- `macro_f1`
- confusion matrix + fold-level metrics

Interpretability outputs are supporting model-behavior robustness evidence only; they are not
direct neural localization claims.

Mode-level manifests:
- locked comparison executions write `comparison_runs/<comparison_id>__<comparison_version>/...`
- confirmatory protocol executions write `protocol_runs/<protocol_id>__<protocol_version>/...`
- locked comparison manifests include `comparison_decision.json` with
  `winner_selected` / `inconclusive` / `invalid_comparison`.
- locked comparison evidence artifacts include:
  - `repeated_run_metrics.csv`
  - `repeated_run_summary.json`
  - `confidence_intervals.json`
  - `metric_intervals.csv`
  - `paired_model_comparisons.json`
  - `paired_model_comparisons.csv`
- confirmatory protocol evidence artifacts include:
  - `repeated_run_metrics.csv`
  - `repeated_run_summary.json`
  - `confidence_intervals.json`
  - `metric_intervals.csv`

## Adding a new model to the registry

Edit `src/Thesis_ML/experiments/model_factory.py`:
- Add a new case in `make_model`.
- Keep preprocessing/model composition inside `build_pipeline` so transforms are fit only on
  training folds.
- Add the model name to `MODEL_NAMES`.
- Keep output schema unchanged (`metrics.json`, `fold_metrics.csv`, `predictions.csv`) for
  comparability.

## Quality checks

```powershell
python -m uv run python -m mypy
python -m uv run python -m ruff check src/Thesis_ML/artifacts src/Thesis_ML/comparisons src/Thesis_ML/protocols src/Thesis_ML/orchestration src/Thesis_ML/workbook src/Thesis_ML/experiments/segment_execution.py src/Thesis_ML/experiments/sections.py src/Thesis_ML/experiments/run_experiment.py src/Thesis_ML/cli/comparison_runner.py src/Thesis_ML/cli/protocol_runner.py --exclude src/Thesis_ML/workbook/template_builder.py
python -m uv run python -m ruff format --check src/Thesis_ML/artifacts src/Thesis_ML/comparisons src/Thesis_ML/protocols src/Thesis_ML/orchestration src/Thesis_ML/workbook src/Thesis_ML/experiments/segment_execution.py src/Thesis_ML/experiments/sections.py src/Thesis_ML/experiments/run_experiment.py src/Thesis_ML/cli/comparison_runner.py src/Thesis_ML/cli/protocol_runner.py --exclude src/Thesis_ML/workbook/template_builder.py
python -m uv run python -m pytest -q
python -m uv run python scripts/acceptance_smoke.py
```

`mypy` in CI is required and checks:
- `src/Thesis_ML/artifacts`
- `src/Thesis_ML/orchestration`
- `src/Thesis_ML/workbook`
- `src/Thesis_ML/spm/extract_glm.py`
- `src/Thesis_ML/features/nifti_features.py`
- `src/Thesis_ML/experiments/run_experiment.py`
- `src/Thesis_ML/comparisons`
- `src/Thesis_ML/protocols`
- `src/Thesis_ML/cli/comparison_runner.py`
- `src/Thesis_ML/cli/protocol_runner.py`
- `src/Thesis_ML/experiments/segment_execution.py`
- `src/Thesis_ML/experiments/sections.py`

## Compatibility wrappers (deprecated)

The following script paths are still supported but deprecated:

- `run_decision_support_experiments.py` -> use `thesisml-run-decision-support`
- `create_thesis_experiment_workbook.py` -> use `thesisml-workbook`
- `scripts/create_thesis_experiment_workbook.py` -> use `thesisml-workbook`
- `scripts/run_baseline.py` -> use `thesisml-run-baseline`

## Existing synthetic baseline

The original synthetic baseline remains available and unchanged:

```powershell
thesisml-run-baseline
```

Outputs:
- `outputs/reports/metrics.json`
- `outputs/models/baseline_model.npz`
