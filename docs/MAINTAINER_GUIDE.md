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

## Merged content: Script Inventory (from `docs/maintainer/script_inventory.md`)

The following content was merged from `docs/maintainer/script_inventory.md` on 2026-04-02. A backup of the original inventory is available at `docs/archived/script_inventory_backup_2026-04-02.md`.

---

# Script Inventory

Current Wave-2 inventory of `scripts/` with canonical vs wrapper ownership.

| Script | Current role | Operator-facing | Canonical or wrapper | Responsibility note |
|---|---|---|---|---|
| `_common.py` | compatibility wrapper | no | wrapper | Backward-compatible helper import surface; real implementations live in `Thesis_ML.script_support`. |
| `acceptance_smoke.py` | dev-only smoke/performance tool | no | canonical | Local acceptance smoke checks for workbook compile + command readiness. |
| `benchmark_torch_ridge_backend.py` | dev-only smoke/performance tool | no | canonical | Backend micro-benchmark utility for torch ridge timing checks. |
| `bootstrap_env.ps1` | platform helper | yes | canonical | Windows bootstrap for dev env sync and quick validation commands. |
| `bootstrap_env.sh` | platform helper | yes | canonical | POSIX bootstrap for dev env sync and quick validation commands. |
| `build_frozen_confirmatory_registry.py` | canonical operator script | yes | canonical | Build frozen confirmatory registry/manifest/report from reviewed preflight bundle + scope. |
| `build_publishable_bundle.py` | canonical operator script | yes | canonical | Assemble publishable release bundle and manifest. |
| `freeze_workbook_registry.py` | canonical operator script | yes | canonical | Compile workbook to decision-support execution registry JSON. |
| `generate_demo_dataset.py` | dev/specialized tool | yes | canonical | Generate deterministic synthetic demo dataset + index + manifest. |
| `performance_smoke.py` | dev-only smoke/performance tool | no | canonical | Local performance smoke benchmark over workbook/campaign path. |
| `prepare_dependent_rerun_registry.py` | canonical operator script | yes | canonical | Build temporary E07/E08 rerun registry when non-ridge stage-3 winner is selected. |
| `prepare_frozen_campaign.ps1` | platform helper | yes | canonical | PowerShell helper for frozen campaign preparation/archive chores. |
| `profile_runtime_baseline.py` | dev-only smoke/performance tool | no | canonical | Runtime profiling and baseline suite helper. |
| `rc1_release_gate.py` | canonical operator script | yes | canonical | Primary release-gate orchestrator for replay/verification/bundle checks. |
| `release_hygiene_check.py` | release helper | yes | canonical | Focused hygiene/governance helper used by release workflows. |
| `replay_official_paths.py` | canonical operator script | yes | canonical | Canonical replay/reproducibility execution and verification path. |
| `review_preflight_stage.py` | canonical operator script | yes | canonical | Preflight E01-E11 review orchestration and review artifact emission. |
| `run_analysis_server_execution_plan.py` | dev/specialized tool | yes | canonical | Execute decision-support campaign plan via analysis-server style orchestration. |
| `run_frozen_campaign.ps1` | platform helper | yes | canonical | PowerShell helper to run frozen campaign commands with explicit parameters. |
| `verify_project.py` | canonical operator script | yes | canonical | Single canonical verification entrypoint with subcommands for current verification families. |
| `verify_campaign_runtime_profile.py` | compatibility wrapper | yes | wrapper | Thin wrapper routing to `verify_project.py campaign-runtime-profile`. |
| `verify_confirmatory_ready.py` | compatibility wrapper | yes | wrapper | Thin wrapper routing to `verify_project.py confirmatory-ready`. |
| `verify_model_cost_policy_precheck.py` | compatibility wrapper | yes | wrapper | Thin wrapper routing to `verify_project.py model-cost-policy-precheck`. |
| `verify_official_artifacts.py` | compatibility wrapper | yes | wrapper | Thin wrapper routing to `verify_project.py official-artifacts`. |
| `verify_publishable_bundle.py` | compatibility wrapper | yes | wrapper | Thin wrapper routing to `verify_project.py publishable-bundle`. |

## Remaining compatibility surface

- `verify_*.py` wrappers:
  supported compatibility wrapper; do not extend.
  Canonical replacement: `scripts/verify_project.py` subcommands.
  Operator-facing: yes (legacy entrypoint compatibility).
  Retirement status: defer until external callers migrate.

- `scripts/_common.py`:
  supported compatibility wrapper; do not extend.
  Canonical replacement: `Thesis_ML.script_support`.
  Operator-facing: no.
  Retirement status: keep during thesis phase for import stability.

- `run_decision_support_experiments.py` (repo root):
  deprecated shim; plan removal after caller migration.
  Canonical replacement: `thesisml-run-decision-support`.
  Operator-facing: yes (legacy callers only).
  Current status: keep for now (still referenced by docs/tests/internal command generation).

- Internal package/config compatibility aliases:
  internal compatibility alias; leave untouched during thesis phase.

-- Merged on 2026-04-02
