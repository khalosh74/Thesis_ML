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
