# Script Inventory

Wave 1 baseline inventory of current `scripts/` artifacts and their intended role.

| Script | Current role | Operator-facing | Canonical or wrapper | Responsibility note |
|---|---|---|---|---|
| `_common.py` | compatibility wrapper | no | wrapper | Backward-compatible helper import surface; real implementations live in `Thesis_ML.script_support`. |
| `acceptance_smoke.py` | dev-only smoke/performance tool | no | canonical | Local acceptance smoke checks for workbook compile + command readiness. |
| `benchmark_torch_ridge_backend.py` | dev-only smoke/performance tool | no | canonical | Backend micro-benchmark utility for torch ridge timing checks. |
| `bootstrap_env.ps1` | platform helper | yes | canonical | Windows bootstrap for dev env sync and quick validation commands. |
| `bootstrap_env.sh` | platform helper | yes | canonical | POSIX bootstrap for dev env sync and quick validation commands. |
| `build_frozen_confirmatory_registry.py` | build/freeze utility | yes | canonical | Build frozen confirmatory registry/manifest/report from reviewed preflight bundle + scope. |
| `build_publishable_bundle.py` | build/freeze utility | yes | canonical | Assemble publishable release bundle and manifest. |
| `create_thesis_experiment_workbook.py` | compatibility wrapper | yes | wrapper | Deprecated compatibility wrapper that delegates to `thesisml-workbook`. |
| `freeze_workbook_registry.py` | build/freeze utility | yes | canonical | Compile workbook to decision-support execution registry JSON. |
| `generate_demo_dataset.py` | build/freeze utility | yes | canonical | Generate deterministic synthetic demo dataset + index + manifest. |
| `performance_smoke.py` | dev-only smoke/performance tool | no | canonical | Local performance smoke benchmark over workbook/campaign path. |
| `prepare_dependent_rerun_registry.py` | build/freeze utility | yes | canonical | Build temporary E07/E08 rerun registry when non-ridge stage-3 winner is selected. |
| `prepare_frozen_campaign.ps1` | platform helper | yes | canonical | PowerShell helper for frozen campaign preparation/archive chores. |
| `profile_runtime_baseline.py` | dev-only smoke/performance tool | no | canonical | Runtime profiling and baseline suite helper. |
| `rc1_release_gate.py` | release gate/orchestrator | yes | canonical | Release-candidate gate orchestration script. |
| `release_hygiene_check.py` | release gate/orchestrator | yes | canonical | Repo hygiene/governance checks for release readiness. |
| `replay_official_paths.py` | release gate/orchestrator | yes | canonical | Replay official comparison/protocol paths and collect reproducibility evidence. |
| `review_e01_target_lock.py` | compatibility wrapper | yes | wrapper | Thin compatibility wrapper that delegates to generic preflight review for E01. |
| `review_preflight_stage.py` | supported operator entrypoint | yes | canonical | Preflight E01-E11 review orchestration and review artifact emission. |
| `run_analysis_server_execution_plan.py` | supported operator entrypoint | yes | canonical | Execute decision-support campaign plan via analysis-server style orchestration. |
| `run_baseline.py` | compatibility wrapper | yes | wrapper | Deprecated compatibility wrapper that delegates to `thesisml-run-baseline`. |
| `run_frozen_campaign.ps1` | platform helper | yes | canonical | PowerShell helper to run frozen campaign commands with explicit parameters. |
| `verify_campaign_runtime_profile.py` | verification tool | yes | canonical | Runtime-profile verification precheck for official campaign plans. |
| `verify_confirmatory_ready.py` | verification tool | yes | canonical | Confirmatory-ready governance verification for output directories. |
| `verify_model_cost_policy_precheck.py` | verification tool | yes | canonical | Model-cost policy precheck across official specs before execution. |
| `verify_official_artifacts.py` | verification tool | yes | canonical | Official artifact contract/invariant verification tool. |
| `verify_official_reproducibility.py` | verification tool | yes | canonical | Deterministic reproducibility verification by rerun and artifact comparison. |
| `verify_publishable_bundle.py` | verification tool | yes | canonical | Validate publishable bundle contents, hashes, and prerequisite verification summaries. |

