# Scripts Directory Roles

This directory keeps operator-facing entrypoint filenames stable.  
Wave 1 introduces package-level shared helpers under `src/Thesis_ML/script_support/`; top-level scripts remain the runnable entrypoints.

## Role taxonomy

### Supported operator entrypoint
- `review_preflight_stage.py`
- `run_analysis_server_execution_plan.py`

### Verification tool
- `verify_campaign_runtime_profile.py`
- `verify_confirmatory_ready.py`
- `verify_model_cost_policy_precheck.py`
- `verify_official_artifacts.py`
- `verify_official_reproducibility.py`
- `verify_publishable_bundle.py`

### Release gate/orchestrator
- `rc1_release_gate.py`
- `release_hygiene_check.py`
- `replay_official_paths.py`

### Build/freeze utility
- `build_frozen_confirmatory_registry.py`
- `build_publishable_bundle.py`
- `freeze_workbook_registry.py`
- `generate_demo_dataset.py`
- `prepare_dependent_rerun_registry.py`

### Compatibility wrapper
- `_common.py`
- `create_thesis_experiment_workbook.py`
- `review_e01_target_lock.py`
- `run_baseline.py`

### Dev-only smoke/performance tool
- `acceptance_smoke.py`
- `benchmark_torch_ridge_backend.py`
- `performance_smoke.py`
- `profile_runtime_baseline.py`

### Platform helper
- `bootstrap_env.ps1`
- `bootstrap_env.sh`
- `prepare_frozen_campaign.ps1`
- `run_frozen_campaign.ps1`

## Canonical extension guidance

- Extend canonical operator flows in the primary scripts for that role (for example `review_preflight_stage.py`, `build_frozen_confirmatory_registry.py`, and `verify_*.py` tools).
- Put reusable low-level helpers in `src/Thesis_ML/script_support/`, not in wrapper scripts.
- Keep compatibility wrappers thin and stable; do not accumulate new business/scientific logic in wrappers.
- Keep platform helper scripts focused on environment/bootstrap/run wiring only.

