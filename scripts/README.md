# Scripts Directory Roles

This directory keeps top-level entrypoint filenames stable while concentrating real logic in a smaller canonical set.

## Canonical operator scripts (extend these)
- `review_preflight_stage.py`
- `build_frozen_confirmatory_registry.py`
- `replay_official_paths.py`
- `build_publishable_bundle.py`
- `rc1_release_gate.py`
- `freeze_workbook_registry.py`
- `prepare_dependent_rerun_registry.py`
- `verify_project.py`

## Dev/specialized tools (not main operator path)
- `acceptance_smoke.py`
- `performance_smoke.py`
- `profile_runtime_baseline.py`
- `benchmark_torch_ridge_backend.py`
- `generate_demo_dataset.py`
- `run_analysis_server_execution_plan.py`

## Platform helpers
- `bootstrap_env.ps1` (`-InstallGpuTorch [-TorchIndexUrl ...]` for explicit CUDA-enabled torch provisioning)
- `bootstrap_env.sh` (`--install-gpu-torch [--torch-index-url ...]` for explicit CUDA-enabled torch provisioning)
- `prepare_frozen_campaign.ps1`
- `run_frozen_campaign.ps1`

## Compatibility wrappers (do not add new logic)
- `_common.py`
- `verify_official_artifacts.py` (routes to `verify_project.py official-artifacts`)
- `verify_confirmatory_ready.py` (routes to `verify_project.py confirmatory-ready`)
- `verify_model_cost_policy_precheck.py` (routes to `verify_project.py model-cost-policy-precheck`)
- `verify_publishable_bundle.py` (routes to `verify_project.py publishable-bundle`)
- `verify_campaign_runtime_profile.py` (routes to `verify_project.py campaign-runtime-profile`)

## Remaining compatibility surface
- `verify_*.py` thin wrappers:
  supported compatibility wrapper; do not extend.
  Canonical replacement is `verify_project.py` subcommands.
  Operator-facing: yes (legacy entrypoint stability only).
  Retirement status: keep during thesis phase, retire after caller migration.
- `scripts/_common.py`:
  supported compatibility wrapper; do not extend.
  Canonical replacement is `Thesis_ML.script_support.*`.
  Operator-facing: no.
  Retirement status: keep during thesis phase for import compatibility.
- `run_decision_support_experiments.py` (repo root, external legacy shim):
  deprecated shim; plan removal after caller migration.
  Canonical replacement is `thesisml-run-decision-support`.
  Operator-facing: yes (legacy callers only).
  Retirement status: keep for now because docs/tests/internal command generation still reference it.
- Internal package/config compatibility aliases:
  internal compatibility alias; leave untouched during thesis phase.
  Canonical replacement: not applicable in this pass.

## Release helper
- `release_hygiene_check.py` remains a focused helper; `rc1_release_gate.py` is the canonical release orchestrator.

## Guidance
- Add or change operator behavior in canonical scripts only.
- Keep wrappers thin and import-forwarding only.
- Put shared helper code in `src/Thesis_ML/script_support/`.
