"""Backward-compatible facade for decision-support orchestration.

Primary implementation now lives in ``Thesis_ML.orchestration.campaign_runner``.
"""

from __future__ import annotations

from typing import Any

from Thesis_ML.orchestration import campaign_runner as _campaign_runner

# Preserve public and test-facing symbols.
run_experiment = _campaign_runner.run_experiment
_expand_experiment_variants = _campaign_runner._expand_experiment_variants
_expand_template_variants = _campaign_runner._expand_template_variants
_collect_dataset_scope = _campaign_runner._collect_dataset_scope
_select_experiments = _campaign_runner._select_experiments
_status_snapshot = _campaign_runner._status_snapshot


def _sync_runtime_hooks() -> None:
    # Allow monkeypatching decision_support.run_experiment in tests and wrappers.
    _campaign_runner.run_experiment = run_experiment


def run_decision_support_campaign(
    *args: Any,
    runtime_profile_summary: Any | None = None,
    quiet_progress: bool | None = None,
    progress_interval_seconds: float | None = None,
    progress_ui: str | None = None,
    progress_detail: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if runtime_profile_summary is not None:
        kwargs["runtime_profile_summary"] = runtime_profile_summary
    if quiet_progress is not None:
        kwargs["quiet_progress"] = bool(quiet_progress)
    if progress_interval_seconds is not None:
        kwargs["progress_interval_seconds"] = float(progress_interval_seconds)
    if progress_ui is not None:
        kwargs["progress_ui"] = str(progress_ui)
    if progress_detail is not None:
        kwargs["progress_detail"] = str(progress_detail)
    _sync_runtime_hooks()
    return _campaign_runner.run_decision_support_campaign(*args, **kwargs)


def run_workbook_decision_support_campaign(
    *args: Any,
    runtime_profile_summary: Any | None = None,
    quiet_progress: bool | None = None,
    progress_interval_seconds: float | None = None,
    progress_ui: str | None = None,
    progress_detail: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if runtime_profile_summary is not None:
        kwargs["runtime_profile_summary"] = runtime_profile_summary
    if quiet_progress is not None:
        kwargs["quiet_progress"] = bool(quiet_progress)
    if progress_interval_seconds is not None:
        kwargs["progress_interval_seconds"] = float(progress_interval_seconds)
    if progress_ui is not None:
        kwargs["progress_ui"] = str(progress_ui)
    if progress_detail is not None:
        kwargs["progress_detail"] = str(progress_detail)
    _sync_runtime_hooks()
    return _campaign_runner.run_workbook_decision_support_campaign(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    _sync_runtime_hooks()
    return _campaign_runner.main(argv)


__all__ = [
    "run_experiment",
    "_collect_dataset_scope",
    "_expand_template_variants",
    "_expand_experiment_variants",
    "_select_experiments",
    "_status_snapshot",
    "run_decision_support_campaign",
    "run_workbook_decision_support_campaign",
    "main",
]
