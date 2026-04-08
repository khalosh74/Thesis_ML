from __future__ import annotations

from Thesis_ML.release.models import RunClass, RunStatus

ALLOWED_RUN_CLASSES_FOR_RUNNER = {
    RunClass.SCRATCH,
    RunClass.EXPLORATORY,
    RunClass.CANDIDATE,
}

PROMOTABLE_SOURCE_RUN_CLASS = RunClass.CANDIDATE


def is_runner_allowed_run_class(run_class: RunClass) -> bool:
    return run_class in ALLOWED_RUN_CLASSES_FOR_RUNNER


def is_terminal_status(status: RunStatus) -> bool:
    return status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.PROMOTED}


__all__ = [
    "ALLOWED_RUN_CLASSES_FOR_RUNNER",
    "PROMOTABLE_SOURCE_RUN_CLASS",
    "is_runner_allowed_run_class",
    "is_terminal_status",
]

