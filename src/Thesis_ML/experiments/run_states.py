from __future__ import annotations

from typing import Final, Literal

RUN_STATUS_PENDING: Final[Literal["pending"]] = "pending"
RUN_STATUS_PLANNED: Final[Literal["planned"]] = "planned"
RUN_STATUS_RUNNING: Final[Literal["running"]] = "running"
RUN_STATUS_SUCCESS: Final[Literal["success"]] = "success"
RUN_STATUS_FAILED: Final[Literal["failed"]] = "failed"
RUN_STATUS_TIMED_OUT: Final[Literal["timed_out"]] = "timed_out"
RUN_STATUS_SKIPPED_DUE_TO_POLICY: Final[Literal["skipped_due_to_policy"]] = "skipped_due_to_policy"

# Backward-compatibility alias for legacy artifacts/tests.
RUN_STATUS_COMPLETED_LEGACY: Final[Literal["completed"]] = "completed"

SUCCESS_RUN_STATUS_ALIASES: Final[frozenset[str]] = frozenset(
    {RUN_STATUS_SUCCESS, RUN_STATUS_COMPLETED_LEGACY}
)

TERMINAL_RUN_STATUSES: Final[frozenset[str]] = frozenset(
    {
        RUN_STATUS_SUCCESS,
        RUN_STATUS_FAILED,
        RUN_STATUS_TIMED_OUT,
        RUN_STATUS_SKIPPED_DUE_TO_POLICY,
    }
)

RUN_RESULT_STATUS_ORDER: Final[tuple[str, ...]] = (
    RUN_STATUS_PLANNED,
    RUN_STATUS_SUCCESS,
    RUN_STATUS_FAILED,
    RUN_STATUS_TIMED_OUT,
    RUN_STATUS_SKIPPED_DUE_TO_POLICY,
)


def normalize_run_status(status: str | None) -> str:
    raw = str(status or "").strip().lower()
    if raw == RUN_STATUS_COMPLETED_LEGACY:
        return RUN_STATUS_SUCCESS
    return raw


def is_run_success_status(status: str | None) -> bool:
    return str(status or "").strip().lower() in SUCCESS_RUN_STATUS_ALIASES


def is_terminal_run_status(status: str | None) -> bool:
    return normalize_run_status(status) in TERMINAL_RUN_STATUSES


def initialized_run_status_counts() -> dict[str, int]:
    return {status: 0 for status in RUN_RESULT_STATUS_ORDER}


def increment_run_status_count(counts: dict[str, int], status: str | None) -> None:
    normalized = normalize_run_status(status)
    if normalized in counts:
        counts[normalized] = int(counts.get(normalized, 0)) + 1
        return
    if normalized:
        counts[normalized] = int(counts.get(normalized, 0)) + 1


__all__ = [
    "RUN_RESULT_STATUS_ORDER",
    "RUN_STATUS_COMPLETED_LEGACY",
    "RUN_STATUS_FAILED",
    "RUN_STATUS_PENDING",
    "RUN_STATUS_PLANNED",
    "RUN_STATUS_RUNNING",
    "RUN_STATUS_SKIPPED_DUE_TO_POLICY",
    "RUN_STATUS_SUCCESS",
    "RUN_STATUS_TIMED_OUT",
    "TERMINAL_RUN_STATUSES",
    "increment_run_status_count",
    "initialized_run_status_counts",
    "is_run_success_status",
    "is_terminal_run_status",
    "normalize_run_status",
]
