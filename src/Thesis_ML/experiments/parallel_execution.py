from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from Thesis_ML.experiments.errors import exception_failure_payload
from Thesis_ML.experiments.timeout_watchdog import execute_run_with_timeout_watchdog

_THREAD_CAP_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class OfficialRunJob:
    order_index: int
    run_id: str
    run_kwargs: dict[str, Any]
    timeout_policy: dict[str, Any]
    phase_name: str
    run_identity: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _run_watchdog_job_process(job: OfficialRunJob) -> dict[str, Any]:
    started_at = _utc_now_iso()
    try:
        watchdog_result = execute_run_with_timeout_watchdog(
            run_kwargs=dict(job.run_kwargs),
            timeout_policy=dict(job.timeout_policy),
            phase_name=str(job.phase_name),
            run_identity=dict(job.run_identity),
            subprocess_env_overrides=dict(_THREAD_CAP_ENV),
        )
        return {
            "order_index": int(job.order_index),
            "run_id": str(job.run_id),
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now_iso(),
            "watchdog_result": watchdog_result,
            "execution_error": None,
        }
    except Exception as exc:  # pragma: no cover - subprocess failure path
        return {
            "order_index": int(job.order_index),
            "run_id": str(job.run_id),
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now_iso(),
            "watchdog_result": None,
            "execution_error": {
                "error": str(exc),
                "failure_payload": exception_failure_payload(exc, default_stage="runtime"),
            },
        }


def _run_watchdog_job_local(
    *,
    job: OfficialRunJob,
    watchdog_executor: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    started_at = _utc_now_iso()
    try:
        watchdog_result = watchdog_executor(
            run_kwargs=dict(job.run_kwargs),
            timeout_policy=dict(job.timeout_policy),
            phase_name=str(job.phase_name),
            run_identity=dict(job.run_identity),
            subprocess_env_overrides=dict(_THREAD_CAP_ENV),
        )
        return {
            "order_index": int(job.order_index),
            "run_id": str(job.run_id),
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now_iso(),
            "watchdog_result": watchdog_result,
            "execution_error": None,
        }
    except Exception as exc:
        return {
            "order_index": int(job.order_index),
            "run_id": str(job.run_id),
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now_iso(),
            "watchdog_result": None,
            "execution_error": {
                "error": str(exc),
                "failure_payload": exception_failure_payload(exc, default_stage="runtime"),
            },
        }


def execute_official_run_jobs(
    *,
    jobs: list[OfficialRunJob],
    max_parallel_runs: int,
    watchdog_executor: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if int(max_parallel_runs) <= 0:
        raise ValueError("max_parallel_runs must be >= 1.")
    if not jobs:
        return []

    resolved_jobs = sorted(
        jobs,
        key=lambda item: (int(item.order_index), str(item.run_id)),
    )
    resolved_parallelism = int(max_parallel_runs)

    if resolved_parallelism <= 1:
        local_executor = (
            watchdog_executor
            if watchdog_executor is not None
            else execute_run_with_timeout_watchdog
        )
        return [
            _run_watchdog_job_local(
                job=job,
                watchdog_executor=local_executor,
            )
            for job in resolved_jobs
        ]

    max_workers = min(int(resolved_parallelism), int(len(resolved_jobs)))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        payloads = list(pool.map(_run_watchdog_job_process, resolved_jobs))
    return sorted(
        payloads,
        key=lambda item: (int(item["order_index"]), str(item["run_id"])),
    )


__all__ = [
    "OfficialRunJob",
    "execute_official_run_jobs",
]
