from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from Thesis_ML.experiments.errors import exception_failure_payload
from Thesis_ML.experiments.timeout_watchdog import (
    WORKER_EXECUTION_MODE_NATIVE,
    WORKER_EXECUTION_MODE_SUBPROCESS,
    execute_run_with_native_worker,
    execute_run_with_timeout_watchdog,
)

_THREAD_CAP_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


ResourceLaneHint = Literal["cpu", "gpu"]
WorkerExecutionMode = Literal["subprocess_worker", "native_worker"]


@dataclass(frozen=True)
class OfficialRunJob:
    order_index: int
    run_id: str
    run_kwargs: dict[str, Any]
    timeout_policy: dict[str, Any]
    phase_name: str
    run_identity: dict[str, Any]
    resource_lane_hint: ResourceLaneHint = "cpu"
    scheduled_compute_assignment: dict[str, Any] | None = None
    worker_execution_mode: WorkerExecutionMode = WORKER_EXECUTION_MODE_SUBPROCESS
    worker_execution_metadata: dict[str, Any] | None = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def resolve_official_job_worker_execution_mode(
    worker_execution_mode: str | None,
) -> WorkerExecutionMode:
    mode = str(worker_execution_mode or "").strip().lower()
    if mode in {WORKER_EXECUTION_MODE_SUBPROCESS, WORKER_EXECUTION_MODE_NATIVE}:
        return mode  # type: ignore[return-value]
    return WORKER_EXECUTION_MODE_SUBPROCESS


def _run_watchdog_job_process(job: OfficialRunJob) -> dict[str, Any]:
    started_at = _utc_now_iso()
    resolved_worker_mode = resolve_official_job_worker_execution_mode(job.worker_execution_mode)
    watchdog_executor: Callable[..., dict[str, Any]]
    if resolved_worker_mode == WORKER_EXECUTION_MODE_NATIVE:
        watchdog_executor = execute_run_with_native_worker
    else:
        watchdog_executor = execute_run_with_timeout_watchdog
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


def resolve_official_job_resource_lane_hint(
    *,
    hardware_mode: str | None,
    scheduled_compute_assignment: dict[str, Any] | None,
) -> ResourceLaneHint:
    if isinstance(scheduled_compute_assignment, dict):
        lane_raw = str(scheduled_compute_assignment.get("assigned_compute_lane", "")).strip().lower()
        if lane_raw in {"cpu", "gpu"}:
            return lane_raw  # type: ignore[return-value]
    return "gpu" if str(hardware_mode or "").strip().lower() == "gpu_only" else "cpu"


def _resolved_lane_from_job(job: OfficialRunJob) -> ResourceLaneHint:
    lane_raw = str(job.resource_lane_hint).strip().lower()
    if lane_raw in {"cpu", "gpu"}:
        return lane_raw  # type: ignore[return-value]
    return resolve_official_job_resource_lane_hint(
        hardware_mode=str(job.run_kwargs.get("hardware_mode", "")),
        scheduled_compute_assignment=(
            dict(job.scheduled_compute_assignment)
            if isinstance(job.scheduled_compute_assignment, dict)
            else None
        ),
    )


def _with_runtime_admission_metadata(
    *,
    payload: dict[str, Any],
    job: OfficialRunJob,
    admitted_at_utc: str,
    completed_at_utc: str,
    max_parallel_gpu_runs_effective: int,
) -> dict[str, Any]:
    lane_hint = _resolved_lane_from_job(job)
    enriched = dict(payload)
    enriched["resource_lane_hint"] = lane_hint
    enriched["gpu_slot_required"] = bool(lane_hint == "gpu")
    enriched["worker_execution_mode"] = resolve_official_job_worker_execution_mode(
        job.worker_execution_mode
    )
    enriched["admitted_at_utc"] = str(admitted_at_utc)
    enriched["completed_at_utc"] = str(completed_at_utc)
    enriched["max_parallel_gpu_runs_effective"] = int(max_parallel_gpu_runs_effective)
    return enriched


def execute_official_run_jobs(
    *,
    jobs: list[OfficialRunJob],
    max_parallel_runs: int,
    max_parallel_gpu_runs: int = 1,
    watchdog_executor: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if int(max_parallel_runs) <= 0:
        raise ValueError("max_parallel_runs must be >= 1.")
    if int(max_parallel_gpu_runs) < 0:
        raise ValueError("max_parallel_gpu_runs must be >= 0.")
    if int(max_parallel_gpu_runs) > int(max_parallel_runs):
        raise ValueError("max_parallel_gpu_runs cannot exceed max_parallel_runs.")
    if not jobs:
        return []

    resolved_jobs = sorted(
        jobs,
        key=lambda item: (int(item.order_index), str(item.run_id)),
    )
    resolved_parallelism = int(max_parallel_runs)
    resolved_max_parallel_gpu_runs = int(max_parallel_gpu_runs)

    if resolved_parallelism <= 1:
        serial_payloads = []
        for job in resolved_jobs:
            if watchdog_executor is not None:
                payload = _run_watchdog_job_local(
                    job=job,
                    watchdog_executor=watchdog_executor,
                )
            else:
                payload = _run_watchdog_job_process(job)
            admitted_at = str(payload.get("started_at_utc") or _utc_now_iso())
            completed_at = str(payload.get("ended_at_utc") or _utc_now_iso())
            serial_payloads.append(
                _with_runtime_admission_metadata(
                    payload=payload,
                    job=job,
                    admitted_at_utc=admitted_at,
                    completed_at_utc=completed_at,
                    max_parallel_gpu_runs_effective=resolved_max_parallel_gpu_runs,
                )
            )
        return serial_payloads

    max_workers = min(int(resolved_parallelism), int(len(resolved_jobs)))
    pending_jobs = list(resolved_jobs)
    running_futures: dict[Future[dict[str, Any]], tuple[OfficialRunJob, str, ResourceLaneHint]] = {}
    completed_payloads: list[dict[str, Any]] = []
    running_gpu_jobs = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        while pending_jobs or running_futures:
            admitted_any = False
            if pending_jobs and len(running_futures) < resolved_parallelism:
                scan_index = 0
                gpu_waiting_blocked = False
                while scan_index < len(pending_jobs) and len(running_futures) < resolved_parallelism:
                    job = pending_jobs[scan_index]
                    lane_hint = _resolved_lane_from_job(job)
                    gpu_slot_required = lane_hint == "gpu"
                    if gpu_slot_required:
                        if gpu_waiting_blocked:
                            scan_index += 1
                            continue
                        if running_gpu_jobs >= resolved_max_parallel_gpu_runs:
                            gpu_waiting_blocked = True
                            scan_index += 1
                            continue

                    admitted_at = _utc_now_iso()
                    future = pool.submit(_run_watchdog_job_process, job)
                    running_futures[future] = (job, admitted_at, lane_hint)
                    if gpu_slot_required:
                        running_gpu_jobs += 1
                    pending_jobs.pop(scan_index)
                    admitted_any = True

            if running_futures:
                done, _ = wait(tuple(running_futures.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    job, admitted_at, lane_hint = running_futures.pop(future)
                    if lane_hint == "gpu":
                        running_gpu_jobs -= 1
                    completed_at = _utc_now_iso()
                    try:
                        payload = dict(future.result())
                    except Exception as exc:  # pragma: no cover - pool worker failure path
                        payload = {
                            "order_index": int(job.order_index),
                            "run_id": str(job.run_id),
                            "started_at_utc": admitted_at,
                            "ended_at_utc": completed_at,
                            "watchdog_result": None,
                            "execution_error": {
                                "error": str(exc),
                                "failure_payload": exception_failure_payload(
                                    exc,
                                    default_stage="runtime",
                                ),
                            },
                        }
                    completed_payloads.append(
                        _with_runtime_admission_metadata(
                            payload=payload,
                            job=job,
                            admitted_at_utc=admitted_at,
                            completed_at_utc=completed_at,
                            max_parallel_gpu_runs_effective=resolved_max_parallel_gpu_runs,
                        )
                    )
                continue

            if pending_jobs and not admitted_any:
                blocked_job_ids = [str(job.run_id) for job in pending_jobs[:5]]
                raise RuntimeError(
                    "No official jobs could be admitted under the configured runtime budgets. "
                    f"Pending run_ids: {', '.join(blocked_job_ids)}."
                )

    return sorted(
        completed_payloads,
        key=lambda item: (int(item["order_index"]), str(item["run_id"])),
    )


__all__ = [
    "OfficialRunJob",
    "resolve_official_job_resource_lane_hint",
    "resolve_official_job_worker_execution_mode",
    "execute_official_run_jobs",
]
