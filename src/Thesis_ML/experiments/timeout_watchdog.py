from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from Thesis_ML.experiments.errors import exception_failure_payload
from Thesis_ML.experiments.execution_policy import read_run_status, write_run_status
from Thesis_ML.experiments.run_states import (
    RUN_STATUS_FAILED,
    RUN_STATUS_SUCCESS,
    RUN_STATUS_TIMED_OUT,
)
from Thesis_ML.observability.process_sampler import ProcessSampler

WORKER_EXECUTION_MODE_SUBPROCESS = "subprocess_worker"
WORKER_EXECUTION_MODE_NATIVE = "native_worker"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _report_dir_from_run_kwargs(run_kwargs: dict[str, Any]) -> Path:
    reports_root = Path(str(run_kwargs.get("reports_root")))
    run_id = str(run_kwargs.get("run_id"))
    return reports_root / run_id


def _run_status_path(report_dir: Path) -> Path:
    return report_dir / "run_status.json"


def _output_tail(value: str, *, max_chars: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _process_profile_artifacts_payload(
    *,
    report_dir: Path,
) -> dict[str, Any]:
    return {
        "process_samples_path": str((report_dir / "process_samples.jsonl").resolve()),
        "process_profile_summary_path": str(
            (report_dir / "process_profile_summary.json").resolve()
        ),
    }


def _attach_process_profile_to_status_file(
    *,
    report_dir: Path,
    run_id: str,
    fallback_status: str,
    process_profile_summary: dict[str, Any],
    process_profile_artifacts: dict[str, Any],
) -> None:
    status_payload = read_run_status(report_dir)
    if isinstance(status_payload, dict):
        status_payload["updated_at_utc"] = _utc_now()
        status_payload["process_profile_summary"] = dict(process_profile_summary)
        status_payload["process_profile_artifacts"] = dict(process_profile_artifacts)
        status_path = _run_status_path(report_dir)
        status_path.write_text(f"{json.dumps(status_payload, indent=2)}\n", encoding="utf-8")
        return
    write_run_status(
        report_dir,
        run_id=run_id,
        status=str(fallback_status),
        message="process profile attached by timeout watchdog",
        process_profile_summary=process_profile_summary,
        process_profile_artifacts=process_profile_artifacts,
    )


@contextmanager
def _temporary_env_overrides(
    overrides: dict[str, str] | None,
) -> Iterator[None]:
    if not isinstance(overrides, dict) or not overrides:
        yield
        return
    previous: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            previous[key_text] = os.environ.get(key_text)
            os.environ[key_text] = str(value)
        yield
    finally:
        for key_text, prior_value in previous.items():
            if prior_value is None:
                os.environ.pop(key_text, None)
            else:
                os.environ[key_text] = str(prior_value)


def _resolve_timeout_watchdog_settings(timeout_policy: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(timeout_policy.get("enabled", True))
    timeout_seconds_raw = timeout_policy.get("effective_timeout_seconds")
    timeout_seconds = int(timeout_seconds_raw) if timeout_seconds_raw is not None else None
    return {
        "enabled": bool(enabled),
        "timeout_seconds": timeout_seconds,
        "grace_seconds": int(timeout_policy.get("shutdown_grace_seconds", 30)),
        "sample_interval_seconds": float(timeout_policy.get("process_sample_interval_seconds", 10.0)),
        "include_io_counters": bool(timeout_policy.get("process_include_io_counters", True)),
    }


def _default_process_profile_summary(
    *,
    elapsed_seconds: float,
    sample_interval_seconds: float,
    terminated_by_watchdog: bool,
    termination_method: str,
    child_pid: int,
    sampling_error: str,
) -> dict[str, Any]:
    return {
        "sampling_enabled": False,
        "sample_interval_seconds": float(sample_interval_seconds),
        "sample_count": 0,
        "first_sample_at_utc": None,
        "last_sample_at_utc": None,
        "wall_clock_elapsed_seconds": float(elapsed_seconds),
        "peak_rss_mb": 0.0,
        "peak_vms_mb": 0.0,
        "peak_thread_count": 0,
        "mean_cpu_percent": 0.0,
        "peak_cpu_percent": 0.0,
        "peak_child_process_count": 0,
        "peak_read_bytes": 0,
        "peak_write_bytes": 0,
        "sampling_errors": [str(sampling_error)],
        "terminated_by_watchdog": bool(terminated_by_watchdog),
        "termination_method": str(termination_method),
        "child_pid": int(child_pid),
    }


def _finalize_process_profile(
    *,
    report_dir: Path,
    sampler: ProcessSampler | None,
    elapsed_seconds: float,
    sample_interval_seconds: float,
    terminated_by_watchdog: bool,
    termination_method: str,
    child_pid: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    process_profile_summary: dict[str, Any] = {}
    if sampler is not None:
        try:
            sampler.stop()
            process_profile_summary = sampler.finalize(
                wall_clock_elapsed_seconds=float(elapsed_seconds),
                terminated_by_watchdog=bool(terminated_by_watchdog),
                termination_method=str(termination_method),
                child_pid=int(child_pid),
            )
        except Exception as exc:
            process_profile_summary = _default_process_profile_summary(
                elapsed_seconds=float(elapsed_seconds),
                sample_interval_seconds=float(sample_interval_seconds),
                terminated_by_watchdog=bool(terminated_by_watchdog),
                termination_method=str(termination_method),
                child_pid=int(child_pid),
                sampling_error=str(exc),
            )

    if not process_profile_summary:
        process_profile_summary = _default_process_profile_summary(
            elapsed_seconds=float(elapsed_seconds),
            sample_interval_seconds=float(sample_interval_seconds),
            terminated_by_watchdog=bool(terminated_by_watchdog),
            termination_method=str(termination_method),
            child_pid=int(child_pid),
            sampling_error="process_sampler_unavailable",
        )

    samples_path = report_dir / "process_samples.jsonl"
    summary_path = report_dir / "process_profile_summary.json"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.touch(exist_ok=True)
    if not summary_path.exists():
        _write_json(summary_path, process_profile_summary)

    process_profile_artifacts = _process_profile_artifacts_payload(report_dir=report_dir)
    return process_profile_summary, process_profile_artifacts


def execute_supervised_worker_payload(
    *,
    run_kwargs: dict[str, Any],
    run_callable: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(run_kwargs, dict):
        return {
            "ok": False,
            "error": "Invalid worker input payload; expected 'run_kwargs' object.",
            "failure_payload": {
                "error_code": "worker_input_invalid",
                "error_type": "ValueError",
                "failure_stage": "watchdog_worker",
                "error_details": {},
            },
        }
    worker_callable = run_callable
    if worker_callable is None:
        from Thesis_ML.experiments.run_experiment import run_experiment

        worker_callable = run_experiment

    try:
        result = worker_callable(**run_kwargs)
    except Exception as exc:  # pragma: no cover - exercised via watchdog tests
        return {
            "ok": False,
            "error": str(exc),
            "failure_payload": exception_failure_payload(exc, default_stage="runtime"),
            "traceback": traceback.format_exc(),
        }
    return {
        "ok": True,
        "result": result,
    }


def _watchdog_result_payload(
    *,
    status: str,
    run_payload: dict[str, Any] | None,
    report_dir: Path,
    run_status_path: Path,
    error: str | None,
    error_code: str | None,
    error_type: str | None,
    failure_stage: str | None,
    error_details: dict[str, Any],
    timeout_seconds: int | None,
    elapsed_seconds: float,
    timeout_diagnostics_path: str | None,
    child_pid: int,
    termination_method: str,
    process_profile_summary: dict[str, Any],
    process_profile_artifacts: dict[str, Any],
    worker_execution_mode: str,
    command: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": str(status),
        "run_payload": dict(run_payload) if isinstance(run_payload, dict) else None,
        "report_dir": str(report_dir.resolve()),
        "run_status_path": str(run_status_path.resolve()),
        "error": error,
        "error_code": error_code,
        "error_type": error_type,
        "failure_stage": failure_stage,
        "error_details": dict(error_details),
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": float(elapsed_seconds),
        "timeout_diagnostics_path": timeout_diagnostics_path,
        "child_pid": int(child_pid),
        "termination_method": str(termination_method),
        "process_profile_summary": dict(process_profile_summary),
        "process_profile_artifacts": dict(process_profile_artifacts),
        "worker_execution_mode": str(worker_execution_mode),
    }
    if isinstance(command, list):
        payload["command"] = list(command)
    return payload


def _finalize_from_worker_payload(
    *,
    worker_payload: dict[str, Any] | None,
    run_id: str,
    report_dir: Path,
    timeout_seconds: int | None,
    elapsed_seconds: float,
    child_pid: int,
    termination_method: str,
    process_profile_summary: dict[str, Any],
    process_profile_artifacts: dict[str, Any],
    worker_execution_mode: str,
    command: list[str] | None = None,
) -> dict[str, Any]:
    run_status_path = _run_status_path(report_dir)

    if isinstance(worker_payload, dict) and bool(worker_payload.get("ok", False)):
        result_payload = worker_payload.get("result")
        if not isinstance(result_payload, dict):
            return _watchdog_result_payload(
                status=RUN_STATUS_FAILED,
                run_payload=None,
                report_dir=report_dir,
                run_status_path=run_status_path,
                error="watchdog_worker_returned_invalid_result",
                error_code="watchdog_worker_invalid_result",
                error_type="RuntimeError",
                failure_stage="watchdog_worker",
                error_details={},
                timeout_seconds=timeout_seconds,
                elapsed_seconds=elapsed_seconds,
                timeout_diagnostics_path=None,
                child_pid=child_pid,
                termination_method=termination_method,
                process_profile_summary=process_profile_summary,
                process_profile_artifacts=process_profile_artifacts,
                worker_execution_mode=worker_execution_mode,
                command=command,
            )

        _attach_process_profile_to_status_file(
            report_dir=report_dir,
            run_id=run_id,
            fallback_status=RUN_STATUS_SUCCESS,
            process_profile_summary=process_profile_summary,
            process_profile_artifacts=process_profile_artifacts,
        )
        return _watchdog_result_payload(
            status=RUN_STATUS_SUCCESS,
            run_payload=result_payload,
            report_dir=report_dir,
            run_status_path=run_status_path,
            error=None,
            error_code=None,
            error_type=None,
            failure_stage=None,
            error_details={},
            timeout_seconds=timeout_seconds,
            elapsed_seconds=elapsed_seconds,
            timeout_diagnostics_path=None,
            child_pid=child_pid,
            termination_method=termination_method,
            process_profile_summary=process_profile_summary,
            process_profile_artifacts=process_profile_artifacts,
            worker_execution_mode=worker_execution_mode,
            command=command,
        )

    failure_payload_raw = worker_payload.get("failure_payload") if isinstance(worker_payload, dict) else None
    failure_payload: dict[str, Any] = (
        dict(failure_payload_raw) if isinstance(failure_payload_raw, dict) else {}
    )
    error_message = (
        str(worker_payload.get("error"))
        if isinstance(worker_payload, dict) and worker_payload.get("error") is not None
        else "run_worker_failed"
    )
    error_code = str(failure_payload.get("error_code") or "run_worker_failed")
    error_type = str(failure_payload.get("error_type") or "RuntimeError")
    failure_stage = str(failure_payload.get("failure_stage") or "runtime")
    failure_error_details_raw = failure_payload.get("error_details")
    error_details = (
        dict(failure_error_details_raw) if isinstance(failure_error_details_raw, dict) else {}
    )

    if not isinstance(read_run_status(report_dir), dict):
        write_run_status(
            report_dir,
            run_id=run_id,
            status=RUN_STATUS_FAILED,
            error=error_message,
            error_code=error_code,
            error_type=error_type,
            failure_stage=failure_stage,
            error_details=error_details,
            stage_timings_seconds={"wall_clock_elapsed": elapsed_seconds},
            process_profile_summary=process_profile_summary,
            process_profile_artifacts=process_profile_artifacts,
        )
    else:
        _attach_process_profile_to_status_file(
            report_dir=report_dir,
            run_id=run_id,
            fallback_status=RUN_STATUS_FAILED,
            process_profile_summary=process_profile_summary,
            process_profile_artifacts=process_profile_artifacts,
        )

    return _watchdog_result_payload(
        status=RUN_STATUS_FAILED,
        run_payload=None,
        report_dir=report_dir,
        run_status_path=run_status_path,
        error=error_message,
        error_code=error_code,
        error_type=error_type,
        failure_stage=failure_stage,
        error_details=error_details,
        timeout_seconds=timeout_seconds,
        elapsed_seconds=elapsed_seconds,
        timeout_diagnostics_path=None,
        child_pid=child_pid,
        termination_method=termination_method,
        process_profile_summary=process_profile_summary,
        process_profile_artifacts=process_profile_artifacts,
        worker_execution_mode=worker_execution_mode,
        command=command,
    )


def execute_run_with_timeout_watchdog(
    *,
    run_kwargs: dict[str, Any],
    timeout_policy: dict[str, Any],
    phase_name: str,
    run_identity: dict[str, Any],
    subprocess_env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    run_id = str(run_kwargs.get("run_id"))
    report_dir = _report_dir_from_run_kwargs(run_kwargs)
    run_status_path = _run_status_path(report_dir)
    resolved = _resolve_timeout_watchdog_settings(timeout_policy)

    started_utc = _utc_now()
    started = perf_counter()
    timeout_diagnostics_path: Path | None = None

    with tempfile.TemporaryDirectory(prefix="thesisml_watchdog_") as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        input_path = temp_dir / "run_input.json"
        output_path = temp_dir / "run_output.json"
        _write_json(input_path, {"run_kwargs": _json_ready(run_kwargs)})

        command = [
            sys.executable,
            "-m",
            "Thesis_ML.experiments.supervised_worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        process_env = os.environ.copy()
        if isinstance(subprocess_env_overrides, dict):
            for key, value in subprocess_env_overrides.items():
                key_text = str(key).strip()
                if not key_text:
                    continue
                process_env[key_text] = str(value)

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=process_env,
        )
        pid = int(process.pid)
        sampler: ProcessSampler | None = ProcessSampler(
            pid=pid,
            report_dir=report_dir,
            sample_interval_seconds=float(resolved["sample_interval_seconds"]),
            include_io_counters=bool(resolved["include_io_counters"]),
        )
        try:
            sampler.start()
        except Exception:
            sampler = None

        timed_out = False
        termination_method = "normal_exit"
        stdout_text = ""
        stderr_text = ""
        try:
            communicate_timeout = (
                float(resolved["timeout_seconds"])
                if bool(resolved["enabled"]) and resolved["timeout_seconds"] is not None
                else None
            )
            stdout_text, stderr_text = process.communicate(timeout=communicate_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            termination_method = "terminate"
            process.terminate()
            try:
                stdout_text, stderr_text = process.communicate(
                    timeout=float(resolved["grace_seconds"])
                )
            except subprocess.TimeoutExpired:
                termination_method = "terminate_then_kill"
                process.kill()
                stdout_text, stderr_text = process.communicate()

        ended = perf_counter()
        ended_utc = _utc_now()
        elapsed_seconds = float(round(ended - started, 6))
        process_profile_summary, process_profile_artifacts = _finalize_process_profile(
            report_dir=report_dir,
            sampler=sampler,
            elapsed_seconds=elapsed_seconds,
            sample_interval_seconds=float(resolved["sample_interval_seconds"]),
            terminated_by_watchdog=bool(timed_out),
            termination_method=str(termination_method),
            child_pid=pid,
        )

        if timed_out:
            existing_status = read_run_status(report_dir)
            timeout_diagnostics_path = report_dir / "timeout_diagnostics.json"
            timeout_payload = {
                "run_id": run_id,
                "phase": str(phase_name),
                "framework_mode": str(run_kwargs.get("framework_mode")),
                "model": str(run_kwargs.get("model")),
                "suite_id": run_identity.get("suite_id"),
                "variant_id": run_identity.get("variant_id"),
                "subject": run_kwargs.get("subject"),
                "train_subject": run_kwargs.get("train_subject"),
                "test_subject": run_kwargs.get("test_subject"),
                "repeat_id": run_kwargs.get("repeat_id"),
                "repeat_count": run_kwargs.get("repeat_count"),
                "timeout_budget_seconds": resolved["timeout_seconds"],
                "elapsed_seconds": elapsed_seconds,
                "last_known_stage": "run_experiment_subprocess",
                "termination_method": termination_method,
                "grace_period_seconds": resolved["grace_seconds"],
                "child_pid": pid,
                "start_utc": started_utc,
                "end_utc": ended_utc,
                "last_run_status_snapshot": existing_status if isinstance(existing_status, dict) else None,
                "stdout_tail": _output_tail(stdout_text),
                "stderr_tail": _output_tail(stderr_text),
                "timeout_policy_effective": dict(timeout_policy),
                "process_profile_summary": dict(process_profile_summary),
                "process_profile_artifacts": dict(process_profile_artifacts),
                "worker_execution_mode": WORKER_EXECUTION_MODE_SUBPROCESS,
            }
            _write_json(timeout_diagnostics_path, timeout_payload)
            write_run_status(
                report_dir,
                run_id=run_id,
                status=RUN_STATUS_TIMED_OUT,
                message=("Run exceeded wall-clock timeout budget and was terminated by watchdog."),
                error="run_exceeded_timeout_budget",
                error_code="run_timeout",
                error_type="RunTimeoutError",
                failure_stage="watchdog_timeout",
                error_details={
                    "timeout_budget_seconds": resolved["timeout_seconds"],
                    "elapsed_seconds": elapsed_seconds,
                    "termination_method": termination_method,
                    "grace_period_seconds": resolved["grace_seconds"],
                    "child_pid": pid,
                    "timeout_diagnostics_path": str(timeout_diagnostics_path.resolve()),
                },
                stage_timings_seconds={"wall_clock_elapsed": elapsed_seconds},
                process_profile_summary=process_profile_summary,
                process_profile_artifacts=process_profile_artifacts,
            )
            return _watchdog_result_payload(
                status=RUN_STATUS_TIMED_OUT,
                run_payload=None,
                report_dir=report_dir,
                run_status_path=run_status_path,
                error="run_exceeded_timeout_budget",
                error_code="run_timeout",
                error_type="RunTimeoutError",
                failure_stage="watchdog_timeout",
                error_details=dict(timeout_payload),
                timeout_seconds=resolved["timeout_seconds"],
                elapsed_seconds=elapsed_seconds,
                timeout_diagnostics_path=str(timeout_diagnostics_path.resolve()),
                child_pid=pid,
                termination_method=termination_method,
                process_profile_summary=process_profile_summary,
                process_profile_artifacts=process_profile_artifacts,
                worker_execution_mode=WORKER_EXECUTION_MODE_SUBPROCESS,
                command=command,
            )

        worker_payload: dict[str, Any] | None = None
        if output_path.exists():
            try:
                loaded = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                worker_payload = loaded

        if not isinstance(worker_payload, dict) and int(process.returncode) != 0:
            worker_payload = {
                "ok": False,
                "error": f"Run subprocess exited with code {process.returncode}.",
                "failure_payload": {
                    "error_code": "run_subprocess_failed",
                    "error_type": "RuntimeError",
                    "failure_stage": "runtime",
                    "error_details": {},
                },
            }

        return _finalize_from_worker_payload(
            worker_payload=worker_payload,
            run_id=run_id,
            report_dir=report_dir,
            timeout_seconds=resolved["timeout_seconds"],
            elapsed_seconds=elapsed_seconds,
            child_pid=pid,
            termination_method=termination_method,
            process_profile_summary=process_profile_summary,
            process_profile_artifacts=process_profile_artifacts,
            worker_execution_mode=WORKER_EXECUTION_MODE_SUBPROCESS,
            command=command,
        )


def execute_run_with_native_worker(
    *,
    run_kwargs: dict[str, Any],
    timeout_policy: dict[str, Any],
    phase_name: str,
    run_identity: dict[str, Any],
    subprocess_env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    del phase_name, run_identity

    run_id = str(run_kwargs.get("run_id"))
    report_dir = _report_dir_from_run_kwargs(run_kwargs)
    resolved = _resolve_timeout_watchdog_settings(timeout_policy)
    pid = int(os.getpid())

    started = perf_counter()
    sampler: ProcessSampler | None = ProcessSampler(
        pid=pid,
        report_dir=report_dir,
        sample_interval_seconds=float(resolved["sample_interval_seconds"]),
        include_io_counters=bool(resolved["include_io_counters"]),
    )
    worker_payload: dict[str, Any] | None = None
    try:
        sampler.start()
    except Exception:
        sampler = None

    with _temporary_env_overrides(subprocess_env_overrides):
        worker_payload = execute_supervised_worker_payload(run_kwargs=run_kwargs)

    elapsed_seconds = float(round(perf_counter() - started, 6))
    process_profile_summary, process_profile_artifacts = _finalize_process_profile(
        report_dir=report_dir,
        sampler=sampler,
        elapsed_seconds=elapsed_seconds,
        sample_interval_seconds=float(resolved["sample_interval_seconds"]),
        terminated_by_watchdog=False,
        termination_method="native_exit",
        child_pid=pid,
    )
    return _finalize_from_worker_payload(
        worker_payload=worker_payload,
        run_id=run_id,
        report_dir=report_dir,
        timeout_seconds=resolved["timeout_seconds"],
        elapsed_seconds=elapsed_seconds,
        child_pid=pid,
        termination_method="native_exit",
        process_profile_summary=process_profile_summary,
        process_profile_artifacts=process_profile_artifacts,
        worker_execution_mode=WORKER_EXECUTION_MODE_NATIVE,
        command=None,
    )


__all__ = [
    "WORKER_EXECUTION_MODE_NATIVE",
    "WORKER_EXECUTION_MODE_SUBPROCESS",
    "execute_run_with_native_worker",
    "execute_run_with_timeout_watchdog",
    "execute_supervised_worker_payload",
]
