from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from Thesis_ML.experiments.execution_policy import read_run_status, write_run_status
from Thesis_ML.experiments.run_states import (
    RUN_STATUS_FAILED,
    RUN_STATUS_SUCCESS,
    RUN_STATUS_TIMED_OUT,
)


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


def _output_tail(value: str, *, max_chars: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


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

    enabled = bool(timeout_policy.get("enabled", True))
    timeout_seconds_raw = timeout_policy.get("effective_timeout_seconds")
    timeout_seconds = int(timeout_seconds_raw) if timeout_seconds_raw is not None else None
    grace_seconds = int(timeout_policy.get("shutdown_grace_seconds", 30))

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

        timed_out = False
        termination_method = "normal_exit"
        stdout_text = ""
        stderr_text = ""
        try:
            communicate_timeout = (
                float(timeout_seconds) if enabled and timeout_seconds is not None else None
            )
            stdout_text, stderr_text = process.communicate(timeout=communicate_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            termination_method = "terminate"
            process.terminate()
            try:
                stdout_text, stderr_text = process.communicate(timeout=float(grace_seconds))
            except subprocess.TimeoutExpired:
                termination_method = "terminate_then_kill"
                process.kill()
                stdout_text, stderr_text = process.communicate()

        ended = perf_counter()
        ended_utc = _utc_now()
        elapsed_seconds = float(round(ended - started, 6))

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
                "timeout_budget_seconds": timeout_seconds,
                "elapsed_seconds": elapsed_seconds,
                "last_known_stage": "run_experiment_subprocess",
                "termination_method": termination_method,
                "grace_period_seconds": grace_seconds,
                "child_pid": pid,
                "start_utc": started_utc,
                "end_utc": ended_utc,
                "last_run_status_snapshot": existing_status
                if isinstance(existing_status, dict)
                else None,
                "stdout_tail": _output_tail(stdout_text),
                "stderr_tail": _output_tail(stderr_text),
                "timeout_policy_effective": dict(timeout_policy),
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
                    "timeout_budget_seconds": timeout_seconds,
                    "elapsed_seconds": elapsed_seconds,
                    "termination_method": termination_method,
                    "grace_period_seconds": grace_seconds,
                    "child_pid": pid,
                    "timeout_diagnostics_path": str(timeout_diagnostics_path.resolve()),
                },
                stage_timings_seconds={"wall_clock_elapsed": elapsed_seconds},
            )
            return {
                "status": RUN_STATUS_TIMED_OUT,
                "run_payload": None,
                "report_dir": str(report_dir.resolve()),
                "error": "run_exceeded_timeout_budget",
                "error_code": "run_timeout",
                "error_type": "RunTimeoutError",
                "failure_stage": "watchdog_timeout",
                "error_details": dict(timeout_payload),
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": elapsed_seconds,
                "timeout_diagnostics_path": str(timeout_diagnostics_path.resolve()),
                "child_pid": pid,
                "termination_method": termination_method,
                "command": command,
            }

        worker_payload: dict[str, Any] | None = None
        if output_path.exists():
            try:
                loaded = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                worker_payload = loaded

        if (
            int(process.returncode) == 0
            and isinstance(worker_payload, dict)
            and bool(worker_payload.get("ok", False))
        ):
            result_payload = worker_payload.get("result")
            if not isinstance(result_payload, dict):
                return {
                    "status": RUN_STATUS_FAILED,
                    "run_payload": None,
                    "report_dir": str(report_dir.resolve()),
                    "error": "watchdog_worker_returned_invalid_result",
                    "error_code": "watchdog_worker_invalid_result",
                    "error_type": "RuntimeError",
                    "failure_stage": "watchdog_worker",
                    "error_details": {},
                    "timeout_seconds": timeout_seconds,
                    "elapsed_seconds": elapsed_seconds,
                    "timeout_diagnostics_path": None,
                    "child_pid": pid,
                    "termination_method": termination_method,
                    "command": command,
                }
            return {
                "status": RUN_STATUS_SUCCESS,
                "run_payload": result_payload,
                "report_dir": str(report_dir.resolve()),
                "error": None,
                "error_code": None,
                "error_type": None,
                "failure_stage": None,
                "error_details": {},
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": elapsed_seconds,
                "timeout_diagnostics_path": None,
                "child_pid": pid,
                "termination_method": termination_method,
                "command": command,
            }

        failure_payload_raw = (
            worker_payload.get("failure_payload") if isinstance(worker_payload, dict) else None
        )
        failure_payload: dict[str, Any] = (
            dict(failure_payload_raw) if isinstance(failure_payload_raw, dict) else {}
        )
        error_message = (
            str(worker_payload.get("error"))
            if isinstance(worker_payload, dict) and worker_payload.get("error") is not None
            else f"Run subprocess exited with code {process.returncode}."
        )
        error_code = str(failure_payload.get("error_code") or "run_subprocess_failed")
        error_type = str(failure_payload.get("error_type") or "RuntimeError")
        failure_stage = str(failure_payload.get("failure_stage") or "runtime")
        failure_error_details_raw = failure_payload.get("error_details")
        error_details = (
            dict(failure_error_details_raw) if isinstance(failure_error_details_raw, dict) else {}
        )
        if not read_run_status(report_dir):
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
            )
        return {
            "status": RUN_STATUS_FAILED,
            "run_payload": None,
            "report_dir": str(report_dir.resolve()),
            "error": error_message,
            "error_code": error_code,
            "error_type": error_type,
            "failure_stage": failure_stage,
            "error_details": error_details,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": elapsed_seconds,
            "timeout_diagnostics_path": None,
            "child_pid": pid,
            "termination_method": termination_method,
            "command": command,
        }


__all__ = ["execute_run_with_timeout_watchdog"]
