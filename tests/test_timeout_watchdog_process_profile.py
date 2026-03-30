from __future__ import annotations

import json
import subprocess
from pathlib import Path

from Thesis_ML.experiments.run_states import RUN_STATUS_SUCCESS
from Thesis_ML.experiments.timeout_watchdog import execute_run_with_timeout_watchdog


def _base_inputs(
    tmp_path: Path, run_id: str
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    reports_root = tmp_path / "reports"
    run_kwargs = {
        "run_id": run_id,
        "reports_root": str(reports_root),
        "framework_mode": "locked_comparison",
        "model": "ridge",
    }
    timeout_policy = {
        "enabled": True,
        "effective_timeout_seconds": 1,
        "shutdown_grace_seconds": 1,
        "process_sample_interval_seconds": 0.01,
    }
    run_identity = {"run_id": run_id, "variant_id": "v1", "suite_id": "s1"}
    return run_kwargs, timeout_policy, run_identity


def test_watchdog_writes_process_profile_artifacts_on_success(tmp_path: Path, monkeypatch) -> None:
    class _SuccessProcess:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            self.pid = 51001
            self.returncode = 0
            output_path = Path(command[command.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "run_id": "success_run",
                            "report_dir": "/tmp/success",
                            "metrics": {"balanced_accuracy": 0.5},
                        },
                    }
                ),
                encoding="utf-8",
            )

        def communicate(self, timeout=None):  # noqa: ANN001
            return ("stdout", "stderr")

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    monkeypatch.setattr(
        "Thesis_ML.experiments.timeout_watchdog.subprocess.Popen",
        lambda *args, **kwargs: _SuccessProcess(args[0], **kwargs),
    )

    run_kwargs, timeout_policy, run_identity = _base_inputs(tmp_path, "success_run")
    result = execute_run_with_timeout_watchdog(
        run_kwargs=run_kwargs,
        timeout_policy=timeout_policy,
        phase_name="confirmatory",
        run_identity=run_identity,
    )

    report_dir = Path(str(run_kwargs["reports_root"])) / str(run_kwargs["run_id"])
    assert result["status"] == RUN_STATUS_SUCCESS
    assert isinstance(result.get("process_profile_summary"), dict)
    assert isinstance(result.get("process_profile_artifacts"), dict)
    assert (report_dir / "process_samples.jsonl").exists()
    assert (report_dir / "process_profile_summary.json").exists()
    run_status = json.loads((report_dir / "run_status.json").read_text(encoding="utf-8"))
    assert isinstance(run_status.get("process_profile_summary"), dict)
    assert isinstance(run_status.get("process_profile_artifacts"), dict)


def test_watchdog_writes_process_profile_artifacts_on_failure(tmp_path: Path, monkeypatch) -> None:
    class _FailureProcess:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            self.pid = 51002
            self.returncode = 1
            output_path = Path(command[command.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "ok": False,
                        "error": "worker_failed",
                        "failure_payload": {
                            "error_code": "worker_failed",
                            "error_type": "RuntimeError",
                            "failure_stage": "runtime",
                            "error_details": {},
                        },
                    }
                ),
                encoding="utf-8",
            )

        def communicate(self, timeout=None):  # noqa: ANN001
            return ("stdout", "stderr")

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    monkeypatch.setattr(
        "Thesis_ML.experiments.timeout_watchdog.subprocess.Popen",
        lambda *args, **kwargs: _FailureProcess(args[0], **kwargs),
    )

    run_kwargs, timeout_policy, run_identity = _base_inputs(tmp_path, "failure_run")
    result = execute_run_with_timeout_watchdog(
        run_kwargs=run_kwargs,
        timeout_policy=timeout_policy,
        phase_name="confirmatory",
        run_identity=run_identity,
    )
    report_dir = Path(str(run_kwargs["reports_root"])) / str(run_kwargs["run_id"])
    assert result["status"] == "failed"
    assert isinstance(result.get("process_profile_summary"), dict)
    assert (report_dir / "process_samples.jsonl").exists()
    assert (report_dir / "process_profile_summary.json").exists()
    run_status = json.loads((report_dir / "run_status.json").read_text(encoding="utf-8"))
    assert isinstance(run_status.get("process_profile_summary"), dict)
    assert isinstance(run_status.get("process_profile_artifacts"), dict)


def test_watchdog_writes_process_profile_artifacts_on_timeout(tmp_path: Path, monkeypatch) -> None:
    class _TimeoutProcess:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            self.pid = 51003
            self.returncode = 0
            self._calls = 0

        def communicate(self, timeout=None):  # noqa: ANN001
            self._calls += 1
            if self._calls == 1:
                raise subprocess.TimeoutExpired(cmd="python", timeout=timeout)
            return ("worker-stdout", "worker-stderr")

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    monkeypatch.setattr(
        "Thesis_ML.experiments.timeout_watchdog.subprocess.Popen",
        lambda *args, **kwargs: _TimeoutProcess(args[0], **kwargs),
    )

    run_kwargs, timeout_policy, run_identity = _base_inputs(tmp_path, "timeout_run")
    result = execute_run_with_timeout_watchdog(
        run_kwargs=run_kwargs,
        timeout_policy=timeout_policy,
        phase_name="confirmatory",
        run_identity=run_identity,
    )
    report_dir = Path(str(run_kwargs["reports_root"])) / str(run_kwargs["run_id"])
    assert result["status"] == "timed_out"
    assert isinstance(result.get("process_profile_summary"), dict)
    assert (report_dir / "process_samples.jsonl").exists()
    assert (report_dir / "process_profile_summary.json").exists()
    timeout_payload = json.loads(
        (report_dir / "timeout_diagnostics.json").read_text(encoding="utf-8")
    )
    assert isinstance(timeout_payload.get("process_profile_summary"), dict)
    assert isinstance(timeout_payload.get("process_profile_artifacts"), dict)
