from __future__ import annotations

import json
import subprocess
from pathlib import Path

from Thesis_ML.experiments.execution_policy import read_run_status
from Thesis_ML.experiments.runtime_policies import resolve_run_timeout_policy
from Thesis_ML.experiments.timeout_watchdog import (
    execute_run_with_native_worker,
    execute_run_with_timeout_watchdog,
)


def test_timeout_policy_defaults_and_model_override() -> None:
    confirmatory = resolve_run_timeout_policy(
        framework_mode="confirmatory",
        model_name="ridge",
        policy_overrides=None,
    )
    assert confirmatory["enabled"] is True
    assert confirmatory["effective_timeout_seconds"] == 45 * 60
    assert confirmatory["shutdown_grace_seconds"] == 30
    assert confirmatory["absolute_hard_ceiling_seconds"] == 180 * 60

    logreg = resolve_run_timeout_policy(
        framework_mode="locked_comparison",
        model_name="logreg",
        policy_overrides=None,
    )
    assert logreg["effective_timeout_seconds"] == 120 * 60
    assert logreg["effective_timeout_source"] == "model_override"


def test_timeout_policy_applies_absolute_hard_ceiling() -> None:
    resolved = resolve_run_timeout_policy(
        framework_mode="locked_comparison",
        model_name="ridge",
        policy_overrides={
            "default_timeout_seconds": 999999,
            "mode_timeouts_seconds": {"locked_comparison": 999999},
        },
    )
    assert resolved["effective_timeout_seconds"] == 180 * 60


def test_watchdog_marks_run_as_timed_out_and_writes_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _TimeoutProcess:
        def __init__(self) -> None:
            self.pid = 4242
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
            self.returncode = -9

    monkeypatch.setattr(
        "Thesis_ML.experiments.timeout_watchdog.subprocess.Popen",
        lambda *args, **kwargs: _TimeoutProcess(),
    )

    reports_root = tmp_path / "reports"
    run_id = "timeout_run_001"
    run_kwargs = {
        "run_id": run_id,
        "reports_root": str(reports_root),
        "framework_mode": "locked_comparison",
        "model": "logreg",
        "repeat_id": 1,
    }
    timeout_policy = {
        "enabled": True,
        "effective_timeout_seconds": 1,
        "shutdown_grace_seconds": 2,
    }
    run_identity = {"run_id": run_id, "variant_id": "logreg", "suite_id": None}

    result = execute_run_with_timeout_watchdog(
        run_kwargs=run_kwargs,
        timeout_policy=timeout_policy,
        phase_name="locked_comparison",
        run_identity=run_identity,
    )

    assert result["status"] == "timed_out"
    assert result["termination_method"] == "terminate"
    report_dir = reports_root / run_id
    run_status = read_run_status(report_dir)
    assert run_status is not None
    assert run_status["status"] == "timed_out"
    assert run_status["error_code"] == "run_timeout"

    timeout_diag_path = report_dir / "timeout_diagnostics.json"
    assert timeout_diag_path.exists()
    timeout_payload = json.loads(timeout_diag_path.read_text(encoding="utf-8"))
    assert timeout_payload["run_id"] == run_id
    assert timeout_payload["timeout_budget_seconds"] == 1
    assert timeout_payload["termination_method"] == "terminate"
    assert timeout_payload["child_pid"] == 4242


def test_native_worker_success_returns_watchdog_compatible_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_id = "native_success_001"
    reports_root = tmp_path / "reports"

    def _stub_worker_payload(*, run_kwargs: dict[str, object], run_callable=None) -> dict[str, object]:
        del run_callable
        kwargs = dict(run_kwargs)
        run_id_local = str(kwargs.get("run_id"))
        report_dir = Path(str(kwargs["reports_root"])) / run_id_local
        report_dir.mkdir(parents=True, exist_ok=True)
        return {
            "ok": True,
            "result": {
                "run_id": run_id_local,
                "report_dir": str(report_dir.resolve()),
                "config_path": str((report_dir / "config.json").resolve()),
                "metrics_path": str((report_dir / "metrics.json").resolve()),
                "metrics": {"balanced_accuracy": 0.5},
            },
        }

    monkeypatch.setattr(
        "Thesis_ML.experiments.timeout_watchdog.execute_supervised_worker_payload",
        _stub_worker_payload,
    )

    result = execute_run_with_native_worker(
        run_kwargs={
            "run_id": run_id,
            "reports_root": str(reports_root),
            "framework_mode": "exploratory",
            "model": "ridge",
        },
        timeout_policy={
            "enabled": True,
            "effective_timeout_seconds": 60,
            "process_sample_interval_seconds": 0.01,
        },
        phase_name="campaign",
        run_identity={"run_id": run_id},
        subprocess_env_overrides={"OMP_NUM_THREADS": "1"},
    )

    assert result["status"] == "success"
    assert result["worker_execution_mode"] == "native_worker"
    assert isinstance(result.get("run_payload"), dict)
    assert isinstance(result.get("process_profile_summary"), dict)
    assert isinstance(result.get("process_profile_artifacts"), dict)
    assert result.get("run_status_path")
    run_status_path = Path(str(result["run_status_path"]))
    assert run_status_path.exists()
    run_status_payload = json.loads(run_status_path.read_text(encoding="utf-8"))
    assert isinstance(run_status_payload.get("process_profile_summary"), dict)
    assert isinstance(run_status_payload.get("process_profile_artifacts"), dict)


def test_native_worker_failure_returns_watchdog_compatible_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_id = "native_failure_001"
    reports_root = tmp_path / "reports"

    def _failing_worker_payload(*, run_kwargs: dict[str, object], run_callable=None) -> dict[str, object]:
        del run_callable
        del run_kwargs
        return {
            "ok": False,
            "error": "synthetic native failure",
            "failure_payload": {
                "error_code": "unhandled_exception",
                "error_type": "RuntimeError",
                "failure_stage": "runtime",
                "error_details": {},
            },
        }

    monkeypatch.setattr(
        "Thesis_ML.experiments.timeout_watchdog.execute_supervised_worker_payload",
        _failing_worker_payload,
    )

    result = execute_run_with_native_worker(
        run_kwargs={
            "run_id": run_id,
            "reports_root": str(reports_root),
            "framework_mode": "exploratory",
            "model": "ridge",
        },
        timeout_policy={
            "enabled": True,
            "effective_timeout_seconds": 60,
            "process_sample_interval_seconds": 0.01,
        },
        phase_name="campaign",
        run_identity={"run_id": run_id},
        subprocess_env_overrides={"OMP_NUM_THREADS": "1"},
    )

    assert result["status"] == "failed"
    assert result["worker_execution_mode"] == "native_worker"
    assert "synthetic native failure" in str(result.get("error"))
    assert isinstance(result.get("process_profile_summary"), dict)
    assert isinstance(result.get("process_profile_artifacts"), dict)
    assert result.get("run_status_path")
    run_status_path = Path(str(result["run_status_path"]))
    assert run_status_path.exists()
    run_status_payload = json.loads(run_status_path.read_text(encoding="utf-8"))
    assert run_status_payload.get("status") == "failed"
    assert isinstance(run_status_payload.get("process_profile_summary"), dict)
