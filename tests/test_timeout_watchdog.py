from __future__ import annotations

import json
import subprocess
from pathlib import Path

from Thesis_ML.experiments.execution_policy import read_run_status
from Thesis_ML.experiments.runtime_policies import resolve_run_timeout_policy
from Thesis_ML.experiments.timeout_watchdog import execute_run_with_timeout_watchdog


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
