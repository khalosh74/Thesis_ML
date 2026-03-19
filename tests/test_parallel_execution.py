from __future__ import annotations

from typing import Any

import pytest

from Thesis_ML.experiments.parallel_execution import OfficialRunJob, execute_official_run_jobs


def _job(*, order_index: int, run_id: str) -> OfficialRunJob:
    return OfficialRunJob(
        order_index=order_index,
        run_id=run_id,
        run_kwargs={
            "run_id": run_id,
            "reports_root": "unused",
            "framework_mode": "locked_comparison",
            "model": "ridge",
        },
        timeout_policy={"enabled": True, "effective_timeout_seconds": 60},
        phase_name="locked_comparison",
        run_identity={"run_id": run_id, "suite_id": None, "variant_id": "ridge"},
    )


def test_execute_official_run_jobs_serial_preserves_order_and_env_caps() -> None:
    seen_env_caps: list[dict[str, str]] = []

    def _watchdog_executor(**kwargs: Any) -> dict[str, Any]:
        env_caps = kwargs.get("subprocess_env_overrides")
        assert isinstance(env_caps, dict)
        seen_env_caps.append(dict(env_caps))
        run_kwargs = kwargs["run_kwargs"]
        run_id = str(run_kwargs["run_id"])
        return {
            "status": "success",
            "run_payload": {
                "report_dir": f"/tmp/{run_id}",
                "config_path": f"/tmp/{run_id}/config.json",
                "metrics_path": f"/tmp/{run_id}/metrics.json",
                "metrics": {"primary_metric_value": 0.5, "n_folds": 1},
            },
        }

    results = execute_official_run_jobs(
        jobs=[
            _job(order_index=2, run_id="run_002"),
            _job(order_index=0, run_id="run_000"),
            _job(order_index=1, run_id="run_001"),
        ],
        max_parallel_runs=1,
        watchdog_executor=_watchdog_executor,
    )

    assert [int(row["order_index"]) for row in results] == [0, 1, 2]
    assert [str(row["run_id"]) for row in results] == ["run_000", "run_001", "run_002"]
    assert seen_env_caps
    for env_caps in seen_env_caps:
        assert env_caps["OMP_NUM_THREADS"] == "1"
        assert env_caps["MKL_NUM_THREADS"] == "1"
        assert env_caps["OPENBLAS_NUM_THREADS"] == "1"
        assert env_caps["NUMEXPR_NUM_THREADS"] == "1"


def test_execute_official_run_jobs_serial_records_execution_error() -> None:
    def _failing_watchdog_executor(**_: Any) -> dict[str, Any]:
        raise RuntimeError("synthetic watchdog failure")

    results = execute_official_run_jobs(
        jobs=[_job(order_index=0, run_id="run_000")],
        max_parallel_runs=1,
        watchdog_executor=_failing_watchdog_executor,
    )

    assert len(results) == 1
    row = results[0]
    assert row["watchdog_result"] is None
    execution_error = row["execution_error"]
    assert isinstance(execution_error, dict)
    assert "synthetic watchdog failure" in str(execution_error.get("error"))
    failure_payload = execution_error.get("failure_payload")
    assert isinstance(failure_payload, dict)
    assert str(failure_payload.get("error_code")) == "unhandled_exception"


def test_execute_official_run_jobs_rejects_invalid_parallelism() -> None:
    with pytest.raises(ValueError, match="max_parallel_runs must be >= 1"):
        execute_official_run_jobs(
            jobs=[_job(order_index=0, run_id="run_000")],
            max_parallel_runs=0,
        )


def test_execute_official_run_jobs_parallel_path_keeps_deterministic_order(monkeypatch) -> None:
    class _FakePool:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> _FakePool:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

        def map(self, fn, jobs):  # noqa: ANN001
            # Simulate out-of-order completion.
            return [fn(job) for job in list(jobs)[::-1]]

    monkeypatch.setattr(
        "Thesis_ML.experiments.parallel_execution.ProcessPoolExecutor",
        _FakePool,
    )

    def _watchdog(**kwargs: Any) -> dict[str, Any]:
        run_kwargs = kwargs["run_kwargs"]
        run_id = str(run_kwargs["run_id"])
        return {
            "status": "success",
            "run_payload": {
                "report_dir": f"/tmp/{run_id}",
                "config_path": f"/tmp/{run_id}/config.json",
                "metrics_path": f"/tmp/{run_id}/metrics.json",
                "metrics": {"primary_metric_value": 0.5, "n_folds": 1},
            },
        }

    monkeypatch.setattr(
        "Thesis_ML.experiments.parallel_execution.execute_run_with_timeout_watchdog",
        _watchdog,
    )

    results = execute_official_run_jobs(
        jobs=[
            _job(order_index=2, run_id="run_002"),
            _job(order_index=0, run_id="run_000"),
            _job(order_index=1, run_id="run_001"),
        ],
        max_parallel_runs=2,
    )

    assert [int(row["order_index"]) for row in results] == [0, 1, 2]
    assert [str(row["run_id"]) for row in results] == ["run_000", "run_001", "run_002"]
