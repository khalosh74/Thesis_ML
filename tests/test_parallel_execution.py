from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED
from typing import Any

import pytest

from Thesis_ML.experiments.parallel_execution import (
    OfficialRunJob,
    execute_official_run_jobs,
)


def _job(
    *,
    order_index: int,
    run_id: str,
    resource_lane_hint: str = "cpu",
    scheduled_compute_assignment: dict[str, Any] | None = None,
    worker_execution_mode: str = "subprocess_worker",
) -> OfficialRunJob:
    return OfficialRunJob(
        order_index=order_index,
        run_id=run_id,
        run_kwargs={
            "run_id": run_id,
            "reports_root": "unused",
            "framework_mode": "locked_comparison",
            "model": "ridge",
            "hardware_mode": "gpu_only" if resource_lane_hint == "gpu" else "cpu_only",
        },
        timeout_policy={"enabled": True, "effective_timeout_seconds": 60},
        phase_name="locked_comparison",
        run_identity={"run_id": run_id, "suite_id": None, "variant_id": "ridge"},
        resource_lane_hint=str(resource_lane_hint),  # type: ignore[arg-type]
        scheduled_compute_assignment=(
            dict(scheduled_compute_assignment)
            if isinstance(scheduled_compute_assignment, dict)
            else None
        ),
        worker_execution_mode=str(worker_execution_mode),  # type: ignore[arg-type]
    )


def _install_fake_parallel_runtime(
    *,
    monkeypatch: pytest.MonkeyPatch,
    completion_order: list[str],
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "admissions": [],
        "running_total": 0,
        "running_gpu": 0,
        "max_running_total": 0,
        "max_running_gpu": 0,
        "native_calls": [],
        "subprocess_calls": [],
    }
    futures_by_run_id: dict[str, Any] = {}
    lane_by_future: dict[Any, str] = {}
    completed_futures: set[Any] = set()

    class _FakeFuture:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = dict(payload)

        def result(self) -> dict[str, Any]:
            return dict(self._payload)

    class _FakePool:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = int(max_workers)

        def __enter__(self) -> _FakePool:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

        def submit(self, fn, job):  # noqa: ANN001
            payload = fn(job)
            future = _FakeFuture(payload)
            run_id = str(job.run_id)
            lane = str(job.resource_lane_hint)
            futures_by_run_id[run_id] = future
            lane_by_future[future] = lane
            state["admissions"].append(run_id)
            state["running_total"] = int(state["running_total"]) + 1
            if lane == "gpu":
                state["running_gpu"] = int(state["running_gpu"]) + 1
            state["max_running_total"] = max(
                int(state["max_running_total"]),
                int(state["running_total"]),
            )
            state["max_running_gpu"] = max(
                int(state["max_running_gpu"]),
                int(state["running_gpu"]),
            )
            return future

    def _fake_wait(futures, return_when):  # noqa: ANN001
        assert return_when == FIRST_COMPLETED
        chosen = None
        for run_id in completion_order:
            candidate = futures_by_run_id.get(str(run_id))
            if candidate in futures and candidate not in completed_futures:
                chosen = candidate
                break
        if chosen is None:
            for candidate in futures:
                if candidate not in completed_futures:
                    chosen = candidate
                    break
        assert chosen is not None
        completed_futures.add(chosen)
        lane = str(lane_by_future.get(chosen, "cpu"))
        state["running_total"] = int(state["running_total"]) - 1
        if lane == "gpu":
            state["running_gpu"] = int(state["running_gpu"]) - 1
        remaining = {item for item in futures if item is not chosen}
        return ({chosen}, remaining)

    def _watchdog_executor(**kwargs: Any) -> dict[str, Any]:
        run_kwargs = kwargs["run_kwargs"]
        run_id = str(run_kwargs["run_id"])
        state["subprocess_calls"].append(run_id)
        return {
            "status": "success",
            "run_payload": {
                "report_dir": f"/tmp/{run_id}",
                "config_path": f"/tmp/{run_id}/config.json",
                "metrics_path": f"/tmp/{run_id}/metrics.json",
                "metrics": {"primary_metric_value": 0.5, "n_folds": 1},
            },
        }

    def _native_executor(**kwargs: Any) -> dict[str, Any]:
        run_kwargs = kwargs["run_kwargs"]
        run_id = str(run_kwargs["run_id"])
        state["native_calls"].append(run_id)
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
        "Thesis_ML.experiments.parallel_execution.ProcessPoolExecutor",
        _FakePool,
    )
    monkeypatch.setattr(
        "Thesis_ML.experiments.parallel_execution.wait",
        _fake_wait,
    )
    monkeypatch.setattr(
        "Thesis_ML.experiments.parallel_execution.execute_run_with_timeout_watchdog",
        _watchdog_executor,
    )
    monkeypatch.setattr(
        "Thesis_ML.experiments.parallel_execution.execute_run_with_native_worker",
        _native_executor,
    )
    return state


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


def test_execute_official_run_jobs_enforces_gpu_admission_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_parallel_runtime(
        monkeypatch=monkeypatch,
        completion_order=["cpu_000", "cpu_001", "gpu_000", "gpu_001", "gpu_002"],
    )

    results = execute_official_run_jobs(
        jobs=[
            _job(order_index=0, run_id="gpu_000", resource_lane_hint="gpu"),
            _job(order_index=1, run_id="gpu_001", resource_lane_hint="gpu"),
            _job(order_index=2, run_id="cpu_000", resource_lane_hint="cpu"),
            _job(order_index=3, run_id="cpu_001", resource_lane_hint="cpu"),
            _job(order_index=4, run_id="gpu_002", resource_lane_hint="gpu"),
        ],
        max_parallel_runs=3,
        max_parallel_gpu_runs=1,
    )

    assert int(state["max_running_gpu"]) <= 1
    assert [int(row["max_parallel_gpu_runs_effective"]) for row in results] == [1, 1, 1, 1, 1]


def test_execute_official_run_jobs_routes_native_worker_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_parallel_runtime(
        monkeypatch=monkeypatch,
        completion_order=["run_native_001", "run_native_000"],
    )

    results = execute_official_run_jobs(
        jobs=[
            _job(
                order_index=0,
                run_id="run_native_000",
                worker_execution_mode="native_worker",
            ),
            _job(
                order_index=1,
                run_id="run_native_001",
                worker_execution_mode="native_worker",
            ),
        ],
        max_parallel_runs=2,
        max_parallel_gpu_runs=0,
    )

    assert state["native_calls"] == ["run_native_000", "run_native_001"]
    assert state["subprocess_calls"] == []
    assert all(str(row["worker_execution_mode"]) == "native_worker" for row in results)


def test_execute_official_run_jobs_routes_subprocess_worker_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_parallel_runtime(
        monkeypatch=monkeypatch,
        completion_order=["run_subprocess_000", "run_subprocess_001"],
    )

    results = execute_official_run_jobs(
        jobs=[
            _job(
                order_index=0,
                run_id="run_subprocess_000",
                worker_execution_mode="subprocess_worker",
            ),
            _job(
                order_index=1,
                run_id="run_subprocess_001",
                worker_execution_mode="subprocess_worker",
            ),
        ],
        max_parallel_runs=2,
        max_parallel_gpu_runs=0,
    )

    assert state["subprocess_calls"] == ["run_subprocess_000", "run_subprocess_001"]
    assert state["native_calls"] == []
    assert all(str(row["worker_execution_mode"]) == "subprocess_worker" for row in results)


def test_execute_official_run_jobs_allows_cpu_backfill_while_gpu_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_parallel_runtime(
        monkeypatch=monkeypatch,
        completion_order=["cpu_000", "gpu_000", "gpu_001"],
    )

    results = execute_official_run_jobs(
        jobs=[
            _job(order_index=0, run_id="gpu_000", resource_lane_hint="gpu"),
            _job(order_index=1, run_id="gpu_001", resource_lane_hint="gpu"),
            _job(order_index=2, run_id="cpu_000", resource_lane_hint="cpu"),
        ],
        max_parallel_runs=2,
        max_parallel_gpu_runs=1,
    )

    admissions = list(state["admissions"])
    assert admissions[:2] == ["gpu_000", "cpu_000"]
    assert admissions.index("cpu_000") < admissions.index("gpu_001")
    assert [str(row["run_id"]) for row in results] == ["gpu_000", "gpu_001", "cpu_000"]


def test_execute_official_run_jobs_parallel_path_keeps_deterministic_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_parallel_runtime(
        monkeypatch=monkeypatch,
        completion_order=["run_002", "run_001", "run_000"],
    )

    results = execute_official_run_jobs(
        jobs=[
            _job(order_index=2, run_id="run_002"),
            _job(order_index=0, run_id="run_000"),
            _job(order_index=1, run_id="run_001"),
        ],
        max_parallel_runs=2,
        max_parallel_gpu_runs=0,
    )

    assert [int(row["order_index"]) for row in results] == [0, 1, 2]
    assert [str(row["run_id"]) for row in results] == ["run_000", "run_001", "run_002"]
