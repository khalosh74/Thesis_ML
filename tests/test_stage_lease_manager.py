from __future__ import annotations

import threading
import time
from pathlib import Path

from Thesis_ML.experiments.stage_lease_manager import StageLeaseManager, StageLeaseRequest


def _request(*, run_id: str, stage_key: str, owner_identity: str) -> StageLeaseRequest:
    return StageLeaseRequest(
        run_id=run_id,
        stage_key=stage_key,
        owner_identity=owner_identity,
        lease_class="gpu",
        lease_reason="test_gpu_stage",
        expected_backend_family="torch_gpu",
        expected_executor_id="model_fit_torch_ridge_gpu_v2",
        expected_compute_lane="gpu",
    )


def test_stage_lease_manager_acquire_release_roundtrip(tmp_path: Path) -> None:
    manager = StageLeaseManager(
        lease_root=tmp_path / "stage_leases",
        max_parallel_gpu_leases=1,
        poll_interval_seconds=0.01,
    )

    handle = manager.acquire(
        _request(
            run_id="run_001",
            stage_key="model_fit",
            owner_identity="run_001:model_fit",
        )
    )
    assert handle.lease_class == "gpu"
    assert handle.wait_seconds >= 0.0

    snapshot_before_release = manager.snapshot()
    assert int(snapshot_before_release["active_gpu_lease_count"]) == 1

    release_result = manager.release(handle)
    assert release_result.released is True

    snapshot_after_release = manager.snapshot()
    assert int(snapshot_after_release["active_gpu_lease_count"]) == 0


def test_stage_lease_manager_respects_gpu_limit_under_contention(tmp_path: Path) -> None:
    manager = StageLeaseManager(
        lease_root=tmp_path / "stage_leases",
        max_parallel_gpu_leases=1,
        poll_interval_seconds=0.01,
    )

    first = manager.acquire(
        _request(
            run_id="run_001",
            stage_key="permutation",
            owner_identity="run_001:permutation",
        )
    )

    started = threading.Event()
    acquired = threading.Event()
    result: dict[str, float] = {}

    def _acquire_second() -> None:
        started.set()
        second = manager.acquire(
            _request(
                run_id="run_002",
                stage_key="permutation",
                owner_identity="run_002:permutation",
            )
        )
        result["wait_seconds"] = float(second.wait_seconds)
        manager.release(second)
        acquired.set()

    thread = threading.Thread(target=_acquire_second, daemon=True)
    thread.start()
    assert started.wait(timeout=1.0)
    time.sleep(0.08)
    assert acquired.is_set() is False

    manager.release(first)
    assert acquired.wait(timeout=2.0)
    thread.join(timeout=2.0)

    assert float(result.get("wait_seconds", 0.0)) >= 0.05


def test_stage_lease_manager_cleanup_run_leases_releases_owned_leases(tmp_path: Path) -> None:
    manager = StageLeaseManager(
        lease_root=tmp_path / "stage_leases",
        max_parallel_gpu_leases=2,
        poll_interval_seconds=0.01,
    )

    handle_a = manager.acquire(
        _request(
            run_id="run_a",
            stage_key="model_fit",
            owner_identity="run_a:model_fit",
        )
    )
    _ = handle_a
    handle_b = manager.acquire(
        _request(
            run_id="run_b",
            stage_key="model_fit",
            owner_identity="run_b:model_fit",
        )
    )

    released_count = manager.cleanup_run_leases(run_id="run_a")
    assert int(released_count) == 1

    snapshot = manager.snapshot()
    assert int(snapshot["active_gpu_lease_count"]) == 1

    manager.release(handle_b)
    assert int(manager.snapshot()["active_gpu_lease_count"]) == 0
