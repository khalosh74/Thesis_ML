from __future__ import annotations

import pytest

from Thesis_ML.experiments.compute_policy import (
    CPU_REFERENCE_BACKEND_STACK_ID,
    ResolvedComputePolicy,
)
from Thesis_ML.experiments.compute_scheduler import (
    ComputeRunAssignment,
    ComputeRunRequest,
    materialize_scheduled_compute_policy,
    plan_compute_schedule,
)


def _max_both_policy(
    *,
    gpu_available: bool,
    allow_backend_fallback: bool,
) -> ResolvedComputePolicy:
    return ResolvedComputePolicy(
        hardware_mode_requested="max_both",
        hardware_mode_effective="max_both" if gpu_available else "cpu_only",
        requested_backend_family="auto_mixed",
        effective_backend_family="sklearn_cpu",
        gpu_device_id=0 if gpu_available else None,
        gpu_device_name="GPU 0" if gpu_available else None,
        gpu_device_total_memory_mb=12288 if gpu_available else None,
        deterministic_compute=False,
        allow_backend_fallback=bool(allow_backend_fallback),
        backend_stack_id=CPU_REFERENCE_BACKEND_STACK_ID,
        backend_fallback_used=False,
        backend_fallback_reason=None,
    )


def _gpu_only_policy(*, allow_backend_fallback: bool) -> ResolvedComputePolicy:
    return ResolvedComputePolicy(
        hardware_mode_requested="gpu_only",
        hardware_mode_effective="gpu_only",
        requested_backend_family="torch_gpu",
        effective_backend_family="torch_gpu",
        gpu_device_id=0,
        gpu_device_name="GPU 0",
        gpu_device_total_memory_mb=12288,
        deterministic_compute=True,
        allow_backend_fallback=bool(allow_backend_fallback),
        backend_stack_id="torch_2.4.1__cuda_12.1",
        backend_fallback_used=False,
        backend_fallback_reason=None,
    )


def test_max_both_assigns_gpu_only_to_supported_models_when_capable() -> None:
    policy = _max_both_policy(gpu_available=True, allow_backend_fallback=False)
    schedule = plan_compute_schedule(
        run_requests=[
            ComputeRunRequest(order_index=0, run_id="run_0", model_name="ridge"),
            ComputeRunRequest(order_index=1, run_id="run_1", model_name="logreg"),
            ComputeRunRequest(order_index=2, run_id="run_2", model_name="linearsvc"),
            ComputeRunRequest(order_index=3, run_id="run_3", model_name="dummy"),
        ],
        base_compute_policy=policy,
        max_parallel_runs=4,
        max_parallel_gpu_runs=2,
    )

    assert [row.run_id for row in schedule] == ["run_0", "run_1", "run_2", "run_3"]
    assert schedule[0].assigned_compute_lane == "gpu"
    assert schedule[0].assigned_backend_family == "torch_gpu"
    assert schedule[1].assigned_compute_lane == "gpu"
    assert schedule[1].assigned_backend_family == "torch_gpu"
    assert schedule[2].assigned_compute_lane == "cpu"
    assert schedule[2].assigned_backend_family == "sklearn_cpu"
    assert schedule[2].lane_assignment_reason == "max_both_model_cpu_only"
    assert schedule[3].assigned_compute_lane == "cpu"
    assert schedule[3].assigned_backend_family == "sklearn_cpu"
    assert schedule[3].lane_assignment_reason == "max_both_model_cpu_only"


def test_scheduler_respects_max_parallel_gpu_runs_per_window() -> None:
    policy = _max_both_policy(gpu_available=True, allow_backend_fallback=False)
    schedule = plan_compute_schedule(
        run_requests=[
            ComputeRunRequest(order_index=0, run_id="run_0", model_name="ridge"),
            ComputeRunRequest(order_index=1, run_id="run_1", model_name="logreg"),
            ComputeRunRequest(order_index=2, run_id="run_2", model_name="ridge"),
        ],
        base_compute_policy=policy,
        max_parallel_runs=2,
        max_parallel_gpu_runs=1,
    )

    assert [row.run_id for row in schedule] == ["run_0", "run_1", "run_2"]
    assert schedule[0].assigned_compute_lane == "gpu"
    assert schedule[1].assigned_compute_lane == "cpu"
    assert schedule[1].lane_assignment_reason == "max_both_gpu_lane_budget_exhausted"
    assert schedule[2].assigned_compute_lane == "gpu"
    assert sum(1 for row in schedule if row.assigned_compute_lane == "gpu") == 2


def test_max_both_gpu_allowlist_restricts_gpu_lanes() -> None:
    policy = _max_both_policy(gpu_available=True, allow_backend_fallback=False)
    schedule = plan_compute_schedule(
        run_requests=[
            ComputeRunRequest(order_index=0, run_id="run_0", model_name="ridge"),
            ComputeRunRequest(order_index=1, run_id="run_1", model_name="logreg"),
        ],
        base_compute_policy=policy,
        max_parallel_runs=2,
        max_parallel_gpu_runs=1,
        gpu_model_allowlist={"ridge"},
    )

    assert schedule[0].assigned_compute_lane == "gpu"
    assert schedule[0].assigned_backend_family == "torch_gpu"
    assert schedule[1].assigned_compute_lane == "cpu"
    assert schedule[1].assigned_backend_family == "sklearn_cpu"
    assert schedule[1].lane_assignment_reason == "max_both_gpu_disallowed_by_policy"


def test_scheduler_outputs_are_deterministically_sorted_by_order_index() -> None:
    policy = _max_both_policy(gpu_available=True, allow_backend_fallback=False)
    schedule = plan_compute_schedule(
        run_requests=[
            ComputeRunRequest(order_index=4, run_id="run_4", model_name="dummy"),
            ComputeRunRequest(order_index=1, run_id="run_1", model_name="ridge"),
            ComputeRunRequest(order_index=3, run_id="run_3", model_name="linearsvc"),
            ComputeRunRequest(order_index=2, run_id="run_2", model_name="logreg"),
        ],
        base_compute_policy=policy,
        max_parallel_runs=3,
        max_parallel_gpu_runs=1,
    )
    assert [row.order_index for row in schedule] == [1, 2, 3, 4]
    assert [row.run_id for row in schedule] == ["run_1", "run_2", "run_3", "run_4"]


def test_max_both_unavailable_gpu_fails_without_fallback() -> None:
    policy = _max_both_policy(gpu_available=False, allow_backend_fallback=False)
    with pytest.raises(ValueError, match="max_both scheduling requires visible GPU capability"):
        plan_compute_schedule(
            run_requests=[ComputeRunRequest(order_index=0, run_id="run_0", model_name="ridge")],
            base_compute_policy=policy,
            max_parallel_runs=1,
            max_parallel_gpu_runs=1,
        )


def test_max_both_unavailable_gpu_routes_to_cpu_when_fallback_allowed() -> None:
    policy = _max_both_policy(gpu_available=False, allow_backend_fallback=True)
    schedule = plan_compute_schedule(
        run_requests=[
            ComputeRunRequest(order_index=0, run_id="run_0", model_name="ridge"),
            ComputeRunRequest(order_index=1, run_id="run_1", model_name="logreg"),
        ],
        base_compute_policy=policy,
        max_parallel_runs=2,
        max_parallel_gpu_runs=1,
    )
    assert all(row.assigned_compute_lane == "cpu" for row in schedule)
    assert all(row.backend_fallback_used is True for row in schedule)
    assert all(row.lane_assignment_reason == "max_both_fallback_cpu_lane" for row in schedule)


def test_gpu_only_fails_for_unsupported_model_without_fallback() -> None:
    policy = _gpu_only_policy(allow_backend_fallback=False)
    with pytest.raises(ValueError, match="gpu_only scheduling requires GPU backend support"):
        plan_compute_schedule(
            run_requests=[ComputeRunRequest(order_index=0, run_id="run_0", model_name="linearsvc")],
            base_compute_policy=policy,
            max_parallel_runs=1,
            max_parallel_gpu_runs=1,
        )


def test_gpu_only_unsupported_model_uses_explicit_cpu_fallback_when_allowed() -> None:
    policy = _gpu_only_policy(allow_backend_fallback=True)
    schedule = plan_compute_schedule(
        run_requests=[ComputeRunRequest(order_index=0, run_id="run_0", model_name="dummy")],
        base_compute_policy=policy,
        max_parallel_runs=1,
        max_parallel_gpu_runs=1,
    )
    assert schedule[0].assigned_compute_lane == "cpu"
    assert schedule[0].assigned_backend_family == "sklearn_cpu"
    assert schedule[0].backend_fallback_used is True
    assert "gpu_backend_unsupported_for_model:dummy" == schedule[0].backend_fallback_reason


def test_materialize_scheduled_compute_policy_stamps_lane_metadata() -> None:
    base_policy = _max_both_policy(gpu_available=True, allow_backend_fallback=False)
    assignment = ComputeRunAssignment(
        order_index=0,
        run_id="run_0",
        model_name="ridge",
        assigned_compute_lane="gpu",
        assigned_backend_family="torch_gpu",
        lane_assignment_reason="max_both_gpu_eligible_assigned_gpu",
        scheduler_mode_effective="max_both",
        gpu_device_id=0,
        backend_fallback_used=False,
        backend_fallback_reason=None,
    )
    resolved = materialize_scheduled_compute_policy(
        base_compute_policy=base_policy,
        assignment=assignment,
    )
    assert resolved.assigned_compute_lane == "gpu"
    assert resolved.assigned_backend_family == "torch_gpu"
    assert resolved.lane_assignment_reason == "max_both_gpu_eligible_assigned_gpu"
    assert resolved.scheduler_mode_effective == "max_both"
    assert resolved.effective_backend_family == "torch_gpu"
    assert resolved.gpu_device_id == 0
    assert resolved.backend_fallback_used is False
