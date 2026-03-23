from __future__ import annotations

import pytest

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy, resolve_compute_policy
from Thesis_ML.experiments.stage_execution import StageKey
from Thesis_ML.experiments.stage_planner import plan_stage_execution
from Thesis_ML.experiments.stage_registry import (
    MODEL_FIT_CPU_EXECUTOR_ID,
    MODEL_FIT_TORCH_LOGREG_EXECUTOR_ID,
    MODEL_FIT_TORCH_RIDGE_EXECUTOR_ID,
    PERMUTATION_REFERENCE_EXECUTOR_ID,
    PERMUTATION_RIDGE_GPU_PREFERRED_EXECUTOR_ID,
    TUNING_GENERIC_EXECUTOR_ID,
)


def _torch_compute_policy() -> ResolvedComputePolicy:
    return ResolvedComputePolicy(
        hardware_mode_requested="gpu_only",
        hardware_mode_effective="gpu_only",
        requested_backend_family="torch_gpu",
        effective_backend_family="torch_gpu",
        gpu_device_id=0,
        gpu_device_name="synthetic_gpu",
        gpu_device_total_memory_mb=4096,
        deterministic_compute=False,
        allow_backend_fallback=False,
        backend_stack_id="torch_gpu_reference_v1",
        backend_fallback_used=False,
        backend_fallback_reason=None,
    )


def _max_both_cpu_lane_policy() -> ResolvedComputePolicy:
    return ResolvedComputePolicy(
        hardware_mode_requested="max_both",
        hardware_mode_effective="max_both",
        requested_backend_family="auto_mixed",
        effective_backend_family="sklearn_cpu",
        gpu_device_id=None,
        gpu_device_name=None,
        gpu_device_total_memory_mb=None,
        deterministic_compute=False,
        allow_backend_fallback=False,
        backend_stack_id="sklearn_cpu_reference_v1",
        backend_fallback_used=False,
        backend_fallback_reason=None,
        assigned_compute_lane="cpu",
        assigned_backend_family="sklearn_cpu",
        lane_assignment_reason="max_both_gpu_lane_budget_exhausted",
        scheduler_mode_effective="max_both",
    )


def _assignment_map(result) -> dict[str, dict[str, object]]:
    return {
        str(assignment.stage): assignment.model_dump(mode="json") for assignment in result.assignments
    }


def test_stage_planner_cpu_only_ridge_uses_cpu_fallback_for_gpu_preferred_stages() -> None:
    policy = resolve_compute_policy(
        framework_mode=FrameworkMode.EXPLORATORY,
        hardware_mode="cpu_only",
    )
    result = plan_stage_execution(
        framework_mode=FrameworkMode.EXPLORATORY,
        compute_policy=policy,
        model_name="ridge",
        methodology_policy_name="grouped_nested_tuning",
        tuning_enabled=True,
        n_permutations=32,
    )
    assignments = _assignment_map(result)

    assert assignments[StageKey.MODEL_FIT.value]["executor_id"] == MODEL_FIT_CPU_EXECUTOR_ID
    assert assignments[StageKey.MODEL_FIT.value]["fallback_used"] is True
    assert isinstance(assignments[StageKey.MODEL_FIT.value]["fallback_reason"], str)

    assert (
        assignments[StageKey.PERMUTATION.value]["executor_id"]
        == PERMUTATION_REFERENCE_EXECUTOR_ID
    )
    assert assignments[StageKey.PERMUTATION.value]["fallback_used"] is True
    assert assignments[StageKey.TUNING.value]["executor_id"] == TUNING_GENERIC_EXECUTOR_ID


def test_stage_planner_gpu_only_ridge_prefers_torch_gpu_executors() -> None:
    result = plan_stage_execution(
        framework_mode=FrameworkMode.EXPLORATORY,
        compute_policy=_torch_compute_policy(),
        model_name="ridge",
        methodology_policy_name="grouped_nested_tuning",
        tuning_enabled=True,
        n_permutations=16,
    )
    assignments = _assignment_map(result)

    assert (
        assignments[StageKey.MODEL_FIT.value]["executor_id"]
        == MODEL_FIT_TORCH_RIDGE_EXECUTOR_ID
    )
    assert assignments[StageKey.MODEL_FIT.value]["fallback_used"] is False
    assert (
        assignments[StageKey.PERMUTATION.value]["executor_id"]
        == PERMUTATION_RIDGE_GPU_PREFERRED_EXECUTOR_ID
    )
    assert assignments[StageKey.PERMUTATION.value]["fallback_used"] is False


def test_stage_planner_gpu_only_logreg_falls_from_cpu_preference_to_torch() -> None:
    result = plan_stage_execution(
        framework_mode=FrameworkMode.EXPLORATORY,
        compute_policy=_torch_compute_policy(),
        model_name="logreg",
        methodology_policy_name="grouped_nested_tuning",
        tuning_enabled=True,
        n_permutations=0,
    )
    assignments = _assignment_map(result)

    assert (
        assignments[StageKey.MODEL_FIT.value]["executor_id"]
        == MODEL_FIT_TORCH_LOGREG_EXECUTOR_ID
    )
    assert assignments[StageKey.MODEL_FIT.value]["fallback_used"] is True
    assert "policy_backend_family_excludes_sklearn_cpu" in str(
        assignments[StageKey.MODEL_FIT.value]["fallback_reason"]
    )


def test_stage_planner_max_both_cpu_lane_keeps_conservative_cpu_behavior() -> None:
    result = plan_stage_execution(
        framework_mode=FrameworkMode.EXPLORATORY,
        compute_policy=_max_both_cpu_lane_policy(),
        model_name="ridge",
        methodology_policy_name="grouped_nested_tuning",
        tuning_enabled=True,
        n_permutations=8,
    )
    assignments = _assignment_map(result)

    assert assignments[StageKey.MODEL_FIT.value]["executor_id"] == MODEL_FIT_CPU_EXECUTOR_ID
    assert assignments[StageKey.MODEL_FIT.value]["compute_lane"] == "cpu"
    assert assignments[StageKey.PERMUTATION.value]["executor_id"] == PERMUTATION_REFERENCE_EXECUTOR_ID


def test_stage_planner_enforces_official_admissibility_for_executor_selection() -> None:
    with pytest.raises(ValueError, match="executor_not_admitted_for_official_path"):
        plan_stage_execution(
            framework_mode=FrameworkMode.CONFIRMATORY,
            compute_policy=_torch_compute_policy(),
            model_name="logreg",
            methodology_policy_name="grouped_nested_tuning",
            tuning_enabled=True,
            n_permutations=8,
        )
