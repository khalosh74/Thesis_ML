from __future__ import annotations

from dataclasses import dataclass

from Thesis_ML.config.framework_mode import FrameworkMode, coerce_framework_mode
from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy
from Thesis_ML.experiments.model_registry import get_model_spec
from Thesis_ML.experiments.stage_execution import (
    StageAssignment,
    StageBackendFamily,
    StageKey,
)
from Thesis_ML.experiments.stage_registry import (
    DATASET_SELECTION_CPU_EXECUTOR_ID,
    EVALUATION_CPU_EXECUTOR_ID,
    FEATURE_CACHE_BUILD_CPU_EXECUTOR_ID,
    FEATURE_MATRIX_LOAD_CPU_EXECUTOR_ID,
    MODEL_FIT_CPU_EXECUTOR_ID,
    MODEL_FIT_TORCH_LOGREG_EXECUTOR_ID,
    MODEL_FIT_TORCH_RIDGE_EXECUTOR_ID,
    MODEL_FIT_XGBOOST_CPU_EXECUTOR_ID,
    MODEL_FIT_XGBOOST_GPU_EXECUTOR_ID,
    PERMUTATION_REFERENCE_EXECUTOR_ID,
    PERMUTATION_RIDGE_GPU_PREFERRED_EXECUTOR_ID,
    PREPROCESS_CPU_EXECUTOR_ID,
    REPORTING_CPU_EXECUTOR_ID,
    SPATIAL_VALIDATION_CPU_EXECUTOR_ID,
    SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID,
    SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID,
    TUNING_GENERIC_EXECUTOR_ID,
    TUNING_SKIPPED_CONTROL_EXECUTOR_ID,
    StageExecutorSelectionContext,
    get_stage_executor,
    stage_executor_support_status,
)

_MODEL_FIT_EXECUTOR_BY_ROUTE: dict[str, str] = {
    "cpu_reference": MODEL_FIT_CPU_EXECUTOR_ID,
    "torch_ridge": MODEL_FIT_TORCH_RIDGE_EXECUTOR_ID,
    "torch_logreg": MODEL_FIT_TORCH_LOGREG_EXECUTOR_ID,
    "xgboost_cpu": MODEL_FIT_XGBOOST_CPU_EXECUTOR_ID,
    "xgboost_gpu": MODEL_FIT_XGBOOST_GPU_EXECUTOR_ID,
}
_TUNING_EXECUTOR_BY_ROUTE: dict[str, str] = {
    "generic": TUNING_GENERIC_EXECUTOR_ID,
    "control_skip": TUNING_SKIPPED_CONTROL_EXECUTOR_ID,
    "linearsvc_specialized": SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID,
    "logreg_specialized": SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID,
}
_PERMUTATION_EXECUTOR_BY_ROUTE: dict[str, str] = {
    "reference": PERMUTATION_REFERENCE_EXECUTOR_ID,
    "ridge_gpu_preferred": PERMUTATION_RIDGE_GPU_PREFERRED_EXECUTOR_ID,
}


@dataclass(frozen=True)
class StagePlanningResult:
    assignments: tuple[StageAssignment, ...]
    runtime_fallback_executor_ids: dict[str, str]


@dataclass(frozen=True)
class _ExecutorResolution:
    assignment: StageAssignment
    runtime_fallback_executor_id: str | None


def _resolve_effective_backend_family(compute_policy: ResolvedComputePolicy) -> str:
    assigned = compute_policy.assigned_backend_family
    if isinstance(assigned, str) and assigned.strip():
        return str(assigned).strip()
    return str(compute_policy.effective_backend_family).strip()


def _compute_lane_for_backend(
    *,
    backend_family: StageBackendFamily,
    compute_policy: ResolvedComputePolicy,
) -> str | None:
    if backend_family in {StageBackendFamily.SKLEARN_CPU, StageBackendFamily.XGBOOST_CPU}:
        return "cpu"
    if backend_family in {StageBackendFamily.TORCH_GPU, StageBackendFamily.XGBOOST_GPU}:
        return "gpu"
    lane = compute_policy.assigned_compute_lane
    if isinstance(lane, str) and lane.strip() in {"cpu", "gpu"}:
        return str(lane).strip()
    return "gpu" if _resolve_effective_backend_family(compute_policy) == "torch_gpu" else "cpu"


_STAGE_GPU_MEMORY_SOFT_LIMITS_MB: dict[StageKey, int] = {
    StageKey.PERMUTATION: 2048,
}


def _resource_admissible(
    *,
    stage_key: StageKey,
    backend_family: StageBackendFamily,
    compute_policy: ResolvedComputePolicy,
) -> tuple[bool, str | None]:
    if backend_family != StageBackendFamily.TORCH_GPU:
        return True, None

    assigned_lane = (
        str(compute_policy.assigned_compute_lane).strip().lower()
        if isinstance(compute_policy.assigned_compute_lane, str)
        else None
    )
    scheduler_mode = (
        str(compute_policy.scheduler_mode_effective).strip().lower()
        if isinstance(compute_policy.scheduler_mode_effective, str)
        else None
    )
    if assigned_lane == "cpu":
        reason = "assigned_cpu_lane_blocks_gpu_executor"
        if scheduler_mode == "max_both":
            reason = "max_both_cpu_lane_budget_exhausted_for_stage"
        return False, reason

    stage_soft_limit = _STAGE_GPU_MEMORY_SOFT_LIMITS_MB.get(stage_key)
    gpu_memory_mb = (
        int(compute_policy.gpu_device_total_memory_mb)
        if compute_policy.gpu_device_total_memory_mb is not None
        else None
    )
    if (
        stage_soft_limit is not None
        and gpu_memory_mb is not None
        and gpu_memory_mb < stage_soft_limit
    ):
        return (
            False,
            (f"gpu_memory_below_stage_soft_limit:{gpu_memory_mb}mb<{int(stage_soft_limit)}mb"),
        )
    return True, None


def _backend_family_admissible(
    *,
    stage_key: StageKey,
    backend_family: StageBackendFamily,
    compute_policy: ResolvedComputePolicy,
) -> tuple[bool, str | None]:
    effective_backend_family = _resolve_effective_backend_family(compute_policy)
    if backend_family == StageBackendFamily.AUTO_MIXED:
        return True, None
    if backend_family == StageBackendFamily.SKLEARN_CPU:
        if stage_key in {
            StageKey.DATASET_SELECTION,
            StageKey.FEATURE_CACHE_BUILD,
            StageKey.FEATURE_MATRIX_LOAD,
            StageKey.SPATIAL_VALIDATION,
            StageKey.PREPROCESS,
            StageKey.TUNING,
            StageKey.EVALUATION,
            StageKey.REPORTING,
        }:
            return True, None
        if effective_backend_family == StageBackendFamily.SKLEARN_CPU.value:
            return True, None
        return False, "policy_backend_family_excludes_sklearn_cpu"
    if backend_family == StageBackendFamily.TORCH_GPU:
        if effective_backend_family == StageBackendFamily.TORCH_GPU.value:
            return True, None
        return False, "policy_backend_family_excludes_torch_gpu"
    if backend_family == StageBackendFamily.XGBOOST_CPU:
        if effective_backend_family == StageBackendFamily.SKLEARN_CPU.value:
            return True, None
        if bool(compute_policy.allow_backend_fallback):
            return True, None
        return False, "policy_backend_family_excludes_xgboost_cpu"
    if backend_family == StageBackendFamily.XGBOOST_GPU:
        if effective_backend_family == StageBackendFamily.TORCH_GPU.value:
            return True, None
        return False, "policy_backend_family_excludes_xgboost_gpu"
    return False, "unknown_backend_family"


def _resolve_executor_for_stage(
    *,
    stage_key: StageKey,
    model_name: str | None,
    framework_mode: FrameworkMode,
    compute_policy: ResolvedComputePolicy,
    ordered_executor_ids: tuple[str, ...],
    assignment_reason: str,
) -> _ExecutorResolution:
    if not ordered_executor_ids:
        raise ValueError(f"Stage planner received no executors for stage '{stage_key.value}'.")

    support_context = StageExecutorSelectionContext(
        stage=stage_key,
        model_name=(str(model_name).strip().lower() if model_name is not None else None),
        framework_mode=framework_mode,
        compute_policy=compute_policy,
    )

    rejection_reasons: list[str] = []
    selected_spec = None
    selected_index = 0
    for index, executor_id in enumerate(ordered_executor_ids):
        spec = get_stage_executor(executor_id)
        if spec.stage_key != stage_key:
            rejection_reasons.append(f"{executor_id}:wrong_stage")
            continue
        backend_ok, backend_reason = _backend_family_admissible(
            stage_key=stage_key,
            backend_family=spec.backend_family,
            compute_policy=compute_policy,
        )
        if not backend_ok:
            rejection_reasons.append(f"{executor_id}:{backend_reason}")
            continue
        resource_ok, resource_reason = _resource_admissible(
            stage_key=stage_key,
            backend_family=spec.backend_family,
            compute_policy=compute_policy,
        )
        if not resource_ok:
            rejection_reasons.append(f"{executor_id}:{resource_reason}")
            continue
        supported, support_reason = stage_executor_support_status(spec, support_context)
        if not supported:
            rejection_reasons.append(
                f"{executor_id}:{support_reason or 'unsupported_for_stage_context'}"
            )
            continue
        selected_spec = spec
        selected_index = int(index)
        break

    if selected_spec is None:
        reasons = "; ".join(rejection_reasons) if rejection_reasons else "no_admissible_executor"
        raise ValueError(
            f"No admissible stage executor for stage='{stage_key.value}', model='{model_name}'. "
            f"reasons={reasons}"
        )

    fallback_used = bool(selected_index > 0)
    fallback_reason = "; ".join(rejection_reasons) if fallback_used and rejection_reasons else None
    runtime_fallback_executor_id = (
        str(ordered_executor_ids[1])
        if stage_key == StageKey.TUNING and selected_index == 0 and len(ordered_executor_ids) > 1
        else None
    )
    return _ExecutorResolution(
        assignment=StageAssignment(
            stage=stage_key,
            backend_family=selected_spec.backend_family,
            compute_lane=_compute_lane_for_backend(
                backend_family=selected_spec.backend_family,
                compute_policy=compute_policy,
            ),
            source="stage_planner_v1",
            reason=str(assignment_reason),
            executor_id=str(selected_spec.executor_id),
            equivalence_class=selected_spec.equivalence_class,
            official_admitted=bool(selected_spec.official_admitted),
            fallback_used=bool(fallback_used),
            fallback_reason=(str(fallback_reason) if fallback_reason else None),
        ),
        runtime_fallback_executor_id=runtime_fallback_executor_id,
    )


def _executor_candidates_from_route_tokens(
    *,
    route_tokens: tuple[str, ...],
    route_map: dict[str, str],
    fallback_executor_id: str,
) -> tuple[str, ...]:
    executor_ids: list[str] = []
    for route_token in route_tokens:
        executor_id = route_map.get(str(route_token))
        if executor_id is None:
            continue
        if executor_id in executor_ids:
            continue
        executor_ids.append(executor_id)
    if not executor_ids:
        executor_ids.append(fallback_executor_id)
    return tuple(executor_ids)


def _model_fit_executor_candidates(model_name: str) -> tuple[str, ...]:
    spec = get_model_spec(model_name)
    return _executor_candidates_from_route_tokens(
        route_tokens=tuple(spec.model_fit_route),
        route_map=_MODEL_FIT_EXECUTOR_BY_ROUTE,
        fallback_executor_id=MODEL_FIT_CPU_EXECUTOR_ID,
    )


def _tuning_executor_candidates(model_name: str) -> tuple[str, ...]:
    spec = get_model_spec(model_name)
    return _executor_candidates_from_route_tokens(
        route_tokens=tuple(spec.tuning_route),
        route_map=_TUNING_EXECUTOR_BY_ROUTE,
        fallback_executor_id=TUNING_GENERIC_EXECUTOR_ID,
    )


def _permutation_executor_candidates(model_name: str) -> tuple[str, ...]:
    spec = get_model_spec(model_name)
    return _executor_candidates_from_route_tokens(
        route_tokens=tuple(spec.permutation_route),
        route_map=_PERMUTATION_EXECUTOR_BY_ROUTE,
        fallback_executor_id=PERMUTATION_REFERENCE_EXECUTOR_ID,
    )


def _plan_stage_assignments_map(
    *,
    model_name: str,
    methodology_policy_name: str,
    tuning_enabled: bool,
    n_permutations: int,
) -> dict[StageKey, tuple[str, tuple[str, ...]]]:
    normalized_model_name = str(model_name).strip().lower()
    normalized_methodology = str(methodology_policy_name).strip()
    preprocess_reason = "phase2_policy_preprocess_cpu_preferred"
    if normalized_model_name == "xgboost":
        preprocess_reason = "phase2_policy_preprocess_cpu_passthrough"
    tuning_reason_suffix = (
        "enabled"
        if bool(tuning_enabled) and normalized_methodology == "grouped_nested_tuning"
        else "disabled_or_not_applicable"
    )
    permutation_reason_suffix = (
        "enabled" if int(n_permutations) > 0 else "disabled_or_not_applicable"
    )
    return {
        StageKey.DATASET_SELECTION: (
            "phase2_policy_dataset_selection_cpu_only",
            (DATASET_SELECTION_CPU_EXECUTOR_ID,),
        ),
        StageKey.FEATURE_CACHE_BUILD: (
            "phase2_policy_feature_cache_build_cpu_only",
            (FEATURE_CACHE_BUILD_CPU_EXECUTOR_ID,),
        ),
        StageKey.FEATURE_MATRIX_LOAD: (
            "phase2_policy_feature_matrix_load_cpu_only",
            (FEATURE_MATRIX_LOAD_CPU_EXECUTOR_ID,),
        ),
        StageKey.SPATIAL_VALIDATION: (
            "phase2_policy_spatial_validation_cpu_only",
            (SPATIAL_VALIDATION_CPU_EXECUTOR_ID,),
        ),
        StageKey.PREPROCESS: (
            preprocess_reason,
            (PREPROCESS_CPU_EXECUTOR_ID,),
        ),
        StageKey.MODEL_FIT: (
            f"phase2_policy_model_fit_{normalized_model_name}_conservative",
            _model_fit_executor_candidates(normalized_model_name),
        ),
        StageKey.TUNING: (
            f"phase2_policy_tuning_{normalized_model_name}_conservative_{tuning_reason_suffix}",
            _tuning_executor_candidates(normalized_model_name),
        ),
        StageKey.PERMUTATION: (
            "phase2_policy_permutation_"
            f"{normalized_model_name}_conservative_{permutation_reason_suffix}",
            _permutation_executor_candidates(normalized_model_name),
        ),
        StageKey.EVALUATION: (
            "phase2_policy_evaluation_cpu_only",
            (EVALUATION_CPU_EXECUTOR_ID,),
        ),
        StageKey.REPORTING: (
            "phase2_policy_reporting_cpu_only",
            (REPORTING_CPU_EXECUTOR_ID,),
        ),
    }


def plan_stage_execution(
    *,
    framework_mode: FrameworkMode | str,
    compute_policy: ResolvedComputePolicy,
    model_name: str,
    methodology_policy_name: str,
    tuning_enabled: bool,
    n_permutations: int,
) -> StagePlanningResult:
    resolved_framework_mode = coerce_framework_mode(framework_mode)
    normalized_model_name = str(model_name).strip().lower()
    if not normalized_model_name:
        raise ValueError("model_name must be non-empty for stage planning.")

    stage_plan_map = _plan_stage_assignments_map(
        model_name=normalized_model_name,
        methodology_policy_name=methodology_policy_name,
        tuning_enabled=bool(tuning_enabled),
        n_permutations=int(n_permutations),
    )
    assignments: list[StageAssignment] = []
    runtime_fallback_executor_ids: dict[str, str] = {}
    for stage_key in StageKey:
        assignment_reason, candidate_executor_ids = stage_plan_map[stage_key]
        resolution = _resolve_executor_for_stage(
            stage_key=stage_key,
            model_name=normalized_model_name,
            framework_mode=resolved_framework_mode,
            compute_policy=compute_policy,
            ordered_executor_ids=candidate_executor_ids,
            assignment_reason=assignment_reason,
        )
        assignments.append(resolution.assignment)
        if resolution.runtime_fallback_executor_id is not None:
            runtime_fallback_executor_ids[stage_key.value] = str(
                resolution.runtime_fallback_executor_id
            )

    return StagePlanningResult(
        assignments=tuple(assignments),
        runtime_fallback_executor_ids=runtime_fallback_executor_ids,
    )


def stage_assignment_for(
    result: StagePlanningResult,
    stage_key: StageKey | str,
) -> StageAssignment:
    resolved_stage = StageKey(str(stage_key))
    for assignment in result.assignments:
        if StageKey(str(assignment.stage)) == resolved_stage:
            return assignment
    raise ValueError(f"Stage planning result has no assignment for stage '{resolved_stage.value}'.")


__all__ = [
    "StagePlanningResult",
    "plan_stage_execution",
    "stage_assignment_for",
]
