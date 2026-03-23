from __future__ import annotations

from dataclasses import dataclass

from Thesis_ML.config.framework_mode import FrameworkMode, coerce_framework_mode
from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy
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
    PERMUTATION_REFERENCE_EXECUTOR_ID,
    PERMUTATION_RIDGE_GPU_PREFERRED_EXECUTOR_ID,
    PREPROCESS_CPU_EXECUTOR_ID,
    REPORTING_CPU_EXECUTOR_ID,
    SPATIAL_VALIDATION_CPU_EXECUTOR_ID,
    SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID,
    TUNING_GENERIC_EXECUTOR_ID,
    TUNING_SKIPPED_CONTROL_EXECUTOR_ID,
    StageExecutorSelectionContext,
    get_stage_executor,
    stage_executor_support_status,
)


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
    if backend_family == StageBackendFamily.SKLEARN_CPU:
        return "cpu"
    if backend_family == StageBackendFamily.TORCH_GPU:
        return "gpu"
    lane = compute_policy.assigned_compute_lane
    if isinstance(lane, str) and lane.strip() in {"cpu", "gpu"}:
        return str(lane).strip()
    return "gpu" if _resolve_effective_backend_family(compute_policy) == "torch_gpu" else "cpu"


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


def _model_fit_executor_candidates(model_name: str) -> tuple[str, ...]:
    normalized = str(model_name).strip().lower()
    if normalized == "dummy":
        return (MODEL_FIT_CPU_EXECUTOR_ID,)
    if normalized == "linearsvc":
        return (MODEL_FIT_CPU_EXECUTOR_ID,)
    if normalized == "logreg":
        return (MODEL_FIT_CPU_EXECUTOR_ID, MODEL_FIT_TORCH_LOGREG_EXECUTOR_ID)
    if normalized == "ridge":
        return (MODEL_FIT_TORCH_RIDGE_EXECUTOR_ID, MODEL_FIT_CPU_EXECUTOR_ID)
    return (MODEL_FIT_CPU_EXECUTOR_ID,)


def _tuning_executor_candidates(model_name: str) -> tuple[str, ...]:
    normalized = str(model_name).strip().lower()
    if normalized == "dummy":
        return (TUNING_SKIPPED_CONTROL_EXECUTOR_ID,)
    if normalized == "linearsvc":
        return (SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID, TUNING_GENERIC_EXECUTOR_ID)
    if normalized in {"logreg", "ridge"}:
        return (TUNING_GENERIC_EXECUTOR_ID,)
    return (TUNING_GENERIC_EXECUTOR_ID,)


def _permutation_executor_candidates(model_name: str) -> tuple[str, ...]:
    normalized = str(model_name).strip().lower()
    if normalized == "ridge":
        return (
            PERMUTATION_RIDGE_GPU_PREFERRED_EXECUTOR_ID,
            PERMUTATION_REFERENCE_EXECUTOR_ID,
        )
    return (PERMUTATION_REFERENCE_EXECUTOR_ID,)


def _plan_stage_assignments_map(
    *,
    model_name: str,
    methodology_policy_name: str,
    tuning_enabled: bool,
    n_permutations: int,
) -> dict[StageKey, tuple[str, tuple[str, ...]]]:
    normalized_model_name = str(model_name).strip().lower()
    normalized_methodology = str(methodology_policy_name).strip()
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
            "phase2_policy_preprocess_cpu_preferred",
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
