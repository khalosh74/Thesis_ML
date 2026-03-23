from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from Thesis_ML.config.framework_mode import FrameworkMode, coerce_framework_mode
from Thesis_ML.config.metric_policy import metric_scorer
from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy
from Thesis_ML.experiments.linearsvc_tuning import (
    SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID,
    run_specialized_linearsvc_grouped_nested_tuning,
)
from Thesis_ML.experiments.stage_execution import (
    StageBackendFamily,
    StageExecutorEquivalence,
    StageKey,
)

StageExecutorSupportPredicate = Callable[
    ["StageExecutorSelectionContext"],
    tuple[bool, str | None],
]
StageExecutorEntrypoint = Callable[..., Any]
FrameworkModeLike = FrameworkMode | str

DATASET_SELECTION_CPU_EXECUTOR_ID = "dataset_selection_cpu_reference_v1"
FEATURE_CACHE_BUILD_CPU_EXECUTOR_ID = "feature_cache_build_cpu_reference_v1"
FEATURE_MATRIX_LOAD_CPU_EXECUTOR_ID = "feature_matrix_load_cpu_reference_v1"
SPATIAL_VALIDATION_CPU_EXECUTOR_ID = "spatial_validation_cpu_reference_v1"
PREPROCESS_CPU_EXECUTOR_ID = "preprocess_cpu_reference_v1"
MODEL_FIT_CPU_EXECUTOR_ID = "model_fit_cpu_reference_v1"
MODEL_FIT_TORCH_RIDGE_EXECUTOR_ID = "model_fit_torch_ridge_v1"
MODEL_FIT_TORCH_LOGREG_EXECUTOR_ID = "model_fit_torch_logreg_v1"
TUNING_GENERIC_EXECUTOR_ID = "gridsearchcv_pipeline_reference_v1"
TUNING_SKIPPED_CONTROL_EXECUTOR_ID = "tuning_skipped_control_model_v1"
PERMUTATION_REFERENCE_EXECUTOR_ID = "permutation_reference_v1"
PERMUTATION_RIDGE_GPU_PREFERRED_EXECUTOR_ID = "permutation_ridge_gpu_preferred_v1"
EVALUATION_CPU_EXECUTOR_ID = "evaluation_cpu_reference_v1"
REPORTING_CPU_EXECUTOR_ID = "reporting_cpu_reference_v1"


@dataclass(frozen=True)
class StageExecutorSelectionContext:
    stage: StageKey
    model_name: str | None
    framework_mode: FrameworkModeLike
    compute_policy: ResolvedComputePolicy | None

    @property
    def framework_mode_resolved(self) -> FrameworkMode:
        return coerce_framework_mode(self.framework_mode)

    @property
    def in_official_path(self) -> bool:
        return self.framework_mode_resolved in {
            FrameworkMode.CONFIRMATORY,
            FrameworkMode.LOCKED_COMPARISON,
        }


@dataclass(frozen=True)
class StageExecutorSpec:
    executor_id: str
    stage_key: StageKey
    backend_family: StageBackendFamily
    supported_model_names: tuple[str, ...] | None
    equivalence_class: StageExecutorEquivalence
    official_admitted: bool
    support_predicate: StageExecutorSupportPredicate
    execute: StageExecutorEntrypoint


def _support_always(_: StageExecutorSelectionContext) -> tuple[bool, str | None]:
    return True, None


def _support_torch_backend(context: StageExecutorSelectionContext) -> tuple[bool, str | None]:
    policy = context.compute_policy
    if policy is None:
        return False, "compute_policy_missing"
    backend_family = str(policy.assigned_backend_family or policy.effective_backend_family).strip()
    if backend_family != StageBackendFamily.TORCH_GPU.value:
        return False, "effective_backend_family_not_torch_gpu"
    if policy.gpu_device_id is None:
        return False, "gpu_device_id_missing"
    return True, None


def _stage_noop_entrypoint(**_: Any) -> dict[str, Any]:
    return {"status": "noop"}


def _fit_estimator_entrypoint(
    *,
    estimator: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    executor_id: str,
) -> dict[str, Any]:
    fit_start = perf_counter()
    estimator.fit(x_train, y_train)
    return {
        "estimator": estimator,
        "estimator_fit_elapsed_seconds": float(perf_counter() - fit_start),
        "executor_id": str(executor_id),
    }


def _model_fit_cpu_entrypoint(**kwargs: Any) -> dict[str, Any]:
    return _fit_estimator_entrypoint(executor_id=MODEL_FIT_CPU_EXECUTOR_ID, **kwargs)


def _model_fit_torch_ridge_entrypoint(**kwargs: Any) -> dict[str, Any]:
    return _fit_estimator_entrypoint(executor_id=MODEL_FIT_TORCH_RIDGE_EXECUTOR_ID, **kwargs)


def _model_fit_torch_logreg_entrypoint(**kwargs: Any) -> dict[str, Any]:
    return _fit_estimator_entrypoint(executor_id=MODEL_FIT_TORCH_LOGREG_EXECUTOR_ID, **kwargs)


def _safe_float_from_cv_results(
    cv_results: dict[str, Any],
    key: str,
) -> float | None:
    raw = cv_results.get(key)
    if raw is None:
        return None
    values = np.asarray(raw, dtype=np.float64)
    if values.size == 0:
        return None
    return float(np.nanmean(values))


def _resolve_profiled_param_grid(
    *,
    param_grid: dict[str, Any],
    candidate_params: list[dict[str, Any]],
    profiled_candidate_count: int,
    configured_candidate_count: int,
) -> dict[str, Any] | list[dict[str, Any]]:
    if int(profiled_candidate_count) >= int(configured_candidate_count):
        return param_grid
    profiled_params = candidate_params[: int(profiled_candidate_count)]
    return [{key: [value] for key, value in params.items()} for params in profiled_params]


def _execute_tuning_generic_entrypoint(
    *,
    pipeline_template: Pipeline,
    x_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    param_grid: dict[str, Any],
    candidate_params: list[dict[str, Any]],
    configured_candidate_count: int,
    configured_inner_fold_count: int,
    profiled_candidate_count: int,
    profiled_inner_fold_count: int,
    primary_metric_name: str,
    tuning_executor_fallback_reason: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    profiled_param_grid = _resolve_profiled_param_grid(
        param_grid=param_grid,
        candidate_params=candidate_params,
        profiled_candidate_count=int(profiled_candidate_count),
        configured_candidate_count=int(configured_candidate_count),
    )
    profiled_inner_splits = inner_splits[: int(profiled_inner_fold_count)]
    search = GridSearchCV(
        estimator=clone(pipeline_template),
        param_grid=profiled_param_grid,
        scoring=metric_scorer(primary_metric_name),
        cv=profiled_inner_splits,
        refit=True,
        n_jobs=1,
    )
    search_start = perf_counter()
    search.fit(x_outer_train, y_outer_train)
    tuned_search_elapsed_seconds = float(perf_counter() - search_start)
    tuned_search_candidate_count = int(len(search.cv_results_["params"]))
    tuned_search_profiled_candidate_count = int(tuned_search_candidate_count)
    tuned_search_configured_candidate_count = int(configured_candidate_count)
    tuned_search_profiled_inner_fold_count = int(profiled_inner_fold_count)
    tuned_search_configured_inner_fold_count = int(configured_inner_fold_count)
    measured_inner_tuning_seconds = float(tuned_search_elapsed_seconds)
    estimated_full_inner_tuning_seconds = float(measured_inner_tuning_seconds)
    tuning_extrapolation_applied = bool(
        tuned_search_profiled_candidate_count != tuned_search_configured_candidate_count
        or tuned_search_profiled_inner_fold_count != tuned_search_configured_inner_fold_count
    )
    if tuning_extrapolation_applied:
        estimated_full_inner_tuning_seconds *= float(
            tuned_search_configured_candidate_count
        ) / float(max(tuned_search_profiled_candidate_count, 1))
        estimated_full_inner_tuning_seconds *= float(
            tuned_search_configured_inner_fold_count
        ) / float(max(tuned_search_profiled_inner_fold_count, 1))
    estimated_full_tuned_search_seconds = float(estimated_full_inner_tuning_seconds)
    return {
        "estimator": search.best_estimator_,
        "tuned_search_elapsed_seconds": float(tuned_search_elapsed_seconds),
        "tuned_search_candidate_count": int(tuned_search_candidate_count),
        "tuned_search_configured_candidate_count": int(tuned_search_configured_candidate_count),
        "tuned_search_profiled_candidate_count": int(tuned_search_profiled_candidate_count),
        "tuned_search_configured_inner_fold_count": int(tuned_search_configured_inner_fold_count),
        "tuned_search_profiled_inner_fold_count": int(tuned_search_profiled_inner_fold_count),
        "tuning_extrapolation_applied": bool(tuning_extrapolation_applied),
        "measured_inner_tuning_seconds": float(measured_inner_tuning_seconds),
        "estimated_full_inner_tuning_seconds": float(estimated_full_inner_tuning_seconds),
        "estimated_full_tuned_search_seconds": float(estimated_full_tuned_search_seconds),
        "tuning_split_scale_seconds": None,
        "tuning_candidate_fit_seconds": None,
        "tuning_candidate_predict_seconds": None,
        "tuning_refit_elapsed_seconds": None,
        "cv_mean_fit_time_seconds": _safe_float_from_cv_results(
            search.cv_results_,
            "mean_fit_time",
        ),
        "cv_std_fit_time_seconds": _safe_float_from_cv_results(
            search.cv_results_,
            "std_fit_time",
        ),
        "cv_mean_score_time_seconds": _safe_float_from_cv_results(
            search.cv_results_,
            "mean_score_time",
        ),
        "cv_std_score_time_seconds": _safe_float_from_cv_results(
            search.cv_results_,
            "std_score_time",
        ),
        "best_score": float(search.best_score_),
        "best_params_json": json.dumps(search.best_params_, sort_keys=True),
        "tuning_executor": TUNING_GENERIC_EXECUTOR_ID,
        "tuning_executor_fallback_reason": tuning_executor_fallback_reason,
        "specialized_linearsvc_tuning_used": False,
    }


def _execute_tuning_linearsvc_specialized_entrypoint(
    *,
    pipeline_template: Pipeline,
    x_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    inner_groups: np.ndarray,
    param_grid: dict[str, Any],
    configured_candidate_count: int,
    configured_inner_fold_count: int,
    profiled_candidate_count: int,
    profiled_inner_fold_count: int,
    primary_metric_name: str,
    **_: Any,
) -> dict[str, Any]:
    specialized_result = run_specialized_linearsvc_grouped_nested_tuning(
        pipeline_template=clone(pipeline_template),
        x_train=x_outer_train,
        y_train=y_outer_train,
        inner_groups=inner_groups,
        param_grid=param_grid,
        primary_metric_name=primary_metric_name,
        profile_inner_folds=(
            int(profiled_inner_fold_count)
            if int(profiled_inner_fold_count) < int(configured_inner_fold_count)
            else None
        ),
        profile_tuning_candidates=(
            int(profiled_candidate_count)
            if int(profiled_candidate_count) < int(configured_candidate_count)
            else None
        ),
    )
    return {
        "estimator": specialized_result.best_estimator,
        "tuned_search_elapsed_seconds": float(specialized_result.tuned_search_elapsed_seconds),
        "tuned_search_candidate_count": int(specialized_result.profiled_candidate_count),
        "tuned_search_configured_candidate_count": int(
            specialized_result.configured_candidate_count
        ),
        "tuned_search_profiled_candidate_count": int(specialized_result.profiled_candidate_count),
        "tuned_search_configured_inner_fold_count": int(
            specialized_result.configured_inner_fold_count
        ),
        "tuned_search_profiled_inner_fold_count": int(
            specialized_result.profiled_inner_fold_count
        ),
        "tuning_extrapolation_applied": bool(specialized_result.tuning_extrapolation_applied),
        "measured_inner_tuning_seconds": float(specialized_result.measured_inner_tuning_seconds),
        "estimated_full_inner_tuning_seconds": float(
            specialized_result.estimated_full_inner_tuning_seconds
        ),
        "estimated_full_tuned_search_seconds": float(
            specialized_result.estimated_full_tuned_search_seconds
        ),
        "tuning_split_scale_seconds": float(specialized_result.split_scale_seconds),
        "tuning_candidate_fit_seconds": float(specialized_result.candidate_fit_seconds),
        "tuning_candidate_predict_seconds": float(specialized_result.candidate_predict_seconds),
        "tuning_refit_elapsed_seconds": float(specialized_result.refit_elapsed_seconds),
        "cv_mean_fit_time_seconds": _safe_float_from_cv_results(
            specialized_result.cv_results,
            "mean_fit_time",
        ),
        "cv_std_fit_time_seconds": _safe_float_from_cv_results(
            specialized_result.cv_results,
            "std_fit_time",
        ),
        "cv_mean_score_time_seconds": _safe_float_from_cv_results(
            specialized_result.cv_results,
            "mean_score_time",
        ),
        "cv_std_score_time_seconds": _safe_float_from_cv_results(
            specialized_result.cv_results,
            "std_score_time",
        ),
        "best_score": float(specialized_result.best_score),
        "best_params_json": json.dumps(specialized_result.best_params, sort_keys=True),
        "tuning_executor": str(specialized_result.executor_id),
        "tuning_executor_fallback_reason": None,
        "specialized_linearsvc_tuning_used": True,
    }


def _execute_tuning_skipped_control_entrypoint() -> dict[str, Any]:
    return {
        "tuning_executor": "skipped_control_model",
        "tuning_executor_fallback_reason": None,
        "specialized_linearsvc_tuning_used": False,
    }


def _execute_permutation_reference_entrypoint(
    *,
    evaluate_permutations_fn: Callable[..., dict[str, Any]],
    build_pipeline_fn: Callable[..., Any],
    model_name: str,
    seed: int,
    x_matrix: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_permutations: int,
    metric_name: str,
    observed_metric: float,
    progress_callback: Any,
    progress_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = evaluate_permutations_fn(
        pipeline_template=build_pipeline_fn(
            model_name=model_name,
            seed=seed,
        ),
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=seed,
        n_permutations=n_permutations,
        metric_name=metric_name,
        observed_metric=observed_metric,
        progress_callback=progress_callback,
        progress_metadata=progress_metadata,
    )
    payload["permutation_executor_id"] = PERMUTATION_REFERENCE_EXECUTOR_ID
    payload["permutation_executor_fallback_reason"] = None
    return payload


def _execute_permutation_ridge_gpu_preferred_entrypoint(
    **kwargs: Any,
) -> dict[str, Any]:
    payload = _execute_permutation_reference_entrypoint(**kwargs)
    fallback_reason = payload.get("specialized_ridge_gpu_fallback_reason")
    payload["permutation_executor_id"] = PERMUTATION_RIDGE_GPU_PREFERRED_EXECUTOR_ID
    payload["permutation_executor_fallback_reason"] = (
        str(fallback_reason) if isinstance(fallback_reason, str) and fallback_reason.strip() else None
    )
    return payload


def _build_registry() -> tuple[StageExecutorSpec, ...]:
    return (
        StageExecutorSpec(
            executor_id=DATASET_SELECTION_CPU_EXECUTOR_ID,
            stage_key=StageKey.DATASET_SELECTION,
            backend_family=StageBackendFamily.SKLEARN_CPU,
            supported_model_names=None,
            equivalence_class="exact_reference_equivalent",
            official_admitted=True,
            support_predicate=_support_always,
            execute=_stage_noop_entrypoint,
        ),
        StageExecutorSpec(
            executor_id=FEATURE_CACHE_BUILD_CPU_EXECUTOR_ID,
            stage_key=StageKey.FEATURE_CACHE_BUILD,
            backend_family=StageBackendFamily.SKLEARN_CPU,
            supported_model_names=None,
            equivalence_class="exact_reference_equivalent",
            official_admitted=True,
            support_predicate=_support_always,
            execute=_stage_noop_entrypoint,
        ),
        StageExecutorSpec(
            executor_id=FEATURE_MATRIX_LOAD_CPU_EXECUTOR_ID,
            stage_key=StageKey.FEATURE_MATRIX_LOAD,
            backend_family=StageBackendFamily.SKLEARN_CPU,
            supported_model_names=None,
            equivalence_class="exact_reference_equivalent",
            official_admitted=True,
            support_predicate=_support_always,
            execute=_stage_noop_entrypoint,
        ),
        StageExecutorSpec(
            executor_id=SPATIAL_VALIDATION_CPU_EXECUTOR_ID,
            stage_key=StageKey.SPATIAL_VALIDATION,
            backend_family=StageBackendFamily.SKLEARN_CPU,
            supported_model_names=None,
            equivalence_class="exact_reference_equivalent",
            official_admitted=True,
            support_predicate=_support_always,
            execute=_stage_noop_entrypoint,
        ),
        StageExecutorSpec(
            executor_id=PREPROCESS_CPU_EXECUTOR_ID,
            stage_key=StageKey.PREPROCESS,
            backend_family=StageBackendFamily.SKLEARN_CPU,
            supported_model_names=None,
            equivalence_class="exact_reference_equivalent",
            official_admitted=True,
            support_predicate=_support_always,
            execute=_stage_noop_entrypoint,
        ),
        StageExecutorSpec(
            executor_id=MODEL_FIT_CPU_EXECUTOR_ID,
            stage_key=StageKey.MODEL_FIT,
            backend_family=StageBackendFamily.SKLEARN_CPU,
            supported_model_names=("dummy", "linearsvc", "logreg", "ridge"),
            equivalence_class="exact_reference_equivalent",
            official_admitted=True,
            support_predicate=_support_always,
            execute=_model_fit_cpu_entrypoint,
        ),
        StageExecutorSpec(
            executor_id=MODEL_FIT_TORCH_RIDGE_EXECUTOR_ID,
            stage_key=StageKey.MODEL_FIT,
            backend_family=StageBackendFamily.TORCH_GPU,
            supported_model_names=("ridge",),
            equivalence_class="validated_variant",
            official_admitted=True,
            support_predicate=_support_torch_backend,
            execute=_model_fit_torch_ridge_entrypoint,
        ),
        StageExecutorSpec(
            executor_id=MODEL_FIT_TORCH_LOGREG_EXECUTOR_ID,
            stage_key=StageKey.MODEL_FIT,
            backend_family=StageBackendFamily.TORCH_GPU,
            supported_model_names=("logreg",),
            equivalence_class="validated_variant",
            official_admitted=False,
            support_predicate=_support_torch_backend,
            execute=_model_fit_torch_logreg_entrypoint,
        ),
        StageExecutorSpec(
            executor_id=TUNING_GENERIC_EXECUTOR_ID,
            stage_key=StageKey.TUNING,
            backend_family=StageBackendFamily.AUTO_MIXED,
            supported_model_names=("linearsvc", "logreg", "ridge"),
            equivalence_class="exact_reference_equivalent",
            official_admitted=True,
            support_predicate=_support_always,
            execute=_execute_tuning_generic_entrypoint,
        ),
        StageExecutorSpec(
            executor_id=SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID,
            stage_key=StageKey.TUNING,
            backend_family=StageBackendFamily.SKLEARN_CPU,
            supported_model_names=("linearsvc",),
            equivalence_class="exact_reference_equivalent",
            official_admitted=True,
            support_predicate=_support_always,
            execute=_execute_tuning_linearsvc_specialized_entrypoint,
        ),
        StageExecutorSpec(
            executor_id=TUNING_SKIPPED_CONTROL_EXECUTOR_ID,
            stage_key=StageKey.TUNING,
            backend_family=StageBackendFamily.SKLEARN_CPU,
            supported_model_names=("dummy",),
            equivalence_class="exact_reference_equivalent",
            official_admitted=True,
            support_predicate=_support_always,
            execute=_execute_tuning_skipped_control_entrypoint,
        ),
        StageExecutorSpec(
            executor_id=PERMUTATION_REFERENCE_EXECUTOR_ID,
            stage_key=StageKey.PERMUTATION,
            backend_family=StageBackendFamily.AUTO_MIXED,
            supported_model_names=("dummy", "linearsvc", "logreg", "ridge"),
            equivalence_class="exact_reference_equivalent",
            official_admitted=True,
            support_predicate=_support_always,
            execute=_execute_permutation_reference_entrypoint,
        ),
        StageExecutorSpec(
            executor_id=PERMUTATION_RIDGE_GPU_PREFERRED_EXECUTOR_ID,
            stage_key=StageKey.PERMUTATION,
            backend_family=StageBackendFamily.TORCH_GPU,
            supported_model_names=("ridge",),
            equivalence_class="validated_variant",
            official_admitted=True,
            support_predicate=_support_torch_backend,
            execute=_execute_permutation_ridge_gpu_preferred_entrypoint,
        ),
        StageExecutorSpec(
            executor_id=EVALUATION_CPU_EXECUTOR_ID,
            stage_key=StageKey.EVALUATION,
            backend_family=StageBackendFamily.SKLEARN_CPU,
            supported_model_names=None,
            equivalence_class="exact_reference_equivalent",
            official_admitted=True,
            support_predicate=_support_always,
            execute=_stage_noop_entrypoint,
        ),
        StageExecutorSpec(
            executor_id=REPORTING_CPU_EXECUTOR_ID,
            stage_key=StageKey.REPORTING,
            backend_family=StageBackendFamily.SKLEARN_CPU,
            supported_model_names=None,
            equivalence_class="exact_reference_equivalent",
            official_admitted=True,
            support_predicate=_support_always,
            execute=_stage_noop_entrypoint,
        ),
    )


STAGE_EXECUTOR_REGISTRY: tuple[StageExecutorSpec, ...] = _build_registry()
_REGISTRY_BY_ID: dict[str, StageExecutorSpec] = {
    spec.executor_id: spec for spec in STAGE_EXECUTOR_REGISTRY
}


def iter_stage_executors(stage_key: StageKey | str) -> tuple[StageExecutorSpec, ...]:
    resolved_stage = StageKey(str(stage_key))
    return tuple(spec for spec in STAGE_EXECUTOR_REGISTRY if spec.stage_key == resolved_stage)


def get_stage_executor(executor_id: str) -> StageExecutorSpec:
    spec = _REGISTRY_BY_ID.get(str(executor_id))
    if spec is None:
        raise ValueError(f"Unknown stage executor_id '{executor_id}'.")
    return spec


def _supports_model(
    spec: StageExecutorSpec,
    *,
    model_name: str | None,
) -> tuple[bool, str | None]:
    if spec.supported_model_names is None:
        return True, None
    if model_name is None:
        return False, "model_name_required"
    normalized_model = str(model_name).strip().lower()
    if normalized_model not in spec.supported_model_names:
        return False, "model_not_supported"
    return True, None


def stage_executor_support_status(
    spec: StageExecutorSpec,
    context: StageExecutorSelectionContext,
) -> tuple[bool, str | None]:
    model_ok, model_reason = _supports_model(spec, model_name=context.model_name)
    if not model_ok:
        return False, model_reason
    predicate_ok, predicate_reason = spec.support_predicate(context)
    if not predicate_ok:
        return False, predicate_reason
    if context.in_official_path and not bool(spec.official_admitted):
        return False, "executor_not_admitted_for_official_path"
    return True, None


def run_model_fit_executor(
    *,
    executor_id: str,
    estimator: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    executor = get_stage_executor(executor_id)
    if executor.stage_key != StageKey.MODEL_FIT:
        raise ValueError(
            f"Stage executor '{executor_id}' cannot run stage '{StageKey.MODEL_FIT.value}'."
        )
    return executor.execute(
        estimator=estimator,
        x_train=x_train,
        y_train=y_train,
    )


def run_tuning_executor(
    *,
    executor_id: str,
    fallback_executor_id: str | None,
    pipeline_template: Pipeline,
    x_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    inner_groups: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    param_grid: dict[str, Any],
    candidate_params: list[dict[str, Any]],
    configured_candidate_count: int,
    configured_inner_fold_count: int,
    profiled_candidate_count: int,
    profiled_inner_fold_count: int,
    primary_metric_name: str,
) -> dict[str, Any]:
    executor = get_stage_executor(executor_id)
    if executor.stage_key != StageKey.TUNING:
        raise ValueError(
            f"Stage executor '{executor_id}' cannot run stage '{StageKey.TUNING.value}'."
        )
    kwargs = {
        "pipeline_template": pipeline_template,
        "x_outer_train": x_outer_train,
        "y_outer_train": y_outer_train,
        "inner_groups": inner_groups,
        "inner_splits": inner_splits,
        "param_grid": param_grid,
        "candidate_params": candidate_params,
        "configured_candidate_count": int(configured_candidate_count),
        "configured_inner_fold_count": int(configured_inner_fold_count),
        "profiled_candidate_count": int(profiled_candidate_count),
        "profiled_inner_fold_count": int(profiled_inner_fold_count),
        "primary_metric_name": str(primary_metric_name),
    }
    if executor_id == TUNING_SKIPPED_CONTROL_EXECUTOR_ID:
        return executor.execute()
    if executor_id != SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID:
        return executor.execute(**kwargs)
    try:
        return executor.execute(**kwargs)
    except ValueError as exc:
        if fallback_executor_id is None:
            raise
        fallback_executor = get_stage_executor(fallback_executor_id)
        if fallback_executor.stage_key != StageKey.TUNING:
            raise
        return fallback_executor.execute(
            **kwargs,
            tuning_executor_fallback_reason=str(exc),
        )


def run_permutation_executor(
    *,
    executor_id: str,
    evaluate_permutations_fn: Callable[..., dict[str, Any]],
    build_pipeline_fn: Callable[..., Any],
    model_name: str,
    seed: int,
    x_matrix: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_permutations: int,
    metric_name: str,
    observed_metric: float,
    progress_callback: Any,
    progress_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    executor = get_stage_executor(executor_id)
    if executor.stage_key != StageKey.PERMUTATION:
        raise ValueError(
            f"Stage executor '{executor_id}' cannot run stage '{StageKey.PERMUTATION.value}'."
        )
    return executor.execute(
        evaluate_permutations_fn=evaluate_permutations_fn,
        build_pipeline_fn=build_pipeline_fn,
        model_name=model_name,
        seed=seed,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        n_permutations=n_permutations,
        metric_name=metric_name,
        observed_metric=observed_metric,
        progress_callback=progress_callback,
        progress_metadata=progress_metadata,
    )


__all__ = [
    "DATASET_SELECTION_CPU_EXECUTOR_ID",
    "EVALUATION_CPU_EXECUTOR_ID",
    "FEATURE_CACHE_BUILD_CPU_EXECUTOR_ID",
    "FEATURE_MATRIX_LOAD_CPU_EXECUTOR_ID",
    "MODEL_FIT_CPU_EXECUTOR_ID",
    "MODEL_FIT_TORCH_LOGREG_EXECUTOR_ID",
    "MODEL_FIT_TORCH_RIDGE_EXECUTOR_ID",
    "PERMUTATION_REFERENCE_EXECUTOR_ID",
    "PERMUTATION_RIDGE_GPU_PREFERRED_EXECUTOR_ID",
    "PREPROCESS_CPU_EXECUTOR_ID",
    "REPORTING_CPU_EXECUTOR_ID",
    "SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID",
    "SPATIAL_VALIDATION_CPU_EXECUTOR_ID",
    "STAGE_EXECUTOR_REGISTRY",
    "TUNING_GENERIC_EXECUTOR_ID",
    "TUNING_SKIPPED_CONTROL_EXECUTOR_ID",
    "StageExecutorSelectionContext",
    "StageExecutorSpec",
    "get_stage_executor",
    "iter_stage_executors",
    "run_model_fit_executor",
    "run_permutation_executor",
    "run_tuning_executor",
    "stage_executor_support_status",
]
