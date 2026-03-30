from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Thesis_ML.config.metric_policy import classification_metric_score
from Thesis_ML.experiments.backends.ridge_exact_kernels import (
    build_ridge_exact_alpha_factorization_state,
    predict_ridge_labels_for_alpha_batch,
    solve_ridge_exact_alpha_batch,
)
from Thesis_ML.experiments.linearsvc_tuning import (
    GroupSufficientStats,
    compute_standard_scaler_stats_from_group_summaries,
)

SPECIALIZED_RIDGE_TUNING_EXECUTOR_ID = "ridge_grouped_nested_exact_v1"


@dataclass(frozen=True)
class RidgeGroupedNestedTuningResult:
    best_estimator: Pipeline
    best_params: dict[str, Any]
    best_score: float
    cv_results: dict[str, Any]
    tuned_search_elapsed_seconds: float
    configured_candidate_count: int
    profiled_candidate_count: int
    configured_inner_fold_count: int
    profiled_inner_fold_count: int
    measured_inner_tuning_seconds: float
    estimated_full_inner_tuning_seconds: float
    estimated_full_tuned_search_seconds: float
    tuning_extrapolation_applied: bool
    split_scale_seconds: float
    candidate_fit_seconds: float
    candidate_predict_seconds: float
    refit_elapsed_seconds: float
    executor_id: str = SPECIALIZED_RIDGE_TUNING_EXECUTOR_ID


def _resolve_pipeline_components(
    pipeline_template: Pipeline,
) -> tuple[StandardScaler, RidgeClassifier]:
    if not isinstance(pipeline_template, Pipeline):
        raise ValueError("Specialized ridge tuning requires a sklearn Pipeline template.")
    scaler = pipeline_template.named_steps.get("scaler")
    model = pipeline_template.named_steps.get("model")
    if not isinstance(scaler, StandardScaler):
        raise ValueError("preprocessor_not_plain_standard_scaler")
    if not bool(getattr(scaler, "with_mean", False)) or not bool(
        getattr(scaler, "with_std", False)
    ):
        raise ValueError("preprocessor_not_plain_standard_scaler")
    if not isinstance(model, RidgeClassifier):
        raise ValueError("Specialized ridge tuning requires pipeline step 'model' to be RidgeClassifier.")
    class_weight = getattr(model, "class_weight", None)
    if class_weight is not None:
        normalized_class_weight = str(class_weight).strip().lower()
        if normalized_class_weight not in {"balanced", "none", ""}:
            raise ValueError("specialized_ridge_requires_class_weight_none_or_balanced")
    return scaler, model


def _resolve_candidate_params_and_values(
    param_grid: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float]]:
    candidate_params = list(ParameterGrid(param_grid))
    if not candidate_params:
        raise ValueError("Specialized ridge tuning requires at least one alpha candidate.")

    alpha_values: list[float] = []
    for params in candidate_params:
        if set(params) != {"model__alpha"}:
            raise ValueError(
                "Specialized ridge tuning only supports param_grid with key 'model__alpha'."
            )
        alpha_raw = params.get("model__alpha")
        if not isinstance(alpha_raw, (int, float)):
            raise ValueError("Specialized ridge tuning requires numeric alpha candidates.")
        alpha_value = float(alpha_raw)
        if not np.isfinite(alpha_value) or alpha_value < 0.0:
            raise ValueError("Specialized ridge tuning requires finite alpha >= 0.")
        alpha_values.append(alpha_value)

    return candidate_params, alpha_values


def _resolve_profiled_count(
    *,
    configured_count: int,
    override_value: int | None,
    label: str,
) -> int:
    if override_value is None:
        return int(configured_count)
    resolved_override = int(override_value)
    if resolved_override <= 0:
        raise ValueError(f"{label} must be > 0 when provided.")
    return int(min(configured_count, resolved_override))


def _build_group_sufficient_statistics(
    *,
    x_train: np.ndarray,
    groups: np.ndarray,
) -> dict[str, GroupSufficientStats]:
    x_array = np.asarray(x_train, dtype=np.float64)
    group_array = np.asarray(groups).astype(str, copy=False)
    stats_by_group: dict[str, GroupSufficientStats] = {}
    for group_id in np.unique(group_array).tolist():
        group_mask = group_array == str(group_id)
        group_rows = x_array[group_mask]
        feature_sum = np.sum(group_rows, axis=0, dtype=np.float64)
        feature_sum_squares = np.sum(group_rows * group_rows, axis=0, dtype=np.float64)
        stats_by_group[str(group_id)] = GroupSufficientStats(
            group_id=str(group_id),
            n_samples=int(group_rows.shape[0]),
            feature_sum=np.asarray(feature_sum, dtype=np.float64),
            feature_sum_squares=np.asarray(feature_sum_squares, dtype=np.float64),
        )
    return stats_by_group


def _load_and_scale_rows(
    *,
    x_train: np.ndarray,
    row_indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    buffer: np.ndarray,
) -> np.ndarray:
    row_count = int(row_indices.shape[0])
    view = buffer[:row_count]
    np.take(x_train, row_indices, axis=0, out=view)
    np.subtract(view, mean, out=view)
    np.divide(view, scale, out=view)
    return view


def _resolve_sample_weights(
    *,
    y_labels: np.ndarray,
    class_weight: str | dict[str, float] | None,
) -> np.ndarray:
    if class_weight is None:
        return np.ones(int(y_labels.shape[0]), dtype=np.float64)

    if isinstance(class_weight, dict):
        weights = np.asarray(
            [float(class_weight[str(label)]) for label in y_labels.astype(str, copy=False).tolist()],
            dtype=np.float64,
        )
        return weights

    normalized = str(class_weight).strip().lower()
    if normalized in {"none", ""}:
        return np.ones(int(y_labels.shape[0]), dtype=np.float64)
    if normalized != "balanced":
        raise ValueError("specialized_ridge_requires_class_weight_none_or_balanced")

    labels = np.asarray(y_labels).astype(str, copy=False)
    classes, counts = np.unique(labels, return_counts=True)
    n_samples = int(labels.shape[0])
    n_classes = int(classes.shape[0])
    class_weights = {
        label: float(n_samples / (n_classes * int(count)))
        for label, count in zip(classes.tolist(), counts.tolist(), strict=True)
    }
    return np.asarray([class_weights[label] for label in labels.tolist()], dtype=np.float64)


def _stable_descending_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order)
    current_rank = 1
    previous_value: float | None = None
    for sorted_position, candidate_index in enumerate(order):
        value = float(values[candidate_index])
        if previous_value is not None and value < previous_value:
            current_rank = sorted_position + 1
        ranks[candidate_index] = int(current_rank)
        previous_value = value
    return ranks.astype(int, copy=False)


def is_specialized_ridge_grouped_nested_supported(
    *,
    model_name: str,
    pipeline_template: Pipeline,
    param_grid: dict[str, Any],
) -> tuple[bool, str | None]:
    if str(model_name).strip().lower() != "ridge":
        return False, "model_not_ridge"
    try:
        _resolve_pipeline_components(pipeline_template)
        _resolve_candidate_params_and_values(param_grid)
    except ValueError as exc:
        return False, str(exc)
    return True, None


def run_specialized_ridge_grouped_nested_tuning(
    *,
    pipeline_template: Pipeline,
    x_train: np.ndarray,
    y_train: np.ndarray,
    inner_groups: np.ndarray,
    param_grid: dict[str, Any],
    primary_metric_name: str,
    profile_inner_folds: int | None = None,
    profile_tuning_candidates: int | None = None,
) -> RidgeGroupedNestedTuningResult:
    _, model_template = _resolve_pipeline_components(pipeline_template)
    candidate_params, alpha_values = _resolve_candidate_params_and_values(param_grid)

    x_train_array = np.asarray(x_train, dtype=np.float64)
    y_train_array = np.asarray(y_train).astype(str, copy=False)
    groups_array = np.asarray(inner_groups).astype(str, copy=False)
    if x_train_array.ndim != 2:
        raise ValueError("Specialized ridge tuning requires a 2D training matrix.")
    if y_train_array.ndim != 1:
        raise ValueError("Specialized ridge tuning requires a 1D training label vector.")
    if x_train_array.shape[0] != y_train_array.shape[0]:
        raise ValueError("Specialized ridge tuning received mismatched X/y row counts.")
    if x_train_array.shape[0] != groups_array.shape[0]:
        raise ValueError("Specialized ridge tuning received mismatched group rows.")

    splitter = LeaveOneGroupOut()
    all_inner_splits = list(splitter.split(x_train_array, y_train_array, groups=groups_array))
    configured_inner_fold_count = int(len(all_inner_splits))
    if configured_inner_fold_count < 2:
        raise ValueError("Specialized ridge tuning requires at least two inner folds.")

    configured_candidate_count = int(len(alpha_values))
    profiled_inner_fold_count = _resolve_profiled_count(
        configured_count=configured_inner_fold_count,
        override_value=profile_inner_folds,
        label="profile_inner_folds",
    )
    profiled_candidate_count = _resolve_profiled_count(
        configured_count=configured_candidate_count,
        override_value=profile_tuning_candidates,
        label="profile_tuning_candidates",
    )
    profiled_splits = all_inner_splits[:profiled_inner_fold_count]
    profiled_params = candidate_params[:profiled_candidate_count]
    profiled_alphas = np.asarray(alpha_values[:profiled_candidate_count], dtype=np.float64)

    n_features = int(x_train_array.shape[1])
    max_inner_train_rows = max(int(len(train_idx)) for train_idx, _ in profiled_splits)
    max_inner_valid_rows = max(int(len(valid_idx)) for _, valid_idx in profiled_splits)
    train_buffer = np.empty((max_inner_train_rows, n_features), dtype=np.float64)
    valid_buffer = np.empty((max_inner_valid_rows, n_features), dtype=np.float64)
    group_stats = _build_group_sufficient_statistics(
        x_train=x_train_array,
        groups=groups_array,
    )

    candidate_scores = np.empty(
        (profiled_candidate_count, profiled_inner_fold_count), dtype=np.float64
    )
    candidate_fit_times = np.empty(
        (profiled_candidate_count, profiled_inner_fold_count), dtype=np.float64
    )
    candidate_score_times = np.empty(
        (profiled_candidate_count, profiled_inner_fold_count), dtype=np.float64
    )

    split_scale_seconds = 0.0
    candidate_fit_seconds = 0.0
    candidate_predict_seconds = 0.0
    inner_tuning_start = perf_counter()

    for split_index, (inner_train_idx, inner_valid_idx) in enumerate(profiled_splits):
        inner_train_idx_array = np.asarray(inner_train_idx, dtype=np.int64)
        inner_valid_idx_array = np.asarray(inner_valid_idx, dtype=np.int64)
        train_group_ids = (
            np.unique(groups_array[inner_train_idx_array]).astype(str, copy=False).tolist()
        )

        split_scale_start = perf_counter()
        mean, scale = compute_standard_scaler_stats_from_group_summaries(
            train_group_ids=train_group_ids,
            group_stats=group_stats,
            n_features=n_features,
        )
        x_train_scaled = _load_and_scale_rows(
            x_train=x_train_array,
            row_indices=inner_train_idx_array,
            mean=mean,
            scale=scale,
            buffer=train_buffer,
        )
        x_valid_scaled = _load_and_scale_rows(
            x_train=x_train_array,
            row_indices=inner_valid_idx_array,
            mean=mean,
            scale=scale,
            buffer=valid_buffer,
        )
        split_scale_seconds += float(perf_counter() - split_scale_start)

        y_inner_train = y_train_array[inner_train_idx_array]
        y_inner_valid = y_train_array[inner_valid_idx_array]
        sample_weights = _resolve_sample_weights(
            y_labels=y_inner_train,
            class_weight=getattr(model_template, "class_weight", None),
        )

        fit_start = perf_counter()
        factorization_state = build_ridge_exact_alpha_factorization_state(
            x_train=x_train_scaled,
            y_train=y_inner_train,
            fit_intercept=bool(getattr(model_template, "fit_intercept", True)),
            sample_weights=sample_weights,
        )
        weight_batch, intercept_batch = solve_ridge_exact_alpha_batch(
            state=factorization_state,
            alphas=profiled_alphas,
        )
        fit_seconds_total = float(perf_counter() - fit_start)
        candidate_fit_seconds += float(fit_seconds_total)

        predict_start = perf_counter()
        prediction_batch, _ = predict_ridge_labels_for_alpha_batch(
            x_eval=x_valid_scaled,
            weight_batch=weight_batch,
            intercept_batch=intercept_batch,
            classes=factorization_state.classes,
            binary_mode=bool(factorization_state.binary_mode),
        )
        predict_seconds_total = float(perf_counter() - predict_start)
        candidate_predict_seconds += float(predict_seconds_total)

        fit_seconds_per_candidate = float(fit_seconds_total) / float(profiled_candidate_count)
        predict_seconds_per_candidate = float(predict_seconds_total) / float(profiled_candidate_count)

        for candidate_index in range(profiled_candidate_count):
            predictions = np.asarray(prediction_batch[candidate_index]).astype(str, copy=False)
            score_value = classification_metric_score(
                y_true=y_inner_valid,
                y_pred=predictions,
                metric_name=primary_metric_name,
            )
            candidate_scores[candidate_index, split_index] = float(score_value)
            candidate_fit_times[candidate_index, split_index] = float(fit_seconds_per_candidate)
            candidate_score_times[candidate_index, split_index] = float(
                predict_seconds_per_candidate
            )

    measured_inner_tuning_seconds = float(perf_counter() - inner_tuning_start)

    mean_scores = np.mean(candidate_scores, axis=1, dtype=np.float64)
    std_scores = np.std(candidate_scores, axis=1, dtype=np.float64)
    mean_fit_time = np.mean(candidate_fit_times, axis=1, dtype=np.float64)
    std_fit_time = np.std(candidate_fit_times, axis=1, dtype=np.float64)
    mean_score_time = np.mean(candidate_score_times, axis=1, dtype=np.float64)
    std_score_time = np.std(candidate_score_times, axis=1, dtype=np.float64)

    best_candidate_index = int(np.argmax(mean_scores))
    best_params = dict(profiled_params[best_candidate_index])
    best_score = float(mean_scores[best_candidate_index])

    refit_start = perf_counter()
    best_estimator = clone(pipeline_template)
    best_estimator.set_params(**best_params)
    best_estimator.fit(x_train_array, y_train_array)
    refit_elapsed_seconds = float(perf_counter() - refit_start)

    tuned_search_elapsed_seconds = float(measured_inner_tuning_seconds + refit_elapsed_seconds)
    estimated_full_inner_tuning_seconds = float(measured_inner_tuning_seconds)
    tuning_extrapolation_applied = bool(
        profiled_inner_fold_count != configured_inner_fold_count
        or profiled_candidate_count != configured_candidate_count
    )
    if tuning_extrapolation_applied:
        estimated_full_inner_tuning_seconds *= float(configured_inner_fold_count) / float(
            max(profiled_inner_fold_count, 1)
        )
        estimated_full_inner_tuning_seconds *= float(configured_candidate_count) / float(
            max(profiled_candidate_count, 1)
        )
    estimated_full_tuned_search_seconds = float(
        estimated_full_inner_tuning_seconds + refit_elapsed_seconds
    )

    cv_results = {
        "params": [dict(params) for params in profiled_params],
        "mean_test_score": [float(value) for value in mean_scores.tolist()],
        "std_test_score": [float(value) for value in std_scores.tolist()],
        "mean_fit_time": [float(value) for value in mean_fit_time.tolist()],
        "std_fit_time": [float(value) for value in std_fit_time.tolist()],
        "mean_score_time": [float(value) for value in mean_score_time.tolist()],
        "std_score_time": [float(value) for value in std_score_time.tolist()],
        "rank_test_score": [int(value) for value in _stable_descending_rank(mean_scores).tolist()],
    }

    return RidgeGroupedNestedTuningResult(
        best_estimator=best_estimator,
        best_params=best_params,
        best_score=float(best_score),
        cv_results=cv_results,
        tuned_search_elapsed_seconds=float(tuned_search_elapsed_seconds),
        configured_candidate_count=int(configured_candidate_count),
        profiled_candidate_count=int(profiled_candidate_count),
        configured_inner_fold_count=int(configured_inner_fold_count),
        profiled_inner_fold_count=int(profiled_inner_fold_count),
        measured_inner_tuning_seconds=float(measured_inner_tuning_seconds),
        estimated_full_inner_tuning_seconds=float(estimated_full_inner_tuning_seconds),
        estimated_full_tuned_search_seconds=float(estimated_full_tuned_search_seconds),
        tuning_extrapolation_applied=bool(tuning_extrapolation_applied),
        split_scale_seconds=float(split_scale_seconds),
        candidate_fit_seconds=float(candidate_fit_seconds),
        candidate_predict_seconds=float(candidate_predict_seconds),
        refit_elapsed_seconds=float(refit_elapsed_seconds),
    )


__all__ = [
    "RidgeGroupedNestedTuningResult",
    "SPECIALIZED_RIDGE_TUNING_EXECUTOR_ID",
    "is_specialized_ridge_grouped_nested_supported",
    "run_specialized_ridge_grouped_nested_tuning",
]
