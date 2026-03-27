from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from Thesis_ML.config.metric_policy import classification_metric_score

SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID = "linearsvc_grouped_nested_exact_v1"


@dataclass(frozen=True)
class GroupSufficientStats:
    group_id: str
    n_samples: int
    feature_sum: np.ndarray
    feature_sum_squares: np.ndarray


@dataclass(frozen=True)
class LinearsvcGroupedNestedTuningResult:
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
    executor_id: str = SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID


def _resolve_pipeline_components(
    pipeline_template: Pipeline,
) -> tuple[StandardScaler, LinearSVC]:
    if not isinstance(pipeline_template, Pipeline):
        raise ValueError("Specialized linearsvc tuning requires a sklearn Pipeline template.")
    scaler = pipeline_template.named_steps.get("scaler")
    model = pipeline_template.named_steps.get("model")
    if not isinstance(scaler, StandardScaler):
        raise ValueError("preprocessor_not_plain_standard_scaler")
    if not bool(getattr(scaler, "with_mean", False)) or not bool(getattr(scaler, "with_std", False)):
        raise ValueError("preprocessor_not_plain_standard_scaler")
    if not isinstance(model, LinearSVC):
        raise ValueError(
            "Specialized linearsvc tuning requires pipeline step 'model' to be LinearSVC."
        )
    return scaler, model


def _resolve_candidate_params_and_values(
    param_grid: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float]]:
    candidate_params = list(ParameterGrid(param_grid))
    if not candidate_params:
        raise ValueError("Specialized linearsvc tuning requires at least one C candidate.")
    candidate_values: list[float] = []
    for params in candidate_params:
        if set(params) != {"model__C"}:
            raise ValueError(
                "Specialized linearsvc tuning only supports param_grid with key 'model__C'."
            )
        candidate_raw = params.get("model__C")
        if not isinstance(candidate_raw, (int, float)):
            raise ValueError("Specialized linearsvc tuning requires numeric C candidates.")
        candidate_values.append(float(candidate_raw))
    return candidate_params, candidate_values


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


def compute_standard_scaler_stats_from_group_summaries(
    *,
    train_group_ids: list[str],
    group_stats: dict[str, GroupSufficientStats],
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not train_group_ids:
        raise ValueError("Specialized linearsvc tuning requires non-empty train groups.")

    total_n = 0
    total_sum = np.zeros(int(n_features), dtype=np.float64)
    total_sum_squares = np.zeros(int(n_features), dtype=np.float64)
    for group_id in train_group_ids:
        stats = group_stats.get(str(group_id))
        if stats is None:
            raise ValueError(
                f"Missing group sufficient statistics for group_id '{group_id}'."
            )
        total_n += int(stats.n_samples)
        total_sum += np.asarray(stats.feature_sum, dtype=np.float64)
        total_sum_squares += np.asarray(stats.feature_sum_squares, dtype=np.float64)

    if total_n <= 0:
        raise ValueError("Specialized linearsvc tuning produced an empty inner training slice.")

    mean = total_sum / float(total_n)
    var = (total_sum_squares / float(total_n)) - (mean * mean)
    var = np.maximum(var, 0.0)
    scale = np.sqrt(var).astype(np.float64, copy=False)
    scale = np.where(scale == 0.0, 1.0, scale).astype(np.float64, copy=False)
    return mean.astype(np.float64, copy=False), scale


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


def is_specialized_linearsvc_grouped_nested_supported(
    *,
    model_name: str,
    pipeline_template: Pipeline,
    param_grid: dict[str, Any],
) -> tuple[bool, str | None]:
    if str(model_name).strip().lower() != "linearsvc":
        return False, "model_not_linearsvc"
    try:
        _resolve_pipeline_components(pipeline_template)
        _resolve_candidate_params_and_values(param_grid)
    except ValueError as exc:
        return False, str(exc)
    return True, None


def run_specialized_linearsvc_grouped_nested_tuning(
    *,
    pipeline_template: Pipeline,
    x_train: np.ndarray,
    y_train: np.ndarray,
    inner_groups: np.ndarray,
    param_grid: dict[str, Any],
    primary_metric_name: str,
    profile_inner_folds: int | None = None,
    profile_tuning_candidates: int | None = None,
) -> LinearsvcGroupedNestedTuningResult:
    _, model_template = _resolve_pipeline_components(pipeline_template)
    candidate_params, candidate_values = _resolve_candidate_params_and_values(param_grid)

    x_train_array = np.asarray(x_train, dtype=np.float64)
    y_train_array = np.asarray(y_train).astype(str, copy=False)
    groups_array = np.asarray(inner_groups).astype(str, copy=False)
    if x_train_array.ndim != 2:
        raise ValueError("Specialized linearsvc tuning requires a 2D training matrix.")
    if y_train_array.ndim != 1:
        raise ValueError("Specialized linearsvc tuning requires a 1D training label vector.")
    if x_train_array.shape[0] != y_train_array.shape[0]:
        raise ValueError("Specialized linearsvc tuning received mismatched X/y row counts.")
    if x_train_array.shape[0] != groups_array.shape[0]:
        raise ValueError("Specialized linearsvc tuning received mismatched group rows.")

    splitter = LeaveOneGroupOut()
    all_inner_splits = list(splitter.split(x_train_array, y_train_array, groups=groups_array))
    configured_inner_fold_count = int(len(all_inner_splits))
    if configured_inner_fold_count < 2:
        raise ValueError("Specialized linearsvc tuning requires at least two inner folds.")

    configured_candidate_count = int(len(candidate_values))
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
    profiled_values = candidate_values[:profiled_candidate_count]

    n_features = int(x_train_array.shape[1])
    max_inner_train_rows = max(int(len(train_idx)) for train_idx, _ in profiled_splits)
    max_inner_valid_rows = max(int(len(valid_idx)) for _, valid_idx in profiled_splits)
    train_buffer = np.empty((max_inner_train_rows, n_features), dtype=np.float64)
    valid_buffer = np.empty((max_inner_valid_rows, n_features), dtype=np.float64)
    group_stats = _build_group_sufficient_statistics(
        x_train=x_train_array,
        groups=groups_array,
    )

    candidate_scores = np.empty((profiled_candidate_count, profiled_inner_fold_count), dtype=np.float64)
    candidate_fit_times = np.empty((profiled_candidate_count, profiled_inner_fold_count), dtype=np.float64)
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

        for candidate_index, c_value in enumerate(profiled_values):
            model = clone(model_template)
            model.set_params(C=float(c_value))

            fit_start = perf_counter()
            model.fit(x_train_scaled, y_inner_train)
            fit_seconds = float(perf_counter() - fit_start)

            predict_start = perf_counter()
            predictions = np.asarray(model.predict(x_valid_scaled))
            score_seconds = float(perf_counter() - predict_start)

            score_value = classification_metric_score(
                y_true=y_inner_valid,
                y_pred=predictions,
                metric_name=primary_metric_name,
            )

            candidate_scores[candidate_index, split_index] = float(score_value)
            candidate_fit_times[candidate_index, split_index] = float(fit_seconds)
            candidate_score_times[candidate_index, split_index] = float(score_seconds)
            candidate_fit_seconds += float(fit_seconds)
            candidate_predict_seconds += float(score_seconds)

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
        estimated_full_inner_tuning_seconds = float(measured_inner_tuning_seconds)
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

    return LinearsvcGroupedNestedTuningResult(
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
    "GroupSufficientStats",
    "LinearsvcGroupedNestedTuningResult",
    "SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID",
    "compute_standard_scaler_stats_from_group_summaries",
    "is_specialized_linearsvc_grouped_nested_supported",
    "run_specialized_linearsvc_grouped_nested_tuning",
]
