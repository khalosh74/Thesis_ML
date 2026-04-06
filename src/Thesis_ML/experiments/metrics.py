from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline

from Thesis_ML.config.metric_policy import classification_metric_score as policy_metric_score
from Thesis_ML.experiments.backends.ridge_exact_kernels import (
    build_ridge_cpu_permutation_core_state,
    solve_ridge_cpu_permutation_core_batch,
)
from Thesis_ML.experiments.backends.torch_ridge import (
    build_ridge_gpu_permutation_fold_state,
    solve_ridge_gpu_permutation_batch,
    supports_ridge_gpu_batched_dual,
)
from Thesis_ML.experiments.progress import ProgressCallback, emit_progress

PermutationExecutionMode = Literal[
    "analytic_shortcut",
    "cached_scaled_cpu",
    "cached_scaled_hybrid_gpu",
    "ridge_cpu_exact_alpha_batched",
    "ridge_gpu_batched_dual",
]
PRIMARY_METRIC_AGGREGATIONS = frozenset({"mean_fold_scores", "pooled_held_out_predictions"})


@dataclass(frozen=True)
class PermutationFoldCache:
    fold_index: int
    x_train_scaled: np.ndarray
    x_test_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class PermutationExecutionPlan:
    execution_mode: PermutationExecutionMode
    estimator_name: str
    backend_family: str
    shortcut_applied: bool
    shortcut_reason: str | None
    shortcut_strategy: str | None


@dataclass(frozen=True)
class BalancedAccuracyBatchCache:
    present_class_indices: np.ndarray
    present_class_counts: np.ndarray
    class_row_indices: tuple[np.ndarray, ...]


def classification_metric_score(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    metric_name: str,
) -> float:
    return policy_metric_score(y_true=y_true, y_pred=y_pred, metric_name=metric_name)


def scores_for_predictions(estimator: Pipeline, x_test: np.ndarray) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {
        "decision_value": [pd.NA] * len(x_test),
        "decision_vector": [pd.NA] * len(x_test),
        "proba_value": [pd.NA] * len(x_test),
        "proba_vector": [pd.NA] * len(x_test),
    }

    if hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(x_test)
        decision_array = np.asarray(decision)
        if decision_array.ndim == 1:
            result["decision_value"] = decision_array.astype(float).tolist()
        else:
            result["decision_vector"] = [json.dumps(row.tolist()) for row in decision_array]

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x_test)
        proba_array = np.asarray(proba)
        result["proba_value"] = proba_array.max(axis=1).astype(float).tolist()
        result["proba_vector"] = [json.dumps(row.tolist()) for row in proba_array]

    return result


def _supports_balanced_accuracy_batch_fast_path(
    *,
    metric_name: str,
    primary_metric_aggregation: str,
) -> bool:
    return (
        str(metric_name).strip().lower() == "balanced_accuracy"
        and str(primary_metric_aggregation).strip().lower() == "mean_fold_scores"
    )


def _build_balanced_accuracy_batch_cache(
    *,
    y_true: np.ndarray,
    classes: np.ndarray,
) -> BalancedAccuracyBatchCache:
    class_lookup = {str(label): int(index) for index, label in enumerate(np.asarray(classes))}
    y_true_text = np.asarray(y_true).astype(str, copy=False)
    y_true_indices = np.asarray(
        [class_lookup[str(label)] for label in y_true_text],
        dtype=np.int64,
    )
    present_class_indices, present_class_counts = np.unique(y_true_indices, return_counts=True)
    class_row_indices = tuple(
        np.flatnonzero(y_true_indices == int(class_index)).astype(np.int64, copy=False)
        for class_index in np.asarray(present_class_indices, dtype=np.int64)
    )
    return BalancedAccuracyBatchCache(
        present_class_indices=np.asarray(present_class_indices, dtype=np.int64),
        present_class_counts=np.asarray(present_class_counts, dtype=np.float64),
        class_row_indices=class_row_indices,
    )


def _score_balanced_accuracy_batch(
    *,
    predicted_label_indices_batch: np.ndarray,
    cache: BalancedAccuracyBatchCache,
) -> np.ndarray:
    prediction_batch = np.asarray(predicted_label_indices_batch, dtype=np.int64)
    if prediction_batch.ndim == 1:
        prediction_batch = prediction_batch.reshape(-1, 1)
    recalls_sum = np.zeros(int(prediction_batch.shape[1]), dtype=np.float64)
    for class_index, class_count, row_indices in zip(
        np.asarray(cache.present_class_indices, dtype=np.int64),
        np.asarray(cache.present_class_counts, dtype=np.float64),
        cache.class_row_indices,
        strict=True,
    ):
        if int(row_indices.size) == 0:
            continue
        recalls_sum += (
            np.sum(prediction_batch[row_indices, :] == int(class_index), axis=0, dtype=np.int64)
            / float(class_count)
        )
    if int(cache.present_class_indices.shape[0]) == 0:
        return recalls_sum
    return recalls_sum / float(cache.present_class_indices.shape[0])


def _resolve_final_estimator(pipeline_template: Pipeline) -> Any:
    if not isinstance(pipeline_template, Pipeline):
        raise TypeError("evaluate_permutations expects pipeline_template to be a sklearn Pipeline.")
    if not pipeline_template.steps:
        raise ValueError("evaluate_permutations received an empty pipeline template.")
    if "model" in pipeline_template.named_steps:
        return pipeline_template.named_steps["model"]
    return pipeline_template.steps[-1][1]


def _resolve_estimator_name(estimator: Any) -> str:
    return type(estimator).__name__


def _resolve_backend_family(estimator: Any) -> str:
    backend_id = getattr(estimator, "backend_id", None)
    if isinstance(backend_id, str):
        normalized_backend_id = backend_id.strip().lower()
        if "torch" in normalized_backend_id or "gpu" in normalized_backend_id:
            return "torch_gpu"
    module_name = str(type(estimator).__module__).strip().lower()
    class_name = str(type(estimator).__name__).strip().lower()
    if "thesis_ml.experiments.backends.torch" in module_name:
        return "torch_gpu"
    if class_name.startswith("torch"):
        return "torch_gpu"
    return "sklearn_cpu"


def _resolve_dummy_shortcut_strategy(estimator: Any) -> str | None:
    if not isinstance(estimator, DummyClassifier):
        return None
    strategy = str(getattr(estimator, "strategy", "")).strip().lower()
    if strategy in {"most_frequent", "prior"}:
        return strategy
    return None


def _binary_fold_targets_supported(
    *,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> bool:
    y_array = np.asarray(y).astype(str, copy=False)
    if np.unique(y_array).shape[0] != 2:
        return False
    for train_idx, _ in splits:
        train_idx_array = np.asarray(train_idx, dtype=np.int64)
        if train_idx_array.size == 0:
            return False
        if np.unique(y_array[train_idx_array]).shape[0] != 2:
            return False
    return True


def _non_constant_fold_targets_supported(
    *,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> bool:
    y_array = np.asarray(y).astype(str, copy=False)
    for train_idx, _ in splits:
        train_idx_array = np.asarray(train_idx, dtype=np.int64)
        if train_idx_array.size == 0:
            return False
        if np.unique(y_array[train_idx_array]).shape[0] < 2:
            return False
    return True


def _supports_ridge_cpu_exact_alpha_batched(estimator: Any) -> tuple[bool, str | None]:
    if not isinstance(estimator, RidgeClassifier):
        return False, "estimator_not_sklearn_ridge"

    alpha_raw = getattr(estimator, "alpha", None)
    if not isinstance(alpha_raw, (int, float)):
        return False, "ridge_alpha_must_be_numeric"
    alpha = float(alpha_raw)
    if not np.isfinite(alpha) or alpha < 0.0:
        return False, "ridge_alpha_must_be_finite_and_non_negative"

    fit_intercept_raw = getattr(estimator, "fit_intercept", None)
    if not isinstance(fit_intercept_raw, (bool, np.bool_)):
        return False, "ridge_fit_intercept_must_be_boolean"

    class_weight = getattr(estimator, "class_weight", None)
    if class_weight is not None:
        normalized_class_weight = str(class_weight).strip().lower()
        if normalized_class_weight not in {"none", ""}:
            return False, "ridge_cpu_exact_alpha_batched_requires_class_weight_none"

    return True, None


def _resolve_permutation_execution_plan(
    pipeline_template: Pipeline,
    *,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> PermutationExecutionPlan:
    final_estimator = _resolve_final_estimator(pipeline_template)
    estimator_name = _resolve_estimator_name(final_estimator)
    backend_family = _resolve_backend_family(final_estimator)
    shortcut_strategy = _resolve_dummy_shortcut_strategy(final_estimator)
    if shortcut_strategy is not None:
        return PermutationExecutionPlan(
            execution_mode="analytic_shortcut",
            estimator_name=estimator_name,
            backend_family=backend_family,
            shortcut_applied=True,
            shortcut_reason="dummy_label_count_invariant",
            shortcut_strategy=shortcut_strategy,
        )
    if backend_family == "torch_gpu":
        ridge_specialized_supported, _ = supports_ridge_gpu_batched_dual(final_estimator)
        if ridge_specialized_supported and _binary_fold_targets_supported(y=y, splits=splits):
            mode: PermutationExecutionMode = "ridge_gpu_batched_dual"
        else:
            mode = "cached_scaled_hybrid_gpu"
    else:
        ridge_cpu_specialized_supported, _ = _supports_ridge_cpu_exact_alpha_batched(
            final_estimator
        )
        mode = (
            "ridge_cpu_exact_alpha_batched"
            if ridge_cpu_specialized_supported and _non_constant_fold_targets_supported(y=y, splits=splits)
            else "cached_scaled_cpu"
        )
    return PermutationExecutionPlan(
        execution_mode=mode,
        estimator_name=estimator_name,
        backend_family=backend_family,
        shortcut_applied=False,
        shortcut_reason=None,
        shortcut_strategy=None,
    )


def _build_permutation_fold_cache(
    *,
    pipeline_template: Pipeline,
    x_matrix: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[PermutationFoldCache]:
    scaler_template = pipeline_template.named_steps.get("scaler")
    fold_caches: list[PermutationFoldCache] = []
    for fold_index, (train_idx, test_idx) in enumerate(splits):
        train_idx_array = np.asarray(train_idx, dtype=np.int64)
        test_idx_array = np.asarray(test_idx, dtype=np.int64)
        x_train = np.asarray(x_matrix[train_idx_array])
        x_test = np.asarray(x_matrix[test_idx_array])
        y_train = np.asarray(y[train_idx_array]).copy()
        y_test = np.asarray(y[test_idx_array]).copy()

        if scaler_template is not None:
            scaler = clone(scaler_template)
            if hasattr(scaler, "fit_transform"):
                x_train_scaled = np.asarray(scaler.fit_transform(x_train))
            else:
                scaler.fit(x_train)
                x_train_scaled = np.asarray(scaler.transform(x_train))
            x_test_scaled = np.asarray(scaler.transform(x_test))
        else:
            x_train_scaled = x_train
            x_test_scaled = x_test
        x_train_scaled = np.ascontiguousarray(x_train_scaled)
        x_test_scaled = np.ascontiguousarray(x_test_scaled)

        fold_caches.append(
            PermutationFoldCache(
                fold_index=int(fold_index),
                x_train_scaled=x_train_scaled,
                x_test_scaled=x_test_scaled,
                y_train=y_train,
                y_test=y_test,
            )
        )
    return fold_caches


def _execute_cached_scaled_permutations(
    *,
    final_estimator_template: Any,
    fold_caches: list[PermutationFoldCache],
    rng: np.random.Generator,
    n_permutations: int,
    metric_name: str,
    primary_metric_aggregation: str,
    progress_callback: ProgressCallback | None,
    progress_base: dict[str, Any],
) -> list[float]:
    permutation_scores: list[float] = []
    fold_test_label_lists: list[list[str]] = [
        np.asarray(fold_cache.y_test).tolist() for fold_cache in fold_caches
    ]
    y_true_all_constant: list[str] = [
        str(label) for labels in fold_test_label_lists for label in labels
    ]
    for permutation_index in range(int(n_permutations)):
        emit_progress(
            progress_callback,
            stage="permutation",
            message=f"running permutation {permutation_index + 1}/{n_permutations}",
            completed_units=float(permutation_index),
            total_units=float(n_permutations),
            metadata={
                **progress_base,
                "permutation_index": int(permutation_index + 1),
                "n_permutations": int(n_permutations),
            },
        )
        y_pred_all: list[str] = []
        fold_scores: list[float] = []
        for fold_cache in fold_caches:
            y_train_permuted = fold_cache.y_train.copy()
            rng.shuffle(y_train_permuted)
            estimator = clone(final_estimator_template)
            estimator.fit(fold_cache.x_train_scaled, y_train_permuted)
            pred = np.asarray(estimator.predict(fold_cache.x_test_scaled))
            predicted_labels = pred.tolist()
            if primary_metric_aggregation == "mean_fold_scores":
                fold_scores.append(
                    float(
                        classification_metric_score(
                            y_true=fold_cache.y_test,
                            y_pred=predicted_labels,
                            metric_name=metric_name,
                        )
                    )
                )
            else:
                y_pred_all.extend(predicted_labels)

        if primary_metric_aggregation == "mean_fold_scores":
            permutation_scores.append(float(np.mean(fold_scores)))
        else:
            permutation_scores.append(
                classification_metric_score(
                    y_true=y_true_all_constant,
                    y_pred=y_pred_all,
                    metric_name=metric_name,
                )
            )
        emit_progress(
            progress_callback,
            stage="permutation",
            message=f"finished permutation {permutation_index + 1}/{n_permutations}",
            completed_units=float(permutation_index + 1),
            total_units=float(n_permutations),
            metadata={
                **progress_base,
                "permutation_index": int(permutation_index + 1),
                "n_permutations": int(n_permutations),
            },
        )
    return permutation_scores


def _execute_ridge_cpu_exact_alpha_batched_permutations(
    *,
    final_estimator_template: Any,
    fold_caches: list[PermutationFoldCache],
    rng: np.random.Generator,
    n_permutations: int,
    metric_name: str,
    primary_metric_aggregation: str,
    progress_callback: ProgressCallback | None,
    progress_base: dict[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    batch_size = max(1, int(getattr(final_estimator_template, "permutation_batch_size", 64)))

    fold_states = []
    fold_cpu_state_build_seconds = 0.0
    fold_factorization_seconds = 0.0
    for fold_cache in fold_caches:
        fold_state, timing_metadata = build_ridge_cpu_permutation_core_state(
            x_train_scaled=np.asarray(fold_cache.x_train_scaled, dtype=np.float64),
            x_test_scaled=np.asarray(fold_cache.x_test_scaled, dtype=np.float64),
            y_train=np.asarray(fold_cache.y_train),
            y_test=np.asarray(fold_cache.y_test),
            alpha=float(getattr(final_estimator_template, "alpha", 1.0)),
            fit_intercept=bool(getattr(final_estimator_template, "fit_intercept", True)),
        )
        fold_states.append(fold_state)
        fold_cpu_state_build_seconds += float(
            timing_metadata.get("fold_cpu_state_build_seconds", 0.0)
        )
        fold_factorization_seconds += float(timing_metadata.get("fold_factorization_seconds", 0.0))

    y_true_all_constant: list[str] = [
        str(label) for fold_state in fold_states for label in np.asarray(fold_state["y_test"]).tolist()
    ]
    permutation_scores: list[float] = []
    batched_target_encode_seconds = 0.0
    batched_solve_seconds = 0.0
    batched_predict_seconds = 0.0
    batched_metric_seconds = 0.0
    use_balanced_accuracy_batch_fast_path = _supports_balanced_accuracy_batch_fast_path(
        metric_name=metric_name,
        primary_metric_aggregation=primary_metric_aggregation,
    )
    if use_balanced_accuracy_batch_fast_path:
        for fold_state in fold_states:
            fold_state["balanced_accuracy_batch_cache"] = _build_balanced_accuracy_batch_cache(
                y_true=np.asarray(fold_state["y_test"]).astype(str, copy=False),
                classes=np.asarray(fold_state["classes"]).astype(str, copy=False),
            )

    permutation_index = 0
    while permutation_index < int(n_permutations):
        current_batch_size = min(batch_size, int(n_permutations) - permutation_index)
        permutation_index_batches: list[np.ndarray] = []
        for fold_state in fold_states:
            n_train = int(fold_state["n_train"])
            index_batch = np.broadcast_to(
                np.arange(n_train, dtype=np.int64)[:, None],
                (n_train, int(current_batch_size)),
            ).copy()
            permutation_index_batches.append(index_batch)

        for local_perm_index in range(int(current_batch_size)):
            for fold_index in range(len(fold_states)):
                rng.shuffle(permutation_index_batches[fold_index][:, local_perm_index])

        prediction_batches_by_fold: list[np.ndarray] = []
        prediction_label_index_batches_by_fold: list[np.ndarray] = []
        for fold_index, fold_state in enumerate(fold_states):
            permuted_label_indices_batch = np.asarray(
                fold_state["y_train_label_indices"],
                dtype=np.int64,
            )[permutation_index_batches[fold_index]]
            prediction_batch, prediction_label_index_batch, timing_metadata = (
                solve_ridge_cpu_permutation_core_batch(
                    state=fold_state,
                    permuted_train_label_indices_batch=permuted_label_indices_batch,
                    return_string_predictions=not use_balanced_accuracy_batch_fast_path,
                )
            )
            if prediction_batch is not None:
                prediction_batches_by_fold.append(
                    np.asarray(prediction_batch).astype(str, copy=False)
                )
            prediction_label_index_batches_by_fold.append(
                np.asarray(prediction_label_index_batch, dtype=np.int64)
            )
            batched_target_encode_seconds += float(
                timing_metadata.get("batched_target_encode_seconds", 0.0)
            )
            batched_solve_seconds += float(timing_metadata.get("batched_solve_seconds", 0.0))
            batched_predict_seconds += float(timing_metadata.get("batched_predict_seconds", 0.0))

        metric_start = perf_counter()
        if use_balanced_accuracy_batch_fast_path:
            batch_score_array = np.zeros(int(current_batch_size), dtype=np.float64)
            for fold_index, fold_state in enumerate(fold_states):
                batch_score_array += _score_balanced_accuracy_batch(
                    predicted_label_indices_batch=prediction_label_index_batches_by_fold[
                        fold_index
                    ],
                    cache=fold_state["balanced_accuracy_batch_cache"],
                )
            batch_scores = (batch_score_array / float(len(fold_states))).tolist()
        else:
            batch_scores = []
            for local_perm_index in range(int(current_batch_size)):
                y_pred_all: list[str] = []
                per_fold_scores: list[float] = []
                for fold_index, fold_state in enumerate(fold_states):
                    predicted_labels = (
                        np.asarray(prediction_batches_by_fold[fold_index][:, local_perm_index])
                        .astype(str, copy=False)
                        .tolist()
                    )
                    if primary_metric_aggregation == "mean_fold_scores":
                        per_fold_scores.append(
                            float(
                                classification_metric_score(
                                    y_true=fold_state["y_test"],
                                    y_pred=predicted_labels,
                                    metric_name=metric_name,
                                )
                            )
                        )
                    else:
                        y_pred_all.extend(predicted_labels)
                if primary_metric_aggregation == "mean_fold_scores":
                    batch_scores.append(float(np.mean(per_fold_scores)))
                else:
                    batch_scores.append(
                        classification_metric_score(
                            y_true=y_true_all_constant,
                            y_pred=y_pred_all,
                            metric_name=metric_name,
                        )
                    )
        batched_metric_seconds += float(perf_counter() - metric_start)

        for local_perm_index, score in enumerate(batch_scores):
            absolute_permutation_index = permutation_index + local_perm_index
            emit_progress(
                progress_callback,
                stage="permutation",
                message=f"running permutation {absolute_permutation_index + 1}/{n_permutations}",
                completed_units=float(absolute_permutation_index),
                total_units=float(n_permutations),
                metadata={
                    **progress_base,
                    "permutation_index": int(absolute_permutation_index + 1),
                    "n_permutations": int(n_permutations),
                },
            )
            permutation_scores.append(float(score))
            emit_progress(
                progress_callback,
                stage="permutation",
                message=f"finished permutation {absolute_permutation_index + 1}/{n_permutations}",
                completed_units=float(absolute_permutation_index + 1),
                total_units=float(n_permutations),
                metadata={
                    **progress_base,
                    "permutation_index": int(absolute_permutation_index + 1),
                    "n_permutations": int(n_permutations),
                },
            )

        permutation_index += int(current_batch_size)

    return permutation_scores, {
        "specialized_ridge_cpu_path_used": True,
        "permutation_batch_size": int(batch_size),
        "fold_cpu_state_build_seconds": float(fold_cpu_state_build_seconds),
        "fold_factorization_seconds": float(fold_factorization_seconds),
        "batched_target_encode_seconds": float(batched_target_encode_seconds),
        "batched_solve_seconds": float(batched_solve_seconds),
        "batched_predict_seconds": float(batched_predict_seconds),
        "batched_metric_seconds": float(batched_metric_seconds),
    }


def _execute_ridge_gpu_batched_dual_permutations(
    *,
    final_estimator_template: Any,
    fold_caches: list[PermutationFoldCache],
    rng: np.random.Generator,
    n_permutations: int,
    metric_name: str,
    primary_metric_aggregation: str,
    progress_callback: ProgressCallback | None,
    progress_base: dict[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    batch_size = max(1, int(getattr(final_estimator_template, "permutation_batch_size", 16)))

    fold_states = []
    fold_gpu_state_build_seconds = 0.0
    fold_factorization_seconds = 0.0
    for fold_cache in fold_caches:
        fold_state, timing_metadata = build_ridge_gpu_permutation_fold_state(
            fold_index=int(fold_cache.fold_index),
            x_train_scaled=np.asarray(fold_cache.x_train_scaled, dtype=np.float64),
            x_test_scaled=np.asarray(fold_cache.x_test_scaled, dtype=np.float64),
            y_train=np.asarray(fold_cache.y_train),
            y_test=np.asarray(fold_cache.y_test),
            estimator=final_estimator_template,
            batch_size_hint=batch_size,
        )
        fold_states.append(fold_state)
        fold_gpu_state_build_seconds += float(
            timing_metadata.get("fold_gpu_state_build_seconds", 0.0)
        )
        fold_factorization_seconds += float(timing_metadata.get("fold_factorization_seconds", 0.0))

    y_true_all_constant: list[str] = [
        str(label) for fold_state in fold_states for label in np.asarray(fold_state.y_test).tolist()
    ]
    permutation_scores: list[float] = []
    batched_solve_seconds = 0.0
    batched_predict_seconds = 0.0

    permutation_index = 0
    while permutation_index < int(n_permutations):
        current_batch_size = min(batch_size, int(n_permutations) - permutation_index)
        permuted_labels_by_fold: list[np.ndarray] = []
        for fold_cache in fold_caches:
            y_train_fold = np.asarray(fold_cache.y_train)
            labels_batch = np.empty(
                (int(y_train_fold.shape[0]), int(current_batch_size)), dtype=y_train_fold.dtype
            )
            permuted_labels_by_fold.append(labels_batch)

        for local_perm_index in range(int(current_batch_size)):
            for fold_index, fold_cache in enumerate(fold_caches):
                y_train_permuted = np.asarray(fold_cache.y_train).copy()
                rng.shuffle(y_train_permuted)
                permuted_labels_by_fold[fold_index][:, local_perm_index] = y_train_permuted

        fold_score_batches: list[np.ndarray] = []
        for fold_index, fold_state in enumerate(fold_states):
            fold_score_batch, timing_metadata = solve_ridge_gpu_permutation_batch(
                state=fold_state,
                permuted_train_labels_batch=permuted_labels_by_fold[fold_index],
            )
            fold_score_batches.append(np.asarray(fold_score_batch, dtype=np.float64))
            batched_solve_seconds += float(timing_metadata.get("batched_solve_seconds", 0.0))
            batched_predict_seconds += float(timing_metadata.get("batched_predict_seconds", 0.0))

        batch_scores: list[float] = []
        for local_perm_index in range(int(current_batch_size)):
            y_pred_all: list[str] = []
            per_fold_scores: list[float] = []
            for fold_index, fold_state in enumerate(fold_states):
                fold_score_matrix = np.asarray(fold_score_batches[fold_index], dtype=np.float64)
                fold_scores_column = fold_score_matrix[:, local_perm_index]
                classes = np.asarray(fold_state.classes).astype(str, copy=False)
                predictions = np.where(
                    fold_scores_column >= 0.0,
                    str(classes[1]),
                    str(classes[0]),
                )
                predicted_labels = predictions.tolist()
                if primary_metric_aggregation == "mean_fold_scores":
                    per_fold_scores.append(
                        float(
                            classification_metric_score(
                                y_true=fold_state.y_test,
                                y_pred=predicted_labels,
                                metric_name=metric_name,
                            )
                        )
                    )
                else:
                    y_pred_all.extend(predicted_labels)
            if primary_metric_aggregation == "mean_fold_scores":
                batch_scores.append(float(np.mean(per_fold_scores)))
            else:
                batch_scores.append(
                    classification_metric_score(
                        y_true=y_true_all_constant,
                        y_pred=y_pred_all,
                        metric_name=metric_name,
                    )
                )

        for local_perm_index, score in enumerate(batch_scores):
            absolute_permutation_index = permutation_index + local_perm_index
            emit_progress(
                progress_callback,
                stage="permutation",
                message=f"running permutation {absolute_permutation_index + 1}/{n_permutations}",
                completed_units=float(absolute_permutation_index),
                total_units=float(n_permutations),
                metadata={
                    **progress_base,
                    "permutation_index": int(absolute_permutation_index + 1),
                    "n_permutations": int(n_permutations),
                },
            )
            permutation_scores.append(float(score))
            emit_progress(
                progress_callback,
                stage="permutation",
                message=f"finished permutation {absolute_permutation_index + 1}/{n_permutations}",
                completed_units=float(absolute_permutation_index + 1),
                total_units=float(n_permutations),
                metadata={
                    **progress_base,
                    "permutation_index": int(absolute_permutation_index + 1),
                    "n_permutations": int(n_permutations),
                },
            )

        permutation_index += int(current_batch_size)

    return permutation_scores, {
        "specialized_ridge_gpu_path_used": True,
        "permutation_batch_size": int(batch_size),
        "fold_gpu_state_build_seconds": float(fold_gpu_state_build_seconds),
        "fold_factorization_seconds": float(fold_factorization_seconds),
        "batched_solve_seconds": float(batched_solve_seconds),
        "batched_predict_seconds": float(batched_predict_seconds),
    }


def _build_permutation_payload(
    *,
    permutation_scores: list[float],
    n_permutations: int,
    metric_name: str,
    observed_metric: float,
    primary_metric_aggregation: str,
    seed: int,
    execution_plan: PermutationExecutionPlan,
    n_folds: int,
    fold_cache_build_seconds: float,
    permutation_loop_seconds: float,
    additional_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ge_count = sum(score >= observed_metric for score in permutation_scores)
    p_value = (ge_count + 1.0) / (n_permutations + 1.0)
    payload: dict[str, Any] = {
        "n_permutations": int(n_permutations),
        "metric_name": metric_name,
        "observed_score": float(observed_metric),
        "primary_metric_aggregation": str(primary_metric_aggregation),
        "permutation_seed": int(seed),
        "p_value": float(p_value),
        "null_summary": {
            "mean": float(np.mean(permutation_scores)),
            "std": float(np.std(permutation_scores)),
            "min": float(np.min(permutation_scores)),
            "max": float(np.max(permutation_scores)),
            "q25": float(np.quantile(permutation_scores, 0.25)),
            "q50": float(np.quantile(permutation_scores, 0.50)),
            "q75": float(np.quantile(permutation_scores, 0.75)),
        },
        "null_scores": [float(value) for value in permutation_scores],
    }
    # Backward-compatible fields retained for existing consumers.
    payload["observed_metric"] = payload["observed_score"]
    payload["permutation_metric_mean"] = payload["null_summary"]["mean"]
    payload["permutation_metric_std"] = payload["null_summary"]["std"]
    payload["permutation_p_value"] = payload["p_value"]
    # Additive metadata.
    payload["execution_mode"] = str(execution_plan.execution_mode)
    payload["backend_family"] = str(execution_plan.backend_family)
    payload["estimator_name"] = str(execution_plan.estimator_name)
    payload["shortcut_applied"] = bool(execution_plan.shortcut_applied)
    payload["shortcut_reason"] = (
        str(execution_plan.shortcut_reason) if execution_plan.shortcut_reason else None
    )
    payload["shortcut_strategy"] = (
        str(execution_plan.shortcut_strategy) if execution_plan.shortcut_strategy else None
    )
    payload["n_folds"] = int(n_folds)
    payload["fold_cache_build_seconds"] = float(fold_cache_build_seconds)
    payload["permutation_loop_seconds"] = float(permutation_loop_seconds)
    payload["specialized_ridge_cpu_path_used"] = bool(
        execution_plan.execution_mode == "ridge_cpu_exact_alpha_batched"
    )
    payload["specialized_ridge_cpu_fallback_reason"] = None
    payload["specialized_ridge_gpu_path_used"] = bool(
        execution_plan.execution_mode == "ridge_gpu_batched_dual"
    )
    payload["specialized_ridge_gpu_fallback_reason"] = None
    payload["permutation_batch_size"] = None
    payload["fold_cpu_state_build_seconds"] = None
    payload["fold_gpu_state_build_seconds"] = None
    payload["fold_factorization_seconds"] = None
    payload["batched_target_encode_seconds"] = None
    payload["batched_solve_seconds"] = None
    payload["batched_predict_seconds"] = None
    payload["batched_metric_seconds"] = None
    if additional_metadata:
        payload.update(additional_metadata)
    return payload


def evaluate_permutations(
    pipeline_template: Pipeline,
    x_matrix: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_permutations: int,
    metric_name: str,
    observed_metric: float,
    primary_metric_aggregation: str = "pooled_held_out_predictions",
    progress_callback: ProgressCallback | None = None,
    progress_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if int(n_permutations) <= 0:
        raise ValueError("evaluate_permutations requires n_permutations > 0.")
    resolved_aggregation = str(primary_metric_aggregation).strip()
    if resolved_aggregation not in PRIMARY_METRIC_AGGREGATIONS:
        raise ValueError(
            "evaluate_permutations requires primary_metric_aggregation to be "
            "'mean_fold_scores' or 'pooled_held_out_predictions'."
        )

    execution_plan = _resolve_permutation_execution_plan(
        pipeline_template,
        y=y,
        splits=splits,
    )
    final_estimator = _resolve_final_estimator(pipeline_template)
    rng = np.random.default_rng(seed)
    fold_cache_build_seconds = 0.0
    permutation_loop_seconds = 0.0
    additional_metadata: dict[str, Any] | None = None
    progress_base = dict(progress_metadata or {})
    emit_progress(
        progress_callback,
        stage="permutation",
        message=f"starting permutation test with {n_permutations} permutations",
        completed_units=0.0,
        total_units=float(n_permutations),
        metadata=progress_base,
    )

    permutation_scores: list[float]
    if execution_plan.execution_mode == "analytic_shortcut":
        permutation_scores = [float(observed_metric)] * int(n_permutations)
        for permutation_index in range(int(n_permutations)):
            emit_progress(
                progress_callback,
                stage="permutation",
                message=f"running permutation {permutation_index + 1}/{n_permutations}",
                completed_units=float(permutation_index),
                total_units=float(n_permutations),
                metadata={
                    **progress_base,
                    "permutation_index": int(permutation_index + 1),
                    "n_permutations": int(n_permutations),
                },
            )
            emit_progress(
                progress_callback,
                stage="permutation",
                message=f"finished permutation {permutation_index + 1}/{n_permutations}",
                completed_units=float(permutation_index + 1),
                total_units=float(n_permutations),
                metadata={
                    **progress_base,
                    "permutation_index": int(permutation_index + 1),
                    "n_permutations": int(n_permutations),
                },
            )
    elif execution_plan.execution_mode in {
        "cached_scaled_cpu",
        "cached_scaled_hybrid_gpu",
        "ridge_cpu_exact_alpha_batched",
    }:
        fold_cache_start = perf_counter()
        fold_caches = _build_permutation_fold_cache(
            pipeline_template=pipeline_template,
            x_matrix=x_matrix,
            y=y,
            splits=splits,
        )
        fold_cache_build_seconds = float(perf_counter() - fold_cache_start)
        permutation_loop_start = perf_counter()
        if execution_plan.execution_mode == "ridge_cpu_exact_alpha_batched":
            permutation_scores, additional_metadata = (
                _execute_ridge_cpu_exact_alpha_batched_permutations(
                    final_estimator_template=final_estimator,
                    fold_caches=fold_caches,
                    rng=rng,
                    n_permutations=int(n_permutations),
                    metric_name=metric_name,
                    primary_metric_aggregation=resolved_aggregation,
                    progress_callback=progress_callback,
                    progress_base=progress_base,
                )
            )
        else:
            permutation_scores = _execute_cached_scaled_permutations(
                final_estimator_template=final_estimator,
                fold_caches=fold_caches,
                rng=rng,
                n_permutations=int(n_permutations),
                metric_name=metric_name,
                primary_metric_aggregation=resolved_aggregation,
                progress_callback=progress_callback,
                progress_base=progress_base,
            )
        permutation_loop_seconds = float(perf_counter() - permutation_loop_start)
    else:
        fold_cache_start = perf_counter()
        fold_caches = _build_permutation_fold_cache(
            pipeline_template=pipeline_template,
            x_matrix=x_matrix,
            y=y,
            splits=splits,
        )
        fold_cache_build_seconds = float(perf_counter() - fold_cache_start)
        permutation_loop_start = perf_counter()
        try:
            permutation_scores, additional_metadata = _execute_ridge_gpu_batched_dual_permutations(
                final_estimator_template=final_estimator,
                fold_caches=fold_caches,
                rng=rng,
                n_permutations=int(n_permutations),
                metric_name=metric_name,
                primary_metric_aggregation=resolved_aggregation,
                progress_callback=progress_callback,
                progress_base=progress_base,
            )
        except ValueError as exc:
            execution_plan = PermutationExecutionPlan(
                execution_mode="cached_scaled_hybrid_gpu",
                estimator_name=execution_plan.estimator_name,
                backend_family=execution_plan.backend_family,
                shortcut_applied=execution_plan.shortcut_applied,
                shortcut_reason=execution_plan.shortcut_reason,
                shortcut_strategy=execution_plan.shortcut_strategy,
            )
            permutation_scores = _execute_cached_scaled_permutations(
                final_estimator_template=final_estimator,
                fold_caches=fold_caches,
                rng=rng,
                n_permutations=int(n_permutations),
                metric_name=metric_name,
                primary_metric_aggregation=resolved_aggregation,
                progress_callback=progress_callback,
                progress_base=progress_base,
            )
            additional_metadata = {
                "specialized_ridge_gpu_path_used": False,
                "specialized_ridge_gpu_fallback_reason": str(exc),
            }
        permutation_loop_seconds = float(perf_counter() - permutation_loop_start)

    payload = _build_permutation_payload(
        permutation_scores=permutation_scores,
        n_permutations=int(n_permutations),
        metric_name=metric_name,
        observed_metric=float(observed_metric),
        primary_metric_aggregation=resolved_aggregation,
        seed=int(seed),
        execution_plan=execution_plan,
        n_folds=int(len(splits)),
        fold_cache_build_seconds=float(fold_cache_build_seconds),
        permutation_loop_seconds=float(permutation_loop_seconds),
        additional_metadata=additional_metadata,
    )
    emit_progress(
        progress_callback,
        stage="permutation",
        message="finished permutation test",
        completed_units=float(n_permutations),
        total_units=float(n_permutations),
        metadata=progress_base,
    )
    return payload


def extract_linear_coefficients(estimator: Pipeline) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model = estimator.named_steps.get("model")
    if model is None or not hasattr(model, "coef_"):
        raise ValueError(
            "Interpretability export requires a fitted linear model with a 'coef_' attribute."
        )

    coef_array = np.asarray(model.coef_, dtype=np.float64)
    if coef_array.ndim == 1:
        coef_array = coef_array.reshape(1, -1)
    if coef_array.ndim != 2:
        raise ValueError(f"Unsupported coefficient shape for interpretability: {coef_array.shape}")

    intercept_raw = getattr(model, "intercept_", np.zeros(coef_array.shape[0], dtype=np.float64))
    intercept_array = np.asarray(intercept_raw, dtype=np.float64).reshape(-1)
    if intercept_array.size == 1 and coef_array.shape[0] > 1:
        intercept_array = np.repeat(intercept_array, coef_array.shape[0])

    classes = getattr(model, "classes_", None)
    if classes is None:
        class_labels = [f"class_{idx}" for idx in range(coef_array.shape[0])]
    else:
        class_labels = [str(value) for value in np.asarray(classes).tolist()]

    return coef_array, intercept_array, class_labels


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    if std_a == 0.0 or std_b == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_interpretability_stability(coef_vectors: list[np.ndarray]) -> dict[str, Any]:
    if not coef_vectors:
        return {
            "status": "no_coefficients",
            "n_folds": 0,
            "n_pairs": 0,
            "mean_pairwise_correlation": None,
            "mean_sign_consistency": None,
            "top_k": 0,
            "mean_top_k_overlap": None,
        }

    lengths = {int(vector.size) for vector in coef_vectors}
    if len(lengths) != 1:
        return {
            "status": "incompatible_coefficient_shapes",
            "n_folds": int(len(coef_vectors)),
            "n_pairs": 0,
            "mean_pairwise_correlation": None,
            "mean_sign_consistency": None,
            "top_k": 0,
            "mean_top_k_overlap": None,
        }

    stacked = np.vstack(coef_vectors).astype(np.float64, copy=False)
    n_folds, n_coeffs = stacked.shape
    pair_indices = list(itertools.combinations(range(n_folds), 2))

    pairwise_corrs = [
        _safe_corr(stacked[left_idx], stacked[right_idx]) for left_idx, right_idx in pair_indices
    ]
    mean_pairwise_corr = float(np.mean(pairwise_corrs)) if pairwise_corrs else None

    sign_matrix = np.sign(stacked)
    sign_consistency = np.maximum.reduce(
        [
            np.mean(sign_matrix == -1.0, axis=0),
            np.mean(sign_matrix == 0.0, axis=0),
            np.mean(sign_matrix == 1.0, axis=0),
        ]
    )
    mean_sign_consistency = float(np.mean(sign_consistency))

    top_k = int(min(100, n_coeffs))
    if top_k > 0:
        top_k_sets = [
            set(np.argpartition(np.abs(row), -top_k)[-top_k:].tolist()) for row in stacked
        ]
        top_k_overlaps = []
        for left_idx, right_idx in pair_indices:
            left = top_k_sets[left_idx]
            right = top_k_sets[right_idx]
            denom = len(left | right)
            overlap = float(len(left & right) / denom) if denom > 0 else 0.0
            top_k_overlaps.append(overlap)
        mean_top_k_overlap = float(np.mean(top_k_overlaps)) if top_k_overlaps else None
    else:
        mean_top_k_overlap = None

    return {
        "status": "ok",
        "n_folds": int(n_folds),
        "n_pairs": int(len(pair_indices)),
        "mean_pairwise_correlation": mean_pairwise_corr,
        "mean_sign_consistency": mean_sign_consistency,
        "top_k": top_k,
        "mean_top_k_overlap": mean_top_k_overlap,
    }
