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
from sklearn.pipeline import Pipeline

from Thesis_ML.config.metric_policy import classification_metric_score as policy_metric_score
from Thesis_ML.experiments.progress import ProgressCallback, emit_progress

PermutationExecutionMode = Literal[
    "analytic_shortcut",
    "cached_scaled_cpu",
    "cached_scaled_hybrid_gpu",
]


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


def _resolve_permutation_execution_plan(pipeline_template: Pipeline) -> PermutationExecutionPlan:
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
        mode: PermutationExecutionMode = "cached_scaled_hybrid_gpu"
    else:
        mode = "cached_scaled_cpu"
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
    progress_callback: ProgressCallback | None,
    progress_base: dict[str, Any],
) -> list[float]:
    permutation_scores: list[float] = []
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
        y_true_all: list[str] = []
        y_pred_all: list[str] = []
        for fold_cache in fold_caches:
            y_train_permuted = fold_cache.y_train.copy()
            rng.shuffle(y_train_permuted)
            estimator = clone(final_estimator_template)
            estimator.fit(fold_cache.x_train_scaled, y_train_permuted)
            pred = np.asarray(estimator.predict(fold_cache.x_test_scaled))
            y_true_all.extend(np.asarray(fold_cache.y_test).tolist())
            y_pred_all.extend(pred.tolist())

        permutation_scores.append(
            classification_metric_score(
                y_true=y_true_all,
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


def _build_permutation_payload(
    *,
    permutation_scores: list[float],
    n_permutations: int,
    metric_name: str,
    observed_metric: float,
    seed: int,
    execution_plan: PermutationExecutionPlan,
    n_folds: int,
    fold_cache_build_seconds: float,
    permutation_loop_seconds: float,
) -> dict[str, Any]:
    ge_count = sum(score >= observed_metric for score in permutation_scores)
    p_value = (ge_count + 1.0) / (n_permutations + 1.0)
    payload: dict[str, Any] = {
        "n_permutations": int(n_permutations),
        "metric_name": metric_name,
        "observed_score": float(observed_metric),
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
    progress_callback: ProgressCallback | None = None,
    progress_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if int(n_permutations) <= 0:
        raise ValueError("evaluate_permutations requires n_permutations > 0.")

    execution_plan = _resolve_permutation_execution_plan(pipeline_template)
    final_estimator = _resolve_final_estimator(pipeline_template)
    rng = np.random.default_rng(seed)
    fold_cache_build_seconds = 0.0
    permutation_loop_seconds = 0.0
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
        permutation_scores = _execute_cached_scaled_permutations(
            final_estimator_template=final_estimator,
            fold_caches=fold_caches,
            rng=rng,
            n_permutations=int(n_permutations),
            metric_name=metric_name,
            progress_callback=progress_callback,
            progress_base=progress_base,
        )
        permutation_loop_seconds = float(perf_counter() - permutation_loop_start)

    payload = _build_permutation_payload(
        permutation_scores=permutation_scores,
        n_permutations=int(n_permutations),
        metric_name=metric_name,
        observed_metric=float(observed_metric),
        seed=int(seed),
        execution_plan=execution_plan,
        n_folds=int(len(splits)),
        fold_cache_build_seconds=float(fold_cache_build_seconds),
        permutation_loop_seconds=float(permutation_loop_seconds),
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
