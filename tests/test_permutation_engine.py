from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import Thesis_ML.experiments.metrics as metrics_module
from Thesis_ML.experiments.metrics import classification_metric_score, evaluate_permutations


def _toy_permutation_inputs() -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    x_matrix = np.asarray(
        [
            [2.0, 0.1, 0.0],
            [1.8, 0.0, 0.1],
            [-2.0, -0.1, 0.0],
            [-1.9, 0.0, -0.2],
            [2.1, 0.2, 0.1],
            [-2.2, -0.2, -0.1],
        ],
        dtype=np.float64,
    )
    y = np.asarray(["positive", "positive", "negative", "negative", "positive", "negative"])
    splits = [
        (np.asarray([0, 1, 2, 3]), np.asarray([4, 5])),
        (np.asarray([2, 3, 4, 5]), np.asarray([0, 1])),
        (np.asarray([0, 1, 4, 5]), np.asarray([2, 3])),
    ]
    return x_matrix, y, splits


def _reference_generic_loop(
    *,
    pipeline_template: Pipeline,
    x_matrix: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_permutations: int,
    metric_name: str,
    observed_metric: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    permutation_scores: list[float] = []
    for _ in range(n_permutations):
        y_true_all: list[str] = []
        y_pred_all: list[str] = []
        for train_idx, test_idx in splits:
            y_train = y[train_idx].copy()
            rng.shuffle(y_train)
            estimator = clone(pipeline_template)
            estimator.fit(x_matrix[train_idx], y_train)
            pred = estimator.predict(x_matrix[test_idx])
            y_true_all.extend(y[test_idx].tolist())
            y_pred_all.extend(np.asarray(pred).tolist())
        permutation_scores.append(
            classification_metric_score(
                y_true=y_true_all,
                y_pred=y_pred_all,
                metric_name=metric_name,
            )
        )
    ge_count = sum(score >= observed_metric for score in permutation_scores)
    p_value = (ge_count + 1.0) / (n_permutations + 1.0)
    return {
        "null_scores": permutation_scores,
        "p_value": p_value,
    }


def test_dummy_most_frequent_uses_analytic_shortcut(monkeypatch) -> None:
    x_matrix, y, splits = _toy_permutation_inputs()
    pipeline_template = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )

    def _forbid_clone(*_args, **_kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("clone() should not be called for analytic dummy shortcut.")

    monkeypatch.setattr(metrics_module, "clone", _forbid_clone)

    payload = evaluate_permutations(
        pipeline_template=pipeline_template,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=7,
        n_permutations=9,
        metric_name="balanced_accuracy",
        observed_metric=0.625,
    )

    assert payload["execution_mode"] == "analytic_shortcut"
    assert payload["shortcut_applied"] is True
    assert payload["shortcut_reason"] == "dummy_label_count_invariant"
    assert payload["shortcut_strategy"] == "most_frequent"
    assert payload["null_scores"] == [0.625] * 9
    assert payload["p_value"] == 1.0


def test_dummy_prior_uses_analytic_shortcut() -> None:
    x_matrix, y, splits = _toy_permutation_inputs()
    pipeline_template = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )

    payload = evaluate_permutations(
        pipeline_template=pipeline_template,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=11,
        n_permutations=5,
        metric_name="balanced_accuracy",
        observed_metric=0.5,
    )

    assert payload["execution_mode"] == "analytic_shortcut"
    assert payload["shortcut_applied"] is True
    assert payload["shortcut_strategy"] == "prior"
    assert payload["null_scores"] == [0.5] * 5


def test_dummy_stratified_does_not_use_shortcut() -> None:
    x_matrix, y, splits = _toy_permutation_inputs()
    pipeline_template = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", DummyClassifier(strategy="stratified", random_state=13)),
        ]
    )

    payload = evaluate_permutations(
        pipeline_template=pipeline_template,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=13,
        n_permutations=4,
        metric_name="balanced_accuracy",
        observed_metric=0.5,
    )

    assert payload["execution_mode"] == "cached_scaled_cpu"
    assert payload["shortcut_applied"] is False
    assert payload["shortcut_strategy"] is None
    assert len(payload["null_scores"]) == 4


def test_non_dummy_cpu_cached_path_matches_reference_generic_loop() -> None:
    x_matrix, y, splits = _toy_permutation_inputs()
    pipeline_template = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", RidgeClassifier(random_state=17)),
        ]
    )

    observed_metric = 0.5
    n_permutations = 12
    metric_name = "balanced_accuracy"
    seed = 17

    payload = evaluate_permutations(
        pipeline_template=pipeline_template,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=seed,
        n_permutations=n_permutations,
        metric_name=metric_name,
        observed_metric=observed_metric,
    )
    reference = _reference_generic_loop(
        pipeline_template=pipeline_template,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=seed,
        n_permutations=n_permutations,
        metric_name=metric_name,
        observed_metric=observed_metric,
    )

    assert payload["execution_mode"] == "cached_scaled_cpu"
    np.testing.assert_allclose(payload["null_scores"], reference["null_scores"], atol=0.0, rtol=0.0)
    assert payload["p_value"] == reference["p_value"]


class _FakeTorchPermutationEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, backend_id: str = "torch_fake_gpu_v1") -> None:
        self.backend_id = backend_id

    def fit(self, x_matrix: np.ndarray, y_labels: np.ndarray) -> "_FakeTorchPermutationEstimator":
        labels = np.asarray(y_labels)
        classes = np.unique(labels)
        self.classes_ = classes
        self._prediction_label_ = str(classes[0])
        return self

    def predict(self, x_matrix: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x_matrix)
        return np.asarray([self._prediction_label_] * int(x_array.shape[0]))


def test_torch_gpu_capable_estimator_routes_to_cached_scaled_hybrid_gpu_mode() -> None:
    x_matrix, y, splits = _toy_permutation_inputs()
    pipeline_template = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", _FakeTorchPermutationEstimator()),
        ]
    )

    payload = evaluate_permutations(
        pipeline_template=pipeline_template,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=21,
        n_permutations=3,
        metric_name="balanced_accuracy",
        observed_metric=0.5,
    )

    assert payload["execution_mode"] == "cached_scaled_hybrid_gpu"
    assert payload["backend_family"] == "torch_gpu"
    assert payload["shortcut_applied"] is False


def test_permutation_payload_preserves_backward_compatible_fields() -> None:
    x_matrix, y, splits = _toy_permutation_inputs()
    pipeline_template = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", RidgeClassifier(random_state=33)),
        ]
    )

    payload = evaluate_permutations(
        pipeline_template=pipeline_template,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=33,
        n_permutations=6,
        metric_name="balanced_accuracy",
        observed_metric=0.5,
    )

    required_existing_fields = {
        "n_permutations",
        "metric_name",
        "observed_score",
        "p_value",
        "null_summary",
        "null_scores",
        "observed_metric",
        "permutation_metric_mean",
        "permutation_metric_std",
        "permutation_p_value",
    }
    assert required_existing_fields <= set(payload)
    assert {"mean", "std", "min", "max", "q25", "q50", "q75"} <= set(payload["null_summary"])
