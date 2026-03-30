from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import Thesis_ML.experiments.metrics as metrics_module
from Thesis_ML.experiments.metrics import classification_metric_score, evaluate_permutations
from Thesis_ML.verification.backend_parity import compare_permutation_parity


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


def _toy_permutation_inputs_unequal_folds() -> tuple[
    np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]
]:
    x_matrix = np.asarray(
        [
            [2.2, 0.1, 0.0],
            [2.0, 0.0, 0.2],
            [-2.1, -0.2, 0.0],
            [-1.9, 0.0, -0.1],
            [1.8, 0.2, 0.1],
            [-1.7, -0.2, -0.1],
            [2.4, 0.3, 0.2],
        ],
        dtype=np.float64,
    )
    y = np.asarray(["positive", "positive", "negative", "negative", "positive", "negative", "positive"])
    splits = [
        (np.asarray([2, 3, 4, 5], dtype=np.int64), np.asarray([0, 1, 6], dtype=np.int64)),
        (np.asarray([0, 1, 4, 6], dtype=np.int64), np.asarray([2, 3, 5], dtype=np.int64)),
        (np.asarray([0, 1, 2, 3, 5, 6], dtype=np.int64), np.asarray([4], dtype=np.int64)),
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
    primary_metric_aggregation: str = "pooled_held_out_predictions",
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    permutation_scores: list[float] = []
    for _ in range(n_permutations):
        y_true_all: list[str] = []
        y_pred_all: list[str] = []
        fold_scores: list[float] = []
        for train_idx, test_idx in splits:
            y_train = y[train_idx].copy()
            rng.shuffle(y_train)
            estimator = clone(pipeline_template)
            estimator.fit(x_matrix[train_idx], y_train)
            pred = estimator.predict(x_matrix[test_idx])
            if primary_metric_aggregation == "mean_fold_scores":
                fold_scores.append(
                    classification_metric_score(
                        y_true=y[test_idx],
                        y_pred=np.asarray(pred).tolist(),
                        metric_name=metric_name,
                    )
                )
            else:
                y_true_all.extend(y[test_idx].tolist())
                y_pred_all.extend(np.asarray(pred).tolist())
        if primary_metric_aggregation == "mean_fold_scores":
            permutation_scores.append(float(np.mean(np.asarray(fold_scores, dtype=np.float64))))
        else:
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


def test_permutation_null_scores_follow_selected_primary_metric_aggregation() -> None:
    x_matrix, y, splits = _toy_permutation_inputs_unequal_folds()
    pipeline_template = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", RidgeClassifier(random_state=71)),
        ]
    )
    metric_name = "accuracy"
    n_permutations = 9
    seed = 71

    pooled_payload = evaluate_permutations(
        pipeline_template=pipeline_template,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=seed,
        n_permutations=n_permutations,
        metric_name=metric_name,
        observed_metric=0.5,
        primary_metric_aggregation="pooled_held_out_predictions",
    )
    mean_fold_payload = evaluate_permutations(
        pipeline_template=pipeline_template,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=seed,
        n_permutations=n_permutations,
        metric_name=metric_name,
        observed_metric=0.5,
        primary_metric_aggregation="mean_fold_scores",
    )

    pooled_reference = _reference_generic_loop(
        pipeline_template=pipeline_template,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=seed,
        n_permutations=n_permutations,
        metric_name=metric_name,
        observed_metric=0.5,
        primary_metric_aggregation="pooled_held_out_predictions",
    )
    mean_fold_reference = _reference_generic_loop(
        pipeline_template=pipeline_template,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=seed,
        n_permutations=n_permutations,
        metric_name=metric_name,
        observed_metric=0.5,
        primary_metric_aggregation="mean_fold_scores",
    )

    assert pooled_payload["primary_metric_aggregation"] == "pooled_held_out_predictions"
    assert mean_fold_payload["primary_metric_aggregation"] == "mean_fold_scores"
    np.testing.assert_allclose(
        pooled_payload["null_scores"],
        pooled_reference["null_scores"],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        mean_fold_payload["null_scores"],
        mean_fold_reference["null_scores"],
        atol=0.0,
        rtol=0.0,
    )
    assert not np.allclose(pooled_payload["null_scores"], mean_fold_payload["null_scores"])


class _FakeTorchPermutationEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, backend_id: str = "torch_fake_gpu_v1") -> None:
        self.backend_id = backend_id

    def fit(self, x_matrix: np.ndarray, y_labels: np.ndarray) -> _FakeTorchPermutationEstimator:
        labels = np.asarray(y_labels)
        classes = np.unique(labels)
        self.classes_ = classes
        self._prediction_label_ = str(classes[0])
        return self

    def predict(self, x_matrix: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x_matrix)
        return np.asarray([self._prediction_label_] * int(x_array.shape[0]))


class _FakeTorchRidgePermutationEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        class_weight: str | None = None,
        backend_id: str = "torch_ridge_gpu_v2",
    ) -> None:
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.class_weight = class_weight
        self.backend_id = str(backend_id)

    def fit(
        self,
        x_matrix: np.ndarray,
        y_labels: np.ndarray,
    ) -> _FakeTorchRidgePermutationEstimator:
        x_array = np.asarray(x_matrix, dtype=np.float64)
        labels = np.asarray(y_labels).astype(str, copy=False)
        classes = np.unique(labels)
        if classes.shape[0] != 2:
            raise ValueError("fake ridge estimator supports binary labels only for this test.")
        encoded = np.where(labels == str(classes[1]), 1.0, -1.0).astype(np.float64)
        if bool(self.fit_intercept):
            feature_mean = np.mean(x_array, axis=0, dtype=np.float64)
            target_mean = float(encoded.mean())
            x_effective = x_array - feature_mean
            y_effective = encoded - target_mean
        else:
            feature_mean = np.zeros(x_array.shape[1], dtype=np.float64)
            target_mean = 0.0
            x_effective = x_array
            y_effective = encoded
        system = x_effective.T @ x_effective
        system = np.asarray(system, dtype=np.float64, copy=True)
        system.flat[:: system.shape[0] + 1] += float(self.alpha)
        rhs = x_effective.T @ y_effective
        coef = np.linalg.solve(system, rhs)
        intercept = float(target_mean - feature_mean @ coef) if bool(self.fit_intercept) else 0.0
        self.coef_ = np.asarray(coef, dtype=np.float64).reshape(1, -1)
        self.intercept_ = np.asarray([intercept], dtype=np.float64)
        self.classes_ = classes
        self.n_features_in_ = int(x_array.shape[1])
        return self

    def decision_function(self, x_matrix: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x_matrix, dtype=np.float64)
        return x_array @ np.asarray(self.coef_, dtype=np.float64).reshape(-1) + float(
            np.asarray(self.intercept_, dtype=np.float64)[0]
        )

    def predict(self, x_matrix: np.ndarray) -> np.ndarray:
        scores = np.asarray(self.decision_function(x_matrix), dtype=np.float64)
        classes = np.asarray(self.classes_).astype(str, copy=False)
        return np.where(scores >= 0.0, str(classes[1]), str(classes[0]))


@dataclass(frozen=True)
class _FakeRidgeGpuFoldState:
    fold_index: int
    n_train: int
    n_test: int
    classes: np.ndarray
    y_test: np.ndarray
    fit_intercept: bool
    alpha: float
    target_mean: float
    k_test: np.ndarray
    cholesky_factor: np.ndarray


def _patch_fake_ridge_gpu_specialized_primitives(monkeypatch) -> None:
    def _fake_build_fold_state(
        *,
        fold_index: int,
        x_train_scaled: np.ndarray,
        x_test_scaled: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        estimator: Any,
        batch_size_hint: int = 16,
    ) -> tuple[_FakeRidgeGpuFoldState, dict[str, float]]:
        del batch_size_hint
        x_train = np.asarray(x_train_scaled, dtype=np.float64)
        x_test = np.asarray(x_test_scaled, dtype=np.float64)
        y_train_text = np.asarray(y_train).astype(str, copy=False)
        y_test_text = np.asarray(y_test).astype(str, copy=False)
        classes = np.unique(y_train_text)
        if classes.shape[0] != 2:
            raise ValueError("fake ridge gpu fold state requires binary labels")
        encoded = np.where(y_train_text == str(classes[1]), 1.0, -1.0).astype(np.float64)
        fit_intercept = bool(getattr(estimator, "fit_intercept", True))
        if fit_intercept:
            feature_mean = np.mean(x_train, axis=0, dtype=np.float64)
            x_train_effective = x_train - feature_mean
            x_test_effective = x_test - feature_mean
            target_mean = float(encoded.mean())
        else:
            x_train_effective = x_train
            x_test_effective = x_test
            target_mean = 0.0
        k_train = x_train_effective @ x_train_effective.T
        system = np.asarray(k_train, dtype=np.float64, copy=True)
        alpha = float(getattr(estimator, "alpha", 1.0))
        system.flat[:: system.shape[0] + 1] += alpha
        cholesky_factor = np.linalg.cholesky(system)
        k_test = x_test_effective @ x_train_effective.T
        state = _FakeRidgeGpuFoldState(
            fold_index=int(fold_index),
            n_train=int(x_train.shape[0]),
            n_test=int(x_test.shape[0]),
            classes=np.asarray(classes).astype(str, copy=False),
            y_test=np.asarray(y_test_text).astype(str, copy=False),
            fit_intercept=fit_intercept,
            alpha=alpha,
            target_mean=float(target_mean),
            k_test=np.asarray(k_test, dtype=np.float64),
            cholesky_factor=np.asarray(cholesky_factor, dtype=np.float64),
        )
        return state, {
            "fold_gpu_state_build_seconds": 0.0,
            "fold_factorization_seconds": 0.0,
        }

    def _fake_solve_batch(
        *,
        state: _FakeRidgeGpuFoldState,
        permuted_train_labels_batch: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        labels_batch = np.asarray(permuted_train_labels_batch)
        if labels_batch.ndim == 1:
            labels_batch = labels_batch.reshape(-1, 1)
        encoded = np.where(
            labels_batch.astype(str, copy=False) == str(np.asarray(state.classes)[1]),
            1.0,
            -1.0,
        ).astype(np.float64)
        if bool(state.fit_intercept):
            encoded = encoded - float(state.target_mean)
        intermediate = np.linalg.solve(state.cholesky_factor, encoded)
        dual = np.linalg.solve(state.cholesky_factor.T, intermediate)
        scores = np.asarray(state.k_test @ dual, dtype=np.float64)
        if bool(state.fit_intercept):
            scores = scores + float(state.target_mean)
        return scores, {
            "batched_solve_seconds": 0.0,
            "batched_predict_seconds": 0.0,
        }

    monkeypatch.setattr(
        metrics_module,
        "build_ridge_gpu_permutation_fold_state",
        _fake_build_fold_state,
    )
    monkeypatch.setattr(
        metrics_module,
        "solve_ridge_gpu_permutation_batch",
        _fake_solve_batch,
    )


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


def test_ridge_gpu_batched_dual_routes_for_supported_torch_ridge_case(monkeypatch) -> None:
    x_matrix, y, splits = _toy_permutation_inputs()
    pipeline_template = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", _FakeTorchRidgePermutationEstimator()),
        ]
    )
    monkeypatch.setattr(
        metrics_module, "supports_ridge_gpu_batched_dual", lambda _estimator: (True, None)
    )

    execution_plan = metrics_module._resolve_permutation_execution_plan(
        pipeline_template,
        y=y,
        splits=splits,
    )

    assert execution_plan.execution_mode == "ridge_gpu_batched_dual"


def test_ridge_gpu_batched_dual_falls_back_for_unsupported_ridge_configuration() -> None:
    x_matrix, y, splits = _toy_permutation_inputs()
    pipeline_template = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", _FakeTorchRidgePermutationEstimator(class_weight="balanced")),
        ]
    )

    execution_plan = metrics_module._resolve_permutation_execution_plan(
        pipeline_template,
        y=y,
        splits=splits,
    )

    assert execution_plan.execution_mode == "cached_scaled_hybrid_gpu"


def test_ridge_gpu_batched_dual_matches_generic_cached_path_for_binary_case(monkeypatch) -> None:
    _patch_fake_ridge_gpu_specialized_primitives(monkeypatch)
    x_matrix, y, splits = _toy_permutation_inputs()
    metric_name = "balanced_accuracy"
    observed_metric = 0.0
    n_permutations = 11
    seed = 123

    specialized_payload = evaluate_permutations(
        pipeline_template=Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", _FakeTorchRidgePermutationEstimator(alpha=1.3, fit_intercept=True)),
            ]
        ),
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=seed,
        n_permutations=n_permutations,
        metric_name=metric_name,
        observed_metric=observed_metric,
    )

    monkeypatch.setattr(
        metrics_module,
        "supports_ridge_gpu_batched_dual",
        lambda _estimator: (False, "forced_generic_path_for_parity_test"),
    )
    generic_payload = evaluate_permutations(
        pipeline_template=Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", _FakeTorchRidgePermutationEstimator(alpha=1.3, fit_intercept=True)),
            ]
        ),
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=seed,
        n_permutations=n_permutations,
        metric_name=metric_name,
        observed_metric=observed_metric,
    )

    assert specialized_payload["execution_mode"] == "ridge_gpu_batched_dual"
    assert generic_payload["execution_mode"] == "cached_scaled_hybrid_gpu"
    parity = compare_permutation_parity(
        reference_payload=generic_payload,
        candidate_payload=specialized_payload,
        category="exact",
    )
    assert parity.passed is True
    np.testing.assert_allclose(
        specialized_payload["null_scores"],
        generic_payload["null_scores"],
        atol=1e-12,
        rtol=1e-12,
    )
    assert specialized_payload["p_value"] == generic_payload["p_value"]


def test_ridge_gpu_batched_dual_preserves_per_permutation_progress_events(monkeypatch) -> None:
    _patch_fake_ridge_gpu_specialized_primitives(monkeypatch)
    x_matrix, y, splits = _toy_permutation_inputs()
    events: list[Any] = []

    evaluate_permutations(
        pipeline_template=Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", _FakeTorchRidgePermutationEstimator(alpha=0.7, fit_intercept=True)),
            ]
        ),
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=202,
        n_permutations=5,
        metric_name="balanced_accuracy",
        observed_metric=0.0,
        progress_callback=events.append,
    )

    permutation_messages = [
        event.message
        for event in events
        if event.stage == "permutation"
        and (
            (event.message.startswith("running permutation ") and "/" in event.message)
            or (event.message.startswith("finished permutation ") and "/" in event.message)
        )
    ]
    assert len(permutation_messages) == 10
    assert permutation_messages == [
        "running permutation 1/5",
        "finished permutation 1/5",
        "running permutation 2/5",
        "finished permutation 2/5",
        "running permutation 3/5",
        "finished permutation 3/5",
        "running permutation 4/5",
        "finished permutation 4/5",
        "running permutation 5/5",
        "finished permutation 5/5",
    ]


def test_ridge_gpu_batched_dual_payload_keeps_required_fields_and_additive_metadata(
    monkeypatch,
) -> None:
    _patch_fake_ridge_gpu_specialized_primitives(monkeypatch)
    x_matrix, y, splits = _toy_permutation_inputs()

    payload = evaluate_permutations(
        pipeline_template=Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", _FakeTorchRidgePermutationEstimator(alpha=0.5)),
            ]
        ),
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=77,
        n_permutations=4,
        metric_name="balanced_accuracy",
        observed_metric=0.0,
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
    assert payload["execution_mode"] == "ridge_gpu_batched_dual"
    assert payload["specialized_ridge_gpu_path_used"] is True
    assert int(payload["permutation_batch_size"]) > 0
    assert isinstance(payload["fold_gpu_state_build_seconds"], float)
    assert isinstance(payload["fold_factorization_seconds"], float)
    assert isinstance(payload["batched_solve_seconds"], float)
    assert isinstance(payload["batched_predict_seconds"], float)


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
