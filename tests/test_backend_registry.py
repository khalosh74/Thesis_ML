from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC

from Thesis_ML.experiments.backend_registry import (
    resolve_backend_constructor,
    resolve_backend_support,
)
from Thesis_ML.experiments.backends.cpu_reference import (
    CPU_REFERENCE_MODEL_CONSTRUCTORS,
    build_cpu_reference_pipeline,
)
from Thesis_ML.experiments.backends.torch_logreg import (
    TORCH_LOGREG_BACKEND_ID,
    TorchLogisticRegression,
)
from Thesis_ML.experiments.backends.torch_ridge import (
    TORCH_RIDGE_BACKEND_ID,
    TorchRidgeClassifier,
)
from Thesis_ML.experiments.backends.xgboost_cpu import XGBOOST_CPU_BACKEND_ID
from Thesis_ML.experiments.backends.xgboost_gpu import XGBOOST_GPU_BACKEND_ID
from Thesis_ML.experiments.compute_policy import (
    CPU_REFERENCE_BACKEND_STACK_ID,
    ResolvedComputePolicy,
)
from Thesis_ML.experiments.model_factory import build_pipeline, make_model


def _resolved_cpu_policy(
    *,
    hardware_mode_requested: str = "cpu_only",
    hardware_mode_effective: str = "cpu_only",
    requested_backend_family: str = "sklearn_cpu",
) -> ResolvedComputePolicy:
    return ResolvedComputePolicy(
        hardware_mode_requested=hardware_mode_requested,
        hardware_mode_effective=hardware_mode_effective,
        requested_backend_family=requested_backend_family,
        effective_backend_family="sklearn_cpu",
        gpu_device_id=None,
        gpu_device_name=None,
        gpu_device_total_memory_mb=None,
        deterministic_compute=False,
        allow_backend_fallback=False,
        backend_stack_id=CPU_REFERENCE_BACKEND_STACK_ID,
        backend_fallback_used=False,
        backend_fallback_reason=None,
    )


def _unsupported_backend_policy(effective_backend_family: str) -> ResolvedComputePolicy:
    return ResolvedComputePolicy(
        hardware_mode_requested="gpu_only",
        hardware_mode_effective="gpu_only",
        requested_backend_family="torch_gpu",
        effective_backend_family=effective_backend_family,
        gpu_device_id=0,
        gpu_device_name="GPU 0",
        gpu_device_total_memory_mb=8192,
        deterministic_compute=True,
        allow_backend_fallback=False,
        backend_stack_id="future_backend_stack",
        backend_fallback_used=False,
        backend_fallback_reason=None,
    )


def _resolved_torch_policy(*, model_requested: str = "ridge") -> ResolvedComputePolicy:
    return ResolvedComputePolicy(
        hardware_mode_requested="gpu_only",
        hardware_mode_effective="gpu_only",
        requested_backend_family="torch_gpu",
        effective_backend_family="torch_gpu",
        gpu_device_id=0,
        gpu_device_name="GPU 0",
        gpu_device_total_memory_mb=12288,
        deterministic_compute=True,
        allow_backend_fallback=False,
        backend_stack_id=f"torch_stack_for_{model_requested}",
        backend_fallback_used=False,
        backend_fallback_reason=None,
    )


def _toy_dataset() -> tuple[np.ndarray, np.ndarray]:
    x_matrix = np.asarray(
        [
            [0.0, 0.0, 0.1, 0.0],
            [0.1, 0.0, 0.0, 0.1],
            [0.2, 0.1, 0.0, 0.0],
            [0.0, 0.2, 0.1, 0.0],
            [0.2, 0.0, 0.2, 0.1],
            [0.1, 0.1, 0.2, 0.0],
            [2.0, 2.1, 2.0, 2.0],
            [2.2, 2.0, 2.1, 2.0],
            [2.1, 2.2, 2.0, 2.1],
            [2.0, 2.0, 2.2, 2.1],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["neg", "neg", "neg", "neg", "neg", "neg", "pos", "pos", "pos", "pos"])
    return x_matrix, labels


@pytest.mark.parametrize(
    ("model_name", "expected_type"),
    [
        ("logreg", LogisticRegression),
        ("linearsvc", LinearSVC),
        ("ridge", RidgeClassifier),
        ("dummy", DummyClassifier),
    ],
)
def test_supported_models_resolve_to_cpu_reference_backend(
    model_name: str,
    expected_type: type[object],
) -> None:
    compute_policy = _resolved_cpu_policy()

    support = resolve_backend_support(model_name, compute_policy)
    assert support.supported is True
    assert support.backend_id == "cpu_reference"
    assert support.effective_backend_family == "sklearn_cpu"

    resolution = resolve_backend_constructor(model_name, compute_policy)
    estimator = resolution.build_estimator(seed=13, class_weight_policy="balanced")

    assert resolution.backend_id == "cpu_reference"
    assert resolution.model_name == model_name
    assert resolution.compute_policy is compute_policy
    assert isinstance(estimator, expected_type)


@pytest.mark.parametrize(
    (
        "model_name",
        "hardware_mode_requested",
        "hardware_mode_effective",
        "requested_backend_family",
    ),
    [
        ("ridge", "gpu_only", "gpu_only", "torch_gpu"),
        ("logreg", "gpu_only", "gpu_only", "torch_gpu"),
        ("ridge", "max_both", "max_both", "auto_mixed"),
        ("logreg", "max_both", "max_both", "auto_mixed"),
    ],
)
def test_effective_backend_family_controls_resolution_not_hardware_mode_metadata(
    model_name: str,
    hardware_mode_requested: str,
    hardware_mode_effective: str,
    requested_backend_family: str,
) -> None:
    compute_policy = _resolved_cpu_policy(
        hardware_mode_requested=hardware_mode_requested,
        hardware_mode_effective=hardware_mode_effective,
        requested_backend_family=requested_backend_family,
    )

    estimator = make_model(
        name=model_name,
        seed=7,
        class_weight_policy="balanced",
        compute_policy=compute_policy,
    )

    if model_name == "ridge":
        assert isinstance(estimator, RidgeClassifier)
    else:
        assert isinstance(estimator, LogisticRegression)
    assert estimator.class_weight == "balanced"
    assert int(estimator.random_state) == 7


@pytest.mark.parametrize("effective_backend_family", ["torch_gpu", "auto_mixed"])
def test_unsupported_future_backend_combinations_fail_clearly(
    effective_backend_family: str,
) -> None:
    compute_policy = _unsupported_backend_policy(effective_backend_family)

    support = resolve_backend_support("ridge", compute_policy)
    if effective_backend_family == "torch_gpu":
        assert support.supported is True
        assert support.backend_id == TORCH_RIDGE_BACKEND_ID
    else:
        assert support.supported is False
        assert support.backend_id is None
        assert support.reason is not None
        assert "Supported backend families are" in support.reason
        with pytest.raises(ValueError, match="Supported backend families are"):
            resolve_backend_constructor("ridge", compute_policy)


def test_ridge_resolves_to_torch_backend_when_effective_backend_family_is_torch_gpu() -> None:
    compute_policy = _resolved_torch_policy()
    resolution = resolve_backend_constructor("ridge", compute_policy)
    estimator = resolution.build_estimator(seed=5, class_weight_policy="balanced")

    assert resolution.backend_id == TORCH_RIDGE_BACKEND_ID
    assert resolution.effective_backend_family == "torch_gpu"
    assert isinstance(estimator, TorchRidgeClassifier)
    assert estimator.gpu_device_id == 0
    assert estimator.deterministic_compute is True
    assert estimator.class_weight == "balanced"


def test_logreg_resolves_to_torch_backend_when_effective_backend_family_is_torch_gpu() -> None:
    compute_policy = _resolved_torch_policy(model_requested="logreg")
    resolution = resolve_backend_constructor("logreg", compute_policy)
    estimator = resolution.build_estimator(seed=11, class_weight_policy="balanced")

    assert resolution.backend_id == TORCH_LOGREG_BACKEND_ID
    assert resolution.effective_backend_family == "torch_gpu"
    assert isinstance(estimator, TorchLogisticRegression)
    assert estimator.gpu_device_id == 0
    assert estimator.deterministic_compute is True
    assert estimator.class_weight == "balanced"


@pytest.mark.parametrize("model_name", ["linearsvc", "dummy"])
def test_torch_backend_requests_for_unsupported_models_fail_clearly(model_name: str) -> None:
    compute_policy = _resolved_torch_policy(model_requested=model_name)

    support = resolve_backend_support(model_name, compute_policy)
    assert support.supported is False
    assert support.backend_id is None
    assert support.reason is not None
    assert "Supported models: ridge, logreg, xgboost." in support.reason

    with pytest.raises(ValueError, match="Supported models: ridge, logreg, xgboost."):
        resolve_backend_constructor(model_name, compute_policy)


def test_cpu_reference_pipeline_preserves_expected_behavior_contracts() -> None:
    x_matrix, labels = _toy_dataset()
    compute_policy = _resolved_cpu_policy()

    for model_name in sorted(CPU_REFERENCE_MODEL_CONSTRUCTORS):
        pipeline_via_registry = build_pipeline(
            model_name=model_name,
            seed=19,
            class_weight_policy="balanced",
            compute_policy=compute_policy,
        )
        pipeline_cpu_reference = build_cpu_reference_pipeline(
            model_name=model_name,
            seed=19,
            class_weight_policy="balanced",
        )

        assert list(pipeline_via_registry.named_steps) == ["scaler", "model"]
        assert list(pipeline_cpu_reference.named_steps) == ["scaler", "model"]
        assert type(pipeline_via_registry.named_steps["model"]) is type(
            pipeline_cpu_reference.named_steps["model"]
        )

        pipeline_via_registry.fit(x_matrix, labels)
        pipeline_cpu_reference.fit(x_matrix, labels)

        predictions_registry = pipeline_via_registry.predict(x_matrix)
        predictions_cpu_reference = pipeline_cpu_reference.predict(x_matrix)
        assert predictions_registry.tolist() == predictions_cpu_reference.tolist()

        if hasattr(pipeline_via_registry, "decision_function"):
            decision_registry = np.asarray(pipeline_via_registry.decision_function(x_matrix))
            decision_cpu_reference = np.asarray(pipeline_cpu_reference.decision_function(x_matrix))
            assert decision_registry.shape == decision_cpu_reference.shape

        if hasattr(pipeline_via_registry, "predict_proba"):
            proba_registry = np.asarray(pipeline_via_registry.predict_proba(x_matrix))
            proba_cpu_reference = np.asarray(pipeline_cpu_reference.predict_proba(x_matrix))
            assert proba_registry.shape == proba_cpu_reference.shape

        fitted_model = pipeline_via_registry.named_steps["model"]
        assert fitted_model.classes_.tolist() == ["neg", "pos"]
        assert int(fitted_model.n_features_in_) == x_matrix.shape[1]

        if hasattr(fitted_model, "coef_"):
            assert (
                np.asarray(fitted_model.coef_).shape
                == np.asarray(pipeline_cpu_reference.named_steps["model"].coef_).shape
            )
            assert (
                np.asarray(fitted_model.intercept_).shape
                == np.asarray(pipeline_cpu_reference.named_steps["model"].intercept_).shape
            )


def test_xgboost_cpu_support_resolution_unavailable_reports_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "Thesis_ML.experiments.backend_registry.xgboost_cpu_support_status",
        lambda: (False, "xgboost_missing_for_test"),
    )
    support = resolve_backend_support("xgboost", _resolved_cpu_policy())
    assert support.supported is False
    assert support.backend_id is None
    assert support.reason == "xgboost_missing_for_test"


def test_xgboost_cpu_resolution_and_constructor_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SentinelEstimator:
        def __init__(self, *, seed: int, class_weight_policy: str) -> None:
            self.seed = int(seed)
            self.class_weight_policy = str(class_weight_policy)

    monkeypatch.setattr(
        "Thesis_ML.experiments.backend_registry.xgboost_cpu_support_status",
        lambda: (True, None),
    )
    monkeypatch.setattr(
        "Thesis_ML.experiments.backend_registry.make_xgboost_cpu_estimator",
        lambda *, seed, class_weight_policy: _SentinelEstimator(
            seed=int(seed),
            class_weight_policy=str(class_weight_policy),
        ),
    )

    resolution = resolve_backend_constructor("xgboost", _resolved_cpu_policy())
    estimator = resolution.build_estimator(seed=23, class_weight_policy="none")
    assert resolution.backend_id == XGBOOST_CPU_BACKEND_ID
    assert resolution.effective_backend_family == "sklearn_cpu"
    assert isinstance(estimator, _SentinelEstimator)
    assert estimator.seed == 23
    assert estimator.class_weight_policy == "none"


def test_xgboost_gpu_resolution_and_constructor_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SentinelEstimator:
        def __init__(
            self,
            *,
            seed: int,
            class_weight_policy: str,
            gpu_device_id: int,
            deterministic_compute: bool,
        ) -> None:
            self.seed = int(seed)
            self.class_weight_policy = str(class_weight_policy)
            self.gpu_device_id = int(gpu_device_id)
            self.deterministic_compute = bool(deterministic_compute)

    monkeypatch.setattr(
        "Thesis_ML.experiments.backend_registry.xgboost_gpu_support_status",
        lambda *, gpu_device_id: (True, None),
    )
    monkeypatch.setattr(
        "Thesis_ML.experiments.backend_registry.make_xgboost_gpu_estimator",
        lambda *, seed, class_weight_policy, gpu_device_id, deterministic_compute: (
            _SentinelEstimator(
                seed=int(seed),
                class_weight_policy=str(class_weight_policy),
                gpu_device_id=int(gpu_device_id),
                deterministic_compute=bool(deterministic_compute),
            )
        ),
    )

    resolution = resolve_backend_constructor("xgboost", _resolved_torch_policy())
    estimator = resolution.build_estimator(seed=31, class_weight_policy="none")
    assert resolution.backend_id == XGBOOST_GPU_BACKEND_ID
    assert resolution.effective_backend_family == "torch_gpu"
    assert isinstance(estimator, _SentinelEstimator)
    assert estimator.seed == 31
    assert estimator.class_weight_policy == "none"
    assert estimator.gpu_device_id == 0
    assert estimator.deterministic_compute is True


def test_xgboost_gpu_support_resolution_unavailable_reports_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "Thesis_ML.experiments.backend_registry.xgboost_gpu_support_status",
        lambda *, gpu_device_id: (False, "xgboost_gpu_missing_for_test"),
    )
    support = resolve_backend_support("xgboost", _resolved_torch_policy())
    assert support.supported is False
    assert support.backend_id is None
    assert support.reason == "xgboost_gpu_missing_for_test"

    with pytest.raises(ValueError, match="xgboost_gpu_missing_for_test"):
        resolve_backend_constructor("xgboost", _resolved_torch_policy())
