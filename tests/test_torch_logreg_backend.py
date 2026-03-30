from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from Thesis_ML.experiments.backends import torch_logreg


class _FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self._value = np.asarray(value)

    @property
    def T(self) -> _FakeTensor:
        return _FakeTensor(self._value.T)

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray(self._value)

    def __matmul__(self, other: object) -> _FakeTensor:
        return _FakeTensor(self._value @ _to_numpy(other))

    def __add__(self, other: object) -> _FakeTensor:
        return _FakeTensor(self._value + _to_numpy(other))

    def __radd__(self, other: object) -> _FakeTensor:
        return _FakeTensor(_to_numpy(other) + self._value)

    def __sub__(self, other: object) -> _FakeTensor:
        return _FakeTensor(self._value - _to_numpy(other))

    def __rsub__(self, other: object) -> _FakeTensor:
        return _FakeTensor(_to_numpy(other) - self._value)

    def __mul__(self, other: object) -> _FakeTensor:
        return _FakeTensor(self._value * _to_numpy(other))

    def __rmul__(self, other: object) -> _FakeTensor:
        return _FakeTensor(_to_numpy(other) * self._value)

    def __truediv__(self, other: object) -> _FakeTensor:
        return _FakeTensor(self._value / _to_numpy(other))

    def __array__(self) -> np.ndarray:
        return np.asarray(self._value)


def _to_numpy(value: object) -> np.ndarray:
    if isinstance(value, _FakeTensor):
        return value.numpy()
    return np.asarray(value)


class _FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 2

    def get_device_name(self, device_id: int) -> str:
        return f"Fake GPU {device_id}"

    def get_device_properties(self, device_id: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=8 * 1024 * 1024 * 1024)

    def reset_peak_memory_stats(self, device: object) -> None:
        del device

    def max_memory_allocated(self, device: object) -> int:
        del device
        return 384 * 1024 * 1024

    def manual_seed_all(self, seed: int) -> None:
        del seed


class _FakeTorch:
    float64 = np.float64

    def __init__(self) -> None:
        self.__version__ = "2.4.1"
        self.version = SimpleNamespace(cuda="12.1")
        self.cuda = _FakeCuda()
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True))
        self._seed = None
        self._deterministic_algorithms_enabled = False

    @staticmethod
    def device(name: str) -> str:
        return str(name)

    @staticmethod
    def as_tensor(
        value: np.ndarray,
        dtype: object | None = None,
        device: object | None = None,
    ) -> _FakeTensor:
        del device
        return _FakeTensor(np.asarray(value, dtype=dtype))

    @staticmethod
    def sigmoid(tensor: _FakeTensor) -> _FakeTensor:
        values = _to_numpy(tensor).astype(np.float64, copy=False)
        return _FakeTensor(1.0 / (1.0 + np.exp(-values)))

    @staticmethod
    def softmax(tensor: _FakeTensor, dim: int = 1) -> _FakeTensor:
        values = _to_numpy(tensor).astype(np.float64, copy=False)
        shifted = values - np.max(values, axis=dim, keepdims=True)
        exp_values = np.exp(shifted)
        probs = exp_values / np.clip(np.sum(exp_values, axis=dim, keepdims=True), 1e-12, None)
        return _FakeTensor(probs)

    def manual_seed(self, seed: int) -> None:
        self._seed = int(seed)

    def use_deterministic_algorithms(self, enabled: bool) -> None:
        self._deterministic_algorithms_enabled = bool(enabled)


def _patch_fake_torch(monkeypatch: pytest.MonkeyPatch, fake_torch: _FakeTorch) -> None:
    original_import_module = torch_logreg.importlib.import_module

    def _fake_import(name: str):
        if name == "torch":
            return fake_torch
        return original_import_module(name)

    monkeypatch.setattr(torch_logreg.importlib, "import_module", _fake_import)


def _binary_dataset() -> tuple[np.ndarray, np.ndarray]:
    x_matrix = np.asarray(
        [
            [0.0, 0.1, 0.0, 0.1],
            [0.1, 0.0, 0.1, 0.0],
            [0.2, 0.1, 0.0, 0.2],
            [0.0, 0.2, 0.1, 0.1],
            [2.0, 2.1, 2.0, 2.1],
            [2.2, 2.0, 2.1, 2.0],
            [2.1, 2.2, 2.0, 2.1],
            [2.0, 2.0, 2.2, 2.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["neg", "neg", "neg", "neg", "pos", "pos", "pos", "pos"])
    return x_matrix, labels


def test_torch_logreg_fit_predict_contract_with_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    _patch_fake_torch(monkeypatch, fake_torch)
    x_matrix, labels = _binary_dataset()

    estimator = torch_logreg.TorchLogisticRegression(
        class_weight="balanced",
        gpu_device_id=1,
        deterministic_compute=True,
        max_iter=600,
        learning_rate=0.2,
    )
    fitted = estimator.fit(x_matrix, labels)

    assert fitted is estimator
    assert estimator.classes_.tolist() == ["neg", "pos"]
    assert estimator.n_features_in_ == x_matrix.shape[1]
    assert np.asarray(estimator.coef_).shape == (1, x_matrix.shape[1])
    assert np.asarray(estimator.intercept_).shape == (1,)

    predictions = estimator.predict(x_matrix)
    assert predictions.shape == labels.shape
    assert set(predictions.tolist()) <= {"neg", "pos"}

    decision = np.asarray(estimator.decision_function(x_matrix), dtype=np.float64)
    assert decision.shape == (x_matrix.shape[0],)

    probabilities = np.asarray(estimator.predict_proba(x_matrix), dtype=np.float64)
    assert probabilities.shape == (x_matrix.shape[0], 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(x_matrix.shape[0]), atol=1e-8)

    metadata = estimator.get_backend_runtime_metadata()
    assert metadata["backend_id"] == torch_logreg.TORCH_LOGREG_BACKEND_ID
    assert isinstance(metadata["gpu_memory_peak_mb"], float)
    assert isinstance(metadata["device_transfer_seconds"], float)
    assert metadata["torch_deterministic_enforced"] is True
    assert metadata["torch_deterministic_limitations"] is None
    assert fake_torch.backends.cudnn.deterministic is True
    assert fake_torch.backends.cudnn.benchmark is False


def test_torch_logreg_prediction_parity_is_within_reasonable_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    _patch_fake_torch(monkeypatch, fake_torch)
    x_matrix, labels = _binary_dataset()

    torch_estimator = torch_logreg.TorchLogisticRegression(
        class_weight="balanced",
        gpu_device_id=0,
        max_iter=800,
        learning_rate=0.2,
    )
    cpu_estimator = LogisticRegression(
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        random_state=0,
    )
    torch_estimator.fit(x_matrix, labels)
    cpu_estimator.fit(x_matrix, labels)

    torch_predictions = np.asarray(torch_estimator.predict(x_matrix))
    cpu_predictions = np.asarray(cpu_estimator.predict(x_matrix))
    agreement = float(np.mean(torch_predictions == cpu_predictions))
    assert agreement >= 0.9

    torch_proba = np.asarray(torch_estimator.predict_proba(x_matrix), dtype=np.float64)[:, 1]
    cpu_proba = np.asarray(cpu_estimator.predict_proba(x_matrix), dtype=np.float64)[:, 1]
    assert torch_proba.shape == cpu_proba.shape
    assert float(np.corrcoef(torch_proba, cpu_proba)[0, 1]) >= 0.9


def test_torch_logreg_raises_clear_error_when_torch_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import_module = torch_logreg.importlib.import_module

    def _fake_import(name: str):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import_module(name)

    monkeypatch.setattr(torch_logreg.importlib, "import_module", _fake_import)
    x_matrix, labels = _binary_dataset()
    estimator = torch_logreg.TorchLogisticRegression(gpu_device_id=0)

    with pytest.raises(RuntimeError, match="torch is not installed"):
        estimator.fit(x_matrix, labels)


def test_torch_logreg_rejects_invalid_gpu_device_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    _patch_fake_torch(monkeypatch, fake_torch)
    x_matrix, labels = _binary_dataset()
    estimator = torch_logreg.TorchLogisticRegression(gpu_device_id=9)

    with pytest.raises(RuntimeError, match="outside visible CUDA range"):
        estimator.fit(x_matrix, labels)


def test_torch_logreg_reuses_device_parameter_cache_for_repeated_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    _patch_fake_torch(monkeypatch, fake_torch)
    x_matrix, labels = _binary_dataset()

    estimator = torch_logreg.TorchLogisticRegression(
        class_weight="balanced",
        gpu_device_id=0,
        max_iter=500,
        learning_rate=0.2,
    )
    estimator.fit(x_matrix, labels)
    assert int(getattr(estimator, "_device_parameter_cache_build_count_", -1)) == 0

    decision_first = np.asarray(estimator.decision_function(x_matrix), dtype=np.float64)
    proba_first = np.asarray(estimator.predict_proba(x_matrix), dtype=np.float64)
    pred_first = np.asarray(estimator.predict(x_matrix)).astype(str, copy=False)

    assert int(getattr(estimator, "_device_parameter_cache_build_count_", -1)) == 1

    decision_second = np.asarray(estimator.decision_function(x_matrix), dtype=np.float64)
    proba_second = np.asarray(estimator.predict_proba(x_matrix), dtype=np.float64)
    pred_second = np.asarray(estimator.predict(x_matrix)).astype(str, copy=False)

    assert int(getattr(estimator, "_device_parameter_cache_build_count_", -1)) == 1
    np.testing.assert_allclose(decision_first, decision_second, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(proba_first, proba_second, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(pred_first, pred_second)


def test_torch_logreg_device_cache_does_not_break_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    _patch_fake_torch(monkeypatch, fake_torch)
    x_matrix, labels = _binary_dataset()

    estimator = torch_logreg.TorchLogisticRegression(
        class_weight="balanced",
        gpu_device_id=0,
        max_iter=500,
        learning_rate=0.2,
    )
    estimator.fit(x_matrix, labels)
    baseline_predictions = np.asarray(estimator.predict(x_matrix)).astype(str, copy=False)
    _ = estimator.decision_function(x_matrix)
    assert int(getattr(estimator, "_device_parameter_cache_build_count_", -1)) == 1

    restored = pickle.loads(pickle.dumps(estimator))
    restored_predictions = np.asarray(restored.predict(x_matrix)).astype(str, copy=False)
    np.testing.assert_array_equal(restored_predictions, baseline_predictions)
    assert int(getattr(restored, "_device_parameter_cache_build_count_", -1)) == 1
