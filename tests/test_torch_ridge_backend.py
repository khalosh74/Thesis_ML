from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier

from Thesis_ML.experiments.backends import ridge_exact_kernels
from Thesis_ML.experiments.backends import torch_ridge


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
        other_value = _to_numpy(other)
        return _FakeTensor(self._value @ other_value)

    def __add__(self, other: object) -> _FakeTensor:
        return _FakeTensor(self._value + _to_numpy(other))

    def __radd__(self, other: object) -> _FakeTensor:
        return _FakeTensor(_to_numpy(other) + self._value)

    def __sub__(self, other: object) -> _FakeTensor:
        return _FakeTensor(self._value - _to_numpy(other))

    def __mul__(self, other: object) -> _FakeTensor:
        return _FakeTensor(self._value * _to_numpy(other))

    def __rmul__(self, other: object) -> _FakeTensor:
        return _FakeTensor(_to_numpy(other) * self._value)

    def __array__(self) -> np.ndarray:
        return np.asarray(self._value)


def _to_numpy(value: object) -> np.ndarray:
    if isinstance(value, _FakeTensor):
        return value.numpy()
    return np.asarray(value)


class _FakeLinalg:
    @staticmethod
    def solve(system_matrix: _FakeTensor, rhs: _FakeTensor) -> _FakeTensor:
        solution = np.linalg.solve(system_matrix.numpy(), rhs.numpy())
        return _FakeTensor(solution)


class _FakeCuda:
    def __init__(self) -> None:
        self._deterministic_algorithms_enabled = False

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
        return 256 * 1024 * 1024


class _FakeTorch:
    float64 = np.float64

    def __init__(self) -> None:
        self.__version__ = "2.4.1"
        self.version = SimpleNamespace(cuda="12.1")
        self.cuda = _FakeCuda()
        self.linalg = _FakeLinalg()
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True))
        self.deterministic_algorithms_enabled = False

    @staticmethod
    def device(name: str) -> str:
        return str(name)

    @staticmethod
    def as_tensor(
        value: np.ndarray, dtype: object | None = None, device: object | None = None
    ) -> _FakeTensor:
        del device
        array = np.asarray(value, dtype=dtype)
        return _FakeTensor(array)

    def use_deterministic_algorithms(self, enabled: bool) -> None:
        self.deterministic_algorithms_enabled = bool(enabled)


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


def _patch_fake_torch(monkeypatch: pytest.MonkeyPatch, fake_torch: _FakeTorch) -> None:
    original_import_module = torch_ridge.importlib.import_module

    def _fake_import(name: str):
        if name == "torch":
            return fake_torch
        return original_import_module(name)

    monkeypatch.setattr(
        torch_ridge.importlib,
        "import_module",
        _fake_import,
    )


def test_torch_ridge_fit_predict_contract_with_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    _patch_fake_torch(monkeypatch, fake_torch)
    x_matrix, labels = _binary_dataset()

    estimator = torch_ridge.TorchRidgeClassifier(
        class_weight="balanced",
        gpu_device_id=1,
        deterministic_compute=True,
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

    decision = np.asarray(estimator.decision_function(x_matrix))
    assert decision.shape == (x_matrix.shape[0],)

    metadata = estimator.get_backend_runtime_metadata()
    assert metadata["backend_id"] == torch_ridge.TORCH_RIDGE_BACKEND_ID
    assert isinstance(metadata["gpu_memory_peak_mb"], float)
    assert isinstance(metadata["device_transfer_seconds"], float)
    assert metadata["torch_deterministic_enforced"] is True
    assert metadata["torch_deterministic_limitations"] is None
    assert fake_torch.backends.cudnn.deterministic is True
    assert fake_torch.backends.cudnn.benchmark is False


def test_torch_ridge_prediction_parity_is_within_reasonable_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    _patch_fake_torch(monkeypatch, fake_torch)
    x_matrix, labels = _binary_dataset()

    torch_estimator = torch_ridge.TorchRidgeClassifier(class_weight="balanced", gpu_device_id=0)
    cpu_estimator = RidgeClassifier(class_weight="balanced")
    torch_estimator.fit(x_matrix, labels)
    cpu_estimator.fit(x_matrix, labels)

    torch_predictions = np.asarray(torch_estimator.predict(x_matrix))
    cpu_predictions = np.asarray(cpu_estimator.predict(x_matrix))
    agreement = float(np.mean(torch_predictions == cpu_predictions))
    assert agreement >= 0.95

    torch_decision = np.asarray(torch_estimator.decision_function(x_matrix), dtype=np.float64)
    cpu_decision = np.asarray(cpu_estimator.decision_function(x_matrix), dtype=np.float64)
    assert torch_decision.shape == cpu_decision.shape
    assert float(np.corrcoef(torch_decision, cpu_decision)[0, 1]) >= 0.9


def test_torch_ridge_raises_clear_error_when_torch_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import_module = torch_ridge.importlib.import_module

    def _fake_import(name: str):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import_module(name)

    monkeypatch.setattr(
        torch_ridge.importlib,
        "import_module",
        _fake_import,
    )
    x_matrix, labels = _binary_dataset()
    estimator = torch_ridge.TorchRidgeClassifier(gpu_device_id=0)

    with pytest.raises(RuntimeError, match="torch is not installed"):
        estimator.fit(x_matrix, labels)


def test_torch_ridge_rejects_invalid_gpu_device_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    _patch_fake_torch(monkeypatch, fake_torch)
    x_matrix, labels = _binary_dataset()
    estimator = torch_ridge.TorchRidgeClassifier(gpu_device_id=9)

    with pytest.raises(RuntimeError, match="outside visible CUDA range"):
        estimator.fit(x_matrix, labels)


def test_torch_ridge_uses_dual_solver_when_features_exceed_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    _patch_fake_torch(monkeypatch, fake_torch)

    x_matrix = np.asarray(
        [
            [0.0, 0.1, 0.0, 0.1, 0.2, 0.0],
            [0.1, 0.0, 0.1, 0.0, 0.0, 0.2],
            [2.0, 2.1, 2.0, 2.1, 2.0, 2.2],
            [2.1, 2.0, 2.1, 2.0, 2.2, 2.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["neg", "neg", "pos", "pos"])

    estimator = torch_ridge.TorchRidgeClassifier(gpu_device_id=0)
    estimator.fit(x_matrix, labels)

    metadata = estimator.get_backend_runtime_metadata()
    assert metadata["ridge_solver_family"] == "dual"
    assert metadata["ridge_system_dimension"] == x_matrix.shape[0]


def test_torch_ridge_uses_primal_solver_when_samples_exceed_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    _patch_fake_torch(monkeypatch, fake_torch)

    x_matrix = np.asarray(
        [
            [0.0, 0.1],
            [0.1, 0.0],
            [0.2, 0.1],
            [0.0, 0.2],
            [2.0, 2.1],
            [2.2, 2.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["neg", "neg", "neg", "pos", "pos", "pos"])

    estimator = torch_ridge.TorchRidgeClassifier(gpu_device_id=0)
    estimator.fit(x_matrix, labels)

    metadata = estimator.get_backend_runtime_metadata()
    assert metadata["ridge_solver_family"] == "primal"
    assert metadata["ridge_system_dimension"] == x_matrix.shape[1]


def test_torch_ridge_multiclass_prediction_parity_is_reasonable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    _patch_fake_torch(monkeypatch, fake_torch)

    x_matrix = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [2.0, 2.0],
            [2.1, 2.0],
            [4.0, 4.0],
            [4.1, 4.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["a", "a", "b", "b", "c", "c"])

    torch_estimator = torch_ridge.TorchRidgeClassifier(gpu_device_id=0)
    cpu_estimator = RidgeClassifier()

    torch_estimator.fit(x_matrix, labels)
    cpu_estimator.fit(x_matrix, labels)

    torch_predictions = np.asarray(torch_estimator.predict(x_matrix))
    cpu_predictions = np.asarray(cpu_estimator.predict(x_matrix))
    agreement = float(np.mean(torch_predictions == cpu_predictions))
    assert agreement >= 0.95


def test_ridge_exact_kernel_alpha_batch_matches_sklearn_ridge_predictions() -> None:
    x_matrix, labels = _binary_dataset()
    sample_weights = np.ones(x_matrix.shape[0], dtype=np.float64)
    alphas = np.asarray([0.1, 1.0, 10.0], dtype=np.float64)

    state = ridge_exact_kernels.build_ridge_exact_alpha_factorization_state(
        x_train=x_matrix,
        y_train=labels,
        fit_intercept=True,
        sample_weights=sample_weights,
    )
    weight_batch, intercept_batch = ridge_exact_kernels.solve_ridge_exact_alpha_batch(
        state=state,
        alphas=alphas,
    )
    prediction_batch, _ = ridge_exact_kernels.predict_ridge_labels_for_alpha_batch(
        x_eval=x_matrix,
        weight_batch=weight_batch,
        intercept_batch=intercept_batch,
        classes=state.classes,
        binary_mode=state.binary_mode,
    )

    for alpha_index, alpha_value in enumerate(alphas.tolist()):
        reference = RidgeClassifier(alpha=float(alpha_value), fit_intercept=True)
        reference.fit(x_matrix, labels)
        reference_predictions = np.asarray(reference.predict(x_matrix)).astype(str, copy=False)
        candidate_predictions = np.asarray(prediction_batch[alpha_index]).astype(str, copy=False)
        np.testing.assert_array_equal(candidate_predictions, reference_predictions)
