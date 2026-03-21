from __future__ import annotations

import importlib
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

TORCH_RIDGE_BACKEND_ID = "torch_ridge_gpu_v1"


def resolve_torch_ridge_class_weight(class_weight_policy: str) -> str | None:
    normalized = str(class_weight_policy).strip().lower()
    if normalized == "none":
        return None
    if normalized == "balanced":
        return "balanced"
    raise ValueError(
        f"Unsupported class_weight_policy '{class_weight_policy}'. Allowed values: none, balanced."
    )


def _import_torch_module() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Torch backend requested but torch is not installed. Install a CUDA-enabled torch build "
            "to run exploratory gpu_only ridge."
        ) from exc


def _to_numpy_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _as_torch_float_tensor(torch: Any, value: np.ndarray, *, device: Any) -> Any:
    dtype = getattr(torch, "float64", None)
    if dtype is None:
        return torch.as_tensor(value, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _resolve_cuda_device(torch: Any, requested_device_id: int) -> tuple[Any, int]:
    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is None:
        raise RuntimeError("Torch is installed but torch.cuda is unavailable.")
    if not bool(cuda_module.is_available()):
        raise RuntimeError("Torch CUDA backend is unavailable (torch.cuda.is_available()=False).")
    try:
        gpu_count = int(cuda_module.device_count())
    except Exception as exc:  # pragma: no cover - defensive probe guard
        raise RuntimeError(
            f"Failed to probe CUDA device count: {exc.__class__.__name__}."
        ) from exc
    if gpu_count <= 0:
        raise RuntimeError("Torch CUDA is available but no visible GPU devices were found.")
    resolved_device_id = int(requested_device_id)
    if resolved_device_id < 0 or resolved_device_id >= gpu_count:
        raise RuntimeError(
            f"Requested gpu_device_id={resolved_device_id} is outside visible CUDA range "
            f"[0, {gpu_count - 1}]."
        )
    device_name = f"cuda:{resolved_device_id}"
    if hasattr(torch, "device"):
        return torch.device(device_name), resolved_device_id
    return device_name, resolved_device_id


def _resolve_sample_weights(y_labels: np.ndarray, class_weight: str | None) -> np.ndarray:
    if class_weight is None:
        return np.ones(y_labels.shape[0], dtype=np.float64)
    if class_weight != "balanced":
        raise ValueError("TorchRidgeClassifier supports class_weight=None or 'balanced' only.")
    classes, counts = np.unique(y_labels, return_counts=True)
    n_samples = int(y_labels.shape[0])
    n_classes = int(classes.shape[0])
    class_weights = {
        label: float(n_samples / (n_classes * int(count)))
        for label, count in zip(classes.tolist(), counts.tolist(), strict=True)
    }
    return np.asarray([class_weights[label] for label in y_labels.tolist()], dtype=np.float64)


def _configure_torch_determinism(torch: Any, *, enabled: bool) -> tuple[bool, str | None]:
    if not bool(enabled):
        return False, None

    enforced = True
    limitations: list[str] = []

    configure_fn = getattr(torch, "use_deterministic_algorithms", None)
    if callable(configure_fn):
        try:
            configure_fn(True)
        except Exception as exc:  # pragma: no cover - defensive path
            enforced = False
            limitations.append(f"use_deterministic_algorithms_failed:{exc.__class__.__name__}")
    else:
        enforced = False
        limitations.append("use_deterministic_algorithms_unavailable")

    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        try:
            cudnn.deterministic = True
        except Exception as exc:  # pragma: no cover - defensive path
            enforced = False
            limitations.append(f"cudnn_deterministic_flag_failed:{exc.__class__.__name__}")
        try:
            cudnn.benchmark = False
        except Exception as exc:  # pragma: no cover - defensive path
            enforced = False
            limitations.append(f"cudnn_benchmark_flag_failed:{exc.__class__.__name__}")

    return enforced, ";".join(limitations) if limitations else None


class TorchRidgeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        class_weight: str | None = None,
        random_state: int | None = None,
        gpu_device_id: int = 0,
        deterministic_compute: bool = False,
        backend_id: str = TORCH_RIDGE_BACKEND_ID,
    ) -> None:
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.class_weight = class_weight
        self.random_state = random_state
        self.gpu_device_id = int(gpu_device_id)
        self.deterministic_compute = bool(deterministic_compute)
        self.backend_id = str(backend_id)

    def fit(self, x_matrix: np.ndarray, y_labels: np.ndarray) -> TorchRidgeClassifier:
        x_array = np.asarray(x_matrix, dtype=np.float64)
        y_array = np.asarray(y_labels)

        if x_array.ndim != 2:
            raise ValueError("TorchRidgeClassifier.fit requires a 2D feature matrix.")
        if y_array.ndim != 1:
            raise ValueError("TorchRidgeClassifier.fit requires a 1D label vector.")
        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError("TorchRidgeClassifier.fit received mismatched X/y row counts.")
        if x_array.shape[0] == 0:
            raise ValueError("TorchRidgeClassifier.fit received an empty training matrix.")

        y_text = y_array.astype(str, copy=False)
        classes, y_indices = np.unique(y_text, return_inverse=True)
        if classes.shape[0] < 2:
            raise ValueError("TorchRidgeClassifier.fit requires at least two target classes.")

        sample_weights = _resolve_sample_weights(y_text, self.class_weight)
        sample_weight_sqrt = np.sqrt(sample_weights).astype(np.float64, copy=False)

        torch = _import_torch_module()
        device, resolved_device_id = _resolve_cuda_device(torch, self.gpu_device_id)
        deterministic_enforced, deterministic_limitations = _configure_torch_determinism(
            torch,
            enabled=bool(self.deterministic_compute),
        )

        cuda_module = getattr(torch, "cuda", None)
        if cuda_module is not None and hasattr(cuda_module, "reset_peak_memory_stats"):
            try:
                cuda_module.reset_peak_memory_stats(device)
            except Exception:  # pragma: no cover - best effort diagnostic only
                pass

        transfer_start = perf_counter()
        if self.fit_intercept:
            x_augmented = np.concatenate(
                [x_array, np.ones((x_array.shape[0], 1), dtype=np.float64)],
                axis=1,
            )
        else:
            x_augmented = x_array
        x_augmented_tensor = _as_torch_float_tensor(torch, x_augmented, device=device)
        weighted_design_tensor = x_augmented_tensor * _as_torch_float_tensor(
            torch,
            sample_weight_sqrt.reshape(-1, 1),
            device=device,
        )
        sample_weight_sqrt_tensor = _as_torch_float_tensor(
            torch,
            sample_weight_sqrt,
            device=device,
        )
        transfer_elapsed = float(perf_counter() - transfer_start)

        regularization = np.eye(x_augmented.shape[1], dtype=np.float64) * float(self.alpha)
        if self.fit_intercept:
            regularization[-1, -1] = 0.0
        regularization_tensor = _as_torch_float_tensor(torch, regularization, device=device)
        system_matrix = weighted_design_tensor.T @ weighted_design_tensor + regularization_tensor

        class_coefficients: list[np.ndarray] = []
        class_intercepts: list[float] = []
        for class_index in range(int(classes.shape[0])):
            class_target = np.where(y_indices == class_index, 1.0, -1.0).astype(np.float64)
            class_target_tensor = _as_torch_float_tensor(torch, class_target, device=device)
            rhs = weighted_design_tensor.T @ (class_target_tensor * sample_weight_sqrt_tensor)
            solution = torch.linalg.solve(system_matrix, rhs)
            solution_np = _to_numpy_array(solution).astype(np.float64, copy=False)
            if self.fit_intercept:
                class_coefficients.append(solution_np[:-1])
                class_intercepts.append(float(solution_np[-1]))
            else:
                class_coefficients.append(solution_np)
                class_intercepts.append(0.0)

        coefficient_matrix = np.vstack(class_coefficients).astype(np.float64, copy=False)
        intercept_vector = np.asarray(class_intercepts, dtype=np.float64)
        if int(classes.shape[0]) == 2:
            self.coef_ = (coefficient_matrix[1] - coefficient_matrix[0]).reshape(1, -1)
            self.intercept_ = np.asarray(
                [float(intercept_vector[1] - intercept_vector[0])],
                dtype=np.float64,
            )
            self._binary_mode_ = True
        else:
            self.coef_ = coefficient_matrix
            self.intercept_ = intercept_vector
            self._binary_mode_ = False

        gpu_memory_peak_mb: float | None = None
        if cuda_module is not None and hasattr(cuda_module, "max_memory_allocated"):
            try:
                peak_bytes = float(cuda_module.max_memory_allocated(device))
                gpu_memory_peak_mb = float(peak_bytes / (1024.0 * 1024.0))
            except Exception:  # pragma: no cover - best effort diagnostic only
                gpu_memory_peak_mb = None

        self.classes_ = classes
        self.n_features_in_ = int(x_array.shape[1])
        self._torch_device_name_ = f"cuda:{resolved_device_id}"
        self.backend_runtime_metadata_ = {
            "backend_id": self.backend_id,
            "gpu_memory_peak_mb": gpu_memory_peak_mb,
            "device_transfer_seconds": transfer_elapsed,
            "torch_deterministic_enforced": bool(deterministic_enforced),
            "torch_deterministic_limitations": deterministic_limitations,
        }
        return self

    def _decision_scores(self, x_matrix: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["coef_", "intercept_", "classes_", "n_features_in_"])
        x_array = np.asarray(x_matrix, dtype=np.float64)
        if x_array.ndim != 2:
            raise ValueError("TorchRidgeClassifier.decision_function requires a 2D feature matrix.")
        if int(x_array.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                "TorchRidgeClassifier.decision_function received an unexpected number of features."
            )
        torch = _import_torch_module()
        device_name = str(getattr(self, "_torch_device_name_", f"cuda:{int(self.gpu_device_id)}"))
        device = torch.device(device_name) if hasattr(torch, "device") else device_name
        x_tensor = _as_torch_float_tensor(torch, x_array, device=device)
        coef_tensor = _as_torch_float_tensor(
            torch,
            np.asarray(self.coef_, dtype=np.float64),
            device=device,
        )
        score_tensor = x_tensor @ coef_tensor.T
        if np.asarray(self.intercept_).size:
            intercept_tensor = _as_torch_float_tensor(
                torch,
                np.asarray(self.intercept_, dtype=np.float64).reshape(1, -1),
                device=device,
            )
            score_tensor = score_tensor + intercept_tensor
        score_array = _to_numpy_array(score_tensor).astype(np.float64, copy=False)
        if bool(getattr(self, "_binary_mode_", False)):
            return score_array.reshape(-1)
        return score_array

    def decision_function(self, x_matrix: np.ndarray) -> np.ndarray:
        return self._decision_scores(x_matrix)

    def predict(self, x_matrix: np.ndarray) -> np.ndarray:
        scores = self._decision_scores(x_matrix)
        if bool(getattr(self, "_binary_mode_", False)):
            class_indices = (np.asarray(scores, dtype=np.float64) >= 0.0).astype(int)
        else:
            class_indices = np.asarray(scores, dtype=np.float64).argmax(axis=1)
        return np.asarray(self.classes_)[class_indices]

    def get_backend_runtime_metadata(self) -> dict[str, Any]:
        metadata = getattr(self, "backend_runtime_metadata_", None)
        return dict(metadata) if isinstance(metadata, dict) else {}


def make_torch_ridge_estimator(
    *,
    seed: int,
    class_weight_policy: str = "none",
    gpu_device_id: int = 0,
    deterministic_compute: bool = False,
) -> TorchRidgeClassifier:
    return TorchRidgeClassifier(
        random_state=int(seed),
        class_weight=resolve_torch_ridge_class_weight(class_weight_policy),
        gpu_device_id=int(gpu_device_id),
        deterministic_compute=bool(deterministic_compute),
    )


__all__ = [
    "TORCH_RIDGE_BACKEND_ID",
    "TorchRidgeClassifier",
    "make_torch_ridge_estimator",
    "resolve_torch_ridge_class_weight",
]
