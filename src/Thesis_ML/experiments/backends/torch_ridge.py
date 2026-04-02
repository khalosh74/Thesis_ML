from __future__ import annotations

import importlib
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from Thesis_ML.experiments.backends.ridge_exact_kernels import (
    add_alpha_to_diagonal as _kernel_add_alpha_to_diagonal,
)
from Thesis_ML.experiments.backends.ridge_exact_kernels import (
    as_torch_float_tensor as _kernel_as_torch_float_tensor,
)
from Thesis_ML.experiments.backends.ridge_exact_kernels import (
    build_ridge_gpu_permutation_core_state,
    solve_ridge_gpu_permutation_core_batch,
)
from Thesis_ML.experiments.backends.ridge_exact_kernels import (
    build_ridge_target_matrix as _kernel_build_ridge_target_matrix,
)
from Thesis_ML.experiments.backends.ridge_exact_kernels import (
    encode_binary_targets_from_labels as _kernel_encode_binary_targets_from_labels,
)
from Thesis_ML.experiments.backends.ridge_exact_kernels import (
    prepare_weighted_centered_problem as _kernel_prepare_weighted_centered_problem,
)
from Thesis_ML.experiments.backends.ridge_exact_kernels import (
    resolve_cholesky_factor as _kernel_resolve_cholesky_factor,
)
from Thesis_ML.experiments.backends.ridge_exact_kernels import (
    solve_cholesky_system as _kernel_solve_cholesky_system,
)
from Thesis_ML.experiments.backends.ridge_exact_kernels import (
    solve_spd_system as _kernel_solve_spd_system,
)
from Thesis_ML.experiments.backends.ridge_exact_kernels import (
    to_numpy_array as _kernel_to_numpy_array,
)

TORCH_RIDGE_BACKEND_ID = "torch_ridge_gpu_v2"


@dataclass(frozen=True)
class RidgeGpuPermutationFoldState:
    fold_index: int
    n_train: int
    n_test: int
    classes: np.ndarray
    y_test: np.ndarray
    fit_intercept: bool
    alpha: float
    target_mean: float
    batch_size_hint: int
    torch_module: Any
    device: Any
    cholesky_factor: Any
    system_matrix: Any
    k_test: Any


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
    return _kernel_to_numpy_array(value)


def _as_torch_float_tensor(torch: Any, value: np.ndarray, *, device: Any) -> Any:
    return _kernel_as_torch_float_tensor(torch, value, device=device)


def _resolve_cuda_device(torch: Any, requested_device_id: int) -> tuple[Any, int]:
    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is None:
        raise RuntimeError("Torch is installed but torch.cuda is unavailable.")
    if not bool(cuda_module.is_available()):
        raise RuntimeError("Torch CUDA backend is unavailable (torch.cuda.is_available()=False).")
    try:
        gpu_count = int(cuda_module.device_count())
    except Exception as exc:  # pragma: no cover - defensive probe guard
        raise RuntimeError(f"Failed to probe CUDA device count: {exc.__class__.__name__}.") from exc
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


def _build_ridge_target_matrix(
    y_indices: np.ndarray,
    *,
    n_classes: int,
) -> tuple[np.ndarray, bool]:
    return _kernel_build_ridge_target_matrix(y_indices, n_classes=n_classes)


def _prepare_weighted_centered_problem(
    x_array: np.ndarray,
    target_matrix: np.ndarray,
    sample_weights: np.ndarray,
    *,
    fit_intercept: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _kernel_prepare_weighted_centered_problem(
        x_array,
        target_matrix,
        sample_weights,
        fit_intercept=fit_intercept,
    )


def _add_alpha_to_diagonal(
    torch: Any,
    matrix: Any,
    alpha: float,
    *,
    device: Any,
) -> Any:
    return _kernel_add_alpha_to_diagonal(torch, matrix, alpha, device=device)


def _solve_spd_system(
    torch: Any,
    system_matrix: Any,
    rhs: Any,
) -> Any:
    return _kernel_solve_spd_system(torch, system_matrix, rhs)


def _resolve_cholesky_factor(torch: Any, system_matrix: Any) -> Any:
    return _kernel_resolve_cholesky_factor(torch, system_matrix)


def _solve_cholesky_system(
    torch: Any,
    *,
    cholesky_factor: Any,
    rhs: Any,
    system_matrix: Any,
) -> Any:
    return _kernel_solve_cholesky_system(
        torch,
        cholesky_factor=cholesky_factor,
        rhs=rhs,
        system_matrix=system_matrix,
    )


def supports_ridge_gpu_batched_dual(estimator: Any) -> tuple[bool, str | None]:
    backend_id = str(getattr(estimator, "backend_id", "")).strip().lower()
    if backend_id != TORCH_RIDGE_BACKEND_ID:
        return False, "backend_id_not_torch_ridge_gpu"

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
            return False, "ridge_gpu_batched_dual_requires_class_weight_none"

    return True, None


def _encode_binary_targets_from_labels(
    labels: np.ndarray,
    *,
    classes: np.ndarray,
) -> np.ndarray:
    return _kernel_encode_binary_targets_from_labels(labels, classes=classes)


def build_ridge_gpu_permutation_fold_state(
    *,
    fold_index: int,
    x_train_scaled: np.ndarray,
    x_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    estimator: Any,
    batch_size_hint: int = 16,
) -> tuple[RidgeGpuPermutationFoldState, dict[str, float]]:
    supported, reason = supports_ridge_gpu_batched_dual(estimator)
    if not supported:
        raise ValueError(reason or "ridge_gpu_batched_dual_not_supported")

    torch = _import_torch_module()
    if getattr(torch, "float64", None) is None:
        raise ValueError("ridge_gpu_batched_dual_requires_torch_float64_support")
    device, _ = _resolve_cuda_device(torch, int(getattr(estimator, "gpu_device_id", 0)))
    _configure_torch_determinism(
        torch,
        enabled=bool(getattr(estimator, "deterministic_compute", False)),
    )

    core_state, timing_metadata = build_ridge_gpu_permutation_core_state(
        x_train_scaled=np.asarray(x_train_scaled, dtype=np.float64),
        x_test_scaled=np.asarray(x_test_scaled, dtype=np.float64),
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        alpha=float(getattr(estimator, "alpha", 1.0)),
        fit_intercept=bool(getattr(estimator, "fit_intercept", True)),
        torch=torch,
        device=device,
        batch_size_hint=int(batch_size_hint),
    )

    state = RidgeGpuPermutationFoldState(
        fold_index=int(fold_index),
        n_train=int(core_state["n_train"]),
        n_test=int(core_state["n_test"]),
        classes=np.asarray(core_state["classes"]).astype(str, copy=False),
        y_test=np.asarray(core_state["y_test"]).astype(str, copy=False),
        fit_intercept=bool(core_state["fit_intercept"]),
        alpha=float(core_state["alpha"]),
        target_mean=float(core_state["target_mean"]),
        batch_size_hint=int(core_state["batch_size_hint"]),
        torch_module=torch,
        device=device,
        cholesky_factor=core_state["cholesky_factor"],
        system_matrix=core_state["system_matrix"],
        k_test=core_state["k_test"],
    )
    return state, timing_metadata


def solve_ridge_gpu_permutation_batch(
    *,
    state: RidgeGpuPermutationFoldState,
    permuted_train_labels_batch: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    core_state = {
        "n_train": int(state.n_train),
        "classes": np.asarray(state.classes).astype(str, copy=False),
        "fit_intercept": bool(state.fit_intercept),
        "target_mean": float(state.target_mean),
        "cholesky_factor": state.cholesky_factor,
        "system_matrix": state.system_matrix,
        "k_test": state.k_test,
    }
    return solve_ridge_gpu_permutation_core_batch(
        state=core_state,
        permuted_train_labels_batch=permuted_train_labels_batch,
        torch=state.torch_module,
        device=state.device,
    )


def _solve_adaptive_ridge_exact(
    torch: Any,
    x_array: np.ndarray,
    y_indices: np.ndarray,
    sample_weights: np.ndarray,
    *,
    alpha: float,
    fit_intercept: bool,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
    n_samples, n_features = x_array.shape
    n_classes = int(np.max(y_indices)) + 1

    target_matrix, binary_mode = _build_ridge_target_matrix(
        y_indices,
        n_classes=n_classes,
    )

    weighted_x, weighted_target, feature_mean, target_mean = _prepare_weighted_centered_problem(
        x_array,
        target_matrix,
        sample_weights,
        fit_intercept=fit_intercept,
    )

    weighted_x_tensor = _as_torch_float_tensor(torch, weighted_x, device=device)
    weighted_target_tensor = _as_torch_float_tensor(torch, weighted_target, device=device)

    target_columns = int(weighted_target.shape[1])

    if n_features > n_samples:
        solver_family = "dual"
        system_dimension = int(n_samples)

        system_matrix = weighted_x_tensor @ weighted_x_tensor.T
        system_matrix = _add_alpha_to_diagonal(
            torch,
            system_matrix,
            float(alpha),
            device=device,
        )

        dual_solution = _solve_spd_system(
            torch,
            system_matrix,
            weighted_target_tensor,
        )
        weight_tensor = weighted_x_tensor.T @ dual_solution
    else:
        solver_family = "primal"
        system_dimension = int(n_features)

        system_matrix = weighted_x_tensor.T @ weighted_x_tensor
        system_matrix = _add_alpha_to_diagonal(
            torch,
            system_matrix,
            float(alpha),
            device=device,
        )

        rhs = weighted_x_tensor.T @ weighted_target_tensor
        weight_tensor = _solve_spd_system(
            torch,
            system_matrix,
            rhs,
        )

    weight_matrix = _to_numpy_array(weight_tensor).astype(np.float64, copy=False)
    if weight_matrix.ndim == 1:
        weight_matrix = weight_matrix.reshape(-1, 1)

    if fit_intercept:
        intercept_vector = target_mean - feature_mean @ weight_matrix
    else:
        intercept_vector = np.zeros(weight_matrix.shape[1], dtype=np.float64)

    metadata = {
        "ridge_solver_family": solver_family,
        "ridge_system_dimension": system_dimension,
        "ridge_rhs_columns": target_columns,
        "ridge_binary_mode": bool(binary_mode),
    }
    return (
        weight_matrix,
        np.asarray(intercept_vector, dtype=np.float64),
        bool(binary_mode),
        metadata,
    )


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

        if float(self.alpha) < 0.0:
            raise ValueError("TorchRidgeClassifier requires alpha >= 0.")
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
        solver_start = perf_counter()

        coefficient_weights, intercept_vector, binary_mode, ridge_solver_metadata = (
            _solve_adaptive_ridge_exact(
                torch,
                x_array,
                y_indices,
                sample_weights,
                alpha=float(self.alpha),
                fit_intercept=bool(self.fit_intercept),
                device=device,
            )
        )

        solver_elapsed = float(perf_counter() - solver_start)
        transfer_elapsed = float(perf_counter() - transfer_start)

        if binary_mode:
            # coefficient_weights: (p, 1)
            self.coef_ = coefficient_weights.T.astype(np.float64, copy=False)
            self.intercept_ = np.asarray([float(intercept_vector[0])], dtype=np.float64)
            self._binary_mode_ = True
        else:
            # coefficient_weights: (p, K)
            self.coef_ = coefficient_weights.T.astype(np.float64, copy=False)
            self.intercept_ = np.asarray(intercept_vector, dtype=np.float64)
            self._binary_mode_ = False

        gpu_memory_peak_mb: float | None = None
        if cuda_module is not None and hasattr(cuda_module, "max_memory_allocated"):
            try:
                peak_bytes = float(cuda_module.max_memory_allocated(device))
                gpu_memory_peak_mb = float(peak_bytes / (1024.0 * 1024.0))
            except Exception:
                gpu_memory_peak_mb = None

        self.classes_ = classes
        self.n_features_in_ = int(x_array.shape[1])
        self._torch_device_name_ = f"cuda:{resolved_device_id}"
        self.backend_runtime_metadata_ = {
            "backend_id": self.backend_id,
            "gpu_memory_peak_mb": gpu_memory_peak_mb,
            "device_transfer_seconds": transfer_elapsed,
            "ridge_solver_seconds": solver_elapsed,
            "torch_deterministic_enforced": bool(deterministic_enforced),
            "torch_deterministic_limitations": deterministic_limitations,
            **ridge_solver_metadata,
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
    "RidgeGpuPermutationFoldState",
    "TORCH_RIDGE_BACKEND_ID",
    "TorchRidgeClassifier",
    "build_ridge_gpu_permutation_fold_state",
    "make_torch_ridge_estimator",
    "resolve_torch_ridge_class_weight",
    "solve_ridge_gpu_permutation_batch",
    "supports_ridge_gpu_batched_dual",
]
