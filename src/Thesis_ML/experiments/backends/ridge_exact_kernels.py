from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RidgeExactAlphaFactorizationState:
    classes: np.ndarray
    fit_intercept: bool
    binary_mode: bool
    feature_mean: np.ndarray
    target_mean: np.ndarray
    weighted_x: np.ndarray
    solver_family: str
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    rhs_rotated: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.weighted_x.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.weighted_x.shape[1])

    @property
    def target_columns(self) -> int:
        return int(self.rhs_rotated.shape[1])


def to_numpy_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def as_torch_float_tensor(torch: Any, value: np.ndarray, *, device: Any) -> Any:
    dtype = getattr(torch, "float64", None)
    if dtype is None:
        return torch.as_tensor(value, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def build_ridge_target_matrix(
    y_indices: np.ndarray,
    *,
    n_classes: int,
) -> tuple[np.ndarray, bool]:
    n_samples = int(y_indices.shape[0])

    if n_classes == 2:
        target = np.where(y_indices == 1, 1.0, -1.0).astype(np.float64).reshape(n_samples, 1)
        return target, True

    target = -np.ones((n_samples, n_classes), dtype=np.float64)
    target[np.arange(n_samples), y_indices] = 1.0
    return target, False


def prepare_weighted_centered_problem(
    x_array: np.ndarray,
    target_matrix: np.ndarray,
    sample_weights: np.ndarray,
    *,
    fit_intercept: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if weights.ndim != 1 or weights.shape[0] != x_array.shape[0]:
        raise ValueError("sample_weights must be a 1D vector aligned with X rows.")

    if fit_intercept:
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            raise ValueError("Weighted ridge requires a strictly positive total sample weight.")

        feature_mean = np.sum(x_array * weights[:, None], axis=0) / total_weight
        target_mean = np.sum(target_matrix * weights[:, None], axis=0) / total_weight

        x_centered = x_array - feature_mean
        target_centered = target_matrix - target_mean
    else:
        feature_mean = np.zeros(x_array.shape[1], dtype=np.float64)
        target_mean = np.zeros(target_matrix.shape[1], dtype=np.float64)
        x_centered = x_array
        target_centered = target_matrix

    weight_sqrt = np.sqrt(weights).astype(np.float64, copy=False).reshape(-1, 1)
    weighted_x = x_centered * weight_sqrt
    weighted_target = target_centered * weight_sqrt

    return weighted_x, weighted_target, feature_mean, target_mean


def add_alpha_to_diagonal(
    torch: Any,
    matrix: Any,
    alpha: float,
    *,
    device: Any,
) -> Any:
    if float(alpha) < 0.0:
        raise ValueError("Ridge alpha must be >= 0.")

    diagonal_fn = getattr(torch, "diagonal", None)
    if callable(diagonal_fn):
        try:
            diag_view = diagonal_fn(matrix)
            add_in_place = getattr(diag_view, "add_", None)
            if callable(add_in_place):
                add_in_place(float(alpha))
                return matrix
        except Exception:
            pass

    matrix_np = to_numpy_array(matrix).astype(np.float64, copy=True)
    matrix_np.flat[:: matrix_np.shape[0] + 1] += float(alpha)
    return as_torch_float_tensor(torch, matrix_np, device=device)


def solve_spd_system(
    torch: Any,
    system_matrix: Any,
    rhs: Any,
) -> Any:
    linalg = getattr(torch, "linalg", None)

    if linalg is not None:
        cholesky_ex = getattr(linalg, "cholesky_ex", None)
        if callable(cholesky_ex):
            try:
                chol_out = cholesky_ex(system_matrix, check_errors=False)
                chol_factor = chol_out.L if hasattr(chol_out, "L") else chol_out[0]
                chol_info = (
                    chol_out.info
                    if hasattr(chol_out, "info")
                    else (
                        chol_out[1] if isinstance(chol_out, tuple) and len(chol_out) > 1 else None
                    )
                )
                if chol_info is None or np.all(to_numpy_array(chol_info) == 0):
                    cholesky_solve = getattr(torch, "cholesky_solve", None)
                    if callable(cholesky_solve):
                        return cholesky_solve(rhs, chol_factor)
            except Exception:
                pass

        solve_ex = getattr(linalg, "solve_ex", None)
        if callable(solve_ex):
            try:
                solve_out = solve_ex(system_matrix, rhs, check_errors=False)
                solve_result = solve_out.result if hasattr(solve_out, "result") else solve_out[0]
                solve_info = (
                    solve_out.info
                    if hasattr(solve_out, "info")
                    else (
                        solve_out[1] if isinstance(solve_out, tuple) and len(solve_out) > 1 else None
                    )
                )
                if solve_info is None or np.all(to_numpy_array(solve_info) == 0):
                    return solve_result
            except Exception:
                pass

        solve = getattr(linalg, "solve", None)
        if callable(solve):
            return solve(system_matrix, rhs)

    raise RuntimeError("Torch ridge backend could not find a usable linear solver.")


def resolve_cholesky_factor(torch: Any, system_matrix: Any) -> Any:
    linalg = getattr(torch, "linalg", None)

    if linalg is not None:
        cholesky_ex = getattr(linalg, "cholesky_ex", None)
        if callable(cholesky_ex):
            chol_out = cholesky_ex(system_matrix, check_errors=False)
            chol_factor = chol_out.L if hasattr(chol_out, "L") else chol_out[0]
            chol_info = (
                chol_out.info
                if hasattr(chol_out, "info")
                else (chol_out[1] if isinstance(chol_out, tuple) and len(chol_out) > 1 else None)
            )
            if chol_info is None or np.all(to_numpy_array(chol_info) == 0):
                return chol_factor

        cholesky = getattr(linalg, "cholesky", None)
        if callable(cholesky):
            return cholesky(system_matrix)

    matrix_np = to_numpy_array(system_matrix).astype(np.float64, copy=False)
    chol_np = np.linalg.cholesky(matrix_np)
    return as_torch_float_tensor(torch, chol_np, device=getattr(system_matrix, "device", None))


def solve_cholesky_system(
    torch: Any,
    *,
    cholesky_factor: Any,
    rhs: Any,
    system_matrix: Any,
) -> Any:
    cholesky_solve = getattr(torch, "cholesky_solve", None)
    if callable(cholesky_solve):
        return cholesky_solve(rhs, cholesky_factor)

    linalg = getattr(torch, "linalg", None)
    solve_triangular = getattr(linalg, "solve_triangular", None) if linalg is not None else None
    if callable(solve_triangular):
        intermediate = solve_triangular(cholesky_factor, rhs, upper=False)
        return solve_triangular(cholesky_factor.T, intermediate, upper=True)

    return solve_spd_system(torch, system_matrix, rhs)


def encode_binary_targets_from_labels(
    labels: np.ndarray,
    *,
    classes: np.ndarray,
) -> np.ndarray:
    labels_array = np.asarray(labels).astype(str, copy=False)
    class_array = np.asarray(classes).astype(str, copy=False)
    if class_array.shape[0] != 2:
        raise ValueError("Binary ridge encoding requires exactly two classes.")
    positive_class = str(class_array[1])
    return np.where(labels_array == positive_class, 1.0, -1.0).astype(np.float64, copy=False)


def build_ridge_gpu_permutation_core_state(
    *,
    x_train_scaled: np.ndarray,
    x_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    fit_intercept: bool,
    torch: Any,
    device: Any,
    batch_size_hint: int = 16,
) -> tuple[dict[str, Any], dict[str, float]]:
    x_train = np.asarray(x_train_scaled, dtype=np.float64)
    x_test = np.asarray(x_test_scaled, dtype=np.float64)
    y_train_text = np.asarray(y_train).astype(str, copy=False)
    y_test_text = np.asarray(y_test).astype(str, copy=False)

    classes = np.unique(y_train_text)
    if classes.shape[0] != 2:
        raise ValueError("ridge_gpu_batched_dual currently supports binary folds only.")

    encoded_targets = encode_binary_targets_from_labels(y_train_text, classes=classes)
    target_mean = float(encoded_targets.mean()) if bool(fit_intercept) else 0.0

    if bool(fit_intercept):
        feature_mean = np.mean(x_train, axis=0, dtype=np.float64)
        x_train_effective = x_train - feature_mean
        x_test_effective = x_test - feature_mean
    else:
        x_train_effective = x_train
        x_test_effective = x_test

    build_start = perf_counter()
    x_train_tensor = as_torch_float_tensor(torch, x_train_effective, device=device)
    x_test_tensor = as_torch_float_tensor(torch, x_test_effective, device=device)

    k_train = x_train_tensor @ x_train_tensor.T
    system_matrix = add_alpha_to_diagonal(
        torch,
        k_train,
        float(alpha),
        device=device,
    )

    factorization_start = perf_counter()
    cholesky_factor = resolve_cholesky_factor(torch, system_matrix)
    factorization_seconds = float(perf_counter() - factorization_start)

    k_test = x_test_tensor @ x_train_tensor.T

    # Retain only fold-resident dual solve state.
    del x_train_tensor
    del x_test_tensor
    del k_train
    fold_state_build_seconds = float(perf_counter() - build_start)

    return {
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "classes": np.asarray(classes).astype(str, copy=False),
        "y_test": np.asarray(y_test_text).astype(str, copy=False),
        "fit_intercept": bool(fit_intercept),
        "alpha": float(alpha),
        "target_mean": float(target_mean),
        "batch_size_hint": max(1, int(batch_size_hint)),
        "cholesky_factor": cholesky_factor,
        "system_matrix": system_matrix,
        "k_test": k_test,
    }, {
        "fold_gpu_state_build_seconds": float(fold_state_build_seconds),
        "fold_factorization_seconds": float(factorization_seconds),
    }


def solve_ridge_gpu_permutation_core_batch(
    *,
    state: dict[str, Any],
    permuted_train_labels_batch: np.ndarray,
    torch: Any,
    device: Any,
) -> tuple[np.ndarray, dict[str, float]]:
    labels_batch = np.asarray(permuted_train_labels_batch)
    if labels_batch.ndim == 1:
        labels_batch = labels_batch.reshape(-1, 1)
    if labels_batch.ndim != 2:
        raise ValueError("permuted_train_labels_batch must be 2D for batched ridge permutation.")
    if int(labels_batch.shape[0]) != int(state["n_train"]):
        raise ValueError(
            "permuted_train_labels_batch row count does not match cached fold training size."
        )

    encode_start = perf_counter()
    encoded_targets = encode_binary_targets_from_labels(labels_batch, classes=state["classes"])
    if bool(state["fit_intercept"]):
        encoded_targets = encoded_targets - float(state["target_mean"])
    rhs_tensor = as_torch_float_tensor(
        torch,
        np.asarray(encoded_targets, dtype=np.float64),
        device=device,
    )
    encode_seconds = float(perf_counter() - encode_start)

    solve_start = perf_counter()
    dual_coefficients = solve_cholesky_system(
        torch,
        cholesky_factor=state["cholesky_factor"],
        rhs=rhs_tensor,
        system_matrix=state["system_matrix"],
    )
    batched_solve_seconds = float(perf_counter() - solve_start)

    predict_start = perf_counter()
    score_tensor = state["k_test"] @ dual_coefficients
    if bool(state["fit_intercept"]):
        score_tensor = score_tensor + float(state["target_mean"])
    score_array = to_numpy_array(score_tensor).astype(np.float64, copy=False)
    if score_array.ndim == 1:
        score_array = score_array.reshape(-1, 1)
    batched_predict_seconds = float(perf_counter() - predict_start)

    return score_array, {
        "batched_target_encode_seconds": float(encode_seconds),
        "batched_solve_seconds": float(batched_solve_seconds),
        "batched_predict_seconds": float(batched_predict_seconds),
    }


def build_ridge_exact_alpha_factorization_state(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    fit_intercept: bool,
    sample_weights: np.ndarray,
) -> RidgeExactAlphaFactorizationState:
    x_array = np.asarray(x_train, dtype=np.float64)
    y_array = np.asarray(y_train).astype(str, copy=False)
    if x_array.ndim != 2:
        raise ValueError("Ridge exact alpha factorization requires a 2D feature matrix.")
    if y_array.ndim != 1:
        raise ValueError("Ridge exact alpha factorization requires a 1D label vector.")
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("Ridge exact alpha factorization received mismatched X/y row counts.")

    classes, y_indices = np.unique(y_array, return_inverse=True)
    if classes.shape[0] < 2:
        raise ValueError("Ridge exact alpha factorization requires at least two classes.")

    target_matrix, binary_mode = build_ridge_target_matrix(
        y_indices.astype(np.int64, copy=False),
        n_classes=int(classes.shape[0]),
    )
    weighted_x, weighted_target, feature_mean, target_mean = prepare_weighted_centered_problem(
        x_array,
        target_matrix,
        sample_weights,
        fit_intercept=bool(fit_intercept),
    )

    n_samples, n_features = weighted_x.shape
    if int(n_features) > int(n_samples):
        solver_family = "dual"
        system = weighted_x @ weighted_x.T
        rhs = weighted_target
    else:
        solver_family = "primal"
        system = weighted_x.T @ weighted_x
        rhs = weighted_x.T @ weighted_target

    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(system, dtype=np.float64, copy=False))
    rhs_rotated = eigenvectors.T @ rhs

    return RidgeExactAlphaFactorizationState(
        classes=np.asarray(classes).astype(str, copy=False),
        fit_intercept=bool(fit_intercept),
        binary_mode=bool(binary_mode),
        feature_mean=np.asarray(feature_mean, dtype=np.float64),
        target_mean=np.asarray(target_mean, dtype=np.float64),
        weighted_x=np.asarray(weighted_x, dtype=np.float64),
        solver_family=str(solver_family),
        eigenvectors=np.asarray(eigenvectors, dtype=np.float64),
        eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        rhs_rotated=np.asarray(rhs_rotated, dtype=np.float64),
    )


def solve_ridge_exact_alpha_batch(
    *,
    state: RidgeExactAlphaFactorizationState,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    alpha_values = np.asarray(alphas, dtype=np.float64).reshape(-1)
    if alpha_values.size == 0:
        raise ValueError("Ridge exact alpha batch requires at least one alpha.")
    if not np.all(np.isfinite(alpha_values)):
        raise ValueError("Ridge exact alpha batch requires finite alpha values.")
    if np.any(alpha_values < 0.0):
        raise ValueError("Ridge exact alpha batch requires alpha >= 0.")

    denominator = state.eigenvalues[:, None] + alpha_values[None, :]
    if np.any(denominator <= 0.0):
        raise ValueError("Ridge exact alpha batch produced non-positive denominator.")

    rotated_solution = state.rhs_rotated[:, :, None] / denominator[:, None, :]

    if state.solver_family == "dual":
        dual_solution = np.einsum("ij,jka->ika", state.eigenvectors, rotated_solution)
        weight_stack = np.einsum("pn,nka->pka", state.weighted_x.T, dual_solution)
    else:
        weight_stack = np.einsum("ij,jka->ika", state.eigenvectors, rotated_solution)

    # (n_alphas, n_features, n_targets)
    weight_batch = np.moveaxis(np.asarray(weight_stack, dtype=np.float64), 2, 0)

    if bool(state.fit_intercept):
        intercept_batch = state.target_mean.reshape(1, -1) - np.einsum(
            "p,apk->ak",
            state.feature_mean,
            weight_batch,
        )
    else:
        intercept_batch = np.zeros((int(alpha_values.shape[0]), state.target_columns), dtype=np.float64)

    return weight_batch, np.asarray(intercept_batch, dtype=np.float64)


def predict_ridge_labels_for_alpha_batch(
    *,
    x_eval: np.ndarray,
    weight_batch: np.ndarray,
    intercept_batch: np.ndarray,
    classes: np.ndarray,
    binary_mode: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x_eval, dtype=np.float64)
    if x_array.ndim != 2:
        raise ValueError("Ridge alpha batch prediction requires a 2D evaluation matrix.")

    scores = np.einsum("np,apk->ank", x_array, weight_batch)
    scores = scores + intercept_batch[:, None, :]

    classes_array = np.asarray(classes).astype(str, copy=False)
    if bool(binary_mode):
        if classes_array.shape[0] != 2:
            raise ValueError("Binary ridge alpha batch prediction requires exactly two classes.")
        binary_scores = np.asarray(scores[:, :, 0], dtype=np.float64)
        predictions = np.where(
            binary_scores >= 0.0,
            str(classes_array[1]),
            str(classes_array[0]),
        )
        return predictions.astype(str, copy=False), binary_scores

    class_indices = np.argmax(scores, axis=2)
    predictions = classes_array[class_indices]
    return predictions.astype(str, copy=False), np.asarray(scores, dtype=np.float64)


__all__ = [
    "RidgeExactAlphaFactorizationState",
    "add_alpha_to_diagonal",
    "as_torch_float_tensor",
    "build_ridge_exact_alpha_factorization_state",
    "build_ridge_gpu_permutation_core_state",
    "build_ridge_target_matrix",
    "encode_binary_targets_from_labels",
    "predict_ridge_labels_for_alpha_batch",
    "prepare_weighted_centered_problem",
    "resolve_cholesky_factor",
    "solve_cholesky_system",
    "solve_ridge_exact_alpha_batch",
    "solve_ridge_gpu_permutation_core_batch",
    "solve_spd_system",
    "to_numpy_array",
]
