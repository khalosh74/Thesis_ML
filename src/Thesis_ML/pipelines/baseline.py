"""Deterministic synthetic baseline pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Thesis_ML.config.paths import DEFAULT_BASELINE_MODELS_DIR, DEFAULT_BASELINE_REPORTS_DIR


def _validate_config(
    n_samples: int,
    n_features: int,
    noise_std: float,
    test_fraction: float,
) -> None:
    if n_samples < 3:
        raise ValueError("n_samples must be at least 3.")
    if n_features < 1:
        raise ValueError("n_features must be at least 1.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")


def _train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = x.shape[0]
    test_size = int(round(n_samples * test_fraction))
    test_size = max(1, min(test_size, n_samples - 1))

    indices = rng.permutation(n_samples)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def _fit_linear_regression_closed_form(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, np.ndarray]:
    x_augmented = np.column_stack((np.ones(x_train.shape[0]), x_train))
    params, *_ = np.linalg.lstsq(x_augmented, y_train, rcond=None)
    intercept = float(params[0])
    weights = params[1:]
    return intercept, weights


def _predict(x: np.ndarray, intercept: float, weights: np.ndarray) -> np.ndarray:
    return (x @ weights) + intercept


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    total_variance = float(np.sum((y_true - np.mean(y_true)) ** 2))
    residual_variance = float(np.sum(residuals**2))
    r2 = 1.0 - (residual_variance / total_variance) if total_variance > 0 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
    }


def run_baseline(
    seed: int = 42,
    n_samples: int = 500,
    n_features: int = 12,
    noise_std: float = 0.3,
    test_fraction: float = 0.2,
    reports_dir: Path | str = DEFAULT_BASELINE_REPORTS_DIR,
    models_dir: Path | str = DEFAULT_BASELINE_MODELS_DIR,
) -> dict[str, object]:
    """
    Run a deterministic synthetic regression baseline and persist outputs locally.

    Returns a dictionary containing metrics and generated file paths.
    """
    _validate_config(
        n_samples=n_samples,
        n_features=n_features,
        noise_std=noise_std,
        test_fraction=test_fraction,
    )

    rng = np.random.default_rng(seed)

    true_weights = rng.normal(loc=0.0, scale=1.0, size=n_features)
    x = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_samples)
    y = (x @ true_weights) + 0.5 + noise

    x_train, x_test, y_train, y_test = _train_test_split(
        x=x,
        y=y,
        test_fraction=test_fraction,
        rng=rng,
    )
    intercept, weights = _fit_linear_regression_closed_form(x_train=x_train, y_train=y_train)
    y_pred = _predict(x=x_test, intercept=intercept, weights=weights)

    metrics = _calculate_metrics(y_true=y_test, y_pred=y_pred)
    metrics.update(
        {
            "seed": int(seed),
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "noise_std": float(noise_std),
            "test_fraction": float(test_fraction),
        }
    )

    reports_path = Path(reports_dir)
    models_path = Path(models_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    metrics_file = reports_path / "metrics.json"
    metrics_file.write_text(f"{json.dumps(metrics, indent=2)}\n", encoding="utf-8")

    model_file = models_path / "baseline_model.npz"
    np.savez(
        model_file,
        intercept=np.array(intercept),
        weights=weights.astype(float),
        seed=np.int64(seed),
        n_samples=np.int64(n_samples),
        n_features=np.int64(n_features),
        noise_std=np.float64(noise_std),
        test_fraction=np.float64(test_fraction),
    )

    return {
        "metrics": metrics,
        "metrics_path": str(metrics_file),
        "model_path": str(model_file),
    }
