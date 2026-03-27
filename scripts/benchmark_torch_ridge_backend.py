from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Thesis_ML.experiments.backends.torch_ridge import TorchRidgeClassifier  # noqa: E402

CASE_LIBRARY: dict[str, dict[str, Any]] = {
    "voxel_binary_small": {
        "n_samples": 500,
        "n_features": 8000,
        "n_classes": 2,
        "informative_features": 256,
    },
    "voxel_binary_large": {
        "n_samples": 500,
        "n_features": 20000,
        "n_classes": 2,
        "informative_features": 256,
    },
    "voxel_multiclass_small": {
        "n_samples": 500,
        "n_features": 8000,
        "n_classes": 3,
        "informative_features": 256,
    },
    "sample_heavy_control": {
        "n_samples": 5000,
        "n_features": 1000,
        "n_classes": 2,
        "informative_features": 128,
    },
}


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _make_synthetic_dataset(
    *,
    n_samples: int,
    n_features: int,
    n_classes: int,
    informative_features: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    informative = int(min(max(4, informative_features), n_features))
    x_matrix = rng.standard_normal((n_samples, n_features)).astype(np.float64)

    if n_classes == 2:
        weight_vector = rng.standard_normal(informative).astype(np.float64)
        latent = x_matrix[:, :informative] @ weight_vector
        latent += 0.50 * rng.standard_normal(n_samples)
        threshold = float(np.median(latent))
        y_index = (latent >= threshold).astype(int)
        labels = np.where(y_index == 1, "pos", "neg").astype(object)
        return x_matrix, labels

    weight_matrix = rng.standard_normal((informative, n_classes)).astype(np.float64)
    logits = x_matrix[:, :informative] @ weight_matrix
    logits += 0.50 * rng.standard_normal((n_samples, n_classes))
    y_index = np.asarray(np.argmax(logits, axis=1), dtype=int)
    class_names = np.asarray([f"class_{idx}" for idx in range(n_classes)], dtype=object)
    labels = class_names[y_index]
    return x_matrix, labels


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _benchmark_torch_ridge(
    *,
    x_matrix: np.ndarray,
    y_labels: np.ndarray,
    alpha: float,
    gpu_device_id: int,
    deterministic_compute: bool,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    fit_seconds: list[float] = []
    predict_seconds: list[float] = []
    train_accuracy: list[float] = []
    gpu_memory_peak_mb: list[float] = []
    transfer_seconds: list[float] = []
    solver_seconds: list[float] = []
    solver_families: list[str] = []
    system_dimensions: list[int] = []

    total_loops = int(warmup + repeats)
    for loop_index in range(total_loops):
        estimator = TorchRidgeClassifier(
            alpha=float(alpha),
            fit_intercept=True,
            class_weight=None,
            random_state=0,
            gpu_device_id=int(gpu_device_id),
            deterministic_compute=bool(deterministic_compute),
        )

        _sync_cuda()
        fit_start = perf_counter()
        estimator.fit(x_matrix, y_labels)
        _sync_cuda()
        fit_elapsed = float(perf_counter() - fit_start)

        metadata = estimator.get_backend_runtime_metadata()

        _sync_cuda()
        predict_start = perf_counter()
        predictions = estimator.predict(x_matrix)
        _sync_cuda()
        predict_elapsed = float(perf_counter() - predict_start)

        accuracy = float(np.mean(np.asarray(predictions) == np.asarray(y_labels)))

        if loop_index >= warmup:
            fit_seconds.append(fit_elapsed)
            predict_seconds.append(predict_elapsed)
            train_accuracy.append(accuracy)

            gpu_peak = _to_float(metadata.get("gpu_memory_peak_mb"))
            if gpu_peak is not None:
                gpu_memory_peak_mb.append(gpu_peak)

            transfer_time = _to_float(metadata.get("device_transfer_seconds"))
            if transfer_time is not None:
                transfer_seconds.append(transfer_time)

            solver_time = _to_float(metadata.get("ridge_solver_seconds"))
            if solver_time is not None:
                solver_seconds.append(solver_time)

            solver_family = metadata.get("ridge_solver_family")
            if solver_family is not None:
                solver_families.append(str(solver_family))

            system_dimension = metadata.get("ridge_system_dimension")
            if system_dimension is not None:
                system_dimensions.append(int(system_dimension))

    summary: dict[str, Any] = {
        "fit_seconds": _summarize(fit_seconds),
        "predict_seconds": _summarize(predict_seconds),
        "train_accuracy": _summarize(train_accuracy),
        "fit_seconds_all": fit_seconds,
        "predict_seconds_all": predict_seconds,
        "train_accuracy_all": train_accuracy,
    }

    if gpu_memory_peak_mb:
        summary["gpu_memory_peak_mb"] = _summarize(gpu_memory_peak_mb)
        summary["gpu_memory_peak_mb_all"] = gpu_memory_peak_mb

    if transfer_seconds:
        summary["device_transfer_seconds"] = _summarize(transfer_seconds)
        summary["device_transfer_seconds_all"] = transfer_seconds

    if solver_seconds:
        summary["ridge_solver_seconds"] = _summarize(solver_seconds)
        summary["ridge_solver_seconds_all"] = solver_seconds

    if solver_families:
        summary["ridge_solver_families_seen"] = sorted(set(solver_families))

    if system_dimensions:
        summary["ridge_system_dimensions_seen"] = sorted(set(system_dimensions))

    return summary


def _benchmark_cpu_reference(
    *,
    x_matrix: np.ndarray,
    y_labels: np.ndarray,
    alpha: float,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    fit_seconds: list[float] = []
    predict_seconds: list[float] = []
    train_accuracy: list[float] = []

    total_loops = int(warmup + repeats)
    for loop_index in range(total_loops):
        estimator = RidgeClassifier(alpha=float(alpha), fit_intercept=True)

        fit_start = perf_counter()
        estimator.fit(x_matrix, y_labels)
        fit_elapsed = float(perf_counter() - fit_start)

        predict_start = perf_counter()
        predictions = estimator.predict(x_matrix)
        predict_elapsed = float(perf_counter() - predict_start)

        accuracy = float(np.mean(np.asarray(predictions) == np.asarray(y_labels)))

        if loop_index >= warmup:
            fit_seconds.append(fit_elapsed)
            predict_seconds.append(predict_elapsed)
            train_accuracy.append(accuracy)

    return {
        "fit_seconds": _summarize(fit_seconds),
        "predict_seconds": _summarize(predict_seconds),
        "train_accuracy": _summarize(train_accuracy),
        "fit_seconds_all": fit_seconds,
        "predict_seconds_all": predict_seconds,
        "train_accuracy_all": train_accuracy,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Torch ridge backend directly.")
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASE_LIBRARY.keys()),
        help="Repeat to run multiple named benchmark cases.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--deterministic-compute", action="store_true")
    parser.add_argument("--compare-cpu", action="store_true")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    selected_cases = (
        list(args.case)
        if args.case
        else [
            "voxel_binary_small",
            "sample_heavy_control",
            "voxel_multiclass_small",
        ]
    )

    summary: dict[str, Any] = {
        "benchmark": "torch_ridge_backend",
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "gpu_device_id": int(args.gpu_device_id),
        "deterministic_compute": bool(args.deterministic_compute),
        "cases": [],
    }

    for case_index, case_name in enumerate(selected_cases):
        config = dict(CASE_LIBRARY[case_name])
        x_matrix, y_labels = _make_synthetic_dataset(
            n_samples=int(config["n_samples"]),
            n_features=int(config["n_features"]),
            n_classes=int(config["n_classes"]),
            informative_features=int(config["informative_features"]),
            seed=int(args.seed + case_index),
        )

        case_result: dict[str, Any] = {
            "case_name": case_name,
            **config,
        }

        torch_result = _benchmark_torch_ridge(
            x_matrix=x_matrix,
            y_labels=y_labels,
            alpha=float(args.alpha),
            gpu_device_id=int(args.gpu_device_id),
            deterministic_compute=bool(args.deterministic_compute),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
        )
        case_result["torch_ridge"] = torch_result

        if bool(args.compare_cpu):
            cpu_result = _benchmark_cpu_reference(
                x_matrix=x_matrix,
                y_labels=y_labels,
                alpha=float(args.alpha),
                repeats=int(args.repeats),
                warmup=int(args.warmup),
            )
            case_result["cpu_ridge_reference"] = cpu_result

        summary["cases"].append(case_result)

        torch_fit_median = case_result["torch_ridge"]["fit_seconds"]["median"]
        torch_predict_median = case_result["torch_ridge"]["predict_seconds"]["median"]
        solver_seen = case_result["torch_ridge"].get("ridge_solver_families_seen", ["unknown"])
        print(
            f"[{case_name}] "
            f"fit_median={torch_fit_median:.4f}s "
            f"predict_median={torch_predict_median:.4f}s "
            f"solver={solver_seen}"
        )

    rendered = json.dumps(summary, indent=2)
    print(rendered)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(f"{rendered}\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
