from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def build_model_metadata_payload(
    *,
    artifact_role: str,
    experiment_id: str | None,
    run_id: str,
    model_logical_name: str,
    target: str,
    cv_mode: str,
    seed: int,
    variant_id: str | None = None,
    backend_family: str | None = None,
    backend_id: str | None = None,
    subject: str | None = None,
    train_subject: str | None = None,
    test_subject: str | None = None,
    fold_index: int | None = None,
    split_identity: str | None = None,
    feature_space: str | None = None,
    preprocessing_strategy: str | None = None,
    dimensionality_strategy: str | None = None,
    class_labels: list[str] | None = None,
    n_train: int | None = None,
    n_test: int | None = None,
    n_features: int | None = None,
    config_filename: str | None = None,
    runtime_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact_role": str(artifact_role),
        "experiment_id": str(experiment_id) if experiment_id else None,
        "run_id": str(run_id),
        "variant_id": str(variant_id) if variant_id else None,
        "model_logical_name": str(model_logical_name),
        "backend_family": str(backend_family) if backend_family else None,
        "backend_id": str(backend_id) if backend_id else None,
        "seed": int(seed),
        "target": str(target),
        "cv_mode": str(cv_mode),
        "subject": str(subject) if subject else None,
        "train_subject": str(train_subject) if train_subject else None,
        "test_subject": str(test_subject) if test_subject else None,
        "fold_index": int(fold_index) if fold_index is not None else None,
        "split_identity": str(split_identity) if split_identity else None,
        "feature_space": str(feature_space) if feature_space else None,
        "preprocessing_strategy": (
            str(preprocessing_strategy) if preprocessing_strategy is not None else None
        ),
        "dimensionality_strategy": (
            str(dimensionality_strategy) if dimensionality_strategy is not None else None
        ),
        "class_labels": [str(value) for value in class_labels] if class_labels else None,
        "n_train": int(n_train) if n_train is not None else None,
        "n_test": int(n_test) if n_test is not None else None,
        "n_features": int(n_features) if n_features is not None else None,
        "config_filename": str(config_filename) if config_filename else None,
        "runtime_info": dict(runtime_info) if isinstance(runtime_info, dict) else None,
    }
    return payload


def save_trained_estimator(
    *,
    estimator: Any,
    model_path: Path,
    metadata_path: Path,
    metadata_payload: dict[str, Any],
) -> dict[str, Any]:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_path)
    metadata_path.write_text(f"{json.dumps(metadata_payload, indent=2)}\n", encoding="utf-8")
    return {
        "status": "saved",
        "model_path": str(model_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "model_size_bytes": int(model_path.stat().st_size),
        "metadata_size_bytes": int(metadata_path.stat().st_size),
    }


def load_trained_estimator(
    *,
    model_path: Path,
    metadata_path: Path,
) -> tuple[Any, dict[str, Any]]:
    estimator = joblib.load(model_path)
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata_payload, dict):
        raise ValueError("Model metadata payload must be a JSON object.")
    return estimator, metadata_payload


def write_model_summary(
    *,
    models_dir: Path,
    saved_model_rows: list[dict[str, Any]],
    run_metadata: dict[str, Any],
) -> tuple[Path, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    summary_path = models_dir / "model_summary.json"
    csv_path = models_dir / "model_artifacts.csv"

    fold_rows = [
        row for row in saved_model_rows if str(row.get("artifact_role")) == "fold_model"
    ]
    final_rows = [
        row
        for row in saved_model_rows
        if str(row.get("artifact_role")) == "final_refit_after_confirmatory_evaluation"
    ]
    summary_payload = {
        "status": "captured" if saved_model_rows else "disabled_or_empty",
        "n_models": int(len(saved_model_rows)),
        "n_fold_models": int(len(fold_rows)),
        "n_final_refit_models": int(len(final_rows)),
        "models": saved_model_rows,
        "run_metadata": dict(run_metadata),
    }
    summary_path.write_text(f"{json.dumps(summary_payload, indent=2)}\n", encoding="utf-8")

    frame = pd.DataFrame(saved_model_rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "artifact_role",
                "fold",
                "direction",
                "model_path",
                "metadata_path",
                "subject",
                "train_subject",
                "test_subject",
                "held_out_identity",
                "n_train",
                "n_test",
                "n_features",
                "class_labels",
                "status",
            ]
        )
    frame.to_csv(csv_path, index=False)
    return summary_path, csv_path


__all__ = [
    "build_model_metadata_payload",
    "save_trained_estimator",
    "load_trained_estimator",
    "write_model_summary",
]
