"""Leakage-safe experiment runner with grouped cross-validation."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from Thesis_ML.features.nifti_features import build_feature_cache

LOGGER = logging.getLogger(__name__)

_MODEL_NAMES = ("logreg", "linearsvc", "ridge")
_TARGET_ALIASES = {
    "emotion": "emotion",
    "modality": "modality",
    "task": "task",
    "regressor_label": "regressor_label",
}


def _current_git_commit() -> str | None:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return process.stdout.strip() or None


def _make_model(name: str, n_samples: int, n_features: int, seed: int) -> Any:
    if name == "logreg":
        solver = "saga" if n_features > n_samples else "liblinear"
        return LogisticRegression(
            solver=solver,
            max_iter=5000,
            random_state=seed,
            multi_class="auto",
        )
    if name == "linearsvc":
        dual = bool(n_samples <= n_features)
        return LinearSVC(dual=dual, random_state=seed, max_iter=5000)
    if name == "ridge":
        return RidgeClassifier(random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def _build_pipeline(model_name: str, n_samples: int, n_features: int, seed: int) -> Pipeline:
    model = _make_model(name=model_name, n_samples=n_samples, n_features=n_features, seed=seed)
    # fMRI voxel vectors are dense numeric arrays; centered scaling is appropriate.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", model),
        ]
    )


def _resolve_target_column(target: str) -> str:
    if target not in _TARGET_ALIASES:
        allowed = ", ".join(sorted(_TARGET_ALIASES))
        raise ValueError(f"Unsupported target '{target}'. Allowed values: {allowed}")
    return _TARGET_ALIASES[target]


def _load_features_from_cache(
    index_df: pd.DataFrame, cache_manifest_path: Path
) -> tuple[np.ndarray, pd.DataFrame]:
    manifest = pd.read_csv(cache_manifest_path)
    if manifest.empty:
        raise ValueError(f"Cache manifest is empty: {cache_manifest_path}")

    selected_ids = set(index_df["sample_id"].astype(str))
    feature_map: dict[str, np.ndarray] = {}
    metadata_map: dict[str, dict[str, Any]] = {}

    for _, row in manifest.iterrows():
        cache_path = Path(str(row["cache_path"]))
        if not cache_path.exists():
            LOGGER.warning("Skipping missing cache file: %s", cache_path)
            continue

        with np.load(cache_path, allow_pickle=False) as npz:
            x_block = np.asarray(npz["X"], dtype=np.float32)
            metadata_json = str(npz["metadata_json"].item())
            metadata_records = json.loads(metadata_json)

        if x_block.shape[0] != len(metadata_records):
            raise ValueError(
                f"Cache row mismatch in {cache_path}: {x_block.shape[0]} != {len(metadata_records)}"
            )

        for row_idx, metadata in enumerate(metadata_records):
            sample_id = str(metadata.get("sample_id", ""))
            if sample_id and sample_id in selected_ids:
                feature_map[sample_id] = x_block[row_idx]
                metadata_map[sample_id] = metadata

    vectors: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    missing_samples: list[str] = []

    for _, row in index_df.iterrows():
        sample_id = str(row["sample_id"])
        if sample_id not in feature_map:
            missing_samples.append(sample_id)
            continue
        vectors.append(feature_map[sample_id])
        merged = dict(metadata_map[sample_id])
        merged.update(row.to_dict())
        metadata_rows.append(merged)

    if missing_samples:
        preview = ", ".join(missing_samples[:5])
        raise ValueError(
            f"{len(missing_samples)} samples were missing in cache. "
            f"First missing sample_id values: {preview}"
        )

    x_matrix = np.vstack(vectors).astype(np.float32, copy=False)
    metadata_df = pd.DataFrame(metadata_rows)
    return x_matrix, metadata_df


def _scores_for_predictions(estimator: Pipeline, x_test: np.ndarray) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {
        "decision_value": [pd.NA] * len(x_test),
        "decision_vector": [pd.NA] * len(x_test),
        "proba_value": [pd.NA] * len(x_test),
        "proba_vector": [pd.NA] * len(x_test),
    }

    if hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(x_test)
        decision_array = np.asarray(decision)
        if decision_array.ndim == 1:
            result["decision_value"] = decision_array.astype(float).tolist()
        else:
            result["decision_vector"] = [json.dumps(row.tolist()) for row in decision_array]

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x_test)
        proba_array = np.asarray(proba)
        result["proba_value"] = proba_array.max(axis=1).astype(float).tolist()
        result["proba_vector"] = [json.dumps(row.tolist()) for row in proba_array]

    return result


def _evaluate_permutations(
    pipeline_template: Pipeline,
    x_matrix: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_permutations: int,
    observed_accuracy: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    permutation_accuracies: list[float] = []

    for _ in range(n_permutations):
        y_true_all: list[str] = []
        y_pred_all: list[str] = []

        for train_idx, test_idx in splits:
            y_train = y[train_idx].copy()
            rng.shuffle(y_train)

            estimator = clone(pipeline_template)
            estimator.fit(x_matrix[train_idx], y_train)
            pred = estimator.predict(x_matrix[test_idx])

            y_true_all.extend(y[test_idx].tolist())
            y_pred_all.extend(pred.tolist())

        permutation_accuracies.append(float(accuracy_score(y_true_all, y_pred_all)))

    ge_count = sum(score >= observed_accuracy for score in permutation_accuracies)
    p_value = (ge_count + 1.0) / (n_permutations + 1.0)
    return {
        "n_permutations": int(n_permutations),
        "permutation_accuracy_mean": float(np.mean(permutation_accuracies)),
        "permutation_accuracy_std": float(np.std(permutation_accuracies)),
        "permutation_p_value": float(p_value),
    }


def run_experiment(
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    target: str,
    model: str,
    cv: str = "loso_session",
    seed: int = 42,
    filter_task: str | None = None,
    filter_modality: str | None = None,
    n_permutations: int = 0,
    run_id: str | None = None,
    reports_root: Path | str = Path("reports") / "experiments",
) -> dict[str, Any]:
    """Run one leakage-safe grouped-CV experiment and write standardized artifacts."""
    index_csv = Path(index_csv)
    data_root = Path(data_root)
    cache_dir = Path(cache_dir)
    reports_root = Path(reports_root)

    if cv != "loso_session":
        raise ValueError("Only cv='loso_session' is currently supported.")

    target_column = _resolve_target_column(target)

    index_df = pd.read_csv(index_csv)
    if index_df.empty:
        raise ValueError(f"Dataset index is empty: {index_csv}")

    for required in ("sample_id", "subject", "session", "task", "modality", target_column):
        if required not in index_df.columns:
            raise ValueError(f"Dataset index missing required column: {required}")

    if filter_task is not None:
        index_df = index_df[index_df["task"] == filter_task].copy()
    if filter_modality is not None:
        index_df = index_df[index_df["modality"] == filter_modality].copy()

    index_df = index_df.dropna(subset=[target_column]).copy()
    index_df[target_column] = index_df[target_column].astype(str)
    if index_df.empty:
        raise ValueError("No samples left after filtering and target cleanup.")

    manifest_path = build_feature_cache(
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        group_key="subject_session_bas",
        force=False,
    )
    x_matrix, metadata_df = _load_features_from_cache(
        index_df=index_df, cache_manifest_path=manifest_path
    )

    y = metadata_df[target_column].astype(str).to_numpy()
    groups = (
        metadata_df["subject"].astype(str) + "_" + metadata_df["session"].astype(str)
    ).to_numpy()
    unique_groups = np.unique(groups)
    unique_labels = np.unique(y)

    if len(unique_groups) < 2:
        raise ValueError("Grouped CV requires at least 2 unique subject-session groups.")
    if len(unique_labels) < 2:
        raise ValueError("Classification requires at least 2 target classes.")

    splitter = LeaveOneGroupOut()
    splits = list(splitter.split(x_matrix, y, groups))
    if len(splits) < 2:
        raise ValueError("Grouped CV produced fewer than 2 folds.")

    pipeline_template = _build_pipeline(
        model_name=model,
        n_samples=x_matrix.shape[0],
        n_features=x_matrix.shape[1],
        seed=seed,
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    y_true_all: list[str] = []
    y_pred_all: list[str] = []

    for fold_index, (train_idx, test_idx) in enumerate(splits):
        estimator = clone(pipeline_template)
        estimator.fit(x_matrix[train_idx], y[train_idx])

        y_pred = estimator.predict(x_matrix[test_idx])
        y_true = y[test_idx]
        score_payload = _scores_for_predictions(estimator=estimator, x_test=x_matrix[test_idx])

        fold_metrics = {
            "fold": fold_index,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "test_groups": "|".join(sorted(np.unique(groups[test_idx]).tolist())),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        fold_rows.append(fold_metrics)

        fold_meta = metadata_df.iloc[test_idx].reset_index(drop=True)
        for row_idx in range(len(fold_meta)):
            prediction_rows.append(
                {
                    "fold": fold_index,
                    "sample_id": str(fold_meta.loc[row_idx, "sample_id"]),
                    "y_true": str(y_true[row_idx]),
                    "y_pred": str(y_pred[row_idx]),
                    "decision_value": score_payload["decision_value"][row_idx],
                    "decision_vector": score_payload["decision_vector"][row_idx],
                    "proba_value": score_payload["proba_value"][row_idx],
                    "proba_vector": score_payload["proba_vector"][row_idx],
                    "subject": fold_meta.loc[row_idx, "subject"],
                    "session": fold_meta.loc[row_idx, "session"],
                    "bas": fold_meta.loc[row_idx, "bas"],
                    "task": fold_meta.loc[row_idx, "task"],
                    "modality": fold_meta.loc[row_idx, "modality"],
                    "emotion": fold_meta.loc[row_idx, "emotion"],
                }
            )

        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())

    overall_accuracy = float(accuracy_score(y_true_all, y_pred_all))
    overall_balanced = float(balanced_accuracy_score(y_true_all, y_pred_all))
    overall_macro_f1 = float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0))
    labels_sorted = sorted(
        np.unique(np.concatenate([np.asarray(y_true_all), np.asarray(y_pred_all)])).tolist()
    )
    cmatrix = confusion_matrix(y_true_all, y_pred_all, labels=labels_sorted)

    metrics: dict[str, Any] = {
        "model": model,
        "target": target_column,
        "cv": cv,
        "n_samples": int(len(y)),
        "n_features": int(x_matrix.shape[1]),
        "n_folds": int(len(fold_rows)),
        "accuracy": overall_accuracy,
        "balanced_accuracy": overall_balanced,
        "macro_f1": overall_macro_f1,
        "labels": labels_sorted,
        "confusion_matrix": cmatrix.tolist(),
    }

    if n_permutations > 0:
        metrics["permutation_test"] = _evaluate_permutations(
            pipeline_template=pipeline_template,
            x_matrix=x_matrix,
            y=y,
            splits=splits,
            seed=seed,
            n_permutations=n_permutations,
            observed_accuracy=overall_accuracy,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_run_id = run_id or f"{timestamp}_{model}_{target_column}"
    report_dir = reports_root / resolved_run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics_path = report_dir / "fold_metrics.csv"
    predictions_path = report_dir / "predictions.csv"
    metrics_path = report_dir / "metrics.json"
    config_path = report_dir / "config.json"

    pd.DataFrame(fold_rows).to_csv(fold_metrics_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    metrics_path.write_text(f"{json.dumps(metrics, indent=2)}\n", encoding="utf-8")

    config = {
        "run_id": resolved_run_id,
        "timestamp": timestamp,
        "index_csv": str(index_csv.resolve()),
        "data_root": str(data_root.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "target": target_column,
        "model": model,
        "cv": cv,
        "seed": int(seed),
        "filter_task": filter_task,
        "filter_modality": filter_modality,
        "n_permutations": int(n_permutations),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "nibabel_version": nib.__version__,
        "git_commit": _current_git_commit(),
    }
    config_path.write_text(f"{json.dumps(config, indent=2)}\n", encoding="utf-8")

    return {
        "run_id": resolved_run_id,
        "report_dir": str(report_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "fold_metrics_path": str(fold_metrics_path.resolve()),
        "predictions_path": str(predictions_path.resolve()),
        "metrics": metrics,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run leakage-safe grouped-CV fMRI experiments.")
    parser.add_argument("--index-csv", required=True, help="Dataset index CSV.")
    parser.add_argument("--data-root", required=True, help="Root directory for relative paths.")
    parser.add_argument("--cache-dir", required=True, help="Feature cache directory.")
    parser.add_argument(
        "--target", required=True, choices=sorted(_TARGET_ALIASES), help="Target label."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[*_MODEL_NAMES, "all"],
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--cv",
        default="loso_session",
        choices=["loso_session"],
        help="Cross-validation scheme.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--filter-task", default=None, help="Optional task filter.")
    parser.add_argument("--filter-modality", default=None, help="Optional modality filter.")
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=0,
        help="Number of permutation rounds for optional significance testing.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier. If omitted, timestamp-based ID is used.",
    )
    parser.add_argument(
        "--reports-root",
        default=str(Path("reports") / "experiments"),
        help="Root directory for experiment reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    models = list(_MODEL_NAMES) if args.model == "all" else [args.model]
    results: list[dict[str, Any]] = []

    for model_name in models:
        model_run_id = args.run_id
        if args.model == "all":
            base = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            model_run_id = f"{base}_{model_name}_{args.target}"

        result = run_experiment(
            index_csv=Path(args.index_csv),
            data_root=Path(args.data_root),
            cache_dir=Path(args.cache_dir),
            target=args.target,
            model=model_name,
            cv=args.cv,
            seed=args.seed,
            filter_task=args.filter_task,
            filter_modality=args.filter_modality,
            n_permutations=args.n_permutations,
            run_id=model_run_id,
            reports_root=Path(args.reports_root),
        )
        results.append(
            {
                "model": model_name,
                "run_id": result["run_id"],
                "report_dir": result["report_dir"],
                "accuracy": result["metrics"]["accuracy"],
                "balanced_accuracy": result["metrics"]["balanced_accuracy"],
                "macro_f1": result["metrics"]["macro_f1"],
            }
        )

    print(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
