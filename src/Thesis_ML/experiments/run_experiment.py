"""Leakage-safe experiment runner with grouped cross-validation."""

from __future__ import annotations

import argparse
import itertools
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

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_EXPERIMENT_REPORT,
    ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    compute_config_hash,
    register_artifact,
)
from Thesis_ML.experiments.sections import (
    DatasetSelectionInput,
    EvaluationInput,
    FeatureCacheBuildInput,
    FeatureMatrixLoadInput,
    InterpretabilityInput,
    ModelFitInput,
    SpatialValidationInput,
    dataset_selection,
    evaluation,
    feature_cache_build,
    feature_matrix_load,
    interpretability,
    model_fit,
    spatial_validation,
)

LOGGER = logging.getLogger(__name__)

_MODEL_NAMES = ("logreg", "linearsvc", "ridge")
_CV_MODES = ("loso_session", "within_subject_loso_session", "frozen_cross_person_transfer")
_TARGET_ALIASES = {
    "emotion": "emotion",
    "coarse_affect": "coarse_affect",
    "modality": "modality",
    "task": "task",
    "regressor_label": "regressor_label",
}
_SPATIAL_AFFINE_ATOL = 1e-5


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


def _make_model(name: str, seed: int) -> Any:
    # Keep model hyperparameters fixed across runs to avoid hidden dependence on
    # full selected dataset geometry before fold-level train/test splitting.
    if name == "logreg":
        return LogisticRegression(
            solver="saga",
            max_iter=5000,
            random_state=seed,
        )
    if name == "linearsvc":
        return LinearSVC(dual=True, random_state=seed, max_iter=5000)
    if name == "ridge":
        return RidgeClassifier(random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def _build_pipeline(model_name: str, seed: int) -> Pipeline:
    model = _make_model(name=model_name, seed=seed)
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


def _resolve_cv_mode(cv: str) -> str:
    if cv not in _CV_MODES:
        allowed = ", ".join(_CV_MODES)
        raise ValueError(f"Unsupported cv '{cv}'. Allowed values: {allowed}")
    return cv


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _normalize_spatial_signature(raw_signature: Any, cache_path: Path) -> dict[str, Any]:
    if not isinstance(raw_signature, dict):
        raise ValueError(f"invalid spatial signature payload in {cache_path}")

    required = ("image_shape", "affine", "mask_voxel_count", "feature_count", "mask_sha256")
    missing = [key for key in required if key not in raw_signature]
    if missing:
        raise ValueError(
            f"missing required spatial signature field(s) {missing} in {cache_path}"
        )

    image_shape = [int(value) for value in list(raw_signature["image_shape"])]
    affine_array = np.asarray(raw_signature["affine"], dtype=np.float64)
    if affine_array.shape != (4, 4):
        raise ValueError(
            f"invalid affine shape in {cache_path}: expected (4, 4), got {affine_array.shape}"
        )

    voxel_size_raw = raw_signature.get("voxel_size", [])
    voxel_size = [float(value) for value in list(voxel_size_raw)]
    mask_sha256 = str(raw_signature["mask_sha256"]).strip()
    if not mask_sha256:
        raise ValueError(f"empty mask_sha256 in {cache_path}")

    return {
        "signature_version": int(raw_signature.get("signature_version", 1)),
        "image_shape": image_shape,
        "affine": affine_array.tolist(),
        "voxel_size": voxel_size,
        "mask_voxel_count": int(raw_signature["mask_voxel_count"]),
        "feature_count": int(raw_signature["feature_count"]),
        "mask_sha256": mask_sha256,
    }


def _spatial_mismatch_reasons(
    reference_signature: dict[str, Any],
    candidate_signature: dict[str, Any],
    affine_atol: float,
) -> list[str]:
    reasons: list[str] = []

    ref_shape = [int(value) for value in reference_signature["image_shape"]]
    cand_shape = [int(value) for value in candidate_signature["image_shape"]]
    if cand_shape != ref_shape:
        reasons.append(f"image_shape mismatch ({cand_shape} != {ref_shape})")

    ref_affine = np.asarray(reference_signature["affine"], dtype=np.float64)
    cand_affine = np.asarray(candidate_signature["affine"], dtype=np.float64)
    if not np.allclose(cand_affine, ref_affine, rtol=0.0, atol=affine_atol):
        reasons.append("affine mismatch")

    ref_voxel_size = np.asarray(reference_signature.get("voxel_size", []), dtype=np.float64)
    cand_voxel_size = np.asarray(candidate_signature.get("voxel_size", []), dtype=np.float64)
    if ref_voxel_size.size > 0 and cand_voxel_size.size > 0:
        if not np.allclose(cand_voxel_size, ref_voxel_size, rtol=0.0, atol=1e-6):
            reasons.append("voxel_size mismatch")

    ref_mask_voxels = int(reference_signature["mask_voxel_count"])
    cand_mask_voxels = int(candidate_signature["mask_voxel_count"])
    if cand_mask_voxels != ref_mask_voxels:
        reasons.append(f"mask_voxel_count mismatch ({cand_mask_voxels} != {ref_mask_voxels})")

    ref_feature_count = int(reference_signature["feature_count"])
    cand_feature_count = int(candidate_signature["feature_count"])
    if cand_feature_count != ref_feature_count:
        reasons.append(f"feature_count mismatch ({cand_feature_count} != {ref_feature_count})")

    ref_hash = str(reference_signature["mask_sha256"])
    cand_hash = str(candidate_signature["mask_sha256"])
    if cand_hash != ref_hash:
        reasons.append("mask_sha256 mismatch")

    return reasons


def _build_spatial_compatibility_report(
    cache_groups: list[dict[str, Any]],
    affine_atol: float,
) -> dict[str, Any]:
    checked_groups: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    reference_signature: dict[str, Any] | None = None
    reference_group_id: str | None = None

    for group in cache_groups:
        group_id = str(group["group_id"])
        cache_path = str(group["cache_path"])
        n_features = int(group["n_features"])
        n_selected_samples = int(group["n_selected_samples"])
        raw_signature = group.get("raw_signature")
        normalized_signature: dict[str, Any] | None = None
        reasons: list[str] = []

        if raw_signature is None:
            reasons.append("missing spatial signature metadata")
        else:
            try:
                normalized_signature = _normalize_spatial_signature(
                    raw_signature=raw_signature,
                    cache_path=Path(cache_path),
                )
            except ValueError as exc:
                reasons.append(str(exc))

        if normalized_signature is not None:
            signature_feature_count = int(normalized_signature["feature_count"])
            signature_mask_voxels = int(normalized_signature["mask_voxel_count"])
            if signature_feature_count != n_features:
                reasons.append(
                    "feature_count mismatch against cached matrix width "
                    f"({signature_feature_count} != {n_features})"
                )
            if signature_mask_voxels != n_features:
                reasons.append(
                    "mask_voxel_count mismatch against cached matrix width "
                    f"({signature_mask_voxels} != {n_features})"
                )
            if reference_signature is None:
                reference_signature = normalized_signature
                reference_group_id = group_id
            else:
                reasons.extend(
                    _spatial_mismatch_reasons(
                        reference_signature=reference_signature,
                        candidate_signature=normalized_signature,
                        affine_atol=affine_atol,
                    )
                )

        checked_groups.append(
            {
                "group_id": group_id,
                "cache_path": cache_path,
                "n_selected_samples": n_selected_samples,
                "n_features": n_features,
                "spatial_signature": normalized_signature,
            }
        )
        if reasons:
            mismatches.append(
                {
                    "group_id": group_id,
                    "cache_path": cache_path,
                    "reasons": reasons,
                }
            )

    if not cache_groups:
        mismatches.append(
            {
                "group_id": None,
                "cache_path": None,
                "reasons": ["no cache groups matched selected samples"],
            }
        )

    passed = bool(cache_groups) and not mismatches and reference_signature is not None
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "affine_atol": float(affine_atol),
        "n_groups_checked": int(len(cache_groups)),
        "reference_group_id": reference_group_id,
        "reference_signature": reference_signature,
        "checked_groups": checked_groups,
        "mismatches": mismatches,
    }


def _raise_spatial_compatibility_error(report: dict[str, Any]) -> None:
    mismatch_summaries: list[str] = []
    for mismatch in report.get("mismatches", [])[:5]:
        group_id = mismatch.get("group_id")
        group_label = str(group_id) if group_id is not None else "<unknown-group>"
        reasons = mismatch.get("reasons", [])
        if reasons:
            reason_text = "; ".join(str(reason) for reason in reasons)
        else:
            reason_text = "unknown mismatch"
        mismatch_summaries.append(f"{group_label}: {reason_text}")

    details = " | ".join(mismatch_summaries) if mismatch_summaries else "unknown mismatch"
    raise ValueError(
        "Spatial compatibility validation failed before feature stacking. "
        f"{details}. Rebuild cache with thesisml-cache-features --force if metadata is stale."
    )


def _load_features_from_cache(
    index_df: pd.DataFrame,
    cache_manifest_path: Path,
    spatial_report_path: Path | None = None,
    affine_atol: float = _SPATIAL_AFFINE_ATOL,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_csv(cache_manifest_path)
    if manifest.empty:
        raise ValueError(f"Cache manifest is empty: {cache_manifest_path}")

    selected_ids = set(index_df["sample_id"].astype(str))
    feature_map: dict[str, np.ndarray] = {}
    metadata_map: dict[str, dict[str, Any]] = {}
    selected_cache_groups: list[dict[str, Any]] = []

    for _, row in manifest.iterrows():
        cache_path = Path(str(row["cache_path"]))
        if not cache_path.exists():
            LOGGER.warning("Skipping missing cache file: %s", cache_path)
            continue

        with np.load(cache_path, allow_pickle=False) as npz:
            x_block = np.asarray(npz["X"], dtype=np.float32)
            metadata_json = str(npz["metadata_json"].item())
            metadata_records = json.loads(metadata_json)
            raw_signature = None
            if "spatial_signature_json" in npz.files:
                raw_signature = json.loads(str(npz["spatial_signature_json"].item()))
            group_id = (
                str(npz["group_id"].item())
                if "group_id" in npz.files
                else str(row.get("group_id", cache_path.name))
            )

        if x_block.shape[0] != len(metadata_records):
            raise ValueError(
                f"Cache row mismatch in {cache_path}: {x_block.shape[0]} != {len(metadata_records)}"
            )

        selected_in_group = 0
        for row_idx, metadata in enumerate(metadata_records):
            sample_id = str(metadata.get("sample_id", ""))
            if sample_id and sample_id in selected_ids:
                feature_map[sample_id] = x_block[row_idx]
                metadata_map[sample_id] = metadata
                selected_in_group += 1

        if selected_in_group > 0:
            selected_cache_groups.append(
                {
                    "group_id": group_id,
                    "cache_path": str(cache_path.resolve()),
                    "n_selected_samples": int(selected_in_group),
                    "n_features": int(x_block.shape[1]),
                    "raw_signature": raw_signature,
                }
            )

    spatial_report = _build_spatial_compatibility_report(
        cache_groups=selected_cache_groups,
        affine_atol=affine_atol,
    )
    if spatial_report_path is not None:
        _write_json(spatial_report_path, spatial_report)
    if not spatial_report["passed"]:
        _raise_spatial_compatibility_error(spatial_report)

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
    return x_matrix, metadata_df, spatial_report


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


def _extract_linear_coefficients(estimator: Pipeline) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model = estimator.named_steps.get("model")
    if model is None or not hasattr(model, "coef_"):
        raise ValueError(
            "Interpretability export requires a fitted linear model with a 'coef_' attribute."
        )

    coef_array = np.asarray(model.coef_, dtype=np.float64)
    if coef_array.ndim == 1:
        coef_array = coef_array.reshape(1, -1)
    if coef_array.ndim != 2:
        raise ValueError(f"Unsupported coefficient shape for interpretability: {coef_array.shape}")

    intercept_raw = getattr(model, "intercept_", np.zeros(coef_array.shape[0], dtype=np.float64))
    intercept_array = np.asarray(intercept_raw, dtype=np.float64).reshape(-1)
    if intercept_array.size == 1 and coef_array.shape[0] > 1:
        intercept_array = np.repeat(intercept_array, coef_array.shape[0])

    classes = getattr(model, "classes_", None)
    if classes is None:
        class_labels = [f"class_{idx}" for idx in range(coef_array.shape[0])]
    else:
        class_labels = [str(value) for value in np.asarray(classes).tolist()]

    return coef_array, intercept_array, class_labels


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    if std_a == 0.0 or std_b == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _compute_interpretability_stability(coef_vectors: list[np.ndarray]) -> dict[str, Any]:
    if not coef_vectors:
        return {
            "status": "no_coefficients",
            "n_folds": 0,
            "n_pairs": 0,
            "mean_pairwise_correlation": None,
            "mean_sign_consistency": None,
            "top_k": 0,
            "mean_top_k_overlap": None,
        }

    lengths = {int(vector.size) for vector in coef_vectors}
    if len(lengths) != 1:
        return {
            "status": "incompatible_coefficient_shapes",
            "n_folds": int(len(coef_vectors)),
            "n_pairs": 0,
            "mean_pairwise_correlation": None,
            "mean_sign_consistency": None,
            "top_k": 0,
            "mean_top_k_overlap": None,
        }

    stacked = np.vstack(coef_vectors).astype(np.float64, copy=False)
    n_folds, n_coeffs = stacked.shape
    pair_indices = list(itertools.combinations(range(n_folds), 2))

    pairwise_corrs = [
        _safe_corr(stacked[left_idx], stacked[right_idx]) for left_idx, right_idx in pair_indices
    ]
    mean_pairwise_corr = float(np.mean(pairwise_corrs)) if pairwise_corrs else None

    sign_matrix = np.sign(stacked)
    sign_consistency = np.maximum.reduce(
        [
            np.mean(sign_matrix == -1.0, axis=0),
            np.mean(sign_matrix == 0.0, axis=0),
            np.mean(sign_matrix == 1.0, axis=0),
        ]
    )
    mean_sign_consistency = float(np.mean(sign_consistency))

    top_k = int(min(100, n_coeffs))
    if top_k > 0:
        top_k_sets = [
            set(np.argpartition(np.abs(row), -top_k)[-top_k:].tolist()) for row in stacked
        ]
        top_k_overlaps = []
        for left_idx, right_idx in pair_indices:
            left = top_k_sets[left_idx]
            right = top_k_sets[right_idx]
            denom = len(left | right)
            overlap = float(len(left & right) / denom) if denom > 0 else 0.0
            top_k_overlaps.append(overlap)
        mean_top_k_overlap = float(np.mean(top_k_overlaps)) if top_k_overlaps else None
    else:
        mean_top_k_overlap = None

    return {
        "status": "ok",
        "n_folds": int(n_folds),
        "n_pairs": int(len(pair_indices)),
        "mean_pairwise_correlation": mean_pairwise_corr,
        "mean_sign_consistency": mean_sign_consistency,
        "top_k": top_k,
        "mean_top_k_overlap": mean_top_k_overlap,
    }


def run_experiment(
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    target: str,
    model: str,
    cv: str | None = None,
    subject: str | None = None,
    train_subject: str | None = None,
    test_subject: str | None = None,
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

    if cv is None or not str(cv).strip():
        allowed = ", ".join(_CV_MODES)
        raise ValueError(
            "run_experiment requires explicit cv mode selection. "
            f"Provide one of: {allowed}"
        )
    cv_mode = _resolve_cv_mode(str(cv).strip())
    if cv_mode == "within_subject_loso_session":
        if subject is None or not str(subject).strip():
            raise ValueError("cv='within_subject_loso_session' requires a non-empty subject.")
        subject = str(subject).strip()
    if cv_mode == "frozen_cross_person_transfer":
        if train_subject is None or not str(train_subject).strip():
            raise ValueError(
                "cv='frozen_cross_person_transfer' requires a non-empty train_subject."
            )
        if test_subject is None or not str(test_subject).strip():
            raise ValueError(
                "cv='frozen_cross_person_transfer' requires a non-empty test_subject."
            )
        train_subject = str(train_subject).strip()
        test_subject = str(test_subject).strip()
        if train_subject == test_subject:
            raise ValueError("train_subject and test_subject must be different.")

    target_column = _resolve_target_column(target)

    selection_output = dataset_selection(
        DatasetSelectionInput(
            index_csv=index_csv,
            target_column=target_column,
            cv_mode=cv_mode,
            subject=subject,
            train_subject=train_subject,
            test_subject=test_subject,
            filter_task=filter_task,
            filter_modality=filter_modality,
        )
    )
    index_df = selection_output.selected_index_df

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_run_id = run_id or f"{timestamp}_{model}_{target_column}"
    report_dir = reports_root / resolved_run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_registry_path = reports_root / "artifact_registry.sqlite3"
    code_ref = _current_git_commit()
    artifact_ids: dict[str, str] = {}

    fold_metrics_path = report_dir / "fold_metrics.csv"
    fold_splits_path = report_dir / "fold_splits.csv"
    predictions_path = report_dir / "predictions.csv"
    metrics_path = report_dir / "metrics.json"
    config_path = report_dir / "config.json"
    spatial_compatibility_report_path = report_dir / "spatial_compatibility_report.json"
    interpretability_summary_path = report_dir / "interpretability_summary.json"
    interpretability_fold_artifacts_path = report_dir / "interpretability_fold_explanations.csv"

    cache_output = feature_cache_build(
        FeatureCacheBuildInput(
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=cache_dir,
            group_key="subject_session_bas",
            force=False,
            run_id=resolved_run_id,
            artifact_registry_path=artifact_registry_path,
            code_ref=code_ref,
        )
    )
    artifact_ids["feature_cache"] = cache_output.feature_cache_artifact_id

    matrix_output = feature_matrix_load(
        FeatureMatrixLoadInput(
            selected_index_df=index_df,
            cache_manifest_path=cache_output.cache_manifest_path,
            spatial_report_path=spatial_compatibility_report_path,
            affine_atol=_SPATIAL_AFFINE_ATOL,
            run_id=resolved_run_id,
            artifact_registry_path=artifact_registry_path,
            code_ref=code_ref,
            upstream_feature_cache_artifact_id=cache_output.feature_cache_artifact_id,
            target_column=target_column,
            cv_mode=cv_mode,
            subject=subject,
            train_subject=train_subject,
            test_subject=test_subject,
            filter_task=filter_task,
            filter_modality=filter_modality,
            load_features_from_cache_fn=_load_features_from_cache,
        )
    )
    artifact_ids["feature_matrix_bundle"] = matrix_output.feature_matrix_artifact_id
    x_matrix = matrix_output.x_matrix
    metadata_df = matrix_output.metadata_df
    spatial_compatibility = matrix_output.spatial_compatibility

    spatial_validation(
        SpatialValidationInput(spatial_compatibility=spatial_compatibility)
    )
    y = metadata_df[target_column].astype(str).to_numpy()
    def _model_fit_impl(payload: dict[str, Any]) -> dict[str, Any]:
        payload_x = payload["x_matrix"]
        payload_metadata = payload["metadata_df"]
        payload_y = payload["y"]
        payload_model = str(payload["model"])
        payload_cv_mode = str(payload["cv_mode"])
        payload_subject = payload.get("subject")
        payload_train_subject = payload.get("train_subject")
        payload_test_subject = payload.get("test_subject")
        payload_target_column = str(payload["target_column"])
        payload_seed = int(payload["seed"])
        payload_run_id = str(payload["run_id"])
        payload_config_filename = str(payload["config_filename"])
        payload_report_dir = Path(payload["report_dir"])

        if payload_cv_mode == "frozen_cross_person_transfer":
            subjects = payload_metadata["subject"].astype(str)
            train_mask = subjects == str(payload_train_subject)
            test_mask = subjects == str(payload_test_subject)
            train_idx = np.flatnonzero(train_mask.to_numpy())
            test_idx = np.flatnonzero(test_mask.to_numpy())

            if len(train_idx) == 0:
                raise ValueError(
                    f"No cache-aligned samples found for train_subject '{payload_train_subject}'."
                )
            if len(test_idx) == 0:
                raise ValueError(
                    f"No cache-aligned samples found for test_subject '{payload_test_subject}'."
                )

            unique_labels_train = np.unique(payload_y[train_idx])
            if len(unique_labels_train) < 2:
                raise ValueError("Training data requires at least 2 target classes.")

            groups = payload_metadata["session"].astype(str).to_numpy()
            splits: list[tuple[np.ndarray, np.ndarray]] = [(train_idx, test_idx)]
        elif payload_cv_mode == "within_subject_loso_session":
            subjects = payload_metadata["subject"].astype(str)
            unique_subjects = sorted(subjects.unique().tolist())
            if len(unique_subjects) != 1 or unique_subjects[0] != payload_subject:
                raise ValueError(
                    "within_subject_loso_session requires exactly one subject in the "
                    "evaluated data."
                )

            groups = payload_metadata["session"].astype(str).to_numpy()
            unique_groups = np.unique(groups)
            unique_labels = np.unique(payload_y)
            if len(unique_groups) < 2:
                raise ValueError("Grouped CV requires at least 2 unique subject-session groups.")
            if len(unique_labels) < 2:
                raise ValueError("Classification requires at least 2 target classes.")

            splitter = LeaveOneGroupOut()
            splits = list(splitter.split(payload_x, payload_y, groups))
            if len(splits) < 2:
                raise ValueError("Grouped CV produced fewer than 2 folds.")
        else:
            groups = (
                payload_metadata["subject"].astype(str)
                + "_"
                + payload_metadata["session"].astype(str)
            ).to_numpy()
            unique_groups = np.unique(groups)
            unique_labels = np.unique(payload_y)
            if len(unique_groups) < 2:
                raise ValueError("Grouped CV requires at least 2 unique subject-session groups.")
            if len(unique_labels) < 2:
                raise ValueError("Classification requires at least 2 target classes.")

            splitter = LeaveOneGroupOut()
            splits = list(splitter.split(payload_x, payload_y, groups))
            if len(splits) < 2:
                raise ValueError("Grouped CV produced fewer than 2 folds.")

        pipeline_template = _build_pipeline(
            model_name=payload_model,
            seed=payload_seed,
        )

        interpretability_enabled = payload_cv_mode == "within_subject_loso_session"
        interpretability_fold_rows: list[dict[str, Any]] = []
        interpretability_vectors: list[np.ndarray] = []
        interpretability_dir: Path | None = None
        if interpretability_enabled:
            interpretability_dir = payload_report_dir / "interpretability"
            interpretability_dir.mkdir(parents=True, exist_ok=True)

        fold_rows: list[dict[str, Any]] = []
        split_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        y_true_all: list[str] = []
        y_pred_all: list[str] = []

        for fold_index, (train_idx, test_idx) in enumerate(splits):
            train_meta = payload_metadata.iloc[train_idx].reset_index(drop=True)
            test_meta = payload_metadata.iloc[test_idx].reset_index(drop=True)
            train_subjects = sorted(train_meta["subject"].astype(str).unique().tolist())
            test_subjects = sorted(test_meta["subject"].astype(str).unique().tolist())
            train_sessions = sorted(train_meta["session"].astype(str).unique().tolist())
            test_sessions = sorted(test_meta["session"].astype(str).unique().tolist())

            if payload_cv_mode == "within_subject_loso_session":
                expected_subjects = [str(payload_subject)]
                if train_subjects != expected_subjects or test_subjects != expected_subjects:
                    raise ValueError(
                        "within_subject_loso_session produced fold(s) with unexpected "
                        "subject membership."
                    )
                if set(train_sessions) & set(test_sessions):
                    raise ValueError(
                        "within_subject_loso_session produced overlapping train/test sessions."
                    )
            if payload_cv_mode == "frozen_cross_person_transfer":
                expected_train = [str(payload_train_subject)]
                expected_test = [str(payload_test_subject)]
                if train_subjects != expected_train or test_subjects != expected_test:
                    raise ValueError(
                        "frozen_cross_person_transfer produced unexpected train/test subject "
                        "membership."
                    )

            estimator = clone(pipeline_template)
            estimator.fit(payload_x[train_idx], payload_y[train_idx])

            if interpretability_enabled:
                if interpretability_dir is None:
                    raise ValueError("Interpretability directory was not initialized.")
                coef_array, intercept_array, class_labels = _extract_linear_coefficients(estimator)
                coef_path = interpretability_dir / f"fold_{fold_index:03d}_coefficients.npz"
                np.savez_compressed(
                    coef_path,
                    coefficients=coef_array.astype(np.float32, copy=False),
                    intercept=intercept_array.astype(np.float32, copy=False),
                    class_labels=np.asarray(class_labels, dtype=np.str_),
                    feature_index=np.arange(coef_array.shape[1], dtype=np.int32),
                )
                interpretability_fold_rows.append(
                    {
                        "fold": fold_index,
                        "experiment_mode": payload_cv_mode,
                        "subject": str(payload_subject),
                        "held_out_test_sessions": "|".join(test_sessions),
                        "model": payload_model,
                        "target": payload_target_column,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "coef_rows": int(coef_array.shape[0]),
                        "n_features": int(coef_array.shape[1]),
                        "coef_shape": json.dumps(list(coef_array.shape)),
                        "intercept_shape": json.dumps(list(intercept_array.shape)),
                        "class_labels": json.dumps(class_labels),
                        "coefficient_file": str(coef_path.resolve()),
                        "seed": int(payload_seed),
                        "run_id": payload_run_id,
                        "config_file": payload_config_filename,
                    }
                )
                interpretability_vectors.append(coef_array.reshape(-1).astype(np.float64))

            y_pred = estimator.predict(payload_x[test_idx])
            y_true = payload_y[test_idx]
            score_payload = _scores_for_predictions(estimator=estimator, x_test=payload_x[test_idx])

            fold_metrics = {
                "fold": fold_index,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_groups": "|".join(sorted(np.unique(groups[test_idx]).tolist())),
                "experiment_mode": payload_cv_mode,
                "subject": (
                    str(payload_subject)
                    if payload_cv_mode == "within_subject_loso_session"
                    else pd.NA
                ),
                "train_subject": (
                    str(payload_train_subject)
                    if payload_cv_mode == "frozen_cross_person_transfer"
                    else pd.NA
                ),
                "test_subject": (
                    str(payload_test_subject)
                    if payload_cv_mode == "frozen_cross_person_transfer"
                    else pd.NA
                ),
                "train_sessions": "|".join(train_sessions),
                "test_sessions": "|".join(test_sessions),
                "target": payload_target_column,
                "model": payload_model,
                "seed": int(payload_seed),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            }
            fold_rows.append(fold_metrics)

            split_rows.append(
                {
                    "fold": fold_index,
                    "experiment_mode": payload_cv_mode,
                    "subject": (
                        str(payload_subject)
                        if payload_cv_mode == "within_subject_loso_session"
                        else pd.NA
                    ),
                    "train_subject": (
                        str(payload_train_subject)
                        if payload_cv_mode == "frozen_cross_person_transfer"
                        else pd.NA
                    ),
                    "test_subject": (
                        str(payload_test_subject)
                        if payload_cv_mode == "frozen_cross_person_transfer"
                        else pd.NA
                    ),
                    "train_subjects": "|".join(train_subjects),
                    "test_subjects": "|".join(test_subjects),
                    "train_sessions": "|".join(train_sessions),
                    "test_sessions": "|".join(test_sessions),
                    "train_sample_count": int(len(train_idx)),
                    "test_sample_count": int(len(test_idx)),
                    "target": payload_target_column,
                    "model": payload_model,
                    "seed": int(payload_seed),
                }
            )

            for row_idx in range(len(test_meta)):
                fold_row = test_meta.loc[row_idx]
                prediction_rows.append(
                    {
                        "fold": fold_index,
                        "sample_id": str(fold_row["sample_id"]),
                        "y_true": str(y_true[row_idx]),
                        "y_pred": str(y_pred[row_idx]),
                        "decision_value": score_payload["decision_value"][row_idx],
                        "decision_vector": score_payload["decision_vector"][row_idx],
                        "proba_value": score_payload["proba_value"][row_idx],
                        "proba_vector": score_payload["proba_vector"][row_idx],
                        "subject": fold_row["subject"],
                        "session": fold_row["session"],
                        "bas": fold_row["bas"],
                        "task": fold_row["task"],
                        "modality": fold_row["modality"],
                        "emotion": fold_row.get("emotion", pd.NA),
                        "coarse_affect": fold_row.get("coarse_affect", pd.NA),
                        "experiment_mode": payload_cv_mode,
                        "train_subject": (
                            str(payload_train_subject)
                            if payload_cv_mode == "frozen_cross_person_transfer"
                            else pd.NA
                        ),
                        "test_subject": (
                            str(payload_test_subject)
                            if payload_cv_mode == "frozen_cross_person_transfer"
                            else pd.NA
                        ),
                    }
                )

            y_true_all.extend(y_true.tolist())
            y_pred_all.extend(y_pred.tolist())

        return {
            "y": payload_y,
            "splits": splits,
            "fold_rows": fold_rows,
            "split_rows": split_rows,
            "prediction_rows": prediction_rows,
            "y_true_all": y_true_all,
            "y_pred_all": y_pred_all,
            "interpretability_enabled": interpretability_enabled,
            "interpretability_fold_rows": interpretability_fold_rows,
            "interpretability_vectors": interpretability_vectors,
            "interpretability_fold_artifacts_path": payload_report_dir
            / "interpretability_fold_explanations.csv",
            "interpretability_summary_path": payload_report_dir / "interpretability_summary.json",
        }

    fit_output = model_fit(
        ModelFitInput(
            fit_fn=_model_fit_impl,
            payload={
                "x_matrix": x_matrix,
                "metadata_df": metadata_df,
                "y": y,
                "model": model,
                "cv_mode": cv_mode,
                "subject": subject,
                "train_subject": train_subject,
                "test_subject": test_subject,
                "target_column": target_column,
                "seed": int(seed),
                "run_id": resolved_run_id,
                "config_filename": config_path.name,
                "report_dir": report_dir,
            },
        )
    )
    y = fit_output.y
    splits = fit_output.splits
    fold_rows = fit_output.fold_rows
    split_rows = fit_output.split_rows
    prediction_rows = fit_output.prediction_rows
    y_true_all = fit_output.y_true_all
    y_pred_all = fit_output.y_pred_all
    interpretability_enabled = fit_output.interpretability_enabled
    interpretability_fold_rows = fit_output.interpretability_fold_rows
    interpretability_vectors = fit_output.interpretability_vectors
    pipeline_template = _build_pipeline(
        model_name=model,
        seed=seed,
    )

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
        "cv": cv_mode,
        "experiment_mode": cv_mode,
        "subject": str(subject) if cv_mode == "within_subject_loso_session" else None,
        "train_subject": (
            str(train_subject) if cv_mode == "frozen_cross_person_transfer" else None
        ),
        "test_subject": str(test_subject) if cv_mode == "frozen_cross_person_transfer" else None,
        "n_samples": int(len(y_true_all)),
        "n_features": int(x_matrix.shape[1]),
        "n_folds": int(len(fold_rows)),
        "accuracy": overall_accuracy,
        "balanced_accuracy": overall_balanced,
        "macro_f1": overall_macro_f1,
        "labels": labels_sorted,
        "confusion_matrix": cmatrix.tolist(),
        "spatial_compatibility": {
            "status": str(spatial_compatibility["status"]),
            "passed": bool(spatial_compatibility["passed"]),
            "n_groups_checked": int(spatial_compatibility["n_groups_checked"]),
            "reference_group_id": spatial_compatibility["reference_group_id"],
            "affine_atol": float(spatial_compatibility["affine_atol"]),
            "report_path": str(spatial_compatibility_report_path.resolve()),
        },
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

    def _interpretability_impl(payload: dict[str, Any]) -> dict[str, Any]:
        payload_enabled = bool(payload["interpretability_enabled"])
        payload_fold_rows = payload["interpretability_fold_rows"]
        payload_vectors = payload["interpretability_vectors"]
        payload_cv_mode = str(payload["cv_mode"])
        payload_subject = payload.get("subject")
        payload_model = str(payload["model"])
        payload_target = str(payload["target_column"])
        payload_fold_artifacts_path = Path(payload["fold_artifacts_path"])
        payload_summary_path = Path(payload["summary_path"])

        caution_text = (
            "Linear coefficients are reported as model-behavior evidence only and must not be "
            "interpreted as direct neural localization."
        )
        if payload_enabled:
            pd.DataFrame(payload_fold_rows).to_csv(payload_fold_artifacts_path, index=False)
            interpretability_summary: dict[str, Any] = {
                "enabled": True,
                "performed": True,
                "status": "performed",
                "reason": None,
                "caution": caution_text,
                "experiment_mode": payload_cv_mode,
                "subject": str(payload_subject),
                "model": payload_model,
                "target": payload_target,
                "n_fold_artifacts": int(len(payload_fold_rows)),
                "fold_artifacts_path": str(payload_fold_artifacts_path.resolve()),
                "stability": _compute_interpretability_stability(payload_vectors),
            }
        else:
            interpretability_summary = {
                "enabled": False,
                "performed": False,
                "status": "not_applicable",
                "reason": (
                    "Interpretability export is enabled only for "
                    "within_subject_loso_session."
                ),
                "caution": caution_text,
                "experiment_mode": payload_cv_mode,
                "subject": None,
                "model": payload_model,
                "target": payload_target,
                "n_fold_artifacts": 0,
                "fold_artifacts_path": None,
                "stability": None,
            }
        payload_summary_path.write_text(
            f"{json.dumps(interpretability_summary, indent=2)}\n",
            encoding="utf-8",
        )
        return interpretability_summary

    interpretability_output = interpretability(
        InterpretabilityInput(
            interpretability_fn=_interpretability_impl,
            payload={
                "interpretability_enabled": interpretability_enabled,
                "interpretability_fold_rows": interpretability_fold_rows,
                "interpretability_vectors": interpretability_vectors,
                "cv_mode": cv_mode,
                "subject": subject,
                "model": model,
                "target_column": target_column,
                "fold_artifacts_path": interpretability_fold_artifacts_path,
                "summary_path": interpretability_summary_path,
            },
            run_id=resolved_run_id,
            artifact_registry_path=artifact_registry_path,
            code_ref=code_ref,
            upstream_feature_matrix_artifact_id=matrix_output.feature_matrix_artifact_id,
            summary_path=interpretability_summary_path,
            cv_mode=cv_mode,
            model=model,
            target_column=target_column,
            subject=subject,
        )
    )
    interpretability_summary = interpretability_output.interpretability_summary
    artifact_ids[ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE] = (
        interpretability_output.interpretability_artifact_id
    )

    metrics["interpretability"] = {
        "enabled": bool(interpretability_summary["enabled"]),
        "performed": bool(interpretability_summary["performed"]),
        "status": str(interpretability_summary["status"]),
        "summary_path": str(interpretability_summary_path.resolve()),
        "fold_artifacts_path": interpretability_summary["fold_artifacts_path"],
        "stability": interpretability_summary["stability"],
    }

    def _evaluation_impl(payload: dict[str, Any]) -> dict[str, Any]:
        payload_metrics = payload["metrics"]
        payload_fold_rows = [dict(row) for row in payload["fold_rows"]]
        payload_split_rows = [dict(row) for row in payload["split_rows"]]
        payload_prediction_rows = [dict(row) for row in payload["prediction_rows"]]
        payload_run_id = str(payload["run_id"])
        payload_config_filename = str(payload["config_filename"])
        payload_fold_metrics_path = Path(payload["fold_metrics_path"])
        payload_fold_splits_path = Path(payload["fold_splits_path"])
        payload_predictions_path = Path(payload["predictions_path"])
        payload_metrics_path = Path(payload["metrics_path"])

        for row in payload_fold_rows:
            row["run_id"] = payload_run_id
            row["config_file"] = payload_config_filename
        for row in payload_split_rows:
            row["run_id"] = payload_run_id
            row["config_file"] = payload_config_filename

        pd.DataFrame(payload_fold_rows).to_csv(payload_fold_metrics_path, index=False)
        pd.DataFrame(payload_split_rows).to_csv(payload_fold_splits_path, index=False)
        pd.DataFrame(payload_prediction_rows).to_csv(payload_predictions_path, index=False)
        payload_metrics_path.write_text(
            f"{json.dumps(payload_metrics, indent=2)}\n",
            encoding="utf-8",
        )
        return payload_metrics

    evaluation_output = evaluation(
        EvaluationInput(
            evaluate_fn=_evaluation_impl,
            payload={
                "metrics": metrics,
                "fold_rows": fold_rows,
                "split_rows": split_rows,
                "prediction_rows": prediction_rows,
                "run_id": resolved_run_id,
                "config_filename": config_path.name,
                "fold_metrics_path": fold_metrics_path,
                "fold_splits_path": fold_splits_path,
                "predictions_path": predictions_path,
                "metrics_path": metrics_path,
            },
            run_id=resolved_run_id,
            artifact_registry_path=artifact_registry_path,
            code_ref=code_ref,
            upstream_feature_matrix_artifact_id=matrix_output.feature_matrix_artifact_id,
            metrics_path=metrics_path,
            model=model,
            target_column=target_column,
            cv_mode=cv_mode,
            seed=seed,
        )
    )
    metrics = evaluation_output.metrics
    artifact_ids[ARTIFACT_TYPE_METRICS_BUNDLE] = evaluation_output.metrics_artifact_id

    config = {
        "run_id": resolved_run_id,
        "timestamp": timestamp,
        "index_csv": str(index_csv.resolve()),
        "data_root": str(data_root.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "target": target_column,
        "model": model,
        "cv": cv_mode,
        "experiment_mode": cv_mode,
        "subject": str(subject) if cv_mode == "within_subject_loso_session" else None,
        "train_subject": (
            str(train_subject) if cv_mode == "frozen_cross_person_transfer" else None
        ),
        "test_subject": str(test_subject) if cv_mode == "frozen_cross_person_transfer" else None,
        "seed": int(seed),
        "filter_task": filter_task,
        "filter_modality": filter_modality,
        "n_permutations": int(n_permutations),
        "fold_splits_path": str(fold_splits_path.resolve()),
        "spatial_compatibility_status": str(spatial_compatibility["status"]),
        "spatial_compatibility_passed": bool(spatial_compatibility["passed"]),
        "spatial_compatibility_n_groups_checked": int(spatial_compatibility["n_groups_checked"]),
        "spatial_compatibility_reference_group_id": spatial_compatibility["reference_group_id"],
        "spatial_compatibility_affine_atol": float(spatial_compatibility["affine_atol"]),
        "spatial_compatibility_report_path": str(spatial_compatibility_report_path.resolve()),
        "interpretability_enabled": bool(interpretability_summary["enabled"]),
        "interpretability_performed": bool(interpretability_summary["performed"]),
        "interpretability_status": str(interpretability_summary["status"]),
        "interpretability_fold_artifacts_path": interpretability_summary["fold_artifacts_path"],
        "interpretability_summary_path": str(interpretability_summary_path.resolve()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "nibabel_version": nib.__version__,
        "git_commit": code_ref,
    }
    config_path.write_text(f"{json.dumps(config, indent=2)}\n", encoding="utf-8")

    report_upstream = [
        evaluation_output.metrics_artifact_id,
        interpretability_output.interpretability_artifact_id,
    ]
    experiment_report_artifact = register_artifact(
        registry_path=artifact_registry_path,
        artifact_type=ARTIFACT_TYPE_EXPERIMENT_REPORT,
        run_id=resolved_run_id,
        upstream_artifact_ids=report_upstream,
        config_hash=compute_config_hash({"run_id": resolved_run_id, "report_dir": str(report_dir)}),
        code_ref=code_ref,
        path=report_dir,
        status="created",
    )
    artifact_ids[ARTIFACT_TYPE_EXPERIMENT_REPORT] = experiment_report_artifact.artifact_id

    return {
        "run_id": resolved_run_id,
        "report_dir": str(report_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "fold_metrics_path": str(fold_metrics_path.resolve()),
        "fold_splits_path": str(fold_splits_path.resolve()),
        "predictions_path": str(predictions_path.resolve()),
        "spatial_compatibility_report_path": str(spatial_compatibility_report_path.resolve()),
        "interpretability_summary_path": str(interpretability_summary_path.resolve()),
        "interpretability_fold_artifacts_path": interpretability_summary["fold_artifacts_path"],
        "artifact_registry_path": str(artifact_registry_path.resolve()),
        "artifact_ids": artifact_ids,
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
        required=True,
        choices=list(_CV_MODES),
        help=(
            "Experiment mode. Required. within_subject_loso_session=primary thesis mode; "
            "frozen_cross_person_transfer=secondary transfer mode; "
            "loso_session=auxiliary grouped mode."
        ),
    )
    parser.add_argument(
        "--subject",
        default=None,
        help=(
            "Subject identifier (required for cv=within_subject_loso_session; "
            "for example sub-001)."
        ),
    )
    parser.add_argument(
        "--train-subject",
        default=None,
        help=(
            "Training subject identifier (required for "
            "cv=frozen_cross_person_transfer)."
        ),
    )
    parser.add_argument(
        "--test-subject",
        default=None,
        help=(
            "Test subject identifier (required for cv=frozen_cross_person_transfer)."
        ),
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
            subject=args.subject,
            train_subject=args.train_subject,
            test_subject=args.test_subject,
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
