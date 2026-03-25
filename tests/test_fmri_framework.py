from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_EXPERIMENT_REPORT,
    ARTIFACT_TYPE_FEATURE_CACHE,
    ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE,
    ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    list_artifacts_for_run,
)
from Thesis_ML.data.affect_labels import COARSE_AFFECT_BY_EMOTION, derive_coarse_affect
from Thesis_ML.data.index_dataset import build_dataset_index
from Thesis_ML.experiments.run_experiment import _build_parser, _make_model, run_experiment
from Thesis_ML.features.nifti_features import build_feature_cache
from Thesis_ML.spm.extract_glm import extract_glm_session, parse_regressor_label


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.eye(4, dtype=np.float64) if affine is None else np.asarray(affine, dtype=np.float64)
    image = nib.Nifti1Image(data.astype(np.float32), affine=matrix)
    nib.save(image, str(path))


def _create_glm_session(
    glm_dir: Path,
    labels: list[str],
    missing_beta_indexes: set[int] | None = None,
    class_signal: bool = False,
    shape: tuple[int, int, int] = (3, 3, 3),
    affine: np.ndarray | None = None,
    mask_variant: Literal["default", "shift_x", "expanded_x"] = "default",
) -> None:
    glm_dir.mkdir(parents=True, exist_ok=True)
    missing_beta_indexes = missing_beta_indexes or set()

    mask = np.zeros(shape, dtype=np.float32)
    if mask_variant == "default":
        mask[1:, 1:, 1:] = 1.0
    elif mask_variant == "shift_x":
        mask[:-1, 1:, 1:] = 1.0
    elif mask_variant == "expanded_x":
        mask[:, 1:, 1:] = 1.0
    else:
        raise ValueError(f"Unsupported mask_variant: {mask_variant}")
    _write_nifti(glm_dir / "mask.nii", mask, affine=affine)

    pd.Series(labels).to_csv(glm_dir / "regressor_labels.csv", index=False, header=False)

    for idx, label in enumerate(labels, start=1):
        if idx in missing_beta_indexes:
            continue

        beta = np.full(shape, fill_value=float(idx), dtype=np.float32)
        if class_signal:
            if "_anger_" in label:
                beta[1:, 1:, 1:] += 5.0
            if "_joy_" in label or "_happiness_" in label:
                beta[1:, 1:, 1:] -= 5.0
        _write_nifti(glm_dir / f"beta_{idx:04d}.nii", beta, affine=affine)


def test_parse_regressor_label() -> None:
    emotion = parse_regressor_label("run-1_passive_anger_audio")
    assert emotion["regressor_type"] == "emotion_condition"
    assert emotion["run"] == 1
    assert emotion["task"] == "passive"
    assert emotion["emotion"] == "anger"
    assert emotion["modality"] == "audio"

    resp = parse_regressor_label("run-2_emo_resp_movement")
    assert resp["regressor_type"] == "resp_movement"
    assert resp["run"] == 2
    assert resp["task"] == "emo"

    motion = parse_regressor_label("run-3_recog_R1")
    assert motion["regressor_type"] == "motion_param"
    assert motion["motion_param"] == "R1"

    constant = parse_regressor_label("run-1_passive_constant")
    assert constant["regressor_type"] == "constant"

    unknown = parse_regressor_label("run-9_passive_unexpected_token")
    assert unknown["regressor_type"] == "unknown"


def test_derive_coarse_affect_exact_mapping() -> None:
    assert COARSE_AFFECT_BY_EMOTION == {
        "happiness": "positive",
        "pride": "positive",
        "relief": "positive",
        "interest": "positive",
        "neutral": "neutral",
        "anger": "negative",
        "anxiety": "negative",
        "disgust": "negative",
        "sadness": "negative",
    }
    assert derive_coarse_affect("HAPPINESS") == "positive"
    assert derive_coarse_affect(" neutral ") == "neutral"
    assert derive_coarse_affect("sadness") == "negative"
    assert pd.isna(derive_coarse_affect("joy"))
    assert pd.isna(derive_coarse_affect(pd.NA))


def test_extract_glm_session_outputs(tmp_path: Path) -> None:
    glm_dir = tmp_path / "Data" / "sub-001" / "ses-01" / "BAS1"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_joy_video",
        "run-1_passive_constant",
        "run-1_passive_R1",
        "broken_label",
    ]
    _create_glm_session(glm_dir=glm_dir, labels=labels, missing_beta_indexes={3})

    out_dir = tmp_path / "extractions" / "sub-001" / "ses-01" / "BAS1"
    result = extract_glm_session(glm_dir=glm_dir, out_dir=out_dir, absolute_paths=False)

    mapping_path = Path(result["mapping_csv"])
    summary_path = Path(result["summary_json"])
    assert mapping_path.exists()
    assert summary_path.exists()

    mapping = pd.read_csv(mapping_path)
    required_cols = {
        "label",
        "raw_label",
        "beta_index",
        "beta_file",
        "beta_path",
        "run",
        "task",
        "emotion",
        "modality",
        "regressor_type",
        "motion_param",
        "beta_exists",
    }
    assert required_cols <= set(mapping.columns)
    assert mapping.loc[mapping["beta_index"] == 3, "beta_exists"].item() == False  # noqa: E712
    assert mapping.loc[mapping["beta_index"] == 4, "regressor_type"].item() == "motion_param"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["bas_name"] == "BAS1"
    assert summary["n_regressors_csv"] == len(labels)
    assert summary["n_beta_present"] == len(labels) - 1
    assert "emotion_condition" in summary["regressor_type_counts"]


def test_build_dataset_index(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"

    _create_glm_session(
        glm_dir=data_root / "sub-001" / "ses-01" / "BAS2",
        labels=[
            "run-1_passive_anger_audio",
            "run-1_passive_happiness_video",
            "run-1_passive_resp_movement",
        ],
    )
    _create_glm_session(
        glm_dir=data_root / "sub-002" / "ses-02" / "BAS2",
        labels=[
            "run-2_emo_anxiety_audio",
            "run-2_emo_neutral_audiovisual",
            "run-2_emo_constant",
        ],
    )

    out_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=out_csv)
    assert out_csv.exists()

    index_df = pd.read_csv(out_csv)
    assert len(index_df) == 4
    assert {
        "subject",
        "session",
        "bas",
        "beta_path",
        "mask_path",
        "regressor_label",
        "emotion",
        "coarse_affect",
    } <= set(index_df.columns)
    assert set(index_df["coarse_affect"]) == {"negative", "neutral", "positive"}
    mapped = index_df["emotion"].map(derive_coarse_affect)
    pd.testing.assert_series_equal(index_df["coarse_affect"], mapped, check_names=False)
    assert index_df["beta_path"].map(lambda value: not Path(value).is_absolute()).all()
    assert index_df["mask_path"].map(lambda value: not Path(value).is_absolute()).all()

    extraction_csv = (
        data_root
        / "processed"
        / "extractions"
        / "sub-001"
        / "ses-01"
        / "BAS2"
        / "regressor_beta_mapping.csv"
    )
    assert extraction_csv.exists()


def test_feature_cache(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    _create_glm_session(
        glm_dir=data_root / "sub-001" / "ses-01" / "BAS2",
        labels=[
            "run-1_passive_anger_audio",
            "run-1_passive_happiness_video",
        ],
    )
    out_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=out_csv)

    cache_dir = tmp_path / "cache"
    manifest_path = build_feature_cache(index_csv=out_csv, data_root=data_root, cache_dir=cache_dir)
    assert manifest_path.exists()

    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 1
    npz_path = Path(manifest.loc[0, "cache_path"])
    assert npz_path.exists()

    with np.load(npz_path, allow_pickle=False) as npz:
        x_matrix = npz["X"]
        y = npz["y"]
        metadata = json.loads(str(npz["metadata_json"].item()))
        spatial_signature = json.loads(str(npz["spatial_signature_json"].item()))

    assert x_matrix.dtype == np.float32
    assert x_matrix.shape[0] == len(y) == len(metadata)
    assert x_matrix.shape[1] == 8
    assert {"emotion", "coarse_affect"} <= set(metadata[0])
    for row in metadata:
        assert row["coarse_affect"] == derive_coarse_affect(row["emotion"])
    assert spatial_signature["image_shape"] == [3, 3, 3]
    assert spatial_signature["mask_voxel_count"] == 8
    assert spatial_signature["feature_count"] == 8
    assert len(str(spatial_signature["mask_sha256"])) == 64
    assert spatial_signature["voxel_size"] == [1.0, 1.0, 1.0]
    assert {
        "spatial_signature_version",
        "image_shape_json",
        "affine_json",
        "voxel_size_json",
        "mask_voxel_count",
        "feature_count",
        "mask_sha256",
    } <= set(manifest.columns)


def test_feature_cache_passes_with_matching_non_identity_affine(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    _create_glm_session(
        glm_dir=data_root / "sub-001" / "ses-01" / "BAS2",
        labels=[
            "run-1_passive_anger_audio",
            "run-1_passive_happiness_video",
        ],
        affine=affine,
    )
    out_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=out_csv)

    cache_dir = tmp_path / "cache"
    manifest_path = build_feature_cache(index_csv=out_csv, data_root=data_root, cache_dir=cache_dir)
    manifest = pd.read_csv(manifest_path)
    npz_path = Path(manifest.loc[0, "cache_path"])
    with np.load(npz_path, allow_pickle=False) as npz:
        x_matrix = np.asarray(npz["X"], dtype=np.float32)
    assert x_matrix.shape == (2, 8)


def test_feature_cache_rejects_beta_mask_affine_mismatch(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    glm_dir = data_root / "sub-001" / "ses-01" / "BAS2"
    _create_glm_session(
        glm_dir=glm_dir,
        labels=[
            "run-1_passive_anger_audio",
            "run-1_passive_happiness_video",
        ],
    )
    _write_nifti(
        glm_dir / "beta_0002.nii",
        np.full((3, 3, 3), fill_value=2.0, dtype=np.float32),
        affine=np.diag([2.0, 1.0, 1.0, 1.0]),
    )

    out_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=out_csv)

    with pytest.raises(ValueError) as exc:
        build_feature_cache(index_csv=out_csv, data_root=data_root, cache_dir=tmp_path / "cache")

    message = str(exc.value)
    assert "Beta/mask spatial compatibility validation failed" in message
    assert "affine mismatch" in message
    assert "beta_0002.nii" in message
    assert "mask.nii" in message


def test_feature_cache_rejects_beta_mask_shape_mismatch(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    glm_dir = data_root / "sub-001" / "ses-01" / "BAS2"
    _create_glm_session(
        glm_dir=glm_dir,
        labels=[
            "run-1_passive_anger_audio",
            "run-1_passive_happiness_video",
        ],
    )
    _write_nifti(
        glm_dir / "beta_0002.nii",
        np.full((4, 3, 3), fill_value=2.0, dtype=np.float32),
        affine=np.eye(4, dtype=np.float64),
    )

    out_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=out_csv)

    with pytest.raises(ValueError) as exc:
        build_feature_cache(index_csv=out_csv, data_root=data_root, cache_dir=tmp_path / "cache")

    message = str(exc.value)
    assert "Beta/mask spatial compatibility validation failed" in message
    assert "shape mismatch" in message
    assert "beta_0002.nii" in message
    assert "mask.nii" in message


def test_experiment_runner_smoke(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
        "run-1_passive_anger_video",
        "run-1_passive_happiness_video",
    ]

    sessions = [
        ("sub-001", "ses-01", "BAS2"),
        ("sub-001", "ses-02", "BAS2"),
        ("sub-002", "ses-01", "BAS2"),
        ("sub-002", "ses-02", "BAS2"),
    ]
    for subject, session, bas in sessions:
        _create_glm_session(
            glm_dir=data_root / subject / session / bas,
            labels=labels,
            class_signal=True,
        )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    cache_dir = tmp_path / "cache"
    reports_root = tmp_path / "reports" / "experiments"
    result = run_experiment(
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        target="emotion",
        model="ridge",
        cv="loso_session",
        seed=42,
        n_permutations=2,
        run_id="smoke_ridge_emotion",
        reports_root=reports_root,
    )

    report_dir = Path(result["report_dir"])
    assert report_dir.exists()
    assert (report_dir / "config.json").exists()
    assert (report_dir / "metrics.json").exists()
    assert (report_dir / "fold_metrics.csv").exists()
    assert (report_dir / "predictions.csv").exists()
    assert (report_dir / "spatial_compatibility_report.json").exists()

    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    assert {"accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix"} <= set(metrics)
    assert metrics["n_folds"] >= 2
    assert "permutation_test" in metrics
    assert metrics["spatial_compatibility"]["passed"] is True
    assert metrics["spatial_compatibility"]["n_groups_checked"] >= 2

    predictions = pd.read_csv(report_dir / "predictions.csv")
    assert len(predictions) > 0
    assert {
        "y_true",
        "y_pred",
        "subject",
        "session",
        "task",
        "modality",
        "emotion",
        "coarse_affect",
    } <= set(predictions.columns)

    registry_path = reports_root / "artifact_registry.sqlite3"
    artifacts = list_artifacts_for_run(registry_path=registry_path, run_id="smoke_ridge_emotion")
    assert len(artifacts) >= 5
    artifact_types = {artifact.artifact_type for artifact in artifacts}
    assert {
        ARTIFACT_TYPE_FEATURE_CACHE,
        ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE,
        ARTIFACT_TYPE_METRICS_BUNDLE,
        ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
        ARTIFACT_TYPE_EXPERIMENT_REPORT,
    } <= artifact_types
    assert "artifact_registry_path" in result
    assert "artifact_ids" in result


def test_experiment_runner_coarse_affect_target(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
        "run-1_passive_neutral_video",
        "run-1_passive_sadness_video",
    ]
    sessions = [
        ("sub-001", "ses-01", "BAS2"),
        ("sub-001", "ses-02", "BAS2"),
        ("sub-002", "ses-01", "BAS2"),
        ("sub-002", "ses-02", "BAS2"),
    ]
    for subject, session, bas in sessions:
        _create_glm_session(
            glm_dir=data_root / subject / session / bas,
            labels=labels,
            class_signal=True,
        )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    cache_dir = tmp_path / "cache"
    reports_root = tmp_path / "reports" / "experiments"
    result = run_experiment(
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        target="coarse_affect",
        model="ridge",
        cv="loso_session",
        seed=42,
        run_id="smoke_ridge_coarse_affect",
        reports_root=reports_root,
    )

    report_dir = Path(result["report_dir"])
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((report_dir / "config.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(report_dir / "predictions.csv")
    spatial_report = json.loads(
        (report_dir / "spatial_compatibility_report.json").read_text(encoding="utf-8")
    )

    assert metrics["target"] == "coarse_affect"
    assert config["target"] == "coarse_affect"
    assert config["spatial_compatibility_passed"] is True
    assert metrics["spatial_compatibility"]["passed"] is True
    assert spatial_report["passed"] is True
    assert {"emotion", "coarse_affect"} <= set(predictions.columns)
    assert set(predictions["y_true"].unique()) <= {"negative", "neutral", "positive"}
    mapped = predictions["emotion"].map(derive_coarse_affect)
    pd.testing.assert_series_equal(predictions["coarse_affect"], mapped, check_names=False)


def test_experiment_cli_accepts_coarse_affect_target() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--index-csv",
            "dataset_index.csv",
            "--data-root",
            "Data",
            "--cache-dir",
            "cache",
            "--target",
            "coarse_affect",
            "--model",
            "ridge",
            "--cv",
            "loso_session",
        ]
    )
    assert args.target == "coarse_affect"
    assert args.cv == "loso_session"


def test_experiment_cli_accepts_binary_valence_like_target() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--index-csv",
            "dataset_index.csv",
            "--data-root",
            "Data",
            "--cache-dir",
            "cache",
            "--target",
            "binary_valence_like",
            "--model",
            "ridge",
            "--cv",
            "within_subject_loso_session",
            "--subject",
            "sub-001",
        ]
    )
    assert args.target == "binary_valence_like"
    assert args.cv == "within_subject_loso_session"


def test_experiment_runner_binary_valence_like_target(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
        "run-1_passive_neutral_video",
    ]
    for subject in ("sub-001", "sub-002"):
        for session in ("ses-01", "ses-02"):
            _create_glm_session(
                glm_dir=data_root / subject / session / "BAS2",
                labels=labels,
                class_signal=True,
            )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    reports_root = tmp_path / "reports" / "experiments"
    result = run_experiment(
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=tmp_path / "cache",
        target="binary_valence_like",
        model="ridge",
        cv="within_subject_loso_session",
        subject="sub-001",
        seed=42,
        run_id="binary_valence_like_sub001",
        reports_root=reports_root,
    )

    report_dir = Path(result["report_dir"])
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((report_dir / "config.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(report_dir / "predictions.csv")

    assert metrics["target"] == "binary_valence_like"
    assert config["target"] == "binary_valence_like"
    assert set(predictions["y_true"].astype(str).unique().tolist()) <= {"negative", "positive"}
    assert set(predictions["y_pred"].astype(str).unique().tolist()) <= {"negative", "positive"}
    assert "neutral" not in set(predictions["y_true"].astype(str).tolist())


def test_baseline_models_use_fixed_explicit_settings() -> None:
    logreg = _make_model(name="logreg", seed=13)
    linearsvc = _make_model(name="linearsvc", seed=13)
    ridge = _make_model(name="ridge", seed=13)

    assert isinstance(logreg, LogisticRegression)
    assert logreg.solver == "saga"
    assert int(logreg.max_iter) == 5000
    assert int(logreg.random_state) == 13

    assert isinstance(linearsvc, LinearSVC)
    assert linearsvc.dual is True
    assert int(linearsvc.max_iter) == 5000
    assert int(linearsvc.random_state) == 13

    assert isinstance(ridge, RidgeClassifier)
    assert int(ridge.random_state) == 13


def test_experiment_api_requires_explicit_cv(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
    ]
    for session in ("ses-01", "ses-02"):
        _create_glm_session(
            glm_dir=data_root / "sub-001" / session / "BAS2",
            labels=labels,
            class_signal=True,
        )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    with pytest.raises(ValueError) as exc:
        run_experiment(
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=tmp_path / "cache",
            target="coarse_affect",
            model="ridge",
            run_id="missing_cv_programmatic",
            reports_root=tmp_path / "reports" / "experiments",
        )

    message = str(exc.value)
    assert "requires explicit cv mode selection" in message
    assert "within_subject_loso_session" in message
    assert "frozen_cross_person_transfer" in message
    assert "loso_session" in message


def test_experiment_cli_requires_cv(capsys: pytest.CaptureFixture[str]) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "--index-csv",
                "dataset_index.csv",
                "--data-root",
                "Data",
                "--cache-dir",
                "cache",
                "--target",
                "coarse_affect",
                "--model",
                "ridge",
            ]
        )

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "--cv" in captured.err
    assert "required" in captured.err


def test_experiment_cli_help_describes_modes() -> None:
    parser = _build_parser()
    help_text = parser.format_help()
    assert "within_subject_loso_session" in help_text
    assert "frozen_cross_person_transfer" in help_text
    assert "loso_session" in help_text
    assert "record_random_split" in help_text
    assert "primary thesis mode" in help_text


def test_experiment_cli_accepts_within_subject_mode() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--index-csv",
            "dataset_index.csv",
            "--data-root",
            "Data",
            "--cache-dir",
            "cache",
            "--target",
            "coarse_affect",
            "--model",
            "ridge",
            "--cv",
            "within_subject_loso_session",
            "--subject",
            "sub-001",
        ]
    )
    assert args.cv == "within_subject_loso_session"
    assert args.subject == "sub-001"


def test_experiment_cli_accepts_record_random_split_mode() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--index-csv",
            "dataset_index.csv",
            "--data-root",
            "Data",
            "--cache-dir",
            "cache",
            "--target",
            "coarse_affect",
            "--model",
            "ridge",
            "--cv",
            "record_random_split",
        ]
    )
    assert args.cv == "record_random_split"


def test_experiment_runner_record_random_split_is_deterministic(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
        "run-1_passive_anger_video",
        "run-1_passive_happiness_video",
    ]
    for subject in ("sub-001", "sub-002"):
        for session in ("ses-01", "ses-02", "ses-03"):
            _create_glm_session(
                glm_dir=data_root / subject / session / "BAS2",
                labels=labels,
                class_signal=True,
            )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)
    reports_root = tmp_path / "reports" / "experiments"

    first = run_experiment(
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=tmp_path / "cache",
        target="coarse_affect",
        model="ridge",
        cv="record_random_split",
        seed=17,
        run_id="record_random_split_a",
        reports_root=reports_root,
    )
    second = run_experiment(
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=tmp_path / "cache",
        target="coarse_affect",
        model="ridge",
        cv="record_random_split",
        seed=17,
        run_id="record_random_split_b",
        reports_root=reports_root,
    )

    first_report = Path(first["report_dir"])
    second_report = Path(second["report_dir"])
    first_metrics = json.loads((first_report / "metrics.json").read_text(encoding="utf-8"))
    second_metrics = json.loads((second_report / "metrics.json").read_text(encoding="utf-8"))
    first_splits = pd.read_csv(first_report / "fold_splits.csv")
    second_splits = pd.read_csv(second_report / "fold_splits.csv")

    expected_folds = 5
    assert int(first_metrics["n_folds"]) == expected_folds
    assert int(second_metrics["n_folds"]) == expected_folds

    compare_columns = [column for column in first_splits.columns if column != "run_id"]
    pd.testing.assert_frame_equal(
        first_splits[compare_columns].reset_index(drop=True),
        second_splits[compare_columns].reset_index(drop=True),
    )


def test_within_subject_mode_requires_subject(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
    ]
    for session in ("ses-01", "ses-02"):
        _create_glm_session(
            glm_dir=data_root / "sub-001" / session / "BAS2",
            labels=labels,
            class_signal=True,
        )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    try:
        run_experiment(
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=tmp_path / "cache",
            target="coarse_affect",
            model="ridge",
            cv="within_subject_loso_session",
            run_id="missing_subject",
            reports_root=tmp_path / "reports" / "experiments",
        )
        raise AssertionError("Expected ValueError for missing subject in within-subject mode.")
    except ValueError as exc:
        assert "requires a non-empty subject" in str(exc)


def test_experiment_runner_within_subject_mode_auditable(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
        "run-1_passive_anger_video",
        "run-1_passive_happiness_video",
    ]
    for subject in ("sub-001", "sub-002"):
        for session in ("ses-01", "ses-02", "ses-03"):
            _create_glm_session(
                glm_dir=data_root / subject / session / "BAS2",
                labels=labels,
                class_signal=True,
            )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    reports_root = tmp_path / "reports" / "experiments"
    result = run_experiment(
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=tmp_path / "cache",
        target="coarse_affect",
        model="ridge",
        cv="within_subject_loso_session",
        subject="sub-001",
        seed=123,
        run_id="within_sub001_coarse",
        reports_root=reports_root,
    )

    report_dir = Path(result["report_dir"])
    config = json.loads((report_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(report_dir / "predictions.csv")
    fold_splits = pd.read_csv(report_dir / "fold_splits.csv")
    fold_metrics = pd.read_csv(report_dir / "fold_metrics.csv")
    spatial_report = json.loads(
        (report_dir / "spatial_compatibility_report.json").read_text(encoding="utf-8")
    )

    assert config["experiment_mode"] == "within_subject_loso_session"
    assert config["subject"] == "sub-001"
    assert config["spatial_compatibility_passed"] is True
    assert metrics["experiment_mode"] == "within_subject_loso_session"
    assert metrics["subject"] == "sub-001"
    assert metrics["spatial_compatibility"]["passed"] is True
    assert spatial_report["passed"] is True

    assert set(predictions["subject"].astype(str).unique()) == {"sub-001"}
    assert set(fold_splits["subject"].astype(str).unique()) == {"sub-001"}
    assert set(fold_metrics["subject"].astype(str).unique()) == {"sub-001"}

    for _, row in fold_splits.iterrows():
        assert row["train_subjects"] == "sub-001"
        assert row["test_subjects"] == "sub-001"
        train_sessions = set(str(row["train_sessions"]).split("|"))
        test_sessions = set(str(row["test_sessions"]).split("|"))
        assert test_sessions
        assert train_sessions.isdisjoint(test_sessions)
        assert row["target"] == "coarse_affect"
        assert row["model"] == "ridge"
        assert int(row["seed"]) == 123
        assert int(row["train_sample_count"]) > 0
        assert int(row["test_sample_count"]) > 0
        assert row["config_file"] == "config.json"

    test_sessions_all = sorted(fold_splits["test_sessions"].astype(str).tolist())
    assert test_sessions_all == ["ses-01", "ses-02", "ses-03"]


def test_within_subject_mode_rejects_unknown_subject(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
    ]
    for session in ("ses-01", "ses-02"):
        _create_glm_session(
            glm_dir=data_root / "sub-001" / session / "BAS2",
            labels=labels,
            class_signal=True,
        )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    try:
        run_experiment(
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=tmp_path / "cache",
            target="coarse_affect",
            model="ridge",
            cv="within_subject_loso_session",
            subject="sub-999",
            run_id="unknown_subject",
            reports_root=tmp_path / "reports" / "experiments",
        )
        raise AssertionError("Expected ValueError for unknown subject in within-subject mode.")
    except ValueError as exc:
        assert "No samples found for subject 'sub-999'" in str(exc)


def test_experiment_cli_accepts_frozen_cross_person_transfer() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--index-csv",
            "dataset_index.csv",
            "--data-root",
            "Data",
            "--cache-dir",
            "cache",
            "--target",
            "coarse_affect",
            "--model",
            "ridge",
            "--cv",
            "frozen_cross_person_transfer",
            "--train-subject",
            "sub-001",
            "--test-subject",
            "sub-002",
        ]
    )
    assert args.cv == "frozen_cross_person_transfer"
    assert args.train_subject == "sub-001"
    assert args.test_subject == "sub-002"


def test_cross_person_transfer_requires_train_and_test_subject(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
    ]
    for subject in ("sub-001", "sub-002"):
        for session in ("ses-01", "ses-02"):
            _create_glm_session(
                glm_dir=data_root / subject / session / "BAS2",
                labels=labels,
                class_signal=True,
            )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    try:
        run_experiment(
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=tmp_path / "cache",
            target="coarse_affect",
            model="ridge",
            cv="frozen_cross_person_transfer",
            test_subject="sub-002",
            run_id="missing_train_subject",
            reports_root=tmp_path / "reports" / "experiments",
        )
        raise AssertionError("Expected ValueError when train_subject is missing.")
    except ValueError as exc:
        assert "requires a non-empty train_subject" in str(exc)

    try:
        run_experiment(
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=tmp_path / "cache",
            target="coarse_affect",
            model="ridge",
            cv="frozen_cross_person_transfer",
            train_subject="sub-001",
            run_id="missing_test_subject",
            reports_root=tmp_path / "reports" / "experiments",
        )
        raise AssertionError("Expected ValueError when test_subject is missing.")
    except ValueError as exc:
        assert "requires a non-empty test_subject" in str(exc)


def test_cross_person_transfer_rejects_identical_subjects(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
    ]
    for session in ("ses-01", "ses-02"):
        _create_glm_session(
            glm_dir=data_root / "sub-001" / session / "BAS2",
            labels=labels,
            class_signal=True,
        )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    try:
        run_experiment(
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=tmp_path / "cache",
            target="coarse_affect",
            model="ridge",
            cv="frozen_cross_person_transfer",
            train_subject="sub-001",
            test_subject="sub-001",
            run_id="same_subject_transfer",
            reports_root=tmp_path / "reports" / "experiments",
        )
        raise AssertionError("Expected ValueError when train_subject == test_subject.")
    except ValueError as exc:
        assert "must be different" in str(exc)


def test_experiment_runner_frozen_cross_person_transfer_auditable(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
        "run-1_passive_anger_video",
        "run-1_passive_happiness_video",
    ]
    subject_sessions = (
        ("sub-001", ("ses-01", "ses-02", "ses-03")),
        ("sub-002", ("ses-01", "ses-02")),
    )
    for subject, sessions in subject_sessions:
        for session in sessions:
            _create_glm_session(
                glm_dir=data_root / subject / session / "BAS2",
                labels=labels,
                class_signal=True,
            )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)
    index_df = pd.read_csv(index_csv)
    expected_train = int((index_df["subject"] == "sub-001").sum())
    expected_test = int((index_df["subject"] == "sub-002").sum())

    result = run_experiment(
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=tmp_path / "cache",
        target="coarse_affect",
        model="ridge",
        cv="frozen_cross_person_transfer",
        train_subject="sub-001",
        test_subject="sub-002",
        seed=77,
        run_id="transfer_sub001_to_sub002",
        reports_root=tmp_path / "reports" / "experiments",
    )

    report_dir = Path(result["report_dir"])
    config = json.loads((report_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(report_dir / "predictions.csv")
    fold_splits = pd.read_csv(report_dir / "fold_splits.csv")
    fold_metrics = pd.read_csv(report_dir / "fold_metrics.csv")
    spatial_report = json.loads(
        (report_dir / "spatial_compatibility_report.json").read_text(encoding="utf-8")
    )

    assert config["experiment_mode"] == "frozen_cross_person_transfer"
    assert config["train_subject"] == "sub-001"
    assert config["test_subject"] == "sub-002"
    assert config["spatial_compatibility_passed"] is True
    assert metrics["experiment_mode"] == "frozen_cross_person_transfer"
    assert metrics["train_subject"] == "sub-001"
    assert metrics["test_subject"] == "sub-002"
    assert metrics["spatial_compatibility"]["passed"] is True
    assert spatial_report["passed"] is True
    assert metrics["target"] == "coarse_affect"
    assert metrics["n_folds"] == 1

    assert set(predictions["subject"].astype(str).unique()) == {"sub-002"}
    assert set(predictions["train_subject"].astype(str).unique()) == {"sub-001"}
    assert set(predictions["test_subject"].astype(str).unique()) == {"sub-002"}
    assert {"emotion", "coarse_affect"} <= set(predictions.columns)

    assert len(fold_splits) == 1
    split_row = fold_splits.iloc[0]
    assert split_row["experiment_mode"] == "frozen_cross_person_transfer"
    assert split_row["train_subject"] == "sub-001"
    assert split_row["test_subject"] == "sub-002"
    assert split_row["train_subjects"] == "sub-001"
    assert split_row["test_subjects"] == "sub-002"
    assert int(split_row["train_sample_count"]) == expected_train
    assert int(split_row["test_sample_count"]) == expected_test
    assert set(str(split_row["train_sessions"]).split("|")) == {"ses-01", "ses-02", "ses-03"}
    assert set(str(split_row["test_sessions"]).split("|")) == {"ses-01", "ses-02"}
    assert split_row["target"] == "coarse_affect"
    assert split_row["model"] == "ridge"
    assert int(split_row["seed"]) == 77
    assert split_row["config_file"] == "config.json"

    assert len(fold_metrics) == 1
    fold_row = fold_metrics.iloc[0]
    assert fold_row["train_subject"] == "sub-001"
    assert fold_row["test_subject"] == "sub-002"
    assert int(fold_row["n_train"]) == expected_train
    assert int(fold_row["n_test"]) == expected_test


@pytest.mark.parametrize(
    ("session2_kwargs", "expected_reason"),
    [
        pytest.param({"shape": (4, 3, 3)}, "image_shape mismatch", id="shape_mismatch"),
        pytest.param(
            {"affine": np.diag([2.0, 1.0, 1.0, 1.0])},
            "affine mismatch",
            id="affine_mismatch",
        ),
        pytest.param(
            {"mask_variant": "expanded_x"},
            "feature_count mismatch",
            id="feature_mismatch",
        ),
        pytest.param({"mask_variant": "shift_x"}, "mask_sha256 mismatch", id="mask_hash_mismatch"),
    ],
)
def test_spatial_compatibility_mismatch_fails_with_clear_error(
    tmp_path: Path,
    session2_kwargs: dict[str, object],
    expected_reason: str,
) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
    ]
    _create_glm_session(
        glm_dir=data_root / "sub-001" / "ses-01" / "BAS2",
        labels=labels,
        class_signal=True,
    )
    _create_glm_session(
        glm_dir=data_root / "sub-001" / "ses-02" / "BAS2",
        labels=labels,
        class_signal=True,
        **session2_kwargs,
    )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    reports_root = tmp_path / "reports" / "experiments"
    run_id = f"spatial_mismatch_{expected_reason.replace(' ', '_')}"
    with pytest.raises(ValueError, match="Spatial compatibility validation failed"):
        run_experiment(
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=tmp_path / "cache",
            target="coarse_affect",
            model="ridge",
            cv="within_subject_loso_session",
            subject="sub-001",
            run_id=run_id,
            reports_root=reports_root,
        )

    spatial_report_path = reports_root / run_id / "spatial_compatibility_report.json"
    assert spatial_report_path.exists()
    report = json.loads(spatial_report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    reasons = json.dumps(report["mismatches"])
    assert expected_reason in reasons


def test_spatial_compatibility_missing_legacy_signature_fails_explicitly(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
    ]
    for session in ("ses-01", "ses-02"):
        _create_glm_session(
            glm_dir=data_root / "sub-001" / session / "BAS2",
            labels=labels,
            class_signal=True,
        )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    cache_dir = tmp_path / "cache"
    manifest_path = build_feature_cache(
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
    )
    manifest = pd.read_csv(manifest_path)
    legacy_npz_path = Path(manifest.loc[0, "cache_path"])
    with np.load(legacy_npz_path, allow_pickle=False) as npz:
        legacy_payload = {
            "X": np.asarray(npz["X"], dtype=np.float32),
            "y": np.asarray(npz["y"]),
            "metadata_json": np.asarray(npz["metadata_json"]),
            "group_id": np.asarray(npz["group_id"]),
        }
    np.savez_compressed(legacy_npz_path, **legacy_payload)

    reports_root = tmp_path / "reports" / "experiments"
    run_id = "legacy_missing_spatial_signature"
    with pytest.raises(
        ValueError,
        match="missing spatial signature metadata",
    ):
        run_experiment(
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=cache_dir,
            target="coarse_affect",
            model="ridge",
            cv="within_subject_loso_session",
            subject="sub-001",
            run_id=run_id,
            reports_root=reports_root,
        )

    spatial_report_path = reports_root / run_id / "spatial_compatibility_report.json"
    assert spatial_report_path.exists()
    report = json.loads(spatial_report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    reasons = json.dumps(report["mismatches"])
    assert "missing spatial signature metadata" in reasons


@pytest.mark.parametrize("model_name", ["ridge", "linearsvc", "logreg"])
def test_within_subject_interpretability_exports_for_linear_models(
    tmp_path: Path, model_name: str
) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
        "run-1_passive_anger_video",
        "run-1_passive_happiness_video",
    ]
    for subject in ("sub-001", "sub-002"):
        for session in ("ses-01", "ses-02", "ses-03"):
            _create_glm_session(
                glm_dir=data_root / subject / session / "BAS2",
                labels=labels,
                class_signal=True,
            )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    result = run_experiment(
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=tmp_path / "cache",
        target="coarse_affect",
        model=model_name,
        cv="within_subject_loso_session",
        subject="sub-001",
        seed=11,
        run_id=f"within_{model_name}_interpretability",
        reports_root=tmp_path / "reports" / "experiments",
    )

    report_dir = Path(result["report_dir"])
    summary_path = report_dir / "interpretability_summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["enabled"] is True
    assert summary["performed"] is True
    assert summary["experiment_mode"] == "within_subject_loso_session"
    assert summary["subject"] == "sub-001"
    assert summary["model"] == model_name
    assert summary["target"] == "coarse_affect"
    assert summary["stability"] is not None
    assert summary["stability"]["n_folds"] == 3
    assert summary["stability"]["n_pairs"] == 3

    artifacts_path = Path(str(summary["fold_artifacts_path"]))
    assert artifacts_path.exists()
    artifacts = pd.read_csv(artifacts_path)
    assert len(artifacts) == result["metrics"]["n_folds"] == 3
    assert {
        "fold",
        "experiment_mode",
        "subject",
        "held_out_test_sessions",
        "model",
        "target",
        "n_features",
        "coef_shape",
        "coefficient_file",
        "seed",
        "run_id",
        "config_file",
    } <= set(artifacts.columns)
    assert set(artifacts["subject"].astype(str).unique()) == {"sub-001"}
    assert set(artifacts["held_out_test_sessions"].astype(str).tolist()) == {
        "ses-01",
        "ses-02",
        "ses-03",
    }

    for _, row in artifacts.iterrows():
        coef_path = Path(str(row["coefficient_file"]))
        assert coef_path.exists()
        with np.load(coef_path, allow_pickle=False) as npz:
            coefficients = np.asarray(npz["coefficients"])
            intercept = np.asarray(npz["intercept"])
            class_labels = np.asarray(npz["class_labels"])
            feature_index = np.asarray(npz["feature_index"])
        assert coefficients.ndim == 2
        assert intercept.ndim == 1
        assert feature_index.ndim == 1
        assert coefficients.shape[1] == int(row["n_features"])
        assert feature_index.shape[0] == int(row["n_features"])
        assert class_labels.size >= 1


def test_within_subject_interpretability_rejects_unsupported_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
        "run-1_passive_anger_video",
        "run-1_passive_happiness_video",
    ]
    for session in ("ses-01", "ses-02", "ses-03"):
        _create_glm_session(
            glm_dir=data_root / "sub-001" / session / "BAS2",
            labels=labels,
            class_signal=True,
        )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.dummy import DummyClassifier

    def _dummy_pipeline(*_args: object, **_kwargs: object) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        )

    module = importlib.import_module("Thesis_ML.experiments.run_experiment")
    monkeypatch.setattr(module, "_build_pipeline", _dummy_pipeline)
    
    with pytest.raises(
        ValueError,
        match="Interpretability export requires a fitted linear model with a 'coef_' attribute.",
    ):
        run_experiment(
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=tmp_path / "cache",
            target="coarse_affect",
            model="ridge",
            cv="within_subject_loso_session",
            subject="sub-001",
            run_id="unsupported_model_interpretability",
            reports_root=tmp_path / "reports" / "experiments",
        )

def test_feature_cache_rejects_mixed_mask_paths_within_group(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    glm_dir = data_root / "sub-001" / "ses-01" / "BAS2"
    _create_glm_session(
        glm_dir=glm_dir,
        labels=[
            "run-1_passive_anger_audio",
            "run-1_passive_happiness_video",
        ],
    )

    alt_mask = glm_dir / "mask_alt.nii"
    alt_mask_data = np.zeros((3, 3, 3), dtype=np.float32)
    alt_mask_data[:-1, 1:, 1:] = 1.0
    _write_nifti(alt_mask, alt_mask_data)

    out_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=out_csv)

    dataset = pd.read_csv(out_csv)
    assert len(dataset) == 2
    dataset.loc[1, "mask_path"] = alt_mask.relative_to(data_root).as_posix()
    dataset.to_csv(out_csv, index=False)

    with pytest.raises(ValueError) as exc:
        build_feature_cache(
            index_csv=out_csv,
            data_root=data_root,
            cache_dir=tmp_path / "cache",
        )

    message = str(exc.value)
    assert "contains multiple resolved mask paths" in message
    assert "sub-001_ses-01_BAS2" in message
    assert "mask.nii" in message
    assert "mask_alt.nii" in message

def test_feature_cache_rejects_mixed_mask_paths_even_when_cache_exists(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    glm_dir = data_root / "sub-001" / "ses-01" / "BAS2"
    _create_glm_session(
        glm_dir=glm_dir,
        labels=[
            "run-1_passive_anger_audio",
            "run-1_passive_happiness_video",
        ],
    )

    out_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=out_csv)

    cache_dir = tmp_path / "cache"
    build_feature_cache(index_csv=out_csv, data_root=data_root, cache_dir=cache_dir)

    alt_mask = glm_dir / "mask_alt.nii"
    alt_mask_data = np.zeros((3, 3, 3), dtype=np.float32)
    alt_mask_data[:-1, 1:, 1:] = 1.0
    _write_nifti(alt_mask, alt_mask_data)

    dataset = pd.read_csv(out_csv)
    dataset.loc[1, "mask_path"] = alt_mask.relative_to(data_root).as_posix()
    dataset.to_csv(out_csv, index=False)

    with pytest.raises(ValueError) as exc:
        build_feature_cache(
            index_csv=out_csv,
            data_root=data_root,
            cache_dir=cache_dir,
        )

    message = str(exc.value)
    assert "contains multiple resolved mask paths" in message
    assert "sub-001_ses-01_BAS2" in message