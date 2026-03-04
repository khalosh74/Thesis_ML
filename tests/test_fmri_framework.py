from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from Thesis_ML.data.index_dataset import build_dataset_index
from Thesis_ML.experiments.run_experiment import run_experiment
from Thesis_ML.features.nifti_features import build_feature_cache
from Thesis_ML.spm.extract_glm import extract_glm_session, parse_regressor_label


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4))
    nib.save(image, str(path))


def _create_glm_session(
    glm_dir: Path,
    labels: list[str],
    missing_beta_indexes: set[int] | None = None,
    class_signal: bool = False,
) -> None:
    glm_dir.mkdir(parents=True, exist_ok=True)
    missing_beta_indexes = missing_beta_indexes or set()

    mask = np.zeros((3, 3, 3), dtype=np.float32)
    mask[1:, 1:, 1:] = 1.0
    _write_nifti(glm_dir / "mask.nii", mask)

    pd.Series(labels).to_csv(glm_dir / "regressor_labels.csv", index=False, header=False)

    for idx, label in enumerate(labels, start=1):
        if idx in missing_beta_indexes:
            continue

        beta = np.full((3, 3, 3), fill_value=float(idx), dtype=np.float32)
        if class_signal:
            if "_anger_" in label:
                beta[1:, 1:, 1:] += 5.0
            if "_joy_" in label:
                beta[1:, 1:, 1:] -= 5.0
        _write_nifti(glm_dir / f"beta_{idx:04d}.nii", beta)


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
        glm_dir=data_root / "sub-001" / "ses-01" / "BAS1",
        labels=[
            "run-1_passive_anger_audio",
            "run-1_passive_joy_video",
            "run-1_passive_resp_movement",
        ],
    )
    _create_glm_session(
        glm_dir=data_root / "sub-002" / "ses-02" / "BAS2",
        labels=[
            "run-2_emo_anger_audio",
            "run-2_emo_joy_audiovisual",
            "run-2_emo_constant",
        ],
    )

    out_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=out_csv)
    assert out_csv.exists()

    index_df = pd.read_csv(out_csv)
    assert len(index_df) == 4
    assert {"subject", "session", "bas", "beta_path", "mask_path", "regressor_label"} <= set(
        index_df.columns
    )
    assert index_df["beta_path"].map(lambda value: not Path(value).is_absolute()).all()
    assert index_df["mask_path"].map(lambda value: not Path(value).is_absolute()).all()

    extraction_csv = (
        data_root
        / "processed"
        / "extractions"
        / "sub-001"
        / "ses-01"
        / "BAS1"
        / "regressor_beta_mapping.csv"
    )
    assert extraction_csv.exists()


def test_feature_cache(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    _create_glm_session(
        glm_dir=data_root / "sub-001" / "ses-01" / "BAS1",
        labels=[
            "run-1_passive_anger_audio",
            "run-1_passive_joy_video",
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

    assert x_matrix.dtype == np.float32
    assert x_matrix.shape[0] == len(y) == len(metadata)
    assert x_matrix.shape[1] == 8


def test_experiment_runner_smoke(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_joy_audio",
        "run-1_passive_anger_video",
        "run-1_passive_joy_video",
    ]

    sessions = [
        ("sub-001", "ses-01", "BAS1"),
        ("sub-001", "ses-02", "BAS1"),
        ("sub-002", "ses-01", "BAS1"),
        ("sub-002", "ses-02", "BAS1"),
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

    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    assert {"accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix"} <= set(metrics)
    assert metrics["n_folds"] >= 2
    assert "permutation_test" in metrics

    predictions = pd.read_csv(report_dir / "predictions.csv")
    assert len(predictions) > 0
    assert {"y_true", "y_pred", "subject", "session", "task", "modality"} <= set(
        predictions.columns
    )
