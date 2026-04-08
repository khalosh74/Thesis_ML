from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from Thesis_ML.release.loader import load_dataset_manifest
from Thesis_ML.release.validator import validate_dataset_manifest


_REQUIRED_COLUMNS = [
    "sample_id",
    "subject",
    "session",
    "task",
    "modality",
    "emotion",
    "coarse_affect",
    "beta_path",
    "mask_path",
    "subject_session",
    "subject_session_bas",
]


def _write_demo_row_dataset(base: Path) -> tuple[Path, Path, Path]:
    data_root = base / "data_root"
    glm_dir = data_root / "sub-001" / "ses-01" / "BAS2"
    glm_dir.mkdir(parents=True, exist_ok=True)

    mask = np.ones((2, 2, 2), dtype=np.float32)
    beta = np.ones((2, 2, 2), dtype=np.float32)
    nib.save(nib.Nifti1Image(mask, affine=np.eye(4, dtype=np.float64)), str(glm_dir / "mask.nii"))
    nib.save(nib.Nifti1Image(beta, affine=np.eye(4, dtype=np.float64)), str(glm_dir / "beta_0001.nii"))

    index = pd.DataFrame(
        [
            {
                "sample_id": "sub-001_ses-01_BAS2_0001",
                "subject": "sub-001",
                "session": "ses-01",
                "bas": "BAS2",
                "task": "emo",
                "modality": "audiovisual",
                "emotion": "anger",
                "coarse_affect": "negative",
                "beta_path": "sub-001/ses-01/BAS2/beta_0001.nii",
                "mask_path": "sub-001/ses-01/BAS2/mask.nii",
                "subject_session": "sub-001_ses-01",
                "subject_session_bas": "sub-001_ses-01_BAS2",
            }
        ]
    )
    index_path = base / "index.csv"
    index.to_csv(index_path, index=False)

    manifest = {
        "schema_version": "dataset-instance-v1",
        "dataset_id": "synthetic_test",
        "dataset_contract_version": "fmri_beta_dataset_v1",
        "dataset_fingerprint": "synthetic_fingerprint",
        "index_csv": "index.csv",
        "data_root": "data_root",
        "cache_dir": "cache",
        "created_at": "2026-04-08T00:00:00Z",
        "source_extraction_version": "synthetic_test_v1",
        "sample_unit": "beta_event",
        "required_columns": list(_REQUIRED_COLUMNS),
        "subject_count": 1,
        "session_counts_by_subject": {"sub-001": 1},
    }
    manifest_path = base / "dataset_manifest.json"
    manifest_path.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")
    return manifest_path, index_path, data_root


def test_valid_dataset_manifest_passes_validation(tmp_path: Path) -> None:
    manifest_path, _, _ = _write_demo_row_dataset(tmp_path)
    summary = validate_dataset_manifest(manifest_path)
    assert summary["passed"] is True
    assert summary["cache_dir_defaulted"] is False


def test_dataset_manifest_fails_when_required_column_missing(tmp_path: Path) -> None:
    manifest_path, index_path, _ = _write_demo_row_dataset(tmp_path)
    frame = pd.read_csv(index_path)
    frame = frame.drop(columns=["mask_path"])
    frame.to_csv(index_path, index=False)

    summary = validate_dataset_manifest(manifest_path)
    assert summary["passed"] is False
    assert any(
        str(issue.get("code")) in {"dataset_index_validation_failed", "dataset_schema_error"}
        for issue in summary["issues"]
    )


def test_dataset_manifest_relative_path_resolution_works(tmp_path: Path) -> None:
    manifest_path, index_path, data_root = _write_demo_row_dataset(tmp_path)
    loaded = load_dataset_manifest(manifest_path)
    assert loaded.index_csv_path == index_path.resolve()
    assert loaded.data_root_path == data_root.resolve()


def test_dataset_manifest_cache_dir_defaults_to_sibling_cache(tmp_path: Path) -> None:
    manifest_path, _, _ = _write_demo_row_dataset(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("cache_dir", None)
    manifest_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    summary = validate_dataset_manifest(manifest_path)
    assert summary["passed"] is True
    assert summary["cache_dir_defaulted"] is True
    assert Path(str(summary["cache_dir_path"])).name == ".cache"
