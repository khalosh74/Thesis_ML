from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from Thesis_ML.experiments.feature_space_loading import (
    FEATURE_SPACE_ROI_MASKED_PREDEFINED,
    load_feature_matrix,
    normalize_feature_space,
)


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4, dtype=np.float64))
    nib.save(image, str(path))


def _write_roi_spec(
    *,
    path: Path,
    representation: str,
    mask_paths: list[Path],
) -> Path:
    payload = {
        "feature_space_id": "test_feature_space",
        "representation": representation,
        "description": "Synthetic ROI spec for unit tests.",
        "reference_space": "test_reference_space",
        "rois": [
            {
                "roi_id": f"roi_{idx + 1}",
                "type": "mask",
                "mask_path": str(mask_path),
            }
            for idx, mask_path in enumerate(mask_paths)
        ],
    }
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    return path


def _unexpected_cache_loader(**_: object) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    raise AssertionError("whole_brain cache loader should not be called for ROI feature spaces.")


def test_normalize_feature_space_accepts_roi_masked_predefined() -> None:
    assert (
        normalize_feature_space("ROI_MASKED_PREDEFINED")
        == FEATURE_SPACE_ROI_MASKED_PREDEFINED
    )


def test_roi_masked_predefined_loads_expected_matrix_and_spatial_payload(tmp_path: Path) -> None:
    shape = (2, 2, 2)
    beta_1 = np.arange(1, 9, dtype=np.float32).reshape(shape)
    beta_2 = beta_1 + 100.0
    beta_1_path = tmp_path / "betas" / "beta_1.nii.gz"
    beta_2_path = tmp_path / "betas" / "beta_2.nii.gz"
    _write_nifti(beta_1_path, beta_1)
    _write_nifti(beta_2_path, beta_2)

    common_mask = np.zeros(shape, dtype=np.float32)
    common_mask[0, 0, 0] = 1.0
    common_mask[0, 0, 1] = 1.0
    common_mask[1, 0, 0] = 1.0
    common_mask_path = tmp_path / "masks" / "common_mask.nii.gz"
    _write_nifti(common_mask_path, common_mask)

    roi_mask_1 = np.zeros(shape, dtype=np.float32)
    roi_mask_1[0, 0, 0] = 1.0
    roi_mask_1[1, 1, 1] = 1.0
    roi_mask_2 = np.zeros(shape, dtype=np.float32)
    roi_mask_2[0, 0, 1] = 1.0
    roi_mask_1_path = tmp_path / "roi" / "roi_mask_1.nii.gz"
    roi_mask_2_path = tmp_path / "roi" / "roi_mask_2.nii.gz"
    _write_nifti(roi_mask_1_path, roi_mask_1)
    _write_nifti(roi_mask_2_path, roi_mask_2)

    roi_spec_path = _write_roi_spec(
        path=tmp_path / "roi_spec_masked.json",
        representation="roi_masked_predefined",
        mask_paths=[roi_mask_1_path, roi_mask_2_path],
    )

    selected_index_df = pd.DataFrame(
        [
            {
                "sample_id": "sample_2",
                "beta_path_canonical": str(beta_2_path),
                "mask_path_canonical": str(common_mask_path),
                "beta_path": str(beta_2_path),
                "mask_path": str(common_mask_path),
            },
            {
                "sample_id": "sample_1",
                "beta_path_canonical": str(beta_1_path),
                "mask_path_canonical": str(common_mask_path),
                "beta_path": str(beta_1_path),
                "mask_path": str(common_mask_path),
            },
        ]
    )

    spatial_report_path = tmp_path / "spatial_compatibility_report.json"
    x_matrix, metadata_df, spatial = load_feature_matrix(
        selected_index_df=selected_index_df,
        cache_manifest_path=tmp_path / "cache_manifest.csv",
        spatial_report_path=spatial_report_path,
        affine_atol=1e-5,
        data_root=tmp_path,
        feature_space="roi_masked_predefined",
        roi_spec_path=roi_spec_path,
        load_features_from_cache_fn=_unexpected_cache_loader,
    )

    assert x_matrix.shape == (2, 2)
    assert np.allclose(x_matrix[0], np.asarray([101.0, 102.0], dtype=np.float32))
    assert np.allclose(x_matrix[1], np.asarray([1.0, 2.0], dtype=np.float32))
    assert metadata_df["sample_id"].tolist() == ["sample_2", "sample_1"]

    required_keys = {
        "status",
        "passed",
        "feature_space",
        "roi_spec_path",
        "feature_space_id",
        "representation",
        "reference_space",
        "roi_spec_description",
        "roi_ids",
        "roi_count",
        "atlas_union_voxel_count",
        "common_valid_mask_voxel_count",
        "effective_union_voxel_count",
        "image_shape",
        "affine",
        "voxel_size",
        "mask_voxel_count",
        "feature_count",
        "mask_sha256",
        "n_features",
    }
    assert required_keys <= set(spatial.keys())
    assert spatial["status"] == "passed"
    assert spatial["passed"] is True
    assert spatial["feature_space"] == "roi_masked_predefined"
    assert spatial["mask_voxel_count"] == 2
    assert spatial["feature_count"] == 2
    assert spatial["n_features"] == 2
    assert spatial["effective_union_voxel_count"] == 2

    report_payload = json.loads(spatial_report_path.read_text(encoding="utf-8"))
    assert report_payload["mask_sha256"] == spatial["mask_sha256"]


def test_roi_masked_predefined_representation_mismatch_raises(tmp_path: Path) -> None:
    shape = (2, 2, 2)
    beta = np.ones(shape, dtype=np.float32)
    mask = np.ones(shape, dtype=np.float32)
    roi_mask = np.ones(shape, dtype=np.float32)
    beta_path = tmp_path / "beta.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    roi_mask_path = tmp_path / "roi_mask.nii.gz"
    _write_nifti(beta_path, beta)
    _write_nifti(mask_path, mask)
    _write_nifti(roi_mask_path, roi_mask)

    roi_spec_path = _write_roi_spec(
        path=tmp_path / "roi_spec_mean_representation.json",
        representation="roi_mean_predefined",
        mask_paths=[roi_mask_path],
    )
    selected_index_df = pd.DataFrame(
        [
            {
                "sample_id": "sample_1",
                "beta_path_canonical": str(beta_path),
                "mask_path_canonical": str(mask_path),
                "beta_path": str(beta_path),
                "mask_path": str(mask_path),
            }
        ]
    )

    with pytest.raises(ValueError, match="representation disagree"):
        load_feature_matrix(
            selected_index_df=selected_index_df,
            cache_manifest_path=tmp_path / "cache_manifest.csv",
            spatial_report_path=tmp_path / "spatial_report.json",
            affine_atol=1e-5,
            data_root=tmp_path,
            feature_space="roi_masked_predefined",
            roi_spec_path=roi_spec_path,
            load_features_from_cache_fn=_unexpected_cache_loader,
        )


def test_roi_masked_predefined_rejects_mask_signature_disagreement(tmp_path: Path) -> None:
    shape = (2, 2, 2)
    beta_1_path = tmp_path / "beta_1.nii.gz"
    beta_2_path = tmp_path / "beta_2.nii.gz"
    _write_nifti(beta_1_path, np.ones(shape, dtype=np.float32))
    _write_nifti(beta_2_path, np.ones(shape, dtype=np.float32) * 2.0)

    mask_a = np.zeros(shape, dtype=np.float32)
    mask_a[0, 0, 0] = 1.0
    mask_b = np.zeros(shape, dtype=np.float32)
    mask_b[0, 0, 1] = 1.0
    mask_a_path = tmp_path / "mask_a.nii.gz"
    mask_b_path = tmp_path / "mask_b.nii.gz"
    _write_nifti(mask_a_path, mask_a)
    _write_nifti(mask_b_path, mask_b)

    roi_mask = np.zeros(shape, dtype=np.float32)
    roi_mask[0, 0, 0] = 1.0
    roi_mask_path = tmp_path / "roi_mask.nii.gz"
    _write_nifti(roi_mask_path, roi_mask)
    roi_spec_path = _write_roi_spec(
        path=tmp_path / "roi_spec_masked.json",
        representation="roi_masked_predefined",
        mask_paths=[roi_mask_path],
    )

    selected_index_df = pd.DataFrame(
        [
            {
                "sample_id": "sample_1",
                "beta_path_canonical": str(beta_1_path),
                "mask_path_canonical": str(mask_a_path),
                "beta_path": str(beta_1_path),
                "mask_path": str(mask_a_path),
            },
            {
                "sample_id": "sample_2",
                "beta_path_canonical": str(beta_2_path),
                "mask_path_canonical": str(mask_b_path),
                "beta_path": str(beta_2_path),
                "mask_path": str(mask_b_path),
            },
        ]
    )

    with pytest.raises(ValueError, match="single common valid mask signature"):
        load_feature_matrix(
            selected_index_df=selected_index_df,
            cache_manifest_path=tmp_path / "cache_manifest.csv",
            spatial_report_path=tmp_path / "spatial_report.json",
            affine_atol=1e-5,
            data_root=tmp_path,
            feature_space="roi_masked_predefined",
            roi_spec_path=roi_spec_path,
            load_features_from_cache_fn=_unexpected_cache_loader,
        )


def test_roi_masked_predefined_rejects_zero_voxel_effective_union(tmp_path: Path) -> None:
    shape = (2, 2, 2)
    beta_path = tmp_path / "beta.nii.gz"
    _write_nifti(beta_path, np.ones(shape, dtype=np.float32))

    common_mask = np.zeros(shape, dtype=np.float32)
    common_mask[1, 1, 1] = 1.0
    common_mask_path = tmp_path / "common_mask.nii.gz"
    _write_nifti(common_mask_path, common_mask)

    roi_mask = np.zeros(shape, dtype=np.float32)
    roi_mask[0, 0, 0] = 1.0
    roi_mask_path = tmp_path / "roi_mask.nii.gz"
    _write_nifti(roi_mask_path, roi_mask)
    roi_spec_path = _write_roi_spec(
        path=tmp_path / "roi_spec_masked.json",
        representation="roi_masked_predefined",
        mask_paths=[roi_mask_path],
    )

    selected_index_df = pd.DataFrame(
        [
            {
                "sample_id": "sample_1",
                "beta_path_canonical": str(beta_path),
                "mask_path_canonical": str(common_mask_path),
                "beta_path": str(beta_path),
                "mask_path": str(common_mask_path),
            }
        ]
    )

    with pytest.raises(ValueError, match="zero voxels"):
        load_feature_matrix(
            selected_index_df=selected_index_df,
            cache_manifest_path=tmp_path / "cache_manifest.csv",
            spatial_report_path=tmp_path / "spatial_report.json",
            affine_atol=1e-5,
            data_root=tmp_path,
            feature_space="roi_masked_predefined",
            roi_spec_path=roi_spec_path,
            load_features_from_cache_fn=_unexpected_cache_loader,
        )
