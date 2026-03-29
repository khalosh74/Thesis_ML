from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

_AFFINE_ATOL = 1e-5
_ALLOWED_ROI_TYPES = {"mask", "sphere_mni"}


def _as_float_triplet(value: Any, *, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"ROI field '{field_name}' must be a list of three numeric values.")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except Exception as exc:  # pragma: no cover - defensive conversion guard
        raise ValueError(f"ROI field '{field_name}' must contain numeric values.") from exc


def _resolve_path_candidate(
    raw_path: str,
    *,
    spec_path: Path,
    data_root: Path | None = None,
) -> Path:
    candidate = Path(str(raw_path).strip())
    if candidate.is_absolute():
        return candidate.resolve()

    search_paths = [
        (spec_path.parent / candidate),
        (Path.cwd() / candidate),
    ]
    if data_root is not None:
        search_paths.append(data_root / candidate)

    for search_path in search_paths:
        if search_path.exists():
            return search_path.resolve()
    return (Path.cwd() / candidate).resolve()


def _resolve_beta_path(row: pd.Series, *, data_root: Path) -> Path:
    canonical = row.get("beta_path_canonical")
    if canonical is not None:
        text = str(canonical).strip()
        if text:
            return Path(text).resolve()

    beta_path = row.get("beta_path")
    if beta_path is None or not str(beta_path).strip():
        sample_id = str(row.get("sample_id", "unknown_sample"))
        raise ValueError(f"Selected row '{sample_id}' is missing beta path metadata.")

    beta_candidate = Path(str(beta_path).strip())
    if beta_candidate.is_absolute():
        return beta_candidate.resolve()
    return (data_root / beta_candidate).resolve()


def _validate_roi_spec(roi_spec: dict[str, Any], *, roi_spec_path: Path) -> dict[str, Any]:
    required_top_level = {
        "feature_space_id",
        "representation",
        "description",
        "reference_space",
        "rois",
    }
    missing_top_level = sorted(required_top_level - set(roi_spec.keys()))
    if missing_top_level:
        raise ValueError(
            f"ROI spec '{roi_spec_path}' is missing required keys: {', '.join(missing_top_level)}."
        )

    rois_raw = roi_spec.get("rois")
    if not isinstance(rois_raw, list) or not rois_raw:
        raise ValueError(f"ROI spec '{roi_spec_path}' must define a non-empty 'rois' list.")

    normalized_rois: list[dict[str, Any]] = []
    seen_roi_ids: set[str] = set()
    for index, roi_raw in enumerate(rois_raw, start=1):
        if not isinstance(roi_raw, dict):
            raise ValueError(f"ROI entry #{index} in '{roi_spec_path}' must be an object.")

        roi_id = str(roi_raw.get("roi_id", "")).strip()
        roi_type = str(roi_raw.get("type", "")).strip()
        if not roi_id:
            raise ValueError(f"ROI entry #{index} in '{roi_spec_path}' is missing non-empty roi_id.")
        if roi_id in seen_roi_ids:
            raise ValueError(f"ROI spec '{roi_spec_path}' contains duplicate roi_id '{roi_id}'.")
        if roi_type not in _ALLOWED_ROI_TYPES:
            allowed = ", ".join(sorted(_ALLOWED_ROI_TYPES))
            raise ValueError(
                f"ROI '{roi_id}' in '{roi_spec_path}' has unsupported type '{roi_type}'. "
                f"Allowed: {allowed}."
            )
        seen_roi_ids.add(roi_id)

        normalized_roi: dict[str, Any] = {"roi_id": roi_id, "type": roi_type}
        if roi_type == "mask":
            mask_path = roi_raw.get("mask_path")
            if mask_path is None or not str(mask_path).strip():
                raise ValueError(
                    f"ROI '{roi_id}' in '{roi_spec_path}' requires non-empty mask_path."
                )
            normalized_roi["mask_path"] = str(mask_path).strip()
        else:
            normalized_roi["center_mni"] = _as_float_triplet(
                roi_raw.get("center_mni"),
                field_name=f"{roi_id}.center_mni",
            )
            radius_value = roi_raw.get("radius_mm")
            if radius_value is None:
                raise ValueError(f"ROI '{roi_id}' in '{roi_spec_path}' requires radius_mm.")
            radius_mm = float(radius_value)
            if radius_mm <= 0:
                raise ValueError(f"ROI '{roi_id}' in '{roi_spec_path}' requires radius_mm > 0.")
            normalized_roi["radius_mm"] = radius_mm
        normalized_rois.append(normalized_roi)

    return {
        "feature_space_id": str(roi_spec["feature_space_id"]),
        "representation": str(roi_spec["representation"]),
        "description": str(roi_spec["description"]),
        "reference_space": str(roi_spec["reference_space"]),
        "rois": normalized_rois,
    }


def _load_roi_spec(roi_spec_path: Path) -> dict[str, Any]:
    if not roi_spec_path.exists():
        raise FileNotFoundError(f"ROI spec does not exist: {roi_spec_path}")
    try:
        payload = json.loads(roi_spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"ROI spec is not valid JSON: {roi_spec_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"ROI spec root must be an object: {roi_spec_path}")
    return _validate_roi_spec(payload, roi_spec_path=roi_spec_path)


def _sphere_mask_from_mni(
    *,
    image_shape: tuple[int, int, int],
    image_affine: np.ndarray,
    center_mni: tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    i_idx, j_idx, k_idx = np.indices(image_shape, dtype=np.float64)
    ijk = np.stack([i_idx.ravel(), j_idx.ravel(), k_idx.ravel()], axis=1)
    homogeneous = np.concatenate([ijk, np.ones((ijk.shape[0], 1), dtype=np.float64)], axis=1)
    xyz = (homogeneous @ image_affine.T)[:, :3]
    center = np.asarray(center_mni, dtype=np.float64)
    distances = np.linalg.norm(xyz - center, axis=1)
    sphere_flat = distances <= float(radius_mm)
    return sphere_flat.reshape(image_shape)


def _build_roi_masks(
    *,
    roi_spec: dict[str, Any],
    roi_spec_path: Path,
    data_root: Path,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
) -> tuple[list[str], list[np.ndarray]]:
    roi_ids: list[str] = []
    roi_masks: list[np.ndarray] = []

    for roi in roi_spec["rois"]:
        roi_id = str(roi["roi_id"])
        roi_type = str(roi["type"])
        if roi_type == "mask":
            mask_path = _resolve_path_candidate(
                str(roi["mask_path"]),
                spec_path=roi_spec_path,
                data_root=data_root,
            )
            if not mask_path.exists():
                raise FileNotFoundError(f"ROI mask file does not exist for '{roi_id}': {mask_path}")
            mask_img = nib.load(str(mask_path))
            mask_data = np.asarray(mask_img.get_fdata(dtype=np.float32))
            mask_affine = np.asarray(mask_img.affine, dtype=np.float64)
            if tuple(mask_data.shape) != reference_shape:
                raise ValueError(
                    f"ROI mask '{roi_id}' shape mismatch. "
                    f"Expected {reference_shape}, got {tuple(mask_data.shape)}."
                )
            if not np.allclose(mask_affine, reference_affine, rtol=0.0, atol=_AFFINE_ATOL):
                raise ValueError(f"ROI mask '{roi_id}' affine mismatch with beta reference image.")
            roi_mask = np.isfinite(mask_data) & (mask_data > 0)
        else:
            roi_mask = _sphere_mask_from_mni(
                image_shape=reference_shape,
                image_affine=reference_affine,
                center_mni=tuple(roi["center_mni"]),
                radius_mm=float(roi["radius_mm"]),
            )

        voxel_count = int(np.asarray(roi_mask, dtype=bool).sum())
        if voxel_count <= 0:
            raise ValueError(
                f"ROI '{roi_id}' in '{roi_spec_path}' resolves to zero voxels and is invalid."
            )
        roi_ids.append(roi_id)
        roi_masks.append(np.asarray(roi_mask, dtype=bool))

    return roi_ids, roi_masks


def load_roi_feature_matrix(
    *,
    selected_index_df: pd.DataFrame,
    data_root: Path,
    roi_spec_path: Path,
    spatial_report_path: Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    if selected_index_df.empty:
        raise ValueError("ROI feature loading received an empty selected index dataframe.")

    resolved_spec_path = Path(roi_spec_path).resolve()
    roi_spec = _load_roi_spec(resolved_spec_path)

    first_row = selected_index_df.iloc[0]
    reference_beta_path = _resolve_beta_path(first_row, data_root=Path(data_root))
    if not reference_beta_path.exists():
        raise FileNotFoundError(f"Reference beta file does not exist: {reference_beta_path}")
    reference_beta_img = nib.load(str(reference_beta_path))
    reference_shape = tuple(int(value) for value in reference_beta_img.shape)
    if len(reference_shape) != 3:
        raise ValueError(
            f"ROI feature loading expects 3D beta maps. Got shape {reference_shape} for "
            f"'{reference_beta_path}'."
        )
    reference_affine = np.asarray(reference_beta_img.affine, dtype=np.float64)

    roi_ids, roi_masks = _build_roi_masks(
        roi_spec=roi_spec,
        roi_spec_path=resolved_spec_path,
        data_root=Path(data_root),
        reference_shape=reference_shape,
        reference_affine=reference_affine,
    )

    feature_rows: list[list[float]] = []
    for row_index, (_, row) in enumerate(selected_index_df.iterrows(), start=1):
        beta_path = _resolve_beta_path(row, data_root=Path(data_root))
        if not beta_path.exists():
            raise FileNotFoundError(f"Selected beta file does not exist: {beta_path}")
        beta_img = nib.load(str(beta_path))
        beta_data = np.asarray(beta_img.get_fdata(dtype=np.float32))
        beta_affine = np.asarray(beta_img.affine, dtype=np.float64)
        if tuple(beta_data.shape) != reference_shape:
            raise ValueError(
                f"Beta shape mismatch for sample '{row.get('sample_id', row_index)}': "
                f"expected {reference_shape}, got {tuple(beta_data.shape)}."
            )
        if not np.allclose(beta_affine, reference_affine, rtol=0.0, atol=_AFFINE_ATOL):
            raise ValueError(
                f"Beta affine mismatch for sample '{row.get('sample_id', row_index)}'."
            )

        row_features: list[float] = []
        for roi_id, roi_mask in zip(roi_ids, roi_masks, strict=True):
            roi_values = np.asarray(beta_data[roi_mask], dtype=np.float64)
            finite_values = roi_values[np.isfinite(roi_values)]
            if finite_values.size == 0:
                raise ValueError(
                    f"ROI '{roi_id}' for sample '{row.get('sample_id', row_index)}' has no "
                    "finite voxels after extraction."
                )
            row_features.append(float(np.mean(finite_values)))
        feature_rows.append(row_features)

    x_matrix = np.asarray(feature_rows, dtype=np.float32)
    metadata_df = selected_index_df.reset_index(drop=True).copy()
    spatial_compatibility = {
        "status": "passed",
        "passed": True,
        "n_groups_checked": 1,
        "reference_group_id": "roi_feature_space",
        "affine_atol": _AFFINE_ATOL,
        "feature_space": "roi_mean_predefined",
        "roi_spec_path": str(resolved_spec_path),
        "feature_space_id": roi_spec["feature_space_id"],
        "representation": roi_spec["representation"],
        "reference_space": roi_spec["reference_space"],
        "roi_ids": list(roi_ids),
        "roi_count": int(len(roi_ids)),
        "n_features": int(x_matrix.shape[1]),
    }

    if spatial_report_path is not None:
        Path(spatial_report_path).write_text(
            f"{json.dumps(spatial_compatibility, indent=2)}\n",
            encoding="utf-8",
        )

    return x_matrix, metadata_df, spatial_compatibility

