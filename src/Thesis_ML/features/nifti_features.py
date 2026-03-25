"""Feature extraction from beta maps using session masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.spatialimages import SpatialImage

from Thesis_ML.data.affect_labels import with_coarse_affect

LOGGER = logging.getLogger(__name__)
_SPATIAL_SIGNATURE_VERSION = 1
_BETA_MASK_AFFINE_ATOL = 1e-5

_REQUIRED_INDEX_COLUMNS = {
    "sample_id",
    "subject",
    "session",
    "bas",
    "beta_path",
    "mask_path",
    "regressor_label",
    "emotion",
}


def _sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _resolve_data_path(path_value: str, data_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return data_root / path


def _mask_sha256(mask_bool: np.ndarray) -> str:
    mask_bytes = np.ascontiguousarray(mask_bool.astype(np.uint8, copy=False)).tobytes()
    return hashlib.sha256(mask_bytes).hexdigest()


def _build_spatial_signature(
    mask_img: nib.spatialimages.SpatialImage, mask_bool: np.ndarray
) -> dict[str, Any]:
    affine = np.asarray(mask_img.affine, dtype=np.float64)
    voxel_size = tuple(float(value) for value in mask_img.header.get_zooms()[: mask_bool.ndim])
    voxel_count = int(mask_bool.sum())
    return {
        "signature_version": _SPATIAL_SIGNATURE_VERSION,
        "image_shape": [int(value) for value in mask_bool.shape],
        "affine": affine.tolist(),
        "voxel_size": list(voxel_size),
        "mask_voxel_count": voxel_count,
        "feature_count": voxel_count,
        "mask_sha256": _mask_sha256(mask_bool),
    }


def _load_mask_and_signature(mask_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

    mask_img = cast(SpatialImage, nib.load(str(mask_path)))
    mask_data = np.asarray(mask_img.get_fdata(dtype=np.float32))
    mask_bool = np.isfinite(mask_data) & (mask_data > 0)
    return mask_bool, _build_spatial_signature(mask_img=mask_img, mask_bool=mask_bool)


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a NIfTI mask and return a boolean voxel mask."""
    mask_bool, _ = _load_mask_and_signature(mask_path=mask_path)
    return mask_bool


def _validate_beta_mask_compatibility(
    *,
    beta_path: Path,
    beta_shape: tuple[int, ...],
    beta_affine: np.ndarray,
    mask_path: Path,
    mask_shape: tuple[int, ...],
    mask_affine: np.ndarray,
    affine_atol: float = _BETA_MASK_AFFINE_ATOL,
) -> None:
    mismatch_reasons: list[str] = []
    if beta_shape != mask_shape:
        mismatch_reasons.append(f"shape mismatch (beta {beta_shape} != mask {mask_shape})")

    if not np.allclose(beta_affine, mask_affine, rtol=0.0, atol=affine_atol):
        mismatch_reasons.append("affine mismatch")

    if mismatch_reasons:
        reasons_text = "; ".join(mismatch_reasons)
        raise ValueError(
            "Beta/mask spatial compatibility validation failed "
            f"(beta='{beta_path}', mask='{mask_path}'): {reasons_text}"
        )


def extract_masked_vector(
    beta_path: Path,
    mask_bool: np.ndarray,
    *,
    mask_path: Path,
    mask_affine: np.ndarray,
) -> np.ndarray:
    """Extract a float32 voxel vector from a beta map using a pre-loaded mask."""
    beta_path = Path(beta_path)
    if not beta_path.exists():
        raise FileNotFoundError(f"Beta file does not exist: {beta_path}")

    beta_img = cast(SpatialImage, nib.load(str(beta_path)))
    beta_data = np.asarray(beta_img.get_fdata(dtype=np.float32))
    beta_affine = np.asarray(beta_img.affine, dtype=np.float64)
    _validate_beta_mask_compatibility(
        beta_path=beta_path,
        beta_shape=tuple(int(value) for value in beta_data.shape),
        beta_affine=beta_affine,
        mask_path=Path(mask_path),
        mask_shape=tuple(int(value) for value in mask_bool.shape),
        mask_affine=np.asarray(mask_affine, dtype=np.float64),
    )

    vector = beta_data[mask_bool]
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _cache_path_for_group(cache_dir: Path, group_rows: pd.DataFrame) -> Path:
    first = group_rows.iloc[0]
    subject = _sanitize_token(str(first["subject"]))
    session = _sanitize_token(str(first["session"]))
    bas = _sanitize_token(str(first["bas"]))
    return cache_dir / subject / session / f"{bas}.npz"


def _signature_manifest_fields(signature: dict[str, Any] | None) -> dict[str, object]:
    if not signature:
        return {
            "spatial_signature_version": pd.NA,
            "image_shape_json": pd.NA,
            "affine_json": pd.NA,
            "voxel_size_json": pd.NA,
            "mask_voxel_count": pd.NA,
            "feature_count": pd.NA,
            "mask_sha256": pd.NA,
        }

    image_shape = signature.get("image_shape")
    affine = signature.get("affine")
    voxel_size = signature.get("voxel_size")
    signature_version = int(signature.get("signature_version", _SPATIAL_SIGNATURE_VERSION))
    return {
        "spatial_signature_version": signature_version,
        "image_shape_json": json.dumps(image_shape) if image_shape is not None else pd.NA,
        "affine_json": json.dumps(affine) if affine is not None else pd.NA,
        "voxel_size_json": json.dumps(voxel_size) if voxel_size is not None else pd.NA,
        "mask_voxel_count": (
            int(signature["mask_voxel_count"]) if "mask_voxel_count" in signature else pd.NA
        ),
        "feature_count": int(signature["feature_count"]) if "feature_count" in signature else pd.NA,
        "mask_sha256": str(signature["mask_sha256"]) if "mask_sha256" in signature else pd.NA,
    }


def _read_existing_cache_signature(cache_path: Path) -> dict[str, Any] | None:
    try:
        with np.load(cache_path, allow_pickle=False) as npz:
            signature: dict[str, Any] | None = None
            if "spatial_signature_json" in npz.files:
                raw = str(npz["spatial_signature_json"].item())
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    signature = parsed
            if signature is None:
                return None
            if "X" in npz.files:
                x_matrix = np.asarray(npz["X"])
                if x_matrix.ndim == 2:
                    signature = dict(signature)
                    signature["feature_count"] = int(x_matrix.shape[1])
            return signature
    except Exception as exc:
        LOGGER.warning("Failed to read existing cache signature from %s: %s", cache_path, exc)
        return None

def _resolve_single_group_mask_path(
    *,
    group_rows: pd.DataFrame,
    data_root: Path,
    group_id: Any,
) -> Path:
    if group_rows.empty:
        raise ValueError(f"Feature cache group '{group_id}' is empty.")

    resolved_mask_paths: list[str] = []
    for raw_mask_path in group_rows["mask_path"].tolist():
        if pd.isna(raw_mask_path):
            raise ValueError(
                f"Feature cache group '{group_id}' contains a null mask_path."
            )

        mask_text = str(raw_mask_path).strip()
        if not mask_text:
            raise ValueError(
                f"Feature cache group '{group_id}' contains a blank mask_path."
            )

        resolved_path = _resolve_data_path(mask_text, data_root=data_root).resolve()
        resolved_mask_paths.append(str(resolved_path))

    unique_mask_paths = sorted(set(resolved_mask_paths))
    if len(unique_mask_paths) != 1:
        raise ValueError(
            "Feature cache group "
            f"'{group_id}' contains multiple resolved mask paths; "
            "expected exactly one mask per cache group. "
            f"Found {len(unique_mask_paths)}: {unique_mask_paths}"
        )

    return Path(unique_mask_paths[0])

def build_feature_cache(
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    group_key: str = "subject_session_bas",
    force: bool = False,
) -> Path:
    """
    Build cache files of masked beta vectors grouped by BAS/session.

    Returns cache manifest path.
    """
    index_csv = Path(index_csv)
    data_root = Path(data_root)
    cache_dir = Path(cache_dir)

    if not index_csv.exists():
        raise FileNotFoundError(f"index_csv does not exist: {index_csv}")
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    index_df = pd.read_csv(index_csv)
    missing = _REQUIRED_INDEX_COLUMNS - set(index_df.columns)
    if missing:
        raise ValueError(f"index_csv missing required columns: {sorted(missing)}")
    index_df = with_coarse_affect(index_df, emotion_column="emotion", coarse_column="coarse_affect")

    if group_key not in index_df.columns:
        if group_key == "subject_session_bas":
            index_df[group_key] = (
                index_df["subject"].astype(str)
                + "_"
                + index_df["session"].astype(str)
                + "_"
                + index_df["bas"].astype(str)
            )
        else:
            raise ValueError(f"group_key '{group_key}' not found in index_csv")

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for group_id, group_rows in index_df.groupby(group_key, sort=True):
        group_rows = group_rows.reset_index(drop=True)
        target_path = _cache_path_for_group(cache_dir, group_rows)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        mask_path = _resolve_single_group_mask_path(
            group_rows=group_rows,
            data_root=data_root,
            group_id=group_id,
        )

        if target_path.exists() and not force:
            existing_signature = _read_existing_cache_signature(target_path)
            manifest_rows.append(
                {
                    "group_id": str(group_id),
                    "cache_path": str(target_path.resolve()),
                    "n_samples": int(len(group_rows)),
                    "n_voxels": (
                        int(existing_signature["feature_count"])
                        if existing_signature and "feature_count" in existing_signature
                        else pd.NA
                    ),
                    "skipped_existing": True,
                    **_signature_manifest_fields(existing_signature),
                }
            )
            continue

        mask_bool, spatial_signature = _load_mask_and_signature(mask_path)
        mask_affine = np.asarray(spatial_signature["affine"], dtype=np.float64)

        vectors: list[np.ndarray] = []
        for _, row in group_rows.iterrows():
            beta_path = _resolve_data_path(str(row["beta_path"]), data_root=data_root)
            vectors.append(
                extract_masked_vector(
                    beta_path=beta_path,
                    mask_bool=mask_bool,
                    mask_path=mask_path,
                    mask_affine=mask_affine,
                )
            )

        x_matrix = np.vstack(vectors).astype(np.float32, copy=False)
        if int(spatial_signature["mask_voxel_count"]) != int(x_matrix.shape[1]):
            raise ValueError(
                "Mask voxel count does not match extracted feature count for "
                "group "
                f"'{group_id}': {spatial_signature['mask_voxel_count']} != {x_matrix.shape[1]}"
            )
        spatial_signature["feature_count"] = int(x_matrix.shape[1])
        y = (
            group_rows["emotion"]
            .fillna(group_rows["regressor_label"])
            .astype(str)
            .to_numpy(dtype=np.str_)
        )
        metadata_json = json.dumps(group_rows.to_dict(orient="records"))

        np.savez_compressed(
            target_path,
            X=x_matrix,
            y=y,
            metadata_json=np.array(metadata_json),
            group_id=np.array(str(group_id)),
            spatial_signature_json=np.array(json.dumps(spatial_signature, sort_keys=True)),
        )

        manifest_rows.append(
            {
                "group_id": str(group_id),
                "cache_path": str(target_path.resolve()),
                "n_samples": int(x_matrix.shape[0]),
                "n_voxels": int(x_matrix.shape[1]),
                "skipped_existing": False,
                **_signature_manifest_fields(spatial_signature),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = cache_dir / "cache_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    LOGGER.info("Feature cache manifest written: %s", manifest_path)
    return manifest_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and cache masked NIfTI feature matrices.")
    parser.add_argument("--index-csv", required=True, help="Dataset index CSV.")
    parser.add_argument("--data-root", required=True, help="Root for relative beta/mask paths.")
    parser.add_argument("--cache-dir", required=True, help="Cache output directory.")
    parser.add_argument(
        "--group-key",
        default="subject_session_bas",
        help="Grouping key for cache chunks (default: subject_session_bas).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing cache files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    manifest_path = build_feature_cache(
        index_csv=Path(args.index_csv),
        data_root=Path(args.data_root),
        cache_dir=Path(args.cache_dir),
        group_key=args.group_key,
        force=args.force,
    )

    manifest = pd.read_csv(manifest_path)
    summary = {
        "manifest_path": str(manifest_path.resolve()),
        "n_cache_files": int(len(manifest)),
        "n_built": int((~manifest["skipped_existing"]).sum()) if not manifest.empty else 0,
        "n_skipped_existing": int(manifest["skipped_existing"].sum()) if not manifest.empty else 0,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
