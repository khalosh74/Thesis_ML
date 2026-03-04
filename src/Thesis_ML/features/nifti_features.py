"""Feature extraction from beta maps using session masks."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

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


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a NIfTI mask and return a boolean voxel mask."""
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

    mask_img = nib.load(str(mask_path))
    mask_data = np.asarray(mask_img.get_fdata(dtype=np.float32))
    return np.isfinite(mask_data) & (mask_data > 0)


def extract_masked_vector(beta_path: Path, mask_bool: np.ndarray) -> np.ndarray:
    """Extract a float32 voxel vector from a beta map using a pre-loaded mask."""
    beta_path = Path(beta_path)
    if not beta_path.exists():
        raise FileNotFoundError(f"Beta file does not exist: {beta_path}")

    beta_img = nib.load(str(beta_path))
    beta_data = np.asarray(beta_img.get_fdata(dtype=np.float32))
    if beta_data.shape != mask_bool.shape:
        raise ValueError(
            "Shape mismatch between beta "
            f"{beta_data.shape} and mask {mask_bool.shape} for {beta_path}"
        )

    vector = beta_data[mask_bool]
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _cache_path_for_group(cache_dir: Path, group_rows: pd.DataFrame) -> Path:
    first = group_rows.iloc[0]
    subject = _sanitize_token(str(first["subject"]))
    session = _sanitize_token(str(first["session"]))
    bas = _sanitize_token(str(first["bas"]))
    return cache_dir / subject / session / f"{bas}.npz"


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

        if target_path.exists() and not force:
            manifest_rows.append(
                {
                    "group_id": str(group_id),
                    "cache_path": str(target_path.resolve()),
                    "n_samples": int(len(group_rows)),
                    "n_voxels": pd.NA,
                    "skipped_existing": True,
                }
            )
            continue

        mask_path = _resolve_data_path(str(group_rows.iloc[0]["mask_path"]), data_root=data_root)
        mask_bool = load_mask(mask_path)

        vectors: list[np.ndarray] = []
        for _, row in group_rows.iterrows():
            beta_path = _resolve_data_path(str(row["beta_path"]), data_root=data_root)
            vectors.append(extract_masked_vector(beta_path=beta_path, mask_bool=mask_bool))

        x_matrix = np.vstack(vectors).astype(np.float32, copy=False)
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
        )

        manifest_rows.append(
            {
                "group_id": str(group_id),
                "cache_path": str(target_path.resolve()),
                "n_samples": int(x_matrix.shape[0]),
                "n_voxels": int(x_matrix.shape[1]),
                "skipped_existing": False,
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
