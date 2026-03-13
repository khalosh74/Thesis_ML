"""Session-level SPM GLM extraction utilities and CLI."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.spatialimages import SpatialImage

LOGGER = logging.getLogger(__name__)

_RUN_PATTERN = re.compile(r"^run-(?P<run>\d+)_(?P<task>[^_]+)_(?P<condition>.+)$")
_EMOTION_PATTERN = re.compile(
    r"^(?P<emotion>[A-Za-z0-9-]+)_(?P<modality>audio|video|audiovisual)$",
    flags=re.IGNORECASE,
)
_MOTION_PATTERN = re.compile(r"^R[1-6]$", flags=re.IGNORECASE)


def parse_regressor_label(label: str) -> dict[str, Any]:
    """
    Parse a regressor label into structured fields.

    Expected examples:
    - run-1_passive_anger_audio
    - run-2_emo_resp_movement
    - run-3_recog_R1
    - run-1_passive_constant
    """
    raw_label = str(label).strip()
    parsed: dict[str, Any] = {
        "label": raw_label,
        "raw_label": raw_label,
        "run": pd.NA,
        "task": pd.NA,
        "emotion": pd.NA,
        "modality": pd.NA,
        "regressor_type": "unknown",
        "motion_param": pd.NA,
    }

    match = _RUN_PATTERN.match(raw_label)
    if not match:
        return parsed

    run = int(match.group("run"))
    task = match.group("task")
    condition = match.group("condition")

    parsed["run"] = run
    parsed["task"] = task

    if condition == "resp_movement":
        parsed["regressor_type"] = "resp_movement"
        return parsed

    if condition == "constant":
        parsed["regressor_type"] = "constant"
        return parsed

    if _MOTION_PATTERN.fullmatch(condition):
        parsed["regressor_type"] = "motion_param"
        parsed["motion_param"] = condition.upper()
        return parsed

    emotion_match = _EMOTION_PATTERN.fullmatch(condition)
    if emotion_match:
        parsed["regressor_type"] = "emotion_condition"
        parsed["emotion"] = emotion_match.group("emotion").lower()
        parsed["modality"] = emotion_match.group("modality").lower()
        return parsed

    return parsed


def _safe_get(mapping: Any, *keys: str, default: Any = None) -> Any:
    current = mapping
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _maybe_load_spm_metadata(spm_mat_path: Path) -> tuple[dict[str, Any], str | None]:
    if not spm_mat_path.exists():
        return {}, None

    try:
        from scipy.io import loadmat
    except Exception as exc:
        return {}, f"scipy not available: {exc}"

    try:
        payload = loadmat(spm_mat_path, simplify_cells=True)
        spm = payload.get("SPM")
        if spm is None:
            return {}, "SPM key missing in SPM.mat"
    except Exception as exc:
        return {}, f"failed to parse SPM.mat: {exc}"

    metadata: dict[str, Any] = {}
    tr = _safe_get(spm, "xY", "RT", default=None)
    nscan = _safe_get(spm, "nscan", default=None)
    design_matrix = _safe_get(spm, "xX", "X", default=None)
    xnames = _safe_get(spm, "xX", "name", default=None)
    basis = _safe_get(spm, "xBF", "name", default=None)
    serial = _safe_get(spm, "xVi", "form", default=None)

    hpf = None
    k_block = _safe_get(spm, "xX", "K", default=None)
    if isinstance(k_block, list) and k_block:
        first = k_block[0]
        if isinstance(first, dict):
            hpf = first.get("HParam")
    elif isinstance(k_block, dict):
        hpf = k_block.get("HParam")

    if tr is not None:
        metadata["TR"] = float(tr)
    if nscan is not None:
        nscan_values = np.atleast_1d(nscan).tolist()
        metadata["nscan"] = [int(value) for value in nscan_values]
    if design_matrix is not None:
        metadata["design_matrix_shape"] = list(np.asarray(design_matrix).shape)
    if basis is not None:
        metadata["basis"] = str(basis)
    if hpf is not None:
        metadata["high_pass_seconds"] = float(np.asarray(hpf).flatten()[0])
    if serial is not None:
        metadata["serial_correlation_model"] = str(serial)
    if xnames is not None:
        metadata["design_regressor_count"] = int(len(np.atleast_1d(xnames)))

    return metadata, None


def _load_regressor_labels(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing required file: {labels_path}")

    labels = pd.read_csv(labels_path, header=None, usecols=[0], names=["raw_label"])
    labels = labels.dropna()
    labels["raw_label"] = labels["raw_label"].astype(str).str.strip()
    labels = labels[labels["raw_label"] != ""]

    if labels.empty:
        raise ValueError(f"regressor_labels.csv is empty: {labels_path}")

    if labels.iloc[0]["raw_label"].lower() == "label":
        labels = labels.iloc[1:].reset_index(drop=True)

    if labels.empty:
        raise ValueError(f"regressor_labels.csv has no usable labels: {labels_path}")

    labels["beta_index"] = np.arange(1, len(labels) + 1, dtype=int)
    parsed = labels["raw_label"].apply(parse_regressor_label).apply(pd.Series)
    parsed["beta_index"] = labels["beta_index"]

    mapping = parsed[
        [
            "label",
            "raw_label",
            "beta_index",
            "run",
            "task",
            "emotion",
            "modality",
            "regressor_type",
            "motion_param",
        ]
    ].copy()
    return mapping


def _beta_path_for_row(glm_dir: Path, beta_file: str, absolute_paths: bool) -> str:
    path = glm_dir / beta_file
    if absolute_paths:
        return str(path.resolve())
    return beta_file


def extract_glm_session(
    glm_dir: Path,
    out_dir: Path,
    absolute_paths: bool = False,
) -> dict[str, Any]:
    """Extract a single BAS2 GLM session into tabular outputs."""
    glm_dir = Path(glm_dir)
    out_dir = Path(out_dir)

    if not glm_dir.exists():
        raise FileNotFoundError(f"GLM directory does not exist: {glm_dir}")
    if not glm_dir.is_dir():
        raise NotADirectoryError(f"GLM path is not a directory: {glm_dir}")

    labels_path = glm_dir / "regressor_labels.csv"
    mask_path = glm_dir / "mask.nii"
    spm_mat_path = glm_dir / "SPM.mat"

    if not mask_path.exists():
        raise FileNotFoundError(f"Missing required file: {mask_path}")

    mapping = _load_regressor_labels(labels_path)
    mapping["beta_file"] = mapping["beta_index"].map(lambda idx: f"beta_{idx:04d}.nii")
    mapping["beta_path"] = mapping["beta_file"].map(
        lambda beta_file: _beta_path_for_row(glm_dir, beta_file, absolute_paths=absolute_paths)
    )
    mapping["beta_exists"] = mapping["beta_file"].map(lambda name: (glm_dir / name).exists())

    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = out_dir / "regressor_beta_mapping.csv"
    mapping.to_csv(mapping_path, index=False)

    mask_img = cast(SpatialImage, nib.load(str(mask_path)))
    mask_data = np.asarray(mask_img.get_fdata(dtype=np.float32))
    mask_bool = np.isfinite(mask_data) & (mask_data > 0)

    summary: dict[str, Any] = {
        "glm_dir": str(glm_dir.resolve()),
        "bas_name": glm_dir.name,
        "mask_path": str(mask_path.resolve()) if absolute_paths else mask_path.name,
        "mask_shape": list(mask_bool.shape),
        "voxels_in_mask": int(mask_bool.sum()),
        "n_regressors_csv": int(len(mapping)),
        "n_beta_present": int(mapping["beta_exists"].sum()),
        "regressor_type_counts": {
            str(key): int(value)
            for key, value in mapping["regressor_type"].value_counts(dropna=False).items()
        },
    }

    spm_metadata, spm_error = _maybe_load_spm_metadata(spm_mat_path)
    summary.update(spm_metadata)
    if spm_error:
        summary["spm_parse_warning"] = spm_error

    summary_path = out_dir / "session_summary.json"
    summary_path.write_text(f"{json.dumps(summary, indent=2)}\n", encoding="utf-8")

    return {
        "glm_dir": str(glm_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "mapping_csv": str(mapping_path.resolve()),
        "summary_json": str(summary_path.resolve()),
        "n_regressors_csv": int(len(mapping)),
        "n_beta_present": int(mapping["beta_exists"].sum()),
        "regressor_type_counts": summary["regressor_type_counts"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract regressor-beta mapping from one GLM session."
    )
    parser.add_argument("--glm-dir", required=True, help="Path to BAS2 GLM directory.")
    parser.add_argument("--out-dir", required=True, help="Output directory for CSV/JSON artifacts.")
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths in output files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = extract_glm_session(
        glm_dir=Path(args.glm_dir),
        out_dir=Path(args.out_dir),
        absolute_paths=args.absolute_paths,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
