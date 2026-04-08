from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from Thesis_ML.data.index_dataset import build_dataset_index
from Thesis_ML.script_support.io import file_sha256

DEFAULT_OUTPUT = Path("demo_data") / "synthetic_v1"
DATA_ROOT_DIRNAME = "data_root"
INDEX_FILENAME = "dataset_index.csv"
RELEASE_INDEX_FILENAME = "release_dataset_index.csv"
RELEASE_DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
MANIFEST_FILENAME = "demo_dataset_manifest.json"
README_FILENAME = "README.md"
MANIFEST_SCHEMA_VERSION = "demo-dataset-manifest-v1"

SUBJECTS = ("sub-001", "sub-002")
SESSIONS = ("ses-01", "ses-02", "ses-03")
BAS_NAME = "BAS2"
LABELS = (
    "run-1_passive_anger_audio",
    "run-1_passive_happiness_audio",
    "run-1_passive_anger_video",
    "run-1_passive_happiness_video",
)


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4, dtype=np.float64))
    nib.save(image, str(path))


def _build_mask(shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    mask[1:, 1:, 1:] = 1.0
    return mask


def _build_beta_volume(
    *,
    shape: tuple[int, int, int],
    beta_index: int,
    subject_idx: int,
    session_idx: int,
    label: str,
) -> np.ndarray:
    grid = np.indices(shape).sum(axis=0).astype(np.float32)
    volume = np.full(shape, fill_value=float(beta_index), dtype=np.float32)
    volume += float(subject_idx + 1) * 0.10
    volume += float(session_idx + 1) * 0.01
    volume += grid * 0.001
    if "_anger_" in label:
        volume[1:, 1:, 1:] += 2.0
    if "_happiness_" in label:
        volume[1:, 1:, 1:] -= 2.0
    if "_video" in label:
        volume[1:, 1:, 1:] += 0.25
    return volume


def _write_glm_session(
    *,
    glm_dir: Path,
    subject_idx: int,
    session_idx: int,
) -> None:
    glm_dir.mkdir(parents=True, exist_ok=True)
    shape = (3, 3, 3)
    _write_nifti(glm_dir / "mask.nii", _build_mask(shape))

    pd.Series(list(LABELS)).to_csv(
        glm_dir / "regressor_labels.csv",
        index=False,
        header=False,
    )

    for idx, label in enumerate(LABELS, start=1):
        beta_volume = _build_beta_volume(
            shape=shape,
            beta_index=idx,
            subject_idx=subject_idx,
            session_idx=session_idx,
            label=label,
        )
        _write_nifti(glm_dir / f"beta_{idx:04d}.nii", beta_volume)


def _write_readme(path: Path) -> None:
    text = """# Synthetic Demo Dataset (v1)

This directory contains a tiny deterministic synthetic dataset used for framework reproducibility checks.

- Intended use: validate official comparison/confirmatory workflow execution, artifact generation, and deterministic replay.
- Not intended use: scientific claims, model quality conclusions, or any clinical/cognitive interpretation.

Generation:
- Source script: `scripts/generate_demo_dataset.py`
- Structure: `data_root/sub-*/ses-*/BAS2/` with synthetic `beta_*.nii`, `mask.nii`, and `regressor_labels.csv`
- Index: `dataset_index.csv` built via `Thesis_ML.data.index_dataset.build_dataset_index`
- Manifest: `demo_dataset_manifest.json` with deterministic file hashes
"""
    path.write_text(text, encoding="utf-8")


def _manifest_payload(*, output_dir: Path, index_csv: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for candidate in sorted(output_dir.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.name in {MANIFEST_FILENAME, RELEASE_DATASET_MANIFEST_FILENAME}:
            continue
        relative = candidate.relative_to(output_dir).as_posix()
        files.append(
            {
                "path": relative,
                "sha256": file_sha256(candidate),
                "size_bytes": int(candidate.stat().st_size),
            }
        )

    index_df = pd.read_csv(index_csv)
    required_columns = [
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
        "regressor_label",
        "bas",
    ]
    missing_columns = sorted(set(required_columns) - set(index_df.columns))
    if missing_columns:
        raise ValueError(
            "Generated demo dataset index is missing required columns: "
            + ", ".join(missing_columns)
        )

    per_subject_sessions = {
        str(subject): int(group["session"].nunique())
        for subject, group in index_df.groupby("subject", sort=True)
    }

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": "thesis_ml_synthetic_v1",
        "dataset_version": "synthetic_v1",
        "generator_script": "scripts/generate_demo_dataset.py",
        "data_root": DATA_ROOT_DIRNAME,
        "index_csv": INDEX_FILENAME,
        "subjects": list(SUBJECTS),
        "sessions": list(SESSIONS),
        "labels": list(LABELS),
        "n_rows": int(index_df.shape[0]),
        "n_subjects": int(index_df["subject"].nunique()),
        "n_sessions": int(index_df["subject_session"].nunique()),
        "sessions_per_subject": per_subject_sessions,
        "required_index_columns": required_columns,
        "index_csv_sha256": file_sha256(index_csv),
        "files": files,
    }


def _release_index_payload(index_df: pd.DataFrame) -> pd.DataFrame:
    release_df = index_df.copy()
    release_df["task"] = release_df["modality"].astype(str).map(
        lambda value: "emo" if value == "audio" else "recog"
    )
    release_df["modality"] = "audiovisual"
    return release_df.sort_values(
        by=["subject", "session", "run", "beta_index"],
        kind="mergesort",
    ).reset_index(drop=True)


def _release_dataset_manifest_payload(*, output_dir: Path, release_index_csv: Path) -> dict[str, Any]:
    release_index = pd.read_csv(release_index_csv)
    required_columns = [
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
    missing_columns = sorted(set(required_columns) - set(release_index.columns))
    if missing_columns:
        raise ValueError(
            "Release demo index is missing required columns: " + ", ".join(missing_columns)
        )

    sessions_by_subject = {
        str(subject): int(group["session"].astype(str).nunique())
        for subject, group in release_index.groupby("subject", sort=True)
    }

    return {
        "schema_version": "dataset-instance-v1",
        "dataset_id": "thesis_ml_synthetic_v1",
        "dataset_contract_version": "fmri_beta_dataset_v1",
        "dataset_fingerprint": file_sha256(release_index_csv),
        "index_csv": RELEASE_INDEX_FILENAME,
        "data_root": DATA_ROOT_DIRNAME,
        "cache_dir": "cache",
        "created_at": "2026-04-08T00:00:00Z",
        "source_extraction_version": "synthetic_v1",
        "sample_unit": "beta_event",
        "required_columns": required_columns,
        "subject_count": int(release_index["subject"].astype(str).nunique()),
        "session_counts_by_subject": sessions_by_subject,
    }


def generate_demo_dataset(*, output_dir: Path, force: bool) -> dict[str, Any]:
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --force to overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = output_dir / DATA_ROOT_DIRNAME
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / ".gitkeep").write_text("", encoding="utf-8")
    for subject_idx, subject in enumerate(SUBJECTS):
        for session_idx, session in enumerate(SESSIONS):
            glm_dir = data_root / subject / session / BAS_NAME
            _write_glm_session(
                glm_dir=glm_dir,
                subject_idx=subject_idx,
                session_idx=session_idx,
            )

    index_csv = output_dir / INDEX_FILENAME
    build_dataset_index(
        data_root=data_root,
        out_csv=index_csv,
        use_cache=False,
    )

    index_df = pd.read_csv(index_csv)
    index_df = index_df.sort_values(
        by=["subject", "session", "run", "beta_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    index_df.to_csv(index_csv, index=False)
    release_index_df = _release_index_payload(index_df)
    release_index_csv = output_dir / RELEASE_INDEX_FILENAME
    release_index_df.to_csv(release_index_csv, index=False)

    _write_readme(output_dir / README_FILENAME)
    release_manifest = _release_dataset_manifest_payload(
        output_dir=output_dir,
        release_index_csv=release_index_csv,
    )
    release_manifest_path = output_dir / RELEASE_DATASET_MANIFEST_FILENAME
    release_manifest_path.write_text(f"{json.dumps(release_manifest, indent=2)}\n", encoding="utf-8")
    manifest = _manifest_payload(output_dir=output_dir, index_csv=index_csv)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir.resolve()),
        "data_root": str(data_root.resolve()),
        "index_csv": str(index_csv.resolve()),
        "release_index_csv": str(release_index_csv.resolve()),
        "release_dataset_manifest_path": str(release_manifest_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "n_rows": int(manifest["n_rows"]),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic demo dataset for official reproducibility checks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output dataset directory (default: {DEFAULT_OUTPUT.as_posix()}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = generate_demo_dataset(output_dir=args.output, force=bool(args.force))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
