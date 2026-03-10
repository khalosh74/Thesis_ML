"""Dataset indexing across multiple BAS2 GLM sessions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.data.affect_labels import derive_coarse_affect
from Thesis_ML.spm.extract_glm import extract_glm_session

LOGGER = logging.getLogger(__name__)


def _find_entity(parts: tuple[str, ...], prefix: str) -> str:
    for token in parts:
        if token.startswith(prefix):
            return token
    raise ValueError(f"Could not infer {prefix} from path parts: {parts}")


def _portable_relative(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path.resolve())


def _default_cache_root(data_root: Path) -> Path:
    return data_root / "processed" / "extractions"


def _build_sample_id(row: pd.Series, subject: str, session: str, bas: str) -> str:
    run = row.get("run")
    beta_index = row.get("beta_index")
    if pd.isna(run):
        run = "na"
    if pd.isna(beta_index):
        beta_index = "na"
    return f"{subject}_{session}_{bas}_run-{run}_beta-{beta_index}"


def build_dataset_index(
    data_root: Path,
    out_csv: Path,
    pattern: str = "sub-*/ses-*/**/BAS2",
    use_cache: bool = True,
    cache_root: Path | None = None,
) -> Path:
    """
    Build an ML dataset index from many BAS2 GLM folders.

    Extracted mapping files are cached under:
    data_root/processed/extractions/<sub>/<ses>/<bas>/
    """
    data_root = Path(data_root)
    out_csv = Path(out_csv)
    cache_root = Path(cache_root) if cache_root is not None else _default_cache_root(data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    glm_dirs = sorted(path for path in data_root.glob(pattern) if path.is_dir())
    if not glm_dirs:
        raise FileNotFoundError(
            f"No GLM directories found under {data_root} with pattern '{pattern}'"
        )

    rows: list[dict[str, Any]] = []
    sessions_processed = 0

    for glm_dir in glm_dirs:
        relative = glm_dir.relative_to(data_root)
        subject = _find_entity(relative.parts, "sub-")
        session = _find_entity(relative.parts, "ses-")
        bas = glm_dir.name

        extraction_dir = cache_root / subject / session / bas
        mapping_path = extraction_dir / "regressor_beta_mapping.csv"
        summary_path = extraction_dir / "session_summary.json"

        if not (use_cache and mapping_path.exists() and summary_path.exists()):
            LOGGER.info("Extracting GLM session: %s", glm_dir)
            extract_glm_session(glm_dir=glm_dir, out_dir=extraction_dir, absolute_paths=False)
        else:
            LOGGER.info("Using cached extraction: %s", extraction_dir)

        mapping = pd.read_csv(mapping_path)
        sessions_processed += 1

        if "regressor_type" not in mapping.columns or "beta_exists" not in mapping.columns:
            raise ValueError(f"Mapping missing required columns: {mapping_path}")

        emotion_rows = mapping[
            (mapping["regressor_type"] == "emotion_condition") & (mapping["beta_exists"] == True)  # noqa: E712
        ].copy()
        if emotion_rows.empty:
            continue

        mask_path = glm_dir / "mask.nii"
        mask_rel = _portable_relative(mask_path, data_root)

        for _, row in emotion_rows.iterrows():
            beta_file = str(row["beta_file"])
            beta_path = glm_dir / beta_file
            beta_rel = _portable_relative(beta_path, data_root)
            sample_id = _build_sample_id(row=row, subject=subject, session=session, bas=bas)
            rows.append(
                {
                    "sample_id": sample_id,
                    "subject": subject,
                    "session": session,
                    "bas": bas,
                    "run": int(row["run"]) if not pd.isna(row["run"]) else pd.NA,
                    "task": row["task"],
                    "emotion": row["emotion"],
                    "coarse_affect": derive_coarse_affect(row["emotion"]),
                    "modality": row["modality"],
                    "beta_index": int(row["beta_index"]),
                    "beta_file": beta_file,
                    "beta_path": beta_rel,
                    "mask_path": mask_rel,
                    "regressor_label": row["label"],
                    "raw_label": row["raw_label"],
                    "subject_session": f"{subject}_{session}",
                    "subject_session_bas": f"{subject}_{session}_{bas}",
                }
            )

    dataset = pd.DataFrame(rows)
    required_columns = [
        "sample_id",
        "subject",
        "session",
        "bas",
        "run",
        "task",
        "emotion",
        "coarse_affect",
        "modality",
        "beta_path",
        "mask_path",
        "regressor_label",
    ]
    if dataset.empty:
        dataset = pd.DataFrame(columns=required_columns)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_csv, index=False)

    LOGGER.info(
        "Indexed %s emotion-condition samples from %s sessions.", len(dataset), sessions_processed
    )
    return out_csv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a dataset index across many BAS2 sessions.")
    parser.add_argument(
        "--data-root", required=True, help="Root containing sub-*/ses-*/**/BAS2 folders."
    )
    parser.add_argument("--out-csv", required=True, help="Output CSV path for dataset index.")
    parser.add_argument(
        "--pattern",
        default="sub-*/ses-*/**/BAS2",
        help="Glob pattern relative to data-root (default: sub-*/ses-*/**/BAS2).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable extraction cache reuse and force re-extraction.",
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Optional extraction cache root (default: data-root/processed/extractions).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    out_path = build_dataset_index(
        data_root=Path(args.data_root),
        out_csv=Path(args.out_csv),
        pattern=args.pattern,
        use_cache=not args.no_cache,
        cache_root=Path(args.cache_root) if args.cache_root else None,
    )
    dataset = pd.read_csv(out_path)
    summary = {
        "out_csv": str(out_path.resolve()),
        "n_rows": int(len(dataset)),
        "n_subjects": int(dataset["subject"].nunique()) if not dataset.empty else 0,
        "n_sessions": int(dataset["subject_session"].nunique()) if not dataset.empty else 0,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
