"""Dataset indexing across multiple BAS2 GLM sessions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.data.affect_labels import (
    BINARY_VALENCE_MAPPING_SHA256,
    BINARY_VALENCE_MAPPING_VERSION,
    COARSE_AFFECT_MAPPING_SHA256,
    COARSE_AFFECT_MAPPING_VERSION,
    derive_coarse_affect,
)
from Thesis_ML.data.index_validation import (
    DatasetIndexValidationError,
    file_sha256,
    validate_dataset_index_strict,
)
from Thesis_ML.spm.extract_glm import extract_glm_session

LOGGER = logging.getLogger(__name__)

_EXPECTED_MAPPING_COLUMNS = {
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
    "beta_exists",
}

_REQUIRED_SUMMARY_FIELDS = {
    "n_unknown_regressors",
    "unknown_regressor_labels",
    "unknown_regressor_beta_indexes",
    "has_unknown_regressors",
}


def _find_entity(parts: tuple[str, ...], prefix: str) -> str:
    for token in parts:
        if token.startswith(prefix):
            return token
    raise ValueError(f"Could not infer {prefix} from path parts: {parts}")


def _portable_relative(path: Path, base: Path) -> str:
    resolved_path = Path(path).resolve()
    resolved_base = Path(base).resolve(strict=True)
    try:
        return resolved_path.relative_to(resolved_base).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Path '{resolved_path}' is outside data_root '{resolved_base}'."
        ) from exc


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


def _to_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"Invalid boolean value for {field_name}: {value!r}")


def _ensure_non_blank(value: Any, *, field_name: str, mapping_path: Path) -> str:
    if pd.isna(value):
        raise ValueError(f"{mapping_path} has null {field_name} in emotion-condition row.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{mapping_path} has blank {field_name} in emotion-condition row.")
    return text


def _validate_mapping_structure(
    *, mapping: pd.DataFrame, mapping_path: Path, glm_dir: Path
) -> pd.DataFrame:
    missing_columns = sorted(_EXPECTED_MAPPING_COLUMNS - set(mapping.columns))
    if missing_columns:
        raise ValueError(f"Mapping missing required columns {missing_columns}: {mapping_path}")

    rows = mapping.copy()
    rows["regressor_type"] = rows["regressor_type"].astype(str)

    beta_exists_values: list[bool] = []
    for value in rows["beta_exists"].tolist():
        beta_exists_values.append(_to_bool(value, field_name="beta_exists"))
    rows["beta_exists"] = beta_exists_values

    emotion_rows = rows[
        (rows["regressor_type"] == "emotion_condition") & (rows["beta_exists"] == True)  # noqa: E712
    ].copy()

    if emotion_rows.empty:
        return emotion_rows

    if emotion_rows["beta_index"].isna().any():
        raise ValueError(f"{mapping_path} has null beta_index for emotion-condition rows.")

    duplicate_beta_index = emotion_rows["beta_index"].astype(int).duplicated(keep=False)
    if bool(duplicate_beta_index.any()):
        duplicates = sorted(
            emotion_rows.loc[duplicate_beta_index, "beta_index"].astype(int).unique().tolist()
        )
        raise ValueError(
            f"{mapping_path} has duplicate beta_index values in emotion-condition rows: {duplicates}"
        )

    for row_idx, row in emotion_rows.iterrows():
        run_value = row.get("run")
        if pd.isna(run_value):
            raise ValueError(f"{mapping_path} has null run in emotion-condition row index={row_idx}.")
        try:
            run_int = int(run_value)
        except Exception as exc:
            raise ValueError(
                f"{mapping_path} has invalid run value in emotion-condition row index={row_idx}: {run_value!r}"
            ) from exc
        if run_int <= 0:
            raise ValueError(
                f"{mapping_path} has non-positive run in emotion-condition row index={row_idx}: {run_int}"
            )

        beta_index_int = int(row["beta_index"])
        if beta_index_int <= 0:
            raise ValueError(
                f"{mapping_path} has non-positive beta_index in emotion-condition row index={row_idx}: {beta_index_int}"
            )

        _ensure_non_blank(row.get("task"), field_name="task", mapping_path=mapping_path)
        _ensure_non_blank(row.get("emotion"), field_name="emotion", mapping_path=mapping_path)
        _ensure_non_blank(row.get("modality"), field_name="modality", mapping_path=mapping_path)

        beta_file = _ensure_non_blank(
            row.get("beta_file"), field_name="beta_file", mapping_path=mapping_path
        )
        expected_beta_file = f"beta_{beta_index_int:04d}.nii"
        if beta_file != expected_beta_file:
            raise ValueError(
                f"{mapping_path} has malformed beta_file for beta_index={beta_index_int}: "
                f"expected '{expected_beta_file}', got '{beta_file}'."
            )

        beta_path = glm_dir / beta_file
        if not beta_path.exists() or not beta_path.is_file():
            raise ValueError(
                f"{mapping_path} references missing beta file for emotion-condition row: {beta_path}"
            )

    return emotion_rows


def _load_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists() or not summary_path.is_file():
        raise ValueError(f"Missing extraction summary: {summary_path}")
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in extraction summary: {summary_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Extraction summary must be a JSON object: {summary_path}")

    missing = sorted(_REQUIRED_SUMMARY_FIELDS - set(payload.keys()))
    if missing:
        raise ValueError(
            f"Extraction summary missing required unknown-regressor fields {missing}: {summary_path}"
        )
    return payload


def _normalize_unknown_summary(summary: dict[str, Any], *, summary_path: Path) -> dict[str, Any]:
    try:
        unknown_count = int(summary.get("n_unknown_regressors", 0))
    except Exception as exc:
        raise ValueError(
            f"Invalid n_unknown_regressors in summary '{summary_path}'."
        ) from exc
    if unknown_count < 0:
        raise ValueError(f"n_unknown_regressors must be >= 0 in summary '{summary_path}'.")

    has_unknown = _to_bool(
        summary.get("has_unknown_regressors", False),
        field_name="has_unknown_regressors",
    )
    raw_labels = summary.get("unknown_regressor_labels", [])
    if not isinstance(raw_labels, list):
        raise ValueError(
            f"unknown_regressor_labels must be a JSON list in summary '{summary_path}'."
        )
    labels = [str(value) for value in raw_labels]

    raw_indexes = summary.get("unknown_regressor_beta_indexes", [])
    if not isinstance(raw_indexes, list):
        raise ValueError(
            f"unknown_regressor_beta_indexes must be a JSON list in summary '{summary_path}'."
        )
    indexes: list[int] = []
    for value in raw_indexes:
        indexes.append(int(value))

    if has_unknown and unknown_count <= 0:
        raise ValueError(
            f"has_unknown_regressors=true but n_unknown_regressors<=0 in summary '{summary_path}'."
        )
    if (not has_unknown) and unknown_count != 0:
        raise ValueError(
            f"has_unknown_regressors=false but n_unknown_regressors!=0 in summary '{summary_path}'."
        )
    if unknown_count != len(labels) or unknown_count != len(indexes):
        raise ValueError(
            "Unknown-regressor fields are inconsistent in extraction summary "
            f"'{summary_path}'."
        )

    return {
        "glm_has_unknown_regressors": bool(has_unknown),
        "glm_unknown_regressor_count": int(unknown_count),
        "glm_unknown_regressor_labels_json": json.dumps(labels, sort_keys=False),
    }


def _extract_and_load_session_artifacts(
    *, glm_dir: Path, extraction_dir: Path, absolute_paths: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    extract_glm_session(
        glm_dir=glm_dir,
        out_dir=extraction_dir,
        absolute_paths=absolute_paths,
        fail_on_unknown_regressors=False,
    )
    mapping_path = extraction_dir / "regressor_beta_mapping.csv"
    summary_path = extraction_dir / "session_summary.json"
    mapping = pd.read_csv(mapping_path)
    summary = _load_summary(summary_path)
    return mapping, summary


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

        should_extract = not (use_cache and mapping_path.exists() and summary_path.exists())
        if should_extract:
            LOGGER.info("Extracting GLM session: %s", glm_dir)
            mapping, summary = _extract_and_load_session_artifacts(
                glm_dir=glm_dir,
                extraction_dir=extraction_dir,
                absolute_paths=False,
            )
        else:
            LOGGER.info("Using cached extraction: %s", extraction_dir)
            try:
                mapping = pd.read_csv(mapping_path)
                summary = _load_summary(summary_path)
                _validate_mapping_structure(mapping=mapping, mapping_path=mapping_path, glm_dir=glm_dir)
                _normalize_unknown_summary(summary, summary_path=summary_path)
            except Exception:
                LOGGER.info(
                    "Cached extraction at '%s' failed strict checks; re-extracting session.",
                    extraction_dir,
                )
                mapping, summary = _extract_and_load_session_artifacts(
                    glm_dir=glm_dir,
                    extraction_dir=extraction_dir,
                    absolute_paths=False,
                )

        sessions_processed += 1

        emotion_rows = _validate_mapping_structure(
            mapping=mapping,
            mapping_path=mapping_path,
            glm_dir=glm_dir,
        )
        if emotion_rows.empty:
            continue

        unknown_summary = _normalize_unknown_summary(summary, summary_path=summary_path)

        mask_path = (glm_dir / "mask.nii").resolve()
        if not mask_path.exists() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask file for session index build: {mask_path}")

        mask_rel = _portable_relative(mask_path, data_root)
        mask_sha = file_sha256(mask_path)

        for _, row in emotion_rows.iterrows():
            beta_file = str(row["beta_file"])
            beta_path = (glm_dir / beta_file).resolve()
            beta_rel = _portable_relative(beta_path, data_root)
            beta_sha = file_sha256(beta_path)
            sample_id = _build_sample_id(row=row, subject=subject, session=session, bas=bas)
            rows.append(
                {
                    "sample_id": sample_id,
                    "subject": subject,
                    "session": session,
                    "bas": bas,
                    "run": int(row["run"]),
                    "task": str(row["task"]).strip(),
                    "emotion": str(row["emotion"]).strip().lower(),
                    "coarse_affect": derive_coarse_affect(row["emotion"]),
                    "modality": str(row["modality"]).strip().lower(),
                    "beta_index": int(row["beta_index"]),
                    "beta_file": beta_file,
                    "beta_path": beta_rel,
                    "mask_path": mask_rel,
                    "regressor_label": str(row["label"]),
                    "raw_label": str(row["raw_label"]),
                    "subject_session": f"{subject}_{session}",
                    "subject_session_bas": f"{subject}_{session}_{bas}",
                    "beta_file_sha256": beta_sha,
                    "mask_file_sha256": mask_sha,
                    "coarse_affect_mapping_version": COARSE_AFFECT_MAPPING_VERSION,
                    "coarse_affect_mapping_sha256": COARSE_AFFECT_MAPPING_SHA256,
                    "binary_valence_mapping_version": BINARY_VALENCE_MAPPING_VERSION,
                    "binary_valence_mapping_sha256": BINARY_VALENCE_MAPPING_SHA256,
                    **unknown_summary,
                }
            )

    if not rows:
        raise ValueError(
            "No valid emotion-condition rows were found while indexing GLM sessions."
        )

    dataset = pd.DataFrame(rows)
    try:
        dataset = validate_dataset_index_strict(
            dataset,
            data_root=data_root,
            require_integrity_columns=True,
        )
    except DatasetIndexValidationError as exc:
        raise ValueError(f"Strict dataset index validation failed: {exc}") from exc

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
