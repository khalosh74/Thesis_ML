from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

CANONICAL_BETA_PATH_COLUMN = "beta_path_canonical"
CANONICAL_MASK_PATH_COLUMN = "mask_path_canonical"

STRICT_INDEX_REQUIRED_COLUMNS = {
    "sample_id",
    "subject",
    "session",
    "bas",
    "subject_session",
    "subject_session_bas",
    "task",
    "modality",
    "emotion",
    "beta_path",
    "mask_path",
}

STRICT_INDEX_KEY_COLUMNS = (
    "sample_id",
    "subject",
    "session",
    "task",
    "modality",
    "emotion",
    "beta_path",
    "mask_path",
)

INDEX_INTEGRITY_COLUMNS = (
    "beta_file_sha256",
    "mask_file_sha256",
    "coarse_affect_mapping_version",
    "coarse_affect_mapping_sha256",
    "binary_valence_mapping_version",
    "binary_valence_mapping_sha256",
    "glm_has_unknown_regressors",
    "glm_unknown_regressor_count",
    "glm_unknown_regressor_labels_json",
)


class DatasetIndexValidationError(ValueError):
    """Raised when strict dataset-index validation fails."""


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_blank(value: Any) -> bool:
    if pd.isna(value):
        return True
    return not str(value).strip()


def _require_data_root(data_root: Path) -> Path:
    root = Path(data_root)
    if not root.exists() or not root.is_dir():
        raise DatasetIndexValidationError(
            f"data_root must exist as a directory for strict index validation: {root}"
        )
    return root.resolve(strict=True)


def _require_contained_path(*, resolved_path: Path, data_root_resolved: Path, field_name: str) -> None:
    try:
        resolved_path.relative_to(data_root_resolved)
    except ValueError as exc:
        raise DatasetIndexValidationError(
            f"{field_name} resolves outside data_root and is not allowed: {resolved_path}"
        ) from exc


def resolve_index_data_path_strict(
    path_value: Any,
    *,
    data_root: Path,
    field_name: str,
    require_exists: bool = True,
) -> Path:
    if _is_blank(path_value):
        raise DatasetIndexValidationError(f"{field_name} contains a blank/null path value.")

    data_root_resolved = _require_data_root(Path(data_root))
    raw_text = str(path_value).strip()
    raw_path = Path(raw_text)
    candidate = raw_path if raw_path.is_absolute() else (data_root_resolved / raw_path)

    try:
        resolved = candidate.resolve(strict=require_exists)
    except FileNotFoundError as exc:
        raise DatasetIndexValidationError(
            f"{field_name} points to a missing file: {candidate}"
        ) from exc

    _require_contained_path(
        resolved_path=resolved,
        data_root_resolved=data_root_resolved,
        field_name=field_name,
    )

    if require_exists and (not resolved.exists() or not resolved.is_file()):
        raise DatasetIndexValidationError(
            f"{field_name} must resolve to an existing file under data_root: {resolved}"
        )

    return resolved


def canonicalize_index_paths(
    frame: pd.DataFrame,
    *,
    data_root: Path,
    beta_column: str = "beta_path",
    mask_column: str = "mask_path",
    beta_canonical_column: str = CANONICAL_BETA_PATH_COLUMN,
    mask_canonical_column: str = CANONICAL_MASK_PATH_COLUMN,
    require_exists: bool = True,
) -> pd.DataFrame:
    if beta_column not in frame.columns:
        raise DatasetIndexValidationError(
            f"Dataset index missing required beta path column '{beta_column}'."
        )
    if mask_column not in frame.columns:
        raise DatasetIndexValidationError(
            f"Dataset index missing required mask path column '{mask_column}'."
        )

    result = frame.copy()
    beta_paths: list[str] = []
    mask_paths: list[str] = []
    for row_idx, row in result.iterrows():
        beta_path = resolve_index_data_path_strict(
            row.get(beta_column),
            data_root=data_root,
            field_name=f"{beta_column} (row={row_idx})",
            require_exists=require_exists,
        )
        mask_path = resolve_index_data_path_strict(
            row.get(mask_column),
            data_root=data_root,
            field_name=f"{mask_column} (row={row_idx})",
            require_exists=require_exists,
        )
        beta_paths.append(str(beta_path))
        mask_paths.append(str(mask_path))

    result[beta_canonical_column] = beta_paths
    result[mask_canonical_column] = mask_paths
    return result


def _validate_required_columns(frame: pd.DataFrame, required_columns: set[str]) -> None:
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise DatasetIndexValidationError(
            "Dataset index is missing required columns: " + ", ".join(missing)
        )


def _validate_non_blank_identifiers(frame: pd.DataFrame, key_columns: Iterable[str]) -> None:
    for column_name in key_columns:
        if column_name not in frame.columns:
            continue
        invalid_mask = frame[column_name].map(_is_blank)
        if bool(invalid_mask.any()):
            bad_count = int(invalid_mask.sum())
            raise DatasetIndexValidationError(
                f"Dataset index column '{column_name}' contains {bad_count} blank/null values."
            )


def _validate_unique_series(series: pd.Series, *, label: str) -> None:
    duplicates = series.astype(str).duplicated(keep=False)
    if bool(duplicates.any()):
        dup_values = sorted(series[duplicates].astype(str).unique().tolist())
        preview = dup_values[:5]
        raise DatasetIndexValidationError(
            f"Dataset index has duplicate {label} values. First duplicates: {preview}"
        )


def _normalized_bool(value: Any, *, column_name: str, row_idx: int) -> bool:
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
    raise DatasetIndexValidationError(
        f"Column '{column_name}' has invalid boolean value at row {row_idx}: {value!r}"
    )


def _normalized_int(value: Any, *, column_name: str, row_idx: int) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise DatasetIndexValidationError(
            f"Column '{column_name}' has invalid integer value at row {row_idx}: {value!r}"
        ) from exc
    return parsed


def _normalized_sha256(value: Any, *, column_name: str, row_idx: int) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise DatasetIndexValidationError(
            f"Column '{column_name}' has invalid sha256 value at row {row_idx}: {value!r}"
        )
    return text


def _normalized_json_list(value: Any, *, column_name: str, row_idx: int) -> list[str]:
    if isinstance(value, list):
        payload = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except Exception as exc:
            raise DatasetIndexValidationError(
                f"Column '{column_name}' has invalid JSON at row {row_idx}."
            ) from exc
    if not isinstance(payload, list):
        raise DatasetIndexValidationError(
            f"Column '{column_name}' must contain a JSON list at row {row_idx}."
        )
    return [str(item) for item in payload]


def _validate_subject_session_consistency(frame: pd.DataFrame) -> None:
    if {"subject", "session", "subject_session"} <= set(frame.columns):
        expected = frame["subject"].astype(str) + "_" + frame["session"].astype(str)
        mismatch = frame["subject_session"].astype(str) != expected
        if bool(mismatch.any()):
            raise DatasetIndexValidationError(
                "subject_session values are inconsistent with subject/session columns."
            )

    if {"subject", "session", "bas", "subject_session_bas"} <= set(frame.columns):
        expected = (
            frame["subject"].astype(str)
            + "_"
            + frame["session"].astype(str)
            + "_"
            + frame["bas"].astype(str)
        )
        mismatch = frame["subject_session_bas"].astype(str) != expected
        if bool(mismatch.any()):
            raise DatasetIndexValidationError(
                "subject_session_bas values are inconsistent with subject/session/bas columns."
            )


def _validate_group_mask_uniqueness(
    frame: pd.DataFrame,
    *,
    group_column: str,
    mask_canonical_column: str,
) -> None:
    if group_column not in frame.columns:
        raise DatasetIndexValidationError(
            f"Dataset index missing required group column '{group_column}'."
        )
    mask_counts = frame.groupby(group_column, dropna=False)[mask_canonical_column].nunique(dropna=False)
    bad_groups = mask_counts[mask_counts != 1]
    if not bad_groups.empty:
        details = {str(index): int(value) for index, value in bad_groups.items()}
        raise DatasetIndexValidationError(
            "Each subject_session_bas group must map to exactly one canonical mask path. "
            f"Violations: {details}"
        )


def _validate_integrity_columns(frame: pd.DataFrame) -> None:
    missing = [column_name for column_name in INDEX_INTEGRITY_COLUMNS if column_name not in frame.columns]
    if missing:
        raise DatasetIndexValidationError(
            "Dataset index is missing required integrity columns: " + ", ".join(sorted(missing))
        )

    beta_hash_cache: dict[str, str] = {}
    mask_hash_cache: dict[str, str] = {}

    for row_idx, row in frame.iterrows():
        beta_hash = _normalized_sha256(row.get("beta_file_sha256"), column_name="beta_file_sha256", row_idx=row_idx)
        mask_hash = _normalized_sha256(row.get("mask_file_sha256"), column_name="mask_file_sha256", row_idx=row_idx)
        coarse_version = str(row.get("coarse_affect_mapping_version", "")).strip()
        binary_version = str(row.get("binary_valence_mapping_version", "")).strip()
        if not coarse_version:
            raise DatasetIndexValidationError(
                f"Column 'coarse_affect_mapping_version' is blank at row {row_idx}."
            )
        if not binary_version:
            raise DatasetIndexValidationError(
                f"Column 'binary_valence_mapping_version' is blank at row {row_idx}."
            )

        _normalized_sha256(
            row.get("coarse_affect_mapping_sha256"),
            column_name="coarse_affect_mapping_sha256",
            row_idx=row_idx,
        )
        _normalized_sha256(
            row.get("binary_valence_mapping_sha256"),
            column_name="binary_valence_mapping_sha256",
            row_idx=row_idx,
        )

        has_unknown = _normalized_bool(
            row.get("glm_has_unknown_regressors"),
            column_name="glm_has_unknown_regressors",
            row_idx=row_idx,
        )
        unknown_count = _normalized_int(
            row.get("glm_unknown_regressor_count"),
            column_name="glm_unknown_regressor_count",
            row_idx=row_idx,
        )
        if unknown_count < 0:
            raise DatasetIndexValidationError(
                f"glm_unknown_regressor_count must be >= 0 at row {row_idx}."
            )

        unknown_labels = _normalized_json_list(
            row.get("glm_unknown_regressor_labels_json"),
            column_name="glm_unknown_regressor_labels_json",
            row_idx=row_idx,
        )
        if has_unknown and unknown_count <= 0:
            raise DatasetIndexValidationError(
                f"glm_has_unknown_regressors=true requires positive glm_unknown_regressor_count at row {row_idx}."
            )
        if (not has_unknown) and unknown_count != 0:
            raise DatasetIndexValidationError(
                f"glm_has_unknown_regressors=false requires glm_unknown_regressor_count=0 at row {row_idx}."
            )
        if unknown_count != len(unknown_labels):
            raise DatasetIndexValidationError(
                f"glm_unknown_regressor_count does not match glm_unknown_regressor_labels_json length at row {row_idx}."
            )

        beta_path = str(row[CANONICAL_BETA_PATH_COLUMN])
        mask_path = str(row[CANONICAL_MASK_PATH_COLUMN])
        if beta_path not in beta_hash_cache:
            beta_hash_cache[beta_path] = file_sha256(Path(beta_path))
        if mask_path not in mask_hash_cache:
            mask_hash_cache[mask_path] = file_sha256(Path(mask_path))

        if beta_hash_cache[beta_path] != beta_hash:
            raise DatasetIndexValidationError(
                f"beta_file_sha256 mismatch for canonical beta path '{beta_path}' at row {row_idx}."
            )
        if mask_hash_cache[mask_path] != mask_hash:
            raise DatasetIndexValidationError(
                f"mask_file_sha256 mismatch for canonical mask path '{mask_path}' at row {row_idx}."
            )


def validate_dataset_index_strict(
    frame: pd.DataFrame,
    *,
    data_root: Path,
    required_columns: Iterable[str] | None = None,
    require_integrity_columns: bool = False,
    beta_column: str = "beta_path",
    mask_column: str = "mask_path",
    beta_canonical_column: str = CANONICAL_BETA_PATH_COLUMN,
    mask_canonical_column: str = CANONICAL_MASK_PATH_COLUMN,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise DatasetIndexValidationError("Dataset index payload must be a pandas DataFrame.")
    if frame.empty:
        raise DatasetIndexValidationError("Dataset index must not be empty for strict validation.")

    required = set(STRICT_INDEX_REQUIRED_COLUMNS)
    if required_columns is not None:
        required.update(str(value) for value in required_columns)
    required.update({beta_column, mask_column})
    _validate_required_columns(frame, required_columns=required)

    _validate_non_blank_identifiers(frame, STRICT_INDEX_KEY_COLUMNS)
    _validate_unique_series(frame["sample_id"], label="sample_id")

    canonicalized = canonicalize_index_paths(
        frame,
        data_root=data_root,
        beta_column=beta_column,
        mask_column=mask_column,
        beta_canonical_column=beta_canonical_column,
        mask_canonical_column=mask_canonical_column,
        require_exists=True,
    )

    _validate_unique_series(canonicalized[beta_canonical_column], label="canonical beta_path")
    _validate_group_mask_uniqueness(
        canonicalized,
        group_column="subject_session_bas",
        mask_canonical_column=mask_canonical_column,
    )
    _validate_subject_session_consistency(canonicalized)

    if require_integrity_columns:
        _validate_integrity_columns(canonicalized)

    return canonicalized


__all__ = [
    "CANONICAL_BETA_PATH_COLUMN",
    "CANONICAL_MASK_PATH_COLUMN",
    "DatasetIndexValidationError",
    "INDEX_INTEGRITY_COLUMNS",
    "STRICT_INDEX_REQUIRED_COLUMNS",
    "STRICT_INDEX_KEY_COLUMNS",
    "canonicalize_index_paths",
    "file_sha256",
    "resolve_index_data_path_strict",
    "validate_dataset_index_strict",
]
