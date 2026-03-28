"""Utilities for deriving thesis coarse-affect labels."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from Thesis_ML.config.paths import DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH
from Thesis_ML.data.target_mapping_registry import load_target_mapping

_BINARY_VALENCE_BY_COARSE = {
    "positive": "positive",
    "negative": "negative",
}

COARSE_AFFECT_MAPPING_VERSION = DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH.stem
BINARY_VALENCE_MAPPING_VERSION = "binary_valence_mapping_v1"


def _stable_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_COARSE_AFFECT_MAPPING_ENTRY = load_target_mapping(COARSE_AFFECT_MAPPING_VERSION)
COARSE_AFFECT_BY_EMOTION = dict(_COARSE_AFFECT_MAPPING_ENTRY.mapping)
COARSE_AFFECT_MAPPING_SHA256 = _COARSE_AFFECT_MAPPING_ENTRY.mapping_hash
BINARY_VALENCE_MAPPING_SHA256 = _stable_sha256(_BINARY_VALENCE_BY_COARSE)

COARSE_AFFECT_MAPPING_VERSION_COLUMN = "coarse_affect_mapping_version"
COARSE_AFFECT_MAPPING_SHA256_COLUMN = "coarse_affect_mapping_sha256"
BINARY_VALENCE_MAPPING_VERSION_COLUMN = "binary_valence_mapping_version"
BINARY_VALENCE_MAPPING_SHA256_COLUMN = "binary_valence_mapping_sha256"


def derive_coarse_affect(emotion: Any) -> object:
    """Map a fine-grained emotion label to thesis coarse affect."""
    if pd.isna(emotion):
        return pd.NA

    label = str(emotion).strip().lower()
    if not label:
        return pd.NA

    return COARSE_AFFECT_BY_EMOTION.get(label, pd.NA)


def derive_binary_valence_like(coarse_affect: Any) -> object:
    """Map coarse affect to binary valence-like labels."""
    if pd.isna(coarse_affect):
        return pd.NA

    label = str(coarse_affect).strip().lower()
    if not label:
        return pd.NA

    return _BINARY_VALENCE_BY_COARSE.get(label, pd.NA)


def with_coarse_affect(
    frame: pd.DataFrame,
    *,
    emotion_column: str = "emotion",
    coarse_column: str = "coarse_affect",
    strict_recompute: bool = False,
    attach_mapping_metadata: bool = False,
) -> pd.DataFrame:
    """
    Return a copy of ``frame`` with a derived coarse-affect column.

    Existing non-null values in ``coarse_column`` are preserved unless
    ``strict_recompute`` is true.
    """
    result = frame.copy()
    if emotion_column in result.columns:
        derived = result[emotion_column].map(derive_coarse_affect)
        if strict_recompute or coarse_column not in result.columns:
            result[coarse_column] = derived
        else:
            existing = result[coarse_column]
            result[coarse_column] = existing.where(existing.notna(), derived)

    if attach_mapping_metadata:
        result[COARSE_AFFECT_MAPPING_VERSION_COLUMN] = COARSE_AFFECT_MAPPING_VERSION
        result[COARSE_AFFECT_MAPPING_SHA256_COLUMN] = COARSE_AFFECT_MAPPING_SHA256

    return result


def with_binary_valence_like(
    frame: pd.DataFrame,
    *,
    coarse_column: str = "coarse_affect",
    binary_column: str = "binary_valence_like",
    strict_recompute: bool = False,
    attach_mapping_metadata: bool = False,
) -> pd.DataFrame:
    """
    Return a copy of ``frame`` with a derived binary-valence-like column.

    Existing non-null values in ``binary_column`` are preserved unless
    ``strict_recompute`` is true.
    """
    result = frame.copy()
    if coarse_column in result.columns:
        derived = result[coarse_column].map(derive_binary_valence_like)
        if strict_recompute or binary_column not in result.columns:
            result[binary_column] = derived
        else:
            existing = result[binary_column]
            result[binary_column] = existing.where(existing.notna(), derived)

    if attach_mapping_metadata:
        result[BINARY_VALENCE_MAPPING_VERSION_COLUMN] = BINARY_VALENCE_MAPPING_VERSION
        result[BINARY_VALENCE_MAPPING_SHA256_COLUMN] = BINARY_VALENCE_MAPPING_SHA256

    return result


def with_affect_mapping_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result[COARSE_AFFECT_MAPPING_VERSION_COLUMN] = COARSE_AFFECT_MAPPING_VERSION
    result[COARSE_AFFECT_MAPPING_SHA256_COLUMN] = COARSE_AFFECT_MAPPING_SHA256
    result[BINARY_VALENCE_MAPPING_VERSION_COLUMN] = BINARY_VALENCE_MAPPING_VERSION
    result[BINARY_VALENCE_MAPPING_SHA256_COLUMN] = BINARY_VALENCE_MAPPING_SHA256
    return result


_TARGET_DERIVATION_AUDIT_COLUMNS = [
    "sample_id",
    "subject",
    "session",
    "task",
    "modality",
    "emotion",
    "coarse_affect",
    "binary_valence_like",
    "target_column",
    "source_column",
    "source_value",
    "drop_category",
    "drop_reason",
]

_DERIVED_LABEL_INCONSISTENCY_COLUMNS = [
    "sample_id",
    "subject",
    "session",
    "task",
    "modality",
    "target_column",
    "source_column",
    "source_value",
    "stored_value",
    "expected_value",
    "inconsistency_category",
    "inconsistency_reason",
]

_INTENTIONAL_BINARY_VALENCE_EXCLUSIONS = {"neutral"}


def _is_missing_label(value: Any) -> bool:
    if pd.isna(value):
        return True
    return not str(value).strip()


def _normalize_label(value: Any) -> str | None:
    if _is_missing_label(value):
        return None
    return str(value).strip().lower()


def _empty_target_derivation_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=_TARGET_DERIVATION_AUDIT_COLUMNS)


def _empty_derived_label_inconsistency_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=_DERIVED_LABEL_INCONSISTENCY_COLUMNS)


def build_target_derivation_audit(
    frame: pd.DataFrame,
    *,
    target_column: str,
) -> pd.DataFrame:
    if target_column not in {"coarse_affect", "binary_valence_like"}:
        return _empty_target_derivation_audit()

    working = frame.copy()
    for column_name in _TARGET_DERIVATION_AUDIT_COLUMNS:
        if column_name not in working.columns:
            working[column_name] = pd.NA

    if target_column == "coarse_affect":
        if "emotion" not in working.columns or "coarse_affect" not in working.columns:
            return _empty_target_derivation_audit()

        dropped = working[working["coarse_affect"].isna()].copy()
        if dropped.empty:
            return _empty_target_derivation_audit()

        dropped["target_column"] = "coarse_affect"
        dropped["source_column"] = "emotion"
        dropped["source_value"] = dropped["emotion"]

        categories: list[str] = []
        reasons: list[str] = []
        for emotion in dropped["emotion"].tolist():
            if _is_missing_label(emotion):
                categories.append("source_missing")
                reasons.append("emotion is missing or blank before coarse_affect derivation.")
            else:
                normalized = str(emotion).strip()
                categories.append("unsupported_source_label")
                reasons.append(
                    f"emotion='{normalized}' is not supported by COARSE_AFFECT_BY_EMOTION."
                )

        dropped["drop_category"] = categories
        dropped["drop_reason"] = reasons
        return dropped[_TARGET_DERIVATION_AUDIT_COLUMNS].reset_index(drop=True)

    if "binary_valence_like" not in working.columns:
        return _empty_target_derivation_audit()

    dropped = working[working["binary_valence_like"].isna()].copy()
    if dropped.empty:
        return _empty_target_derivation_audit()

    dropped["target_column"] = "binary_valence_like"
    dropped["source_column"] = "coarse_affect"

    source_values: list[object] = []
    categories = []
    reasons = []

    for _, row in dropped.iterrows():
        coarse_value = row.get("coarse_affect", pd.NA)
        emotion_value = row.get("emotion", pd.NA)

        if not _is_missing_label(coarse_value):
            coarse_text = str(coarse_value).strip()
            coarse_norm = coarse_text.lower()
            source_values.append(coarse_text)

            if coarse_norm in _INTENTIONAL_BINARY_VALENCE_EXCLUSIONS:
                categories.append("intended_target_exclusion")
                reasons.append(
                    "coarse_affect='neutral' is intentionally excluded from binary_valence_like."
                )
            elif coarse_norm in _BINARY_VALENCE_BY_COARSE:
                categories.append("source_missing")
                reasons.append(
                    "binary_valence_like is missing despite a supported coarse_affect source."
                )
            else:
                categories.append("unsupported_source_label")
                reasons.append(
                    f"coarse_affect='{coarse_text}' is not supported by binary valence mapping."
                )
            continue

        if not _is_missing_label(emotion_value):
            emotion_text = str(emotion_value).strip()
            source_values.append(emotion_text)
            derived_coarse = derive_coarse_affect(emotion_value)

            if pd.isna(derived_coarse):
                categories.append("unsupported_upstream_source_label")
                reasons.append(
                    f"emotion='{emotion_text}' is not supported by COARSE_AFFECT_BY_EMOTION, "
                    "so binary_valence_like cannot be derived."
                )
            elif str(derived_coarse).strip().lower() in _INTENTIONAL_BINARY_VALENCE_EXCLUSIONS:
                categories.append("intended_target_exclusion")
                reasons.append(
                    f"emotion='{emotion_text}' maps to coarse_affect='neutral', which is "
                    "intentionally excluded from binary_valence_like."
                )
            else:
                categories.append("source_missing")
                reasons.append("coarse_affect is missing before binary_valence_like derivation.")
            continue

        source_values.append(pd.NA)
        categories.append("source_missing")
        reasons.append("coarse_affect source value is missing or blank before derivation.")

    dropped["source_value"] = source_values
    dropped["drop_category"] = categories
    dropped["drop_reason"] = reasons
    return dropped[_TARGET_DERIVATION_AUDIT_COLUMNS].reset_index(drop=True)


def build_derived_label_inconsistency_audit(
    frame: pd.DataFrame,
    *,
    emotion_column: str = "emotion",
    coarse_column: str = "coarse_affect",
    binary_column: str = "binary_valence_like",
) -> pd.DataFrame:
    if frame.empty:
        return _empty_derived_label_inconsistency_audit()

    working = frame.copy()
    for column_name in _DERIVED_LABEL_INCONSISTENCY_COLUMNS:
        if column_name not in working.columns:
            working[column_name] = pd.NA

    rows: list[dict[str, Any]] = []

    if emotion_column in working.columns and coarse_column in working.columns:
        expected_coarse = working[emotion_column].map(derive_coarse_affect)
        for row_idx, row in working.iterrows():
            source_value = row.get(emotion_column, pd.NA)
            stored_value = row.get(coarse_column, pd.NA)
            expected_value = expected_coarse.iloc[row_idx]
            stored_norm = _normalize_label(stored_value)
            expected_norm = _normalize_label(expected_value)

            if stored_norm == expected_norm:
                continue

            category = "stored_value_mismatch"
            if stored_norm is None and expected_norm is not None:
                category = "stored_value_missing"
            elif stored_norm is not None and expected_norm is None:
                category = "unsupported_source_label_stored_value_present"

            rows.append(
                {
                    "sample_id": row.get("sample_id", pd.NA),
                    "subject": row.get("subject", pd.NA),
                    "session": row.get("session", pd.NA),
                    "task": row.get("task", pd.NA),
                    "modality": row.get("modality", pd.NA),
                    "target_column": coarse_column,
                    "source_column": emotion_column,
                    "source_value": source_value,
                    "stored_value": stored_value,
                    "expected_value": expected_value,
                    "inconsistency_category": category,
                    "inconsistency_reason": (
                        "Stored coarse_affect is inconsistent with emotion-derived coarse_affect."
                    ),
                }
            )

    if coarse_column in working.columns and binary_column in working.columns:
        expected_binary = working[coarse_column].map(derive_binary_valence_like)
        for row_idx, row in working.iterrows():
            source_value = row.get(coarse_column, pd.NA)
            stored_value = row.get(binary_column, pd.NA)
            expected_value = expected_binary.iloc[row_idx]
            stored_norm = _normalize_label(stored_value)
            expected_norm = _normalize_label(expected_value)

            if stored_norm == expected_norm:
                continue

            category = "stored_value_mismatch"
            if stored_norm is None and expected_norm is not None:
                category = "stored_value_missing"
            elif stored_norm is not None and expected_norm is None:
                category = "unsupported_source_label_stored_value_present"

            rows.append(
                {
                    "sample_id": row.get("sample_id", pd.NA),
                    "subject": row.get("subject", pd.NA),
                    "session": row.get("session", pd.NA),
                    "task": row.get("task", pd.NA),
                    "modality": row.get("modality", pd.NA),
                    "target_column": binary_column,
                    "source_column": coarse_column,
                    "source_value": source_value,
                    "stored_value": stored_value,
                    "expected_value": expected_value,
                    "inconsistency_category": category,
                    "inconsistency_reason": (
                        "Stored binary_valence_like is inconsistent with coarse-affect-derived binary label."
                    ),
                }
            )

    if not rows:
        return _empty_derived_label_inconsistency_audit()

    return pd.DataFrame(rows, columns=_DERIVED_LABEL_INCONSISTENCY_COLUMNS)


def blocking_target_derivation_audit_rows(audit_df: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty or "drop_category" not in audit_df.columns:
        return audit_df.copy()

    blocking_categories = {
        "source_missing",
        "unsupported_source_label",
        "unsupported_upstream_source_label",
    }
    return audit_df[audit_df["drop_category"].astype(str).isin(blocking_categories)].copy()


def blocking_derived_label_inconsistency_rows(audit_df: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty:
        return audit_df.copy()
    return audit_df.copy()


def summarize_target_derivation_audit(audit_df: pd.DataFrame) -> dict[str, Any]:
    if audit_df.empty:
        return {
            "n_rows": 0,
            "by_category": {},
            "sample_ids_head": [],
        }

    by_category = (
        audit_df["drop_category"].astype(str).value_counts(dropna=False).sort_index().to_dict()
    )
    sample_ids_head = []
    if "sample_id" in audit_df.columns:
        sample_ids_head = audit_df["sample_id"].dropna().astype(str).tolist()[:10]

    return {
        "n_rows": int(len(audit_df)),
        "by_category": {str(key): int(value) for key, value in by_category.items()},
        "sample_ids_head": sample_ids_head,
    }


def summarize_derived_label_inconsistency_audit(audit_df: pd.DataFrame) -> dict[str, Any]:
    if audit_df.empty:
        return {
            "n_rows": 0,
            "by_category": {},
            "sample_ids_head": [],
        }

    by_category = (
        audit_df["inconsistency_category"]
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    sample_ids_head = []
    if "sample_id" in audit_df.columns:
        sample_ids_head = audit_df["sample_id"].dropna().astype(str).tolist()[:10]

    return {
        "n_rows": int(len(audit_df)),
        "by_category": {str(key): int(value) for key, value in by_category.items()},
        "sample_ids_head": sample_ids_head,
    }


__all__ = [
    "BINARY_VALENCE_MAPPING_SHA256",
    "BINARY_VALENCE_MAPPING_SHA256_COLUMN",
    "BINARY_VALENCE_MAPPING_VERSION",
    "BINARY_VALENCE_MAPPING_VERSION_COLUMN",
    "COARSE_AFFECT_BY_EMOTION",
    "COARSE_AFFECT_MAPPING_SHA256",
    "COARSE_AFFECT_MAPPING_SHA256_COLUMN",
    "COARSE_AFFECT_MAPPING_VERSION",
    "COARSE_AFFECT_MAPPING_VERSION_COLUMN",
    "blocking_derived_label_inconsistency_rows",
    "blocking_target_derivation_audit_rows",
    "build_derived_label_inconsistency_audit",
    "build_target_derivation_audit",
    "derive_binary_valence_like",
    "derive_coarse_affect",
    "summarize_derived_label_inconsistency_audit",
    "summarize_target_derivation_audit",
    "with_affect_mapping_metadata",
    "with_binary_valence_like",
    "with_coarse_affect",
]
