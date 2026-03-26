"""Utilities for deriving thesis coarse-affect labels."""

from __future__ import annotations

from typing import Any

import pandas as pd

COARSE_AFFECT_BY_EMOTION = {
    "happiness": "positive",
    "pride": "positive",
    "relief": "positive",
    "interest": "positive",
    "neutral": "neutral",
    "anger": "negative",
    "anxiety": "negative",
    "disgust": "negative",
    "sadness": "negative",
}

_BINARY_VALENCE_BY_COARSE = {
    "positive": "positive",
    "negative": "negative",
}


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
) -> pd.DataFrame:
    """
    Return a copy of ``frame`` with a derived coarse-affect column.

    Existing non-null values in ``coarse_column`` are preserved.
    """
    if emotion_column not in frame.columns:
        return frame.copy()

    result = frame.copy()
    derived = result[emotion_column].map(derive_coarse_affect)

    if coarse_column in result.columns:
        existing = result[coarse_column]
        result[coarse_column] = existing.where(existing.notna(), derived)
    else:
        result[coarse_column] = derived

    return result


def with_binary_valence_like(
    frame: pd.DataFrame,
    *,
    coarse_column: str = "coarse_affect",
    binary_column: str = "binary_valence_like",
) -> pd.DataFrame:
    """
    Return a copy of ``frame`` with a derived binary-valence-like column.

    Existing non-null values in ``binary_column`` are preserved.
    """
    if coarse_column not in frame.columns:
        return frame.copy()

    result = frame.copy()
    derived = result[coarse_column].map(derive_binary_valence_like)

    if binary_column in result.columns:
        existing = result[binary_column]
        result[binary_column] = existing.where(existing.notna(), derived)
    else:
        result[binary_column] = derived

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

_INTENTIONAL_BINARY_VALENCE_EXCLUSIONS = {"neutral"}


def _is_missing_label(value: Any) -> bool:
    if pd.isna(value):
        return True
    return not str(value).strip()


def _empty_target_derivation_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=_TARGET_DERIVATION_AUDIT_COLUMNS)


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
                reasons.append(
                    "coarse_affect is missing before binary_valence_like derivation."
                )
            continue

        source_values.append(pd.NA)
        categories.append("source_missing")
        reasons.append("coarse_affect source value is missing or blank before derivation.")

    dropped["source_value"] = source_values
    dropped["drop_category"] = categories
    dropped["drop_reason"] = reasons
    return dropped[_TARGET_DERIVATION_AUDIT_COLUMNS].reset_index(drop=True)


def blocking_target_derivation_audit_rows(audit_df: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty or "drop_category" not in audit_df.columns:
        return audit_df.copy()

    blocking_categories = {
        "source_missing",
        "unsupported_source_label",
        "unsupported_upstream_source_label",
    }
    return audit_df[audit_df["drop_category"].astype(str).isin(blocking_categories)].copy()


def summarize_target_derivation_audit(audit_df: pd.DataFrame) -> dict[str, Any]:
    if audit_df.empty:
        return {
            "n_rows": 0,
            "by_category": {},
            "sample_ids_head": [],
        }

    by_category = (
        audit_df["drop_category"]
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