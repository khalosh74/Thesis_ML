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
