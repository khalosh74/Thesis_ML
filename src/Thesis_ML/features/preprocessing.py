from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASELINE_STANDARD_SCALER_RECIPE_ID = "baseline_standard_scaler_v1"
SAMPLE_CENTER_STANDARD_SCALER_RECIPE_ID = "sample_center_standard_scaler_v1"
VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID = "variance_filter_standard_scaler_v1"
SAMPLE_CENTER_VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID = (
    "sample_center_variance_filter_standard_scaler_v1"
)

FEATURE_RECIPE_IDS: tuple[str, ...] = (
    BASELINE_STANDARD_SCALER_RECIPE_ID,
    SAMPLE_CENTER_STANDARD_SCALER_RECIPE_ID,
    VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID,
    SAMPLE_CENTER_VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID,
)


class PerSampleMeanCenter(BaseEstimator, TransformerMixin):
    def fit(self, x: Any, y: Any = None) -> "PerSampleMeanCenter":
        _ = y
        x_array = np.asarray(x)
        if x_array.ndim != 2:
            raise ValueError("PerSampleMeanCenter expects a 2D matrix.")
        return self

    def transform(self, x: Any) -> np.ndarray:
        x_array = np.asarray(x, dtype=np.float64)
        if x_array.ndim != 2:
            raise ValueError("PerSampleMeanCenter expects a 2D matrix.")
        row_means = np.mean(x_array, axis=1, keepdims=True)
        return np.asarray(x_array - row_means, dtype=np.float64)


def resolve_feature_recipe_id(recipe_id: str | None) -> str:
    normalized = str(recipe_id or BASELINE_STANDARD_SCALER_RECIPE_ID).strip().lower()
    if normalized not in set(FEATURE_RECIPE_IDS):
        allowed = ", ".join(sorted(FEATURE_RECIPE_IDS))
        raise ValueError(
            f"Unsupported feature_recipe_id '{recipe_id}'. Allowed values: {allowed}."
        )
    return normalized


def build_feature_preprocessing_recipe(recipe_id: str | None) -> Any:
    resolved = resolve_feature_recipe_id(recipe_id)
    if resolved == BASELINE_STANDARD_SCALER_RECIPE_ID:
        return StandardScaler(with_mean=True, with_std=True)
    if resolved == SAMPLE_CENTER_STANDARD_SCALER_RECIPE_ID:
        return Pipeline(
            steps=[
                ("sample_center", PerSampleMeanCenter()),
                ("standard_scaler", StandardScaler(with_mean=True, with_std=True)),
            ]
        )
    if resolved == VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID:
        return Pipeline(
            steps=[
                ("variance_filter", VarianceThreshold(threshold=0.0)),
                ("standard_scaler", StandardScaler(with_mean=True, with_std=True)),
            ]
        )
    if resolved == SAMPLE_CENTER_VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID:
        return Pipeline(
            steps=[
                ("sample_center", PerSampleMeanCenter()),
                ("variance_filter", VarianceThreshold(threshold=0.0)),
                ("standard_scaler", StandardScaler(with_mean=True, with_std=True)),
            ]
        )
    raise ValueError(f"Unsupported feature_recipe_id '{recipe_id}'.")


__all__ = [
    "BASELINE_STANDARD_SCALER_RECIPE_ID",
    "FEATURE_RECIPE_IDS",
    "PerSampleMeanCenter",
    "SAMPLE_CENTER_STANDARD_SCALER_RECIPE_ID",
    "SAMPLE_CENTER_VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID",
    "VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID",
    "build_feature_preprocessing_recipe",
    "resolve_feature_recipe_id",
]
