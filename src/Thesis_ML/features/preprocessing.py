from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

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
SUPPORTED_PREPROCESSING_STRATEGIES: tuple[str, ...] = ("none", "standardize_zscore")


@dataclass(frozen=True)
class ResolvedPreprocessingConfig:
    strategy: str | None


class PerSampleMeanCenter(BaseEstimator, TransformerMixin):
    def fit(self, x: Any, y: Any = None) -> PerSampleMeanCenter:
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
        raise ValueError(f"Unsupported feature_recipe_id '{recipe_id}'. Allowed values: {allowed}.")
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


def resolve_preprocessing_strategy(preprocessing_strategy: str | None) -> ResolvedPreprocessingConfig:
    if preprocessing_strategy is None:
        return ResolvedPreprocessingConfig(strategy=None)
    normalized = str(preprocessing_strategy).strip().lower()
    if not normalized:
        return ResolvedPreprocessingConfig(strategy=None)
    if normalized not in set(SUPPORTED_PREPROCESSING_STRATEGIES):
        allowed = ", ".join(sorted(SUPPORTED_PREPROCESSING_STRATEGIES))
        raise ValueError(
            "Unsupported preprocessing_strategy "
            f"'{preprocessing_strategy}'. Allowed values: {allowed}."
        )
    return ResolvedPreprocessingConfig(strategy=normalized)


def validate_preprocessing_for_training_data(
    *,
    preprocessing_strategy: str | None,
    x_train: np.ndarray,
) -> None:
    resolved = resolve_preprocessing_strategy(preprocessing_strategy)
    if resolved.strategy != "standardize_zscore":
        return
    x_array = np.asarray(x_train)
    if x_array.ndim != 2:
        raise ValueError("preprocessing_strategy='standardize_zscore' requires a 2D training matrix.")
    n_samples = int(x_array.shape[0])
    n_features = int(x_array.shape[1])
    if n_samples <= 0 or n_features <= 0:
        raise ValueError(
            "preprocessing_strategy='standardize_zscore' requires non-empty training data."
        )


def apply_preprocessing_to_pipeline(
    *,
    pipeline: Any,
    preprocessing_strategy: str | None,
) -> Any:
    resolved = resolve_preprocessing_strategy(preprocessing_strategy)
    if resolved.strategy is None:
        return pipeline
    if not isinstance(pipeline, Pipeline):
        raise ValueError("preprocessing_strategy requires sklearn Pipeline inputs.")
    steps = list(pipeline.steps)
    scaler_index = next((idx for idx, (name, _) in enumerate(steps) if str(name) == "scaler"), None)
    if scaler_index is None:
        raise ValueError("Pipeline is missing required 'scaler' step for preprocessing strategy.")
    if resolved.strategy == "none":
        scaler_transformer = FunctionTransformer(validate=False)
    elif resolved.strategy == "standardize_zscore":
        scaler_transformer = StandardScaler(with_mean=True, with_std=True)
    else:  # pragma: no cover - guarded by resolve_preprocessing_strategy
        raise ValueError(
            f"Unsupported preprocessing_strategy '{resolved.strategy}'."
        )
    steps[scaler_index] = ("scaler", scaler_transformer)
    return Pipeline(steps=steps)


__all__ = [
    "BASELINE_STANDARD_SCALER_RECIPE_ID",
    "FEATURE_RECIPE_IDS",
    "PerSampleMeanCenter",
    "SAMPLE_CENTER_STANDARD_SCALER_RECIPE_ID",
    "SAMPLE_CENTER_VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID",
    "SUPPORTED_PREPROCESSING_STRATEGIES",
    "VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID",
    "ResolvedPreprocessingConfig",
    "apply_preprocessing_to_pipeline",
    "build_feature_preprocessing_recipe",
    "resolve_preprocessing_strategy",
    "resolve_feature_recipe_id",
    "validate_preprocessing_for_training_data",
]
