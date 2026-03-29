from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

DEFAULT_PCA_VARIANCE_RATIO = 0.95
SUPPORTED_DIMENSIONALITY_STRATEGIES = ("none", "pca")


@dataclass(frozen=True)
class ResolvedDimensionalityConfig:
    strategy: str
    pca_n_components: int | None
    pca_variance_ratio: float | None


def resolve_dimensionality_config(
    *,
    dimensionality_strategy: str | None,
    pca_n_components: int | None,
    pca_variance_ratio: float | None,
) -> ResolvedDimensionalityConfig:
    strategy = str(dimensionality_strategy or "none").strip().lower()
    if strategy not in SUPPORTED_DIMENSIONALITY_STRATEGIES:
        allowed = ", ".join(sorted(SUPPORTED_DIMENSIONALITY_STRATEGIES))
        raise ValueError(
            f"Unsupported dimensionality_strategy '{strategy}'. Allowed values: {allowed}."
        )

    resolved_n_components = None
    resolved_variance_ratio = None
    if pca_n_components is not None:
        resolved_n_components = int(pca_n_components)
        if resolved_n_components <= 0:
            raise ValueError("pca_n_components must be > 0 when provided.")
    if pca_variance_ratio is not None:
        resolved_variance_ratio = float(pca_variance_ratio)
        if not (0.0 < resolved_variance_ratio <= 1.0):
            raise ValueError("pca_variance_ratio must be in (0, 1] when provided.")

    if strategy == "none":
        if resolved_n_components is not None or resolved_variance_ratio is not None:
            raise ValueError(
                "dimensionality_strategy='none' cannot be combined with PCA parameters."
            )
        return ResolvedDimensionalityConfig(
            strategy=strategy,
            pca_n_components=None,
            pca_variance_ratio=None,
        )

    if resolved_n_components is not None and resolved_variance_ratio is not None:
        raise ValueError(
            "Use exactly one PCA rule: pca_n_components OR pca_variance_ratio, not both."
        )
    if resolved_n_components is None and resolved_variance_ratio is None:
        resolved_variance_ratio = float(DEFAULT_PCA_VARIANCE_RATIO)

    return ResolvedDimensionalityConfig(
        strategy=strategy,
        pca_n_components=resolved_n_components,
        pca_variance_ratio=resolved_variance_ratio,
    )


def validate_dimensionality_for_training_data(
    *,
    config: ResolvedDimensionalityConfig,
    x_train: np.ndarray,
) -> None:
    if config.strategy != "pca":
        return
    if x_train.ndim != 2:
        raise ValueError("PCA requires a 2D training matrix.")
    n_samples, n_features = int(x_train.shape[0]), int(x_train.shape[1])
    if n_samples < 2 or n_features < 2:
        raise ValueError(
            "PCA requires at least two training samples and two features per fold."
        )
    if config.pca_n_components is not None:
        max_components = min(n_samples, n_features)
        if int(config.pca_n_components) > int(max_components):
            raise ValueError(
                "pca_n_components exceeds fold training limits: "
                f"requested={config.pca_n_components}, max_allowed={max_components}."
            )


def apply_dimensionality_to_pipeline(
    *,
    pipeline: Pipeline,
    config: ResolvedDimensionalityConfig,
) -> Pipeline:
    if config.strategy == "none":
        return pipeline

    if not isinstance(pipeline, Pipeline):
        raise ValueError("PCA dimensionality strategy requires sklearn Pipeline inputs.")

    pca_n_components: int | float
    if config.pca_n_components is not None:
        pca_n_components = int(config.pca_n_components)
    elif config.pca_variance_ratio is not None:
        pca_n_components = float(config.pca_variance_ratio)
    else:  # pragma: no cover - guarded by resolve_dimensionality_config
        pca_n_components = float(DEFAULT_PCA_VARIANCE_RATIO)

    dimensionality_step = (
        "dimensionality",
        PCA(
            n_components=pca_n_components,
            svd_solver="full",
        ),
    )
    steps = list(pipeline.steps)
    model_index = next(
        (idx for idx, (name, _) in enumerate(steps) if str(name) == "model"),
        None,
    )
    if model_index is None:
        raise ValueError("Pipeline is missing required 'model' step for dimensionality insertion.")
    if any(str(name) == "dimensionality" for name, _ in steps):
        raise ValueError("Pipeline already contains a 'dimensionality' step.")

    augmented_steps = steps[:model_index] + [dimensionality_step] + steps[model_index:]
    return Pipeline(steps=augmented_steps)


__all__ = [
    "DEFAULT_PCA_VARIANCE_RATIO",
    "ResolvedDimensionalityConfig",
    "SUPPORTED_DIMENSIONALITY_STRATEGIES",
    "apply_dimensionality_to_pipeline",
    "resolve_dimensionality_config",
    "validate_dimensionality_for_training_data",
]

