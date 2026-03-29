from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from Thesis_ML.features.roi_features import load_roi_feature_matrix

FEATURE_SPACE_WHOLE_BRAIN_MASKED = "whole_brain_masked"
FEATURE_SPACE_ROI_MEAN_PREDEFINED = "roi_mean_predefined"
SUPPORTED_FEATURE_SPACES = (
    FEATURE_SPACE_WHOLE_BRAIN_MASKED,
    FEATURE_SPACE_ROI_MEAN_PREDEFINED,
)


def normalize_feature_space(value: str | None) -> str:
    resolved = str(value or FEATURE_SPACE_WHOLE_BRAIN_MASKED).strip().lower()
    if resolved not in SUPPORTED_FEATURE_SPACES:
        allowed = ", ".join(sorted(SUPPORTED_FEATURE_SPACES))
        raise ValueError(f"Unsupported feature_space '{resolved}'. Allowed values: {allowed}")
    return resolved


def load_feature_matrix(
    *,
    selected_index_df: pd.DataFrame,
    cache_manifest_path: Path,
    spatial_report_path: Path,
    affine_atol: float,
    data_root: Path,
    feature_space: str | None,
    roi_spec_path: Path | None,
    load_features_from_cache_fn: Callable[..., tuple[np.ndarray, pd.DataFrame, dict[str, Any]]],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    resolved_feature_space = normalize_feature_space(feature_space)

    if resolved_feature_space == FEATURE_SPACE_WHOLE_BRAIN_MASKED:
        return load_features_from_cache_fn(
            index_df=selected_index_df,
            cache_manifest_path=cache_manifest_path,
            spatial_report_path=spatial_report_path,
            affine_atol=affine_atol,
        )

    if roi_spec_path is None or not str(roi_spec_path).strip():
        raise ValueError(
            "feature_space='roi_mean_predefined' requires a non-empty roi_spec_path."
        )
    return load_roi_feature_matrix(
        selected_index_df=selected_index_df,
        data_root=Path(data_root),
        roi_spec_path=Path(roi_spec_path),
        spatial_report_path=spatial_report_path,
    )


__all__ = [
    "FEATURE_SPACE_ROI_MEAN_PREDEFINED",
    "FEATURE_SPACE_WHOLE_BRAIN_MASKED",
    "SUPPORTED_FEATURE_SPACES",
    "load_feature_matrix",
    "normalize_feature_space",
]

