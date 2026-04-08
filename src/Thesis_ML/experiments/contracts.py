from __future__ import annotations

from enum import StrEnum


class SectionName(StrEnum):
    DATASET_SELECTION = "dataset_selection"
    FEATURE_CACHE_BUILD = "feature_cache_build"
    FEATURE_MATRIX_LOAD = "feature_matrix_load"
    SPATIAL_VALIDATION = "spatial_validation"
    MODEL_FIT = "model_fit"
    EVALUATION = "evaluation"
    INTERPRETABILITY = "interpretability"


class ReusePolicy(StrEnum):
    AUTO = "auto"
    REQUIRE_EXPLICIT_BASE = "require_explicit_base"
    DISALLOW = "disallow"
