"""NIfTI feature extraction and caching.

This module intentionally avoids eager imports of :mod:`nifti_features` so that
model/comparison/protocol imports do not require nibabel at import time.
"""

from __future__ import annotations

from typing import Any

__all__ = ["load_mask", "extract_masked_vector", "build_feature_cache"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    from Thesis_ML.features import nifti_features as _nifti_features

    return getattr(_nifti_features, name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
