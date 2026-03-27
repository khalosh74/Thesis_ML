"""Dataset utilities with lazy heavy imports.

This module avoids importing :mod:`index_dataset` eagerly so that model/comparison/protocol
imports do not require nibabel at import time.
"""

from __future__ import annotations

from typing import Any

from Thesis_ML.data.affect_labels import (
    derive_binary_valence_like,
    derive_coarse_affect,
    with_binary_valence_like,
    with_coarse_affect,
)

__all__ = [
    "build_dataset_index",
    "derive_coarse_affect",
    "derive_binary_valence_like",
    "with_coarse_affect",
    "with_binary_valence_like",
]


def __getattr__(name: str) -> Any:
    if name != "build_dataset_index":
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    from Thesis_ML.data.index_dataset import build_dataset_index as _build_dataset_index

    return _build_dataset_index


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
