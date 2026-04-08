from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from Thesis_ML.release.hashing import sha256_bytes
from Thesis_ML.release.scope_models import ScopeCounts


def selected_sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    normalized = [str(sample_id) for sample_id in sample_ids]
    payload = "\n".join(normalized).encode("utf-8")
    return sha256_bytes(payload)


def _value_counts(frame: pd.DataFrame, column_name: str) -> dict[str, int]:
    if column_name not in frame.columns:
        return {}
    values = frame[column_name].astype(str).value_counts(dropna=False).sort_index().to_dict()
    return {str(key): int(value) for key, value in values.items()}


def build_scope_counts(frame: pd.DataFrame, *, target_column: str) -> ScopeCounts:
    return ScopeCounts(
        by_subject=_value_counts(frame, "subject"),
        by_session=_value_counts(frame, "session"),
        by_task=_value_counts(frame, "task"),
        by_modality=_value_counts(frame, "modality"),
        by_target=_value_counts(frame, target_column),
    )


__all__ = [
    "build_scope_counts",
    "selected_sample_ids_sha256",
]

