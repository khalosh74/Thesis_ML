from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    message: str
    completed_units: float | None = None
    total_units: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[ProgressEvent], None]


def emit_progress(
    callback: ProgressCallback | None,
    *,
    stage: str,
    message: str,
    completed_units: float | None = None,
    total_units: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if callback is None:
        return
    callback(
        ProgressEvent(
            stage=str(stage),
            message=str(message),
            completed_units=completed_units,
            total_units=total_units,
            metadata=dict(metadata or {}),
        )
    )
