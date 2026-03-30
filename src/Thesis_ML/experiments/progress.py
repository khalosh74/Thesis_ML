from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        return str(value)
    except Exception:
        return repr(value)


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    message: str
    completed_units: float | None = None
    total_units: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    event_name: str = "progress"
    scope: str = "run"
    status: str | None = None
    campaign_id: str | None = None
    phase_name: str | None = None
    experiment_id: str | None = None
    variant_id: str | None = None
    run_id: str | None = None
    timestamp_utc: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_name": str(self.event_name),
            "scope": str(self.scope),
            "status": (None if self.status is None else str(self.status)),
            "stage": str(self.stage),
            "message": str(self.message),
            "completed_units": self.completed_units,
            "total_units": self.total_units,
            "timestamp_utc": (None if self.timestamp_utc is None else str(self.timestamp_utc)),
            "campaign_id": (None if self.campaign_id is None else str(self.campaign_id)),
            "phase_name": (None if self.phase_name is None else str(self.phase_name)),
            "experiment_id": (None if self.experiment_id is None else str(self.experiment_id)),
            "variant_id": (None if self.variant_id is None else str(self.variant_id)),
            "run_id": (None if self.run_id is None else str(self.run_id)),
            "metadata": _json_safe(dict(self.metadata)),
        }


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
            timestamp_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
        )
    )
