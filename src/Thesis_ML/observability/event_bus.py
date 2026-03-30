from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from Thesis_ML.experiments.progress import ProgressEvent
from Thesis_ML.observability.live_status import (
    apply_event_to_live_status,
    initial_live_status,
    merge_eta_payload_into_live_status,
    write_live_status_atomic,
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


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


class ExecutionEventBus:
    def __init__(
        self,
        campaign_root: Path,
        campaign_id: str,
        keep_recent_events: int = 50,
        eta_estimator: Any | None = None,
    ) -> None:
        self.campaign_root = Path(campaign_root)
        self.campaign_id = str(campaign_id)
        self.keep_recent_events = int(keep_recent_events)
        self.eta_estimator = eta_estimator
        self.execution_events_path = self.campaign_root / "execution_events.jsonl"
        self.live_status_path = self.campaign_root / "campaign_live_status.json"
        self.campaign_root.mkdir(parents=True, exist_ok=True)
        self._live_status = initial_live_status(self.campaign_id)
        write_live_status_atomic(self.live_status_path, self._live_status)

    def emit_event(
        self,
        *,
        event_name: str,
        scope: str,
        status: str | None = None,
        stage: str | None = None,
        message: str | None = None,
        completed_units: float | None = None,
        total_units: float | None = None,
        timestamp_utc: str | None = None,
        phase_name: str | None = None,
        experiment_id: str | None = None,
        variant_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        eta_payload: dict[str, Any] | None = None
        event_payload = {
            "event_name": str(event_name),
            "scope": str(scope),
            "status": None if status is None else str(status),
            "stage": None if stage is None else str(stage),
            "message": None if message is None else str(message),
            "completed_units": completed_units,
            "total_units": total_units,
            "timestamp_utc": str(timestamp_utc or _utc_now()),
            "campaign_id": str(self.campaign_id),
            "phase_name": None if phase_name is None else str(phase_name),
            "experiment_id": None if experiment_id is None else str(experiment_id),
            "variant_id": None if variant_id is None else str(variant_id),
            "run_id": None if run_id is None else str(run_id),
            "metadata": _json_safe(dict(metadata or {})),
        }
        if self.eta_estimator is not None:
            try:
                eta_result = self.eta_estimator.ingest_event(event_payload)
                if isinstance(eta_result, dict):
                    eta_payload = dict(eta_result)
            except Exception:
                eta_payload = None
        try:
            with self.execution_events_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{json.dumps(event_payload, ensure_ascii=True)}\n")
            self._live_status = apply_event_to_live_status(
                self._live_status,
                event_payload,
                keep_recent_events=self.keep_recent_events,
            )
            if eta_payload is not None:
                self._live_status = merge_eta_payload_into_live_status(
                    self._live_status,
                    eta_payload,
                )
            write_live_status_atomic(self.live_status_path, self._live_status)
        except Exception:
            return event_payload
        return event_payload

    def build_progress_callback(
        self,
        *,
        phase_name: str | None = None,
        experiment_id: str | None = None,
        variant_id: str | None = None,
        run_id: str | None = None,
    ):
        def _callback(event: ProgressEvent) -> None:
            payload = event.to_payload()
            merged_metadata = dict(payload.get("metadata", {}))
            self.emit_event(
                event_name=str(payload.get("event_name") or "progress"),
                scope=str(payload.get("scope") or "run"),
                status=(
                    payload.get("status")
                    if payload.get("status") is None
                    else str(payload.get("status"))
                ),
                stage=(
                    payload.get("stage")
                    if payload.get("stage") is None
                    else str(payload.get("stage"))
                ),
                message=(
                    payload.get("message")
                    if payload.get("message") is None
                    else str(payload.get("message"))
                ),
                completed_units=payload.get("completed_units"),
                total_units=payload.get("total_units"),
                timestamp_utc=(
                    payload.get("timestamp_utc")
                    if payload.get("timestamp_utc") is None
                    else str(payload.get("timestamp_utc"))
                ),
                phase_name=(payload.get("phase_name") or phase_name),
                experiment_id=(payload.get("experiment_id") or experiment_id),
                variant_id=(payload.get("variant_id") or variant_id),
                run_id=(payload.get("run_id") or run_id),
                metadata=merged_metadata,
            )

        return _callback


__all__ = ["ExecutionEventBus"]
