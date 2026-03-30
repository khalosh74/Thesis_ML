from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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


def initial_live_status(campaign_id: str) -> dict[str, Any]:
    now = _utc_now()
    return {
        "campaign_id": str(campaign_id),
        "status": "running",
        "started_at_utc": now,
        "last_updated_at_utc": now,
        "current_phase": None,
        "current_phase_status": None,
        "counts": {
            "experiments_total": 0,
            "experiments_started": 0,
            "experiments_finished": 0,
            "runs_planned": 0,
            "runs_dispatched": 0,
            "runs_started": 0,
            "runs_completed": 0,
            "runs_failed": 0,
            "runs_blocked": 0,
            "runs_dry_run": 0,
        },
        "active_experiments": [],
        "active_runs": [],
        "recent_events": [],
        "blocked_experiments": [],
        "failed_runs": [],
        "eta_p50_seconds": None,
        "eta_p80_seconds": None,
        "eta_confidence": None,
        "eta_source": None,
        "campaign_eta": None,
        "phase_eta": None,
        "anomalies": [],
    }


def _append_unique(items: list[str], value: str | None) -> None:
    if value is None:
        return
    normalized = str(value).strip()
    if not normalized:
        return
    if normalized not in items:
        items.append(normalized)


def _remove_value(items: list[str], value: str | None) -> None:
    if value is None:
        return
    normalized = str(value).strip()
    if not normalized:
        return
    try:
        items.remove(normalized)
    except ValueError:
        return


def apply_event_to_live_status(
    state: dict[str, Any],
    event: dict[str, Any],
    *,
    keep_recent_events: int = 50,
) -> dict[str, Any]:
    now = str(event.get("timestamp_utc") or _utc_now())
    event_name = str(event.get("event_name") or "progress")
    event_status = event.get("status")
    phase_name = event.get("phase_name")
    experiment_id = event.get("experiment_id")
    run_id = event.get("run_id")
    metadata = event.get("metadata")

    state["last_updated_at_utc"] = now
    counts = state.setdefault("counts", {})

    if event_name == "campaign_started":
        state["status"] = "running"
        state["started_at_utc"] = str(
            event.get("timestamp_utc") or state.get("started_at_utc") or now
        )
        if isinstance(metadata, dict):
            experiments_total = metadata.get("experiments_total")
            if experiments_total is not None:
                counts["experiments_total"] = int(experiments_total)
    elif event_name == "campaign_finished":
        state["status"] = "finished"
    elif event_name == "phase_started":
        state["current_phase"] = phase_name
        state["current_phase_status"] = "running"
    elif event_name == "phase_finished":
        state["current_phase"] = phase_name
        state["current_phase_status"] = str(event_status or "finished")
    elif event_name == "experiment_started":
        counts["experiments_started"] = int(counts.get("experiments_started", 0)) + 1
        _append_unique(
            state.setdefault("active_experiments", []),
            None if experiment_id is None else str(experiment_id),
        )
    elif event_name == "experiment_finished":
        counts["experiments_finished"] = int(counts.get("experiments_finished", 0)) + 1
        _remove_value(
            state.setdefault("active_experiments", []),
            None if experiment_id is None else str(experiment_id),
        )
        if str(event_status or "").strip().lower() == "blocked":
            _append_unique(
                state.setdefault("blocked_experiments", []),
                None if experiment_id is None else str(experiment_id),
            )
    elif event_name == "run_planned":
        counts["runs_planned"] = int(counts.get("runs_planned", 0)) + 1
    elif event_name == "run_dispatched":
        counts["runs_dispatched"] = int(counts.get("runs_dispatched", 0)) + 1
    elif event_name == "run_started":
        counts["runs_started"] = int(counts.get("runs_started", 0)) + 1
        _append_unique(state.setdefault("active_runs", []), None if run_id is None else str(run_id))
    elif event_name == "run_finished":
        counts["runs_completed"] = int(counts.get("runs_completed", 0)) + 1
        _remove_value(state.setdefault("active_runs", []), None if run_id is None else str(run_id))
    elif event_name == "run_failed":
        counts["runs_failed"] = int(counts.get("runs_failed", 0)) + 1
        _remove_value(state.setdefault("active_runs", []), None if run_id is None else str(run_id))
        _append_unique(state.setdefault("failed_runs", []), None if run_id is None else str(run_id))
    elif event_name == "run_blocked":
        counts["runs_blocked"] = int(counts.get("runs_blocked", 0)) + 1
    elif event_name == "run_dry_run":
        counts["runs_dry_run"] = int(counts.get("runs_dry_run", 0)) + 1

    recent_events = state.setdefault("recent_events", [])
    recent_events.append(_json_safe(dict(event)))
    if int(keep_recent_events) > 0 and len(recent_events) > int(keep_recent_events):
        del recent_events[: -int(keep_recent_events)]

    state["counts"] = {
        "experiments_total": int(counts.get("experiments_total", 0)),
        "experiments_started": int(counts.get("experiments_started", 0)),
        "experiments_finished": int(counts.get("experiments_finished", 0)),
        "runs_planned": int(counts.get("runs_planned", 0)),
        "runs_dispatched": int(counts.get("runs_dispatched", 0)),
        "runs_started": int(counts.get("runs_started", 0)),
        "runs_completed": int(counts.get("runs_completed", 0)),
        "runs_failed": int(counts.get("runs_failed", 0)),
        "runs_blocked": int(counts.get("runs_blocked", 0)),
        "runs_dry_run": int(counts.get("runs_dry_run", 0)),
    }
    return state


def merge_eta_payload_into_live_status(
    state: dict[str, Any],
    eta_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(eta_payload, dict):
        return state
    campaign_eta = eta_payload.get("campaign_eta")
    phase_eta = eta_payload.get("phase_eta")
    if isinstance(campaign_eta, dict):
        state["campaign_eta"] = dict(campaign_eta)
        state["eta_p50_seconds"] = campaign_eta.get("eta_p50_seconds")
        state["eta_p80_seconds"] = campaign_eta.get("eta_p80_seconds")
        state["eta_confidence"] = campaign_eta.get("eta_confidence")
        state["eta_source"] = campaign_eta.get("eta_source")
    else:
        if "eta_p50_seconds" in eta_payload:
            state["eta_p50_seconds"] = eta_payload.get("eta_p50_seconds")
        if "eta_p80_seconds" in eta_payload:
            state["eta_p80_seconds"] = eta_payload.get("eta_p80_seconds")
        if "eta_confidence" in eta_payload:
            state["eta_confidence"] = eta_payload.get("eta_confidence")
        if "eta_source" in eta_payload:
            state["eta_source"] = eta_payload.get("eta_source")
    if isinstance(phase_eta, dict):
        state["phase_eta"] = dict(phase_eta)
    if "current_phase" in eta_payload and eta_payload.get("current_phase") is not None:
        state["current_phase"] = eta_payload.get("current_phase")
    return state


def write_live_status_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(f"{json.dumps(_json_safe(payload), indent=2)}\n", encoding="utf-8")
    tmp_path.replace(path)
