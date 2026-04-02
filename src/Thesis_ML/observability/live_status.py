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


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_percent_complete(
    completed_units: float | None,
    total_units: float | None,
) -> float | None:
    if completed_units is None or total_units is None:
        return None
    if total_units <= 0.0:
        return None
    percent = (float(completed_units) / float(total_units)) * 100.0
    return float(min(max(percent, 0.0), 100.0))


def _rebuild_operation_progress_indexes(state: dict[str, Any]) -> None:
    active_operations_raw = state.get("active_operations")
    active_operations = (
        dict(active_operations_raw) if isinstance(active_operations_raw, dict) else {}
    )

    experiment_progress: dict[str, dict[str, Any]] = {}
    phase_progress: dict[str, dict[str, Any]] = {}

    for operation in active_operations.values():
        if not isinstance(operation, dict):
            continue
        experiment_id = _normalize_optional_text(operation.get("experiment_id"))
        phase_name = _normalize_optional_text(operation.get("phase_name"))
        percent_complete = _coerce_float(operation.get("percent_complete"))

        if experiment_id is not None:
            exp_entry = experiment_progress.setdefault(
                experiment_id,
                {
                    "experiment_id": experiment_id,
                    "active_operation_count": 0,
                    "avg_percent_complete": None,
                },
            )
            exp_entry["active_operation_count"] = int(exp_entry["active_operation_count"]) + 1
            if percent_complete is not None:
                current = exp_entry.get("_percent_samples")
                samples = list(current) if isinstance(current, list) else []
                samples.append(float(percent_complete))
                exp_entry["_percent_samples"] = samples

        if phase_name is not None:
            phase_entry = phase_progress.setdefault(
                phase_name,
                {
                    "phase_name": phase_name,
                    "active_operation_count": 0,
                    "avg_percent_complete": None,
                },
            )
            phase_entry["active_operation_count"] = int(phase_entry["active_operation_count"]) + 1
            if percent_complete is not None:
                current = phase_entry.get("_percent_samples")
                samples = list(current) if isinstance(current, list) else []
                samples.append(float(percent_complete))
                phase_entry["_percent_samples"] = samples

    for payload in list(experiment_progress.values()) + list(phase_progress.values()):
        samples_raw = payload.pop("_percent_samples", None)
        samples = (
            [float(value) for value in samples_raw if isinstance(value, (int, float))]
            if isinstance(samples_raw, list)
            else []
        )
        if samples:
            payload["avg_percent_complete"] = float(sum(samples) / float(len(samples)))

    state["experiment_progress"] = _json_safe(experiment_progress)
    state["phase_progress"] = _json_safe(phase_progress)


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
        "active_operations": {},
        "experiment_progress": {},
        "phase_progress": {},
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
        "anomaly_counts": {"total": 0, "by_severity": {}, "by_code": {}, "by_category": {}},
        "latest_anomaly": None,
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
    normalized_run_id = _normalize_optional_text(run_id)

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
        if str(event_status or "").strip().lower() in {"blocked", "skipped"}:
            _append_unique(
                state.setdefault("blocked_experiments", []),
                None if experiment_id is None else str(experiment_id),
            )
    elif event_name == "experiment_skipped":
        counts["experiments_finished"] = int(counts.get("experiments_finished", 0)) + 1
        _remove_value(
            state.setdefault("active_experiments", []),
            None if experiment_id is None else str(experiment_id),
        )
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
        _append_unique(state.setdefault("active_runs", []), normalized_run_id)
    elif event_name == "run_finished":
        counts["runs_completed"] = int(counts.get("runs_completed", 0)) + 1
        _remove_value(state.setdefault("active_runs", []), normalized_run_id)
    elif event_name == "run_failed":
        counts["runs_failed"] = int(counts.get("runs_failed", 0)) + 1
        _remove_value(state.setdefault("active_runs", []), normalized_run_id)
        _append_unique(state.setdefault("failed_runs", []), normalized_run_id)
    elif event_name == "run_blocked":
        counts["runs_blocked"] = int(counts.get("runs_blocked", 0)) + 1
        _remove_value(state.setdefault("active_runs", []), normalized_run_id)
    elif event_name == "run_dry_run":
        counts["runs_dry_run"] = int(counts.get("runs_dry_run", 0)) + 1
        _remove_value(state.setdefault("active_runs", []), normalized_run_id)

    active_operations_raw = state.setdefault("active_operations", {})
    active_operations = (
        active_operations_raw if isinstance(active_operations_raw, dict) else {}
    )
    if event_name == "progress" and normalized_run_id is not None:
        existing_payload = active_operations.get(normalized_run_id)
        existing_operation = (
            dict(existing_payload) if isinstance(existing_payload, dict) else {}
        )
        completed_units = _coerce_float(event.get("completed_units"))
        total_units = _coerce_float(event.get("total_units"))
        operation_payload = {
            "run_id": normalized_run_id,
            "experiment_id": (
                _normalize_optional_text(experiment_id)
                or _normalize_optional_text(existing_operation.get("experiment_id"))
            ),
            "phase_name": (
                _normalize_optional_text(phase_name)
                or _normalize_optional_text(existing_operation.get("phase_name"))
            ),
            "stage": (
                _normalize_optional_text(event.get("stage"))
                or _normalize_optional_text(existing_operation.get("stage"))
            ),
            "message": (
                _normalize_optional_text(event.get("message"))
                or _normalize_optional_text(existing_operation.get("message"))
            ),
            "completed_units": completed_units,
            "total_units": total_units,
            "started_at_utc": str(existing_operation.get("started_at_utc") or now),
            "last_updated_at_utc": str(now),
            "percent_complete": _resolve_percent_complete(completed_units, total_units),
            "metadata": _json_safe(
                dict(metadata)
                if isinstance(metadata, dict)
                else dict(existing_operation.get("metadata", {}))
            ),
        }
        active_operations[normalized_run_id] = _json_safe(operation_payload)

    if event_name in {"run_finished", "run_failed", "run_blocked", "run_dry_run"}:
        if normalized_run_id is not None:
            active_operations.pop(normalized_run_id, None)

    state["active_operations"] = _json_safe(active_operations)
    _rebuild_operation_progress_indexes(state)

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


def merge_anomaly_payload_into_live_status(
    state: dict[str, Any],
    anomaly_payload: dict[str, Any] | None,
    *,
    keep_recent_anomalies: int = 20,
) -> dict[str, Any]:
    if not isinstance(anomaly_payload, dict):
        return state
    anomalies = anomaly_payload.get("anomalies")
    if isinstance(anomalies, list):
        normalized = [_json_safe(dict(item)) for item in anomalies if isinstance(item, dict)]
        if int(keep_recent_anomalies) > 0 and len(normalized) > int(keep_recent_anomalies):
            normalized = normalized[-int(keep_recent_anomalies) :]
        state["anomalies"] = normalized
    anomaly_counts = anomaly_payload.get("anomaly_counts")
    if isinstance(anomaly_counts, dict):
        state["anomaly_counts"] = _json_safe(dict(anomaly_counts))
    else:
        state.setdefault(
            "anomaly_counts",
            {"total": 0, "by_severity": {}, "by_code": {}, "by_category": {}},
        )
    latest = anomaly_payload.get("latest_anomaly")
    state["latest_anomaly"] = _json_safe(dict(latest)) if isinstance(latest, dict) else None
    return state


def write_live_status_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(f"{json.dumps(_json_safe(payload), indent=2)}\n", encoding="utf-8")
    tmp_path.replace(path)
