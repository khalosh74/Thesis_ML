from __future__ import annotations

from Thesis_ML.observability.live_status import apply_event_to_live_status, initial_live_status


def test_live_status_tracks_and_clears_active_operations_from_progress_events() -> None:
    state = initial_live_status("campaign-1")

    state = apply_event_to_live_status(
        state,
        {
            "event_name": "progress",
            "status": "running",
            "stage": "section",
            "message": "starting section dataset_selection",
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "phase_name": "Stage 1 target/scope lock",
            "experiment_id": "E01",
            "run_id": "run_1",
            "completed_units": 2.0,
            "total_units": 10.0,
            "metadata": {"section": "dataset_selection"},
        },
    )

    active_operations = dict(state.get("active_operations", {}))
    assert "run_1" in active_operations
    operation = dict(active_operations["run_1"])
    assert operation["run_id"] == "run_1"
    assert operation["experiment_id"] == "E01"
    assert operation["phase_name"] == "Stage 1 target/scope lock"
    assert operation["stage"] == "section"
    assert operation["message"] == "starting section dataset_selection"
    assert operation["completed_units"] == 2.0
    assert operation["total_units"] == 10.0
    assert operation["percent_complete"] == 20.0
    assert operation["started_at_utc"] == "2026-01-01T00:00:00+00:00"
    assert operation["last_updated_at_utc"] == "2026-01-01T00:00:00+00:00"

    phase_progress = dict(state.get("phase_progress", {}))
    experiment_progress = dict(state.get("experiment_progress", {}))
    assert phase_progress["Stage 1 target/scope lock"]["active_operation_count"] == 1
    assert experiment_progress["E01"]["active_operation_count"] == 1

    state = apply_event_to_live_status(
        state,
        {
            "event_name": "progress",
            "status": "running",
            "stage": "fold",
            "message": "processing fold 2/5",
            "timestamp_utc": "2026-01-01T00:00:05+00:00",
            "phase_name": "Stage 1 target/scope lock",
            "experiment_id": "E01",
            "run_id": "run_1",
            "completed_units": None,
            "total_units": None,
            "metadata": {"fold": 2},
        },
    )

    operation = dict(dict(state.get("active_operations", {}))["run_1"])
    assert operation["stage"] == "fold"
    assert operation["message"] == "processing fold 2/5"
    assert operation["completed_units"] is None
    assert operation["total_units"] is None
    assert operation["percent_complete"] is None
    assert operation["started_at_utc"] == "2026-01-01T00:00:00+00:00"
    assert operation["last_updated_at_utc"] == "2026-01-01T00:00:05+00:00"

    state = apply_event_to_live_status(
        state,
        {
            "event_name": "run_finished",
            "status": "completed",
            "stage": "campaign",
            "message": "run finished",
            "timestamp_utc": "2026-01-01T00:00:10+00:00",
            "phase_name": "Stage 1 target/scope lock",
            "experiment_id": "E01",
            "run_id": "run_1",
            "completed_units": None,
            "total_units": None,
            "metadata": {},
        },
    )

    assert "run_1" not in dict(state.get("active_operations", {}))
    assert dict(state.get("phase_progress", {})) == {}
    assert dict(state.get("experiment_progress", {})) == {}
