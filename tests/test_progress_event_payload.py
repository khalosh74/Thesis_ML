from __future__ import annotations

from Thesis_ML.experiments.progress import ProgressEvent, emit_progress


def test_progress_event_to_payload_is_backward_compatible_and_json_safe() -> None:
    event = ProgressEvent(
        stage="model_fit",
        message="fitting",
        completed_units=1.0,
        total_units=3.0,
        metadata={"path_like": object(), "nested": {"k": object()}},
    )
    payload = event.to_payload()
    assert payload["event_name"] == "progress"
    assert payload["scope"] == "run"
    assert payload["stage"] == "model_fit"
    assert payload["message"] == "fitting"
    assert payload["completed_units"] == 1.0
    assert payload["total_units"] == 3.0
    assert isinstance(payload["metadata"], dict)
    assert isinstance(payload["metadata"]["path_like"], str)
    assert isinstance(payload["metadata"]["nested"]["k"], str)


def test_emit_progress_populates_timestamp_when_absent() -> None:
    captured: list[ProgressEvent] = []

    def _callback(event: ProgressEvent) -> None:
        captured.append(event)

    emit_progress(
        _callback,
        stage="evaluation",
        message="done",
        completed_units=2.0,
        total_units=2.0,
    )
    assert len(captured) == 1
    payload = captured[0].to_payload()
    assert payload["timestamp_utc"] is not None
    assert payload["stage"] == "evaluation"
    assert payload["message"] == "done"
