from __future__ import annotations

import io
import time

import pytest

from Thesis_ML.observability.console_reporter import (
    LegacyLineReporter,
    RichLiveReporter,
    build_progress_reporter,
    rich_available,
)


class _TtyBuffer(io.StringIO):
    def isatty(self) -> bool:  # pragma: no cover - tiny utility
        return True


def _live_status() -> dict[str, object]:
    return {
        "started_at_utc": "2026-01-01T00:00:00+00:00",
        "last_updated_at_utc": "2026-01-01T00:01:40+00:00",
        "current_phase": "Stage 1 target/scope lock",
        "counts": {
            "runs_planned": 10,
            "runs_started": 6,
            "runs_completed": 5,
            "runs_blocked": 1,
            "runs_failed": 0,
            "runs_dry_run": 0,
        },
        "active_runs": ["r6"],
        "anomaly_counts": {"by_severity": {"info": 1, "warning": 2, "error": 0}},
    }


def test_build_progress_reporter_auto_falls_back_to_legacy_non_tty() -> None:
    reporter = build_progress_reporter(
        stream=io.StringIO(),
        interval_seconds=1.0,
        progress_ui="auto",
        progress_detail="experiment_stage",
    )
    assert isinstance(reporter, LegacyLineReporter)


def test_build_progress_reporter_auto_prefers_rich_for_tty_when_available() -> None:
    if not rich_available():
        pytest.skip("rich is not available in this environment")
    reporter = build_progress_reporter(
        stream=_TtyBuffer(),
        interval_seconds=1.0,
        progress_ui="auto",
        progress_detail="experiment_stage",
    )
    assert isinstance(reporter, RichLiveReporter)


def test_build_progress_reporter_quiet_forces_legacy_non_live_output() -> None:
    reporter = build_progress_reporter(
        stream=_TtyBuffer(),
        interval_seconds=1.0,
        progress_ui="bar",
        progress_detail="experiment_stage",
        quiet=True,
    )
    assert isinstance(reporter, LegacyLineReporter)


def test_legacy_reporter_emits_experiment_finished_once() -> None:
    stream = io.StringIO()
    reporter = LegacyLineReporter(
        stream=stream,
        interval_seconds=300.0,
        quiet=False,
        progress_detail="experiment_stage",
    )
    reporter._last_summary_at = time.monotonic()
    event = {
        "event_name": "experiment_finished",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "status": "completed",
        "phase_name": "Stage 1 target/scope lock",
        "experiment_id": "E01",
        "message": "experiment finished",
    }
    reporter.handle_event(event, _live_status())
    reporter.handle_event(event, _live_status())
    assert stream.getvalue().count("[event:experiment_finished]") == 1


def test_legacy_reporter_stage_completion_feed_is_deduplicated() -> None:
    stream = io.StringIO()
    reporter = LegacyLineReporter(
        stream=stream,
        interval_seconds=300.0,
        quiet=False,
        progress_detail="experiment_stage",
    )
    reporter._last_summary_at = time.monotonic()
    event = {
        "event_name": "progress",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "status": "executed",
        "phase_name": "Stage 1 target/scope lock",
        "experiment_id": "E01",
        "run_id": "run_1",
        "stage": "stage",
        "message": "stage_finished stage model_fit",
        "metadata": {
            "event_type": "stage_finished",
            "stage_key": "model_fit",
            "run_id": "run_1",
        },
    }
    reporter.handle_event(event, _live_status())
    reporter.handle_event(event, _live_status())
    assert stream.getvalue().count("[event:stage_finished]") == 1


def test_legacy_reporter_quiet_suppresses_completion_feed() -> None:
    stream = io.StringIO()
    reporter = LegacyLineReporter(
        stream=stream,
        interval_seconds=0.1,
        quiet=True,
        progress_detail="experiment_stage",
    )
    reporter.handle_event(
        {
            "event_name": "progress",
            "status": "executed",
            "phase_name": "Stage 1 target/scope lock",
            "experiment_id": "E01",
            "run_id": "run_1",
            "stage": "stage",
            "metadata": {"event_type": "stage_finished", "stage_key": "model_fit"},
        },
        _live_status(),
    )
    reporter.handle_event({"event_name": "campaign_finished", "status": "finished"}, _live_status())
    text = stream.getvalue()
    assert "[event:stage_finished]" not in text
    assert "[event:campaign_finished]" in text


def test_legacy_reporter_verbose_surfaces_stage_started_events() -> None:
    stream = io.StringIO()
    reporter = LegacyLineReporter(
        stream=stream,
        interval_seconds=300.0,
        quiet=False,
        progress_detail="verbose",
    )
    reporter._last_summary_at = time.monotonic()
    reporter.handle_event(
        {
            "event_name": "progress",
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "status": "started",
            "phase_name": "Stage 1 target/scope lock",
            "experiment_id": "E01",
            "run_id": "run_1",
            "stage": "stage",
            "message": "stage_started stage model_fit",
            "metadata": {
                "event_type": "stage_started",
                "stage_key": "model_fit",
                "run_id": "run_1",
            },
        },
        _live_status(),
    )
    text = stream.getvalue()
    assert "[event:stage_started]" in text
    assert "stage_key=model_fit" in text


def test_legacy_reporter_verbose_operation_progress_emits_start_milestones_and_finish() -> None:
    stream = io.StringIO()
    reporter = LegacyLineReporter(
        stream=stream,
        interval_seconds=300.0,
        quiet=False,
        progress_detail="verbose",
    )
    reporter._last_summary_at = time.monotonic()

    reporter.handle_event(
        {
            "event_name": "progress",
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "status": "running",
            "phase_name": "Stage 1 target/scope lock",
            "experiment_id": "E01",
            "run_id": "run_1",
            "stage": "section",
            "message": "starting section dataset_selection",
            "completed_units": 0.0,
            "total_units": 10.0,
            "metadata": {"section": "dataset_selection"},
        },
        _live_status(),
    )
    reporter.handle_event(
        {
            "event_name": "progress",
            "timestamp_utc": "2026-01-01T00:00:10+00:00",
            "status": "running",
            "phase_name": "Stage 1 target/scope lock",
            "experiment_id": "E01",
            "run_id": "run_1",
            "stage": "section",
            "message": "completed section feature_cache_build",
            "completed_units": 1.0,
            "total_units": 10.0,
            "metadata": {"section": "feature_cache_build"},
        },
        _live_status(),
    )
    reporter.handle_event(
        {
            "event_name": "progress",
            "timestamp_utc": "2026-01-01T00:00:30+00:00",
            "status": "running",
            "phase_name": "Stage 1 target/scope lock",
            "experiment_id": "E01",
            "run_id": "run_1",
            "stage": "section",
            "message": "completed section model_fit",
            "completed_units": 5.0,
            "total_units": 10.0,
            "metadata": {"section": "model_fit"},
        },
        _live_status(),
    )
    reporter.handle_event(
        {
            "event_name": "progress",
            "timestamp_utc": "2026-01-01T00:01:00+00:00",
            "status": "completed",
            "phase_name": "Stage 1 target/scope lock",
            "experiment_id": "E01",
            "run_id": "run_1",
            "stage": "section",
            "message": "completed section evaluation",
            "completed_units": 10.0,
            "total_units": 10.0,
            "metadata": {"section": "evaluation"},
        },
        _live_status(),
    )

    text = stream.getvalue()
    assert "[operation:start]" in text
    assert text.count("[operation:progress]") >= 2
    assert "[operation:finish]" in text
    assert "percent=50%" in text
