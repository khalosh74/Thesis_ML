from __future__ import annotations

import io
import time

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
    reporter = build_progress_reporter(
        stream=_TtyBuffer(),
        interval_seconds=1.0,
        progress_ui="auto",
        progress_detail="experiment_stage",
    )
    if rich_available():
        assert isinstance(reporter, RichLiveReporter)
    else:
        assert isinstance(reporter, LegacyLineReporter)


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
