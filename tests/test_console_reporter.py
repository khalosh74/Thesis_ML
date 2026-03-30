from __future__ import annotations

import io
import time

from Thesis_ML.observability.console_reporter import ConsoleReporter


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


def test_console_reporter_prints_immediate_phase_and_campaign_transitions() -> None:
    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream, interval_seconds=300.0, quiet=False)
    reporter._last_summary_at = time.monotonic()

    for event_name in ("campaign_started", "phase_started", "phase_finished", "campaign_finished"):
        reporter.handle_event(
            {
                "event_name": event_name,
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "status": "running" if event_name.endswith("started") else "finished",
                "phase_name": "Stage 1 target/scope lock",
                "message": "ok",
            },
            _live_status(),
        )

    text = stream.getvalue()
    assert "[event:campaign_started]" in text
    assert "[event:phase_started]" in text
    assert "[event:phase_finished]" in text
    assert "[event:campaign_finished]" in text


def test_console_reporter_heartbeat_is_throttled(monkeypatch) -> None:
    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream, interval_seconds=10.0, quiet=False)
    times = iter([1.0, 2.0, 15.0])
    monkeypatch.setattr("Thesis_ML.observability.console_reporter.time.monotonic", lambda: next(times))
    reporter._last_summary_at = 0.0

    event = {"event_name": "run_started", "status": "running"}
    reporter.handle_event(event, _live_status())
    reporter.handle_event(event, _live_status())
    reporter.handle_event(event, _live_status())

    assert stream.getvalue().count("[progress]") == 1


def test_console_reporter_quiet_mode_suppresses_heartbeat_but_keeps_final_summary() -> None:
    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream, interval_seconds=0.1, quiet=True)
    reporter.handle_event({"event_name": "run_started", "status": "running"}, _live_status())
    reporter.handle_event({"event_name": "campaign_finished", "status": "finished"}, _live_status())

    text = stream.getvalue()
    assert "[progress]" not in text
    assert "[event:campaign_finished]" in text


def test_console_reporter_summary_line_includes_eta_fields_when_present() -> None:
    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream, interval_seconds=15.0, quiet=False)
    live_status = _live_status()
    live_status["eta_p50_seconds"] = 60.0
    live_status["eta_p80_seconds"] = 120.0
    reporter.emit_summary_line(live_status)

    text = stream.getvalue()
    assert "eta_p50=01:00" in text
    assert "eta_p80=02:00" in text


def test_console_reporter_suppresses_repeated_dry_run_blocked_lines_per_experiment() -> None:
    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream, interval_seconds=300.0, quiet=False)
    reporter._last_summary_at = time.monotonic()

    for idx in range(8):
        reporter.handle_event(
            {
                "event_name": "run_blocked",
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "status": "blocked",
                "phase_name": "Context robustness",
                "experiment_id": "E23",
                "run_id": f"run_{idx}",
                "message": "run blocked",
                "metadata": {"dry_run": True, "blocked_reason": "unsupported"},
            },
            _live_status(),
        )

    text = stream.getvalue()
    blocked_lines = [line for line in text.splitlines() if "[event:run_blocked]" in line]
    assert len(blocked_lines) == 4
    assert any("suppressed after 3" in line for line in blocked_lines)
