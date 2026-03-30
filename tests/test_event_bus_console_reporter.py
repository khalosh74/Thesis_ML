from __future__ import annotations

import io
import time
from pathlib import Path

from Thesis_ML.observability.console_reporter import ConsoleReporter
from Thesis_ML.observability.event_bus import ExecutionEventBus


def test_event_bus_calls_console_reporter_for_important_events(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaigns" / "c1"
    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream, interval_seconds=60.0, quiet=False)
    reporter._last_summary_at = time.monotonic()
    bus = ExecutionEventBus(
        campaign_root=campaign_root,
        campaign_id="c1",
        console_reporter=reporter,
    )

    bus.emit_event(
        event_name="campaign_started",
        scope="campaign",
        status="running",
        stage="campaign",
        message="campaign started",
    )

    output = stream.getvalue()
    assert "[event:campaign_started]" in output
