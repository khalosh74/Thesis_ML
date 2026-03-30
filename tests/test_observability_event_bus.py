from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.observability.event_bus import ExecutionEventBus


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_event_bus_writes_jsonl_and_live_status_updates_counts(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaigns" / "c1"
    bus = ExecutionEventBus(campaign_root=campaign_root, campaign_id="c1", keep_recent_events=10)

    bus.emit_event(
        event_name="campaign_started",
        scope="campaign",
        status="running",
        stage="campaign",
        message="start",
        metadata={"experiments_total": 2},
    )
    bus.emit_event(
        event_name="experiment_started",
        scope="experiment",
        status="running",
        stage="campaign",
        experiment_id="E01",
        message="exp start",
    )
    bus.emit_event(
        event_name="run_planned",
        scope="run",
        status="planned",
        stage="campaign",
        experiment_id="E01",
        run_id="r1",
        message="planned",
    )
    bus.emit_event(
        event_name="run_dry_run",
        scope="run",
        status="dry_run",
        stage="campaign",
        experiment_id="E01",
        run_id="r1",
        message="dry",
    )
    bus.emit_event(
        event_name="experiment_finished",
        scope="experiment",
        status="dry_run",
        stage="campaign",
        experiment_id="E01",
        message="exp done",
    )
    bus.emit_event(
        event_name="campaign_finished",
        scope="campaign",
        status="finished",
        stage="campaign",
        message="done",
    )

    events_path = campaign_root / "execution_events.jsonl"
    live_path = campaign_root / "campaign_live_status.json"
    assert events_path.exists()
    assert live_path.exists()

    events = _read_jsonl(events_path)
    assert any(event["event_name"] == "campaign_started" for event in events)
    assert any(event["event_name"] == "campaign_finished" for event in events)

    live_payload = json.loads(live_path.read_text(encoding="utf-8"))
    assert live_payload["campaign_id"] == "c1"
    assert live_payload["status"] == "finished"
    assert live_payload["counts"]["experiments_total"] == 2
    assert live_payload["counts"]["experiments_started"] == 1
    assert live_payload["counts"]["experiments_finished"] == 1
    assert live_payload["counts"]["runs_planned"] == 1
    assert live_payload["counts"]["runs_dry_run"] == 1


def test_event_bus_recent_events_is_bounded(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaigns" / "c2"
    bus = ExecutionEventBus(campaign_root=campaign_root, campaign_id="c2", keep_recent_events=3)
    for idx in range(6):
        bus.emit_event(
            event_name="run_planned",
            scope="run",
            status="planned",
            stage="campaign",
            run_id=f"r{idx}",
            message="planned",
        )
    live_payload = json.loads(
        (campaign_root / "campaign_live_status.json").read_text(encoding="utf-8")
    )
    recent_events = list(live_payload["recent_events"])
    assert len(recent_events) == 3
    assert recent_events[-1]["run_id"] == "r5"
