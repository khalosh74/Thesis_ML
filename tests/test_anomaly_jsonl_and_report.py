from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.observability.anomalies import AnomalyEngine


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_anomaly_jsonl_and_final_report_written(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaigns" / "c1"
    engine = AnomalyEngine(campaign_root=campaign_root, campaign_id="c1")

    engine.ingest_event(
        {
            "event_name": "run_failed",
            "status": "failed",
            "phase_name": "Stage 1 target/scope lock",
            "experiment_id": "E01",
            "run_id": "run_fail",
            "metadata": {"error": "boom"},
        }
    )
    engine.inspect_terminal_run(
        {
            "phase_name": "Stage 1 target/scope lock",
            "experiment_id": "E01",
            "run_id": "run_fail",
            "projected_runtime_seconds": 5.0,
            "actual_runtime_seconds": 20.0,
            "n_permutations": 0,
            "tuning_enabled": False,
            "stage_timings_seconds": {"total": 20.0},
        }
    )
    report = engine.finalize()

    anomalies_path = campaign_root / "anomalies.jsonl"
    report_path = campaign_root / "campaign_anomaly_report.json"
    assert anomalies_path.exists()
    assert report_path.exists()

    rows = _read_jsonl(anomalies_path)
    assert rows
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["campaign_id"] == "c1"
    assert report_payload["anomaly_counts"]["total"] == len(rows)
    assert report_payload == report
    assert "RUN_FAILED" in report_payload["anomaly_counts"]["by_code"]
