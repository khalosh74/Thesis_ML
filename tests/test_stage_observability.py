from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.experiments.stage_observability import (
    StageBoundaryRecorder,
    merge_stage_resource_attribution,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_stage_boundary_recorder_emits_events_and_observed_evidence(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    recorder = StageBoundaryRecorder(report_dir=report_dir, run_id="r1")
    recorder.stage_started(
        "model_fit",
        metadata={
            "planned_backend_family": "sklearn_cpu",
            "planned_compute_lane": "cpu",
            "planned_executor_id": "model_fit_cpu_reference_v1",
        },
        timestamp_utc="2026-01-01T00:00:00+00:00",
    )
    recorder.stage_finished(
        "model_fit",
        metadata={"observed_backend_family": "sklearn_cpu", "observed_compute_lane": "cpu"},
        timestamp_utc="2026-01-01T00:00:04+00:00",
        status="executed",
    )

    stage_events_path = report_dir / "stage_events.jsonl"
    stage_observed_path = report_dir / "stage_observed_evidence.json"
    assert stage_events_path.exists()
    assert stage_observed_path.exists()

    events = _read_jsonl(stage_events_path)
    assert len(events) >= 2
    assert events[0]["event_type"] == "stage_started"
    assert events[-1]["event_type"] == "stage_finished"

    observed_payload = json.loads(stage_observed_path.read_text(encoding="utf-8"))
    assert observed_payload["schema_version"] == "stage-observed-evidence-v1"
    rows = {
        str(row.get("stage_key")): row
        for row in observed_payload.get("stages", [])
        if isinstance(row, dict)
    }
    assert rows["model_fit"]["observed_status"] == "executed"
    assert rows["model_fit"]["started_at_utc"] == "2026-01-01T00:00:00+00:00"
    assert rows["model_fit"]["ended_at_utc"] == "2026-01-01T00:00:04+00:00"


def test_stage_resource_attribution_from_process_samples(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    recorder = StageBoundaryRecorder(report_dir=report_dir, run_id="r2")
    recorder.stage_started("model_fit", timestamp_utc="2026-01-01T00:00:00+00:00")
    recorder.stage_finished("model_fit", timestamp_utc="2026-01-01T00:00:03+00:00")

    samples_path = report_dir / "process_samples.jsonl"
    rows = [
        {
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "cpu_percent": 10.0,
            "rss_mb": 100.0,
            "vms_mb": 200.0,
            "num_threads": 4,
            "read_bytes": 100,
            "write_bytes": 50,
        },
        {
            "timestamp_utc": "2026-01-01T00:00:01+00:00",
            "cpu_percent": 20.0,
            "rss_mb": 140.0,
            "vms_mb": 220.0,
            "num_threads": 6,
            "read_bytes": 220,
            "write_bytes": 110,
        },
        {
            "timestamp_utc": "2026-01-01T00:00:03+00:00",
            "cpu_percent": 15.0,
            "rss_mb": 120.0,
            "vms_mb": 210.0,
            "num_threads": 5,
            "read_bytes": 300,
            "write_bytes": 150,
        },
    ]
    samples_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    attribution = merge_stage_resource_attribution(
        report_dir=report_dir,
        process_profile_summary={
            "sample_interval_seconds": 1.0,
            "gpu_sampling_enabled": False,
            "gpu_sampling_reason": "torch_not_available",
        },
    )

    summaries = attribution["stage_resource_summaries"]
    assert "model_fit" in summaries
    model_fit = summaries["model_fit"]
    assert model_fit["sample_count"] == 3
    assert model_fit["mean_cpu_percent"] == 15.0
    assert model_fit["peak_rss_mb"] == 140.0
    assert model_fit["read_bytes_delta"] == 200
    assert model_fit["write_bytes_delta"] == 100
    assert model_fit["gpu_sample_count"] == 0
    assert model_fit["resource_coverage"] in {"high", "partial"}

    observed_payload = json.loads((report_dir / "stage_observed_evidence.json").read_text("utf-8"))
    observed_rows = {
        str(row.get("stage_key")): row
        for row in observed_payload.get("stages", [])
        if isinstance(row, dict)
    }
    assert "resource_summary" in observed_rows["model_fit"]


def test_stage_resource_attribution_marks_sparse_coverage_honestly(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    recorder = StageBoundaryRecorder(report_dir=report_dir, run_id="r3")
    recorder.stage_started("model_fit", timestamp_utc="2026-01-01T00:00:00+00:00")
    recorder.stage_finished("model_fit", timestamp_utc="2026-01-01T00:00:00.050000+00:00")

    (report_dir / "process_samples.jsonl").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-01-01T00:00:10+00:00",
                "cpu_percent": 12.0,
                "rss_mb": 64.0,
                "vms_mb": 96.0,
                "num_threads": 2,
                "read_bytes": 1,
                "write_bytes": 1,
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    attribution = merge_stage_resource_attribution(
        report_dir=report_dir,
        process_profile_summary={"sample_interval_seconds": 10.0},
    )
    model_fit = attribution["stage_resource_summaries"]["model_fit"]
    assert model_fit["sample_count"] == 0
    assert model_fit["resource_coverage"] == "none"
    assert model_fit["evidence_quality"] == "low"
