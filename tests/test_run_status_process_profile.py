from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.experiments.execution_policy import write_run_status


def test_write_run_status_persists_process_profile_fields(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    status_path = write_run_status(
        report_dir,
        run_id="run_001",
        status="running",
        process_profile_summary={"sample_count": 3, "peak_rss_mb": 12.5},
        process_profile_artifacts={
            "process_samples_path": str((report_dir / "process_samples.jsonl").resolve()),
            "process_profile_summary_path": str(
                (report_dir / "process_profile_summary.json").resolve()
            ),
        },
    )
    assert status_path.exists()
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run_001"
    assert payload["status"] == "running"
    assert payload["process_profile_summary"]["sample_count"] == 3
    assert payload["process_profile_artifacts"]["process_samples_path"].endswith(
        "process_samples.jsonl"
    )
