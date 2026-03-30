from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from Thesis_ML.observability.process_sampler import ProcessSampler


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_process_sampler_writes_samples_and_summary(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(0.4)",
        ]
    )
    sampler = ProcessSampler(
        pid=int(process.pid),
        report_dir=report_dir,
        sample_interval_seconds=0.05,
    )
    sampler.start()
    process.wait(timeout=3.0)
    sampler.stop()
    summary = sampler.finalize(
        wall_clock_elapsed_seconds=0.4,
        terminated_by_watchdog=False,
        termination_method="normal_exit",
        child_pid=int(process.pid),
    )

    samples_path = report_dir / "process_samples.jsonl"
    summary_path = report_dir / "process_profile_summary.json"
    assert samples_path.exists()
    assert summary_path.exists()

    samples = _read_jsonl(samples_path)
    assert len(samples) >= 1
    assert all("timestamp_utc" in row for row in samples)
    assert all("pid" in row for row in samples)
    assert all("cpu_percent" in row for row in samples)
    assert all("rss_mb" in row for row in samples)

    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded_summary["sample_count"] >= 1
    assert loaded_summary["child_pid"] == int(process.pid)
    assert loaded_summary["termination_method"] == "normal_exit"
    assert loaded_summary["sampling_enabled"] in {True, False}
    assert summary["sample_count"] == loaded_summary["sample_count"]
