from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Thesis_ML.cli.protocol_runner import main as protocol_main


def test_execution_status_payload_contains_counts(tmp_path: Path, monkeypatch) -> None:
    index_csv = tmp_path / "dataset_index.csv"
    pd.DataFrame(
        [
            {"subject": "sub-001"},
            {"subject": "sub-002"},
        ]
    ).to_csv(index_csv, index=False)

    data_root = tmp_path / "Data"
    cache_dir = tmp_path / "cache"
    reports_root = tmp_path / "reports" / "experiments"

    monkeypatch.setenv("THESIS_ML_INDEX_CSV", str(index_csv))
    monkeypatch.setenv("THESIS_ML_DATA_ROOT", str(data_root))
    monkeypatch.setenv("THESIS_ML_CACHE_DIR", str(cache_dir))

    exit_code = protocol_main(
        [
            "--protocol-alias",
            "protocol.thesis_canonical_default",
            "--all-suites",
            "--reports-root",
            str(reports_root),
            "--dry-run",
        ]
    )
    assert exit_code == 0

    protocol_output_dir = reports_root / "protocol_runs" / "thesis-canonical-nested__2.0.0"
    assert (protocol_output_dir / "execution_status.json").exists()

    payload = json.loads(
        (protocol_output_dir / "execution_status.json").read_text(encoding="utf-8")
    )

    expected_keys = (
        "n_runs",
        "n_planned",
        "n_success",
        "n_failed",
        "n_timed_out",
        "n_skipped_due_to_policy",
        "n_completed",
    )
    for key in expected_keys:
        assert key in payload, f"Missing key {key} in execution_status payload"
        assert isinstance(payload[key], int)

    assert all(run["status"] == "planned" for run in payload["runs"])
