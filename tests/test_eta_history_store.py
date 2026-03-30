from __future__ import annotations

from pathlib import Path

from Thesis_ML.observability.eta import append_runtime_history, load_runtime_history


def test_runtime_history_append_and_load_roundtrip(tmp_path: Path) -> None:
    history_path = tmp_path / "runtime_history.jsonl"
    append_runtime_history(
        history_path,
        {
            "campaign_id": "c1",
            "run_id": "r1",
            "runtime_key_exact": "exact|a=1",
            "runtime_key_backoff_1": "backoff_1|a=1",
            "runtime_key_backoff_2": "backoff_2|a=1",
            "actual_runtime_seconds": 10.0,
        },
    )
    append_runtime_history(
        history_path,
        {
            "campaign_id": "c2",
            "run_id": "r2",
            "runtime_key_exact": "exact|a=2",
            "runtime_key_backoff_1": "backoff_1|a=2",
            "runtime_key_backoff_2": "backoff_2|a=2",
            "actual_runtime_seconds": 20.0,
        },
    )

    rows = load_runtime_history(history_path)
    assert len(rows) == 2
    assert rows[0]["run_id"] == "r1"
    assert rows[1]["run_id"] == "r2"
