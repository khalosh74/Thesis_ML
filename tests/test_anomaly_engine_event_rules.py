from __future__ import annotations

from pathlib import Path

from Thesis_ML.observability.anomalies import AnomalyEngine


def test_event_rules_emit_timeout_and_failed_and_lock_phase_failure(tmp_path: Path) -> None:
    engine = AnomalyEngine(campaign_root=tmp_path / "campaigns" / "c1", campaign_id="c1")

    timeout_rows = engine.ingest_event(
        {
            "event_name": "run_timeout",
            "status": "timeout",
            "phase_name": "Stage 1 target/scope lock",
            "experiment_id": "E01",
            "run_id": "run_timeout",
            "metadata": {"error": "watchdog_timeout"},
        }
    )
    codes = {row["code"] for row in timeout_rows}
    assert "RUN_TIMEOUT" in codes

    failed_rows = engine.ingest_event(
        {
            "event_name": "run_failed",
            "status": "failed",
            "phase_name": "Stage 1 target/scope lock",
            "experiment_id": "E01",
            "run_id": "run_failed",
            "metadata": {"error": "failed"},
        }
    )
    failed_codes = {row["code"] for row in failed_rows}
    assert "RUN_FAILED" in failed_codes
    assert "LOCK_PHASE_HAS_FAILURES" in failed_codes


def test_event_rules_deduplicate_blocked_special_experiment_warning(tmp_path: Path) -> None:
    engine = AnomalyEngine(campaign_root=tmp_path / "campaigns" / "c2", campaign_id="c2")

    first = engine.ingest_event(
        {
            "event_name": "run_blocked",
            "status": "blocked",
            "phase_name": "Context robustness",
            "experiment_id": "E23",
            "run_id": "run_a",
            "metadata": {"blocked_reason": "no omitted sessions found"},
        }
    )
    second = engine.ingest_event(
        {
            "event_name": "run_blocked",
            "status": "blocked",
            "phase_name": "Context robustness",
            "experiment_id": "E23",
            "run_id": "run_b",
            "metadata": {"blocked_reason": "no omitted sessions found"},
        }
    )

    first_codes = [row["code"] for row in first]
    second_codes = [row["code"] for row in second]
    assert "UNSUPPORTED_SPECIAL_EXPERIMENT_BLOCKED" in first_codes
    assert "UNSUPPORTED_SPECIAL_EXPERIMENT_BLOCKED" not in second_codes


def test_event_rules_lock_phase_blocked_ratio_and_phase_with_no_completions(tmp_path: Path) -> None:
    engine = AnomalyEngine(campaign_root=tmp_path / "campaigns" / "c3", campaign_id="c3")
    phase_name = "Stage 2 split/transfer lock"
    for idx in range(20):
        engine.ingest_event(
            {
                "event_name": "run_planned",
                "status": "planned",
                "phase_name": phase_name,
                "experiment_id": "E04",
                "run_id": f"planned_{idx}",
                "metadata": {"supported": True},
            }
        )
    blocked_rows = engine.ingest_event(
        {
            "event_name": "run_blocked",
            "status": "blocked",
            "phase_name": phase_name,
            "experiment_id": "E04",
            "run_id": "blocked_1",
            "metadata": {"blocked_reason": "unsupported"},
        }
    )
    assert any(row["code"] == "HIGH_BLOCKED_RATIO_LOCK_PHASE" for row in blocked_rows)
    assert any(row["severity"] == "warning" for row in blocked_rows)

    finished_rows = engine.ingest_event(
        {
            "event_name": "phase_finished",
            "status": "blocked",
            "phase_name": phase_name,
            "experiment_id": None,
            "run_id": None,
            "metadata": {},
        }
    )
    assert any(row["code"] == "PHASE_WITH_NO_COMPLETIONS" for row in finished_rows)
