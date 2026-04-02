from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.orchestration.reporting import write_campaign_execution_report


def test_write_campaign_execution_report_uses_existing_artifacts(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaigns" / "c1"
    campaign_root.mkdir(parents=True, exist_ok=True)

    live_status = {
        "campaign_id": "c1",
        "status": "finished",
        "started_at_utc": "2026-01-01T00:00:00+00:00",
        "last_updated_at_utc": "2026-01-01T00:10:00+00:00",
        "current_phase": "Stage 1 target/scope lock",
        "current_phase_status": "completed",
        "counts": {
            "runs_planned": 2,
            "runs_dispatched": 2,
            "runs_started": 2,
            "runs_completed": 1,
            "runs_failed": 1,
            "runs_blocked": 0,
            "runs_dry_run": 0,
        },
        "active_runs": [],
        "blocked_experiments": ["E23"],
        "failed_runs": ["run-2"],
        "anomaly_counts": {"total": 1, "by_severity": {"error": 1}},
        "latest_anomaly": {"code": "RUN_FAILED", "severity": "error"},
    }
    (campaign_root / "campaign_live_status.json").write_text(
        json.dumps(live_status, indent=2) + "\n",
        encoding="utf-8",
    )
    (campaign_root / "eta_state.json").write_text(
        json.dumps({"campaign_eta": {"eta_p50_seconds": 5.0}, "phase_eta": {}}, indent=2) + "\n",
        encoding="utf-8",
    )
    (campaign_root / "campaign_eta_calibration.json").write_text(
        json.dumps({"counts_by_eta_source": {"projected_runtime": 1}}, indent=2) + "\n",
        encoding="utf-8",
    )
    (campaign_root / "campaign_anomaly_report.json").write_text(
        json.dumps({"anomaly_counts": {"total": 1}}, indent=2) + "\n",
        encoding="utf-8",
    )
    (campaign_root / "stage1_lock.json").write_text("{}", encoding="utf-8")
    (campaign_root / "stage1_target_lock_summary.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (campaign_root / "stage1_target_lock_summary.md").write_text("# stage\n", encoding="utf-8")
    (campaign_root / "stage1_target_lock_summary_decision_notes.md").write_text(
        "# notes\n",
        encoding="utf-8",
    )
    (campaign_root / "preflight_reviews").mkdir(parents=True, exist_ok=True)
    (campaign_root / "preflight_reviews" / "confirmatory_selection_bundle.json").write_text(
        json.dumps({"scope_id": "confirmatory_scope_v1"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (campaign_root / "preflight_reviews" / "frozen_confirmatory_outputs.json").write_text(
        json.dumps(
            {
                "registry": "configs/generated/frozen_confirmatory_registry_c1.json",
                "manifest": "configs/generated/frozen_confirmatory_manifest_c1.json",
                "report": "configs/generated/frozen_confirmatory_report_c1.md",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    md_path, json_path = write_campaign_execution_report(campaign_root=campaign_root, campaign_id="c1")

    assert md_path.exists()
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["campaign_id"] == "c1"
    assert payload["final_execution_status"]["status"] == "finished"
    assert "key_generated_artifacts" in payload
    assert "stage1_target_lock_summary.csv" in payload["key_generated_artifacts"]["stage_summaries"]
    assert str(payload["key_generated_artifacts"]["confirmatory_selection_bundle"]).replace(
        "\\", "/"
    ).endswith("preflight_reviews/confirmatory_selection_bundle.json")
    assert isinstance(payload["key_generated_artifacts"]["frozen_confirmatory_artifacts"], dict)

    markdown = md_path.read_text(encoding="utf-8")
    assert "## Campaign overview" in markdown
    assert "## Final execution status" in markdown
