from __future__ import annotations

from pathlib import Path

from Thesis_ML.observability.anomalies import AnomalyEngine


def test_terminal_rules_emit_expected_anomalies(tmp_path: Path) -> None:
    for feature_space in ("roi_mean_predefined", "roi_masked_predefined"):
        engine = AnomalyEngine(
            campaign_root=tmp_path / "campaigns" / f"c1_{feature_space}",
            campaign_id=f"c1_{feature_space}",
        )
        rows = engine.inspect_terminal_run(
            {
                "phase_name": "Stage 4 representation/preprocessing lock",
                "experiment_id": "E10",
                "run_id": f"run_terminal_{feature_space}",
                "feature_space": feature_space,
                "roi_spec_path": str(tmp_path / "missing_roi_spec.json"),
                "tuning_enabled": True,
                "n_permutations": 100,
                "projected_runtime_seconds": 10.0,
                "eta_p80_seconds": 12.0,
                "actual_runtime_seconds": 30.0,
                "stage_timings_seconds": {"total": 30.0, "dataset_selection": 1.0},
                "process_profile_summary": {"peak_rss_mb": 9000.0},
            }
        )
        codes = {row["code"] for row in rows}
        assert "RUN_EXCEEDS_ETA_P80" in codes
        assert "HIGH_MEMORY_PEAK" in codes
        assert "TUNING_EXPECTED_BUT_MISSING" in codes
        assert "PERMUTATION_EXPECTED_BUT_MISSING" in codes
        assert "ROI_FEATURE_SPACE_PROXY_WARNING" in codes
