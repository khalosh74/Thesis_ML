from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.observability.eta import EtaEstimator


def test_eta_calibration_excludes_nonmeasured_terminal_runs(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaigns" / "c1"
    campaign_root.mkdir(parents=True, exist_ok=True)
    history_path = campaign_root.parent / "runtime_history.jsonl"
    estimator = EtaEstimator(
        campaign_root=campaign_root,
        campaign_id="c1",
        history_path=history_path,
    )

    planning = {
        "phase_name": "Confirmatory",
        "experiment_id": "E16",
        "framework_mode": "confirmatory",
        "model_cost_tier": "official_fast",
        "feature_space": "whole_brain_masked",
        "preprocessing_strategy": "none",
        "dimensionality_strategy": "none",
        "tuning_enabled": False,
        "cv_mode": "within_subject_loso_session",
        "n_permutations": 100,
        "projected_runtime_seconds": 20.0,
    }

    estimator.register_planned_run({**planning, "run_id": "measured"})
    estimator.mark_run_finished(
        {
            **planning,
            "run_id": "measured",
            "actual_runtime_seconds": 12.0,
        }
    )
    estimator.mark_run_terminal_nonmeasured({**planning, "run_id": "blocked"})
    estimator.mark_run_terminal_nonmeasured({**planning, "run_id": "dry_run"})

    payload = estimator.finalize()
    calibration_path = campaign_root / "campaign_eta_calibration.json"
    assert calibration_path.exists()
    saved = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert saved == payload
    assert payload["total_actual_seconds_for_measured_runs"] == 12.0
    assert payload["total_estimated_seconds_for_measured_runs"] > 0.0
    assert sum(payload["counts_by_eta_source"].values()) == 1
