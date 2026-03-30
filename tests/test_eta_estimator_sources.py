from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.observability.eta import EtaEstimator, build_runtime_keys


def _make_estimator(
    tmp_path: Path,
    *,
    history_rows: list[dict[str, object]] | None = None,
    runtime_profile_summary: dict[str, object] | None = None,
) -> EtaEstimator:
    campaign_root = tmp_path / "campaigns" / "c1"
    campaign_root.mkdir(parents=True, exist_ok=True)
    history_path = campaign_root.parent / "runtime_history.jsonl"
    if history_rows:
        with history_path.open("w", encoding="utf-8") as handle:
            for row in history_rows:
                handle.write(json.dumps(row) + "\n")
    runtime_profile_path = None
    if runtime_profile_summary is not None:
        runtime_profile_path = campaign_root / "runtime_profile_summary.json"
        runtime_profile_path.write_text(
            json.dumps(runtime_profile_summary, indent=2) + "\n",
            encoding="utf-8",
        )
    return EtaEstimator(
        campaign_root=campaign_root,
        campaign_id="c1",
        history_path=history_path,
        runtime_profile_summary_path=runtime_profile_path,
    )


def test_eta_estimator_prefers_history_exact_then_live_exact(tmp_path: Path) -> None:
    metadata = {
        "phase_name": "Confirmatory",
        "experiment_id": "E16",
        "framework_mode": "confirmatory",
        "model_cost_tier": "official_fast",
        "feature_space": "whole_brain_masked",
        "preprocessing_strategy": "none",
        "dimensionality_strategy": "none",
        "tuning_enabled": False,
        "cv_mode": "within_subject_loso_session",
        "n_permutations": 0,
    }
    keys = build_runtime_keys(metadata)
    estimator = _make_estimator(
        tmp_path,
        history_rows=[
            {
                "runtime_key_exact": keys["exact"],
                "runtime_key_backoff_1": keys["backoff_1"],
                "runtime_key_backoff_2": keys["backoff_2"],
                "actual_runtime_seconds": 30.0,
            }
        ],
    )

    estimator.register_planned_run({**metadata, "run_id": "run_a"})
    payload = estimator.current_eta_payload()
    assert payload["campaign_eta"]["eta_source"] == "history_exact"

    estimator.mark_run_finished({**metadata, "run_id": "run_a", "actual_runtime_seconds": 12.0})
    estimator.register_planned_run({**metadata, "run_id": "run_b"})
    payload_live = estimator.current_eta_payload()
    assert payload_live["campaign_eta"]["eta_source"] == "live_exact"


def test_eta_estimator_history_backoff_runtime_profile_and_projected_fallbacks(
    tmp_path: Path,
) -> None:
    base_metadata = {
        "phase_name": "Confirmatory",
        "framework_mode": "confirmatory",
        "model_cost_tier": "official_fast",
        "feature_space": "whole_brain_masked",
        "preprocessing_strategy": "none",
        "dimensionality_strategy": "none",
        "tuning_enabled": False,
        "cv_mode": "within_subject_loso_session",
        "n_permutations": 0,
    }
    exact_mismatch_keys = build_runtime_keys({**base_metadata, "experiment_id": "E99"})
    estimator_backoff = _make_estimator(
        tmp_path / "backoff",
        history_rows=[
            {
                "runtime_key_exact": exact_mismatch_keys["exact"],
                "runtime_key_backoff_1": exact_mismatch_keys["backoff_1"],
                "runtime_key_backoff_2": exact_mismatch_keys["backoff_2"],
                "actual_runtime_seconds": 45.0,
            }
        ],
    )
    estimator_backoff.register_planned_run(
        {
            **base_metadata,
            "experiment_id": "E16",
            "run_id": "run_backoff",
            "projected_runtime_seconds": 90.0,
        }
    )
    payload_backoff = estimator_backoff.current_eta_payload()
    assert payload_backoff["campaign_eta"]["eta_source"] == "history_backoff"

    estimator_profile = _make_estimator(
        tmp_path / "profile",
        runtime_profile_summary={
            "cohort_estimates": [],
            "phase_estimates": {
                "confirmatory": {
                    "estimated_total_seconds": 100.0,
                    "n_planned_runs": 2,
                }
            },
            "model_estimates": {},
            "estimated_total_wall_time_seconds": 100.0,
        },
    )
    estimator_profile.register_planned_run(
        {
            **base_metadata,
            "experiment_id": "E16",
            "run_id": "run_profile",
        }
    )
    payload_profile = estimator_profile.current_eta_payload()
    assert payload_profile["campaign_eta"]["eta_source"] == "runtime_profile_phase"
    assert payload_profile["campaign_eta"]["eta_p50_seconds"] == 50.0

    estimator_projected = _make_estimator(tmp_path / "projected")
    estimator_projected.register_planned_run(
        {
            **base_metadata,
            "experiment_id": "E16",
            "run_id": "run_projected",
            "projected_runtime_seconds": 22.0,
        }
    )
    payload_projected = estimator_projected.current_eta_payload()
    assert payload_projected["campaign_eta"]["eta_source"] == "projected_runtime"
    assert payload_projected["campaign_eta"]["eta_p50_seconds"] == 22.0
