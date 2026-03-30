from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Thesis_ML.orchestration import campaign_engine


def _write_index(path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "task": "emo",
                "modality": "audio",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-02",
                "task": "emo",
                "modality": "audio",
            },
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_registry(path: Path) -> None:
    payload = {
        "schema_version": "test",
        "experiments": [
            {
                "experiment_id": "E01",
                "title": "Target granularity",
                "stage": "Stage 1 - Target lock",
                "decision_id": "D01",
                "manipulated_factor": "target",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "supported_template",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                        },
                    },
                    {
                        "template_id": "blocked_template",
                        "supported": False,
                        "blocked_reason": "intentionally blocked in test",
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                        },
                    },
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _stub_run_experiment(**kwargs: object) -> dict[str, object]:
    run_id = str(kwargs.get("run_id"))
    return {
        "run_id": run_id,
        "report_dir": f"/tmp/{run_id}",
        "config_path": f"/tmp/{run_id}/config.json",
        "metrics_path": f"/tmp/{run_id}/metrics.json",
        "fold_metrics_path": f"/tmp/{run_id}/fold_metrics.csv",
        "fold_splits_path": f"/tmp/{run_id}/fold_splits.csv",
        "predictions_path": f"/tmp/{run_id}/predictions.csv",
        "spatial_compatibility_report_path": f"/tmp/{run_id}/spatial.json",
        "framework_mode": "exploratory",
        "model_cost_tier": "official_fast",
        "projected_runtime_seconds": 25,
        "tuning_enabled": False,
        "stage_timings_seconds": {"total": 5.0},
        "metrics": {
            "balanced_accuracy": 0.5,
            "macro_f1": 0.5,
            "accuracy": 0.5,
            "n_folds": 2,
        },
    }


def test_campaign_eta_files_and_live_status_fields_are_written(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    index_csv = tmp_path / "index.csv"
    _write_registry(registry_path)
    _write_index(index_csv)

    result = campaign_engine.run_decision_support_campaign(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "outputs",
        experiment_id=None,
        stage=None,
        run_all=True,
        seed=42,
        n_permutations=0,
        dry_run=False,
        run_experiment_fn=_stub_run_experiment,
    )

    campaign_root = Path(result["campaign_root"])
    eta_state_path = campaign_root / "eta_state.json"
    calibration_path = campaign_root / "campaign_eta_calibration.json"
    live_status_path = campaign_root / "campaign_live_status.json"
    runtime_history_path = campaign_root.parent / "runtime_history.jsonl"

    assert eta_state_path.exists()
    assert calibration_path.exists()
    assert live_status_path.exists()
    assert runtime_history_path.exists()

    eta_payload = json.loads(eta_state_path.read_text(encoding="utf-8"))
    assert eta_payload["counts"]["runs_planned"] >= 1
    assert eta_payload["counts"]["runs_completed"] >= 1
    assert "campaign_eta" in eta_payload
    assert "phase_eta" in eta_payload

    live_payload = json.loads(live_status_path.read_text(encoding="utf-8"))
    assert "eta_p50_seconds" in live_payload
    assert "eta_p80_seconds" in live_payload
    assert "eta_confidence" in live_payload
    assert "eta_source" in live_payload
