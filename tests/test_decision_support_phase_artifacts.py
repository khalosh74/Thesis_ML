from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Thesis_ML.orchestration import campaign_engine


def _write_index(path: Path) -> None:
    df = pd.DataFrame(
        [
            {"sample_id": "s1", "subject": "sub-001", "session": "ses-01", "task": "emo", "modality": "audio"},
            {"sample_id": "s2", "subject": "sub-002", "session": "ses-01", "task": "recog", "modality": "video"},
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
                        "template_id": "e01_template",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                        },
                    }
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
        "metrics": {
            "balanced_accuracy": 0.6,
            "macro_f1": 0.58,
            "accuracy": 0.62,
            "n_folds": 2,
        },
    }


def test_phase_artifacts_written_with_required_minimum_fields(tmp_path: Path) -> None:
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
        dry_run=True,
        run_experiment_fn=_stub_run_experiment,
    )
    campaign_root = Path(result["campaign_root"])
    required = [
        "stage1_lock.json",
        "stage2_lock.json",
        "stage3_lock.json",
        "stage4_lock.json",
        "final_confirmatory_pipeline.json",
        "phase_skip_summary.json",
    ]
    for filename in required:
        path = campaign_root / filename
        assert path.exists(), filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["campaign_id"] == result["campaign_id"]
        assert "generated_at" in payload
        assert "phase_name" in payload
        assert "status" in payload
