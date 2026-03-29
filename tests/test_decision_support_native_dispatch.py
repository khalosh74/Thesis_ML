from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.experiments.parallel_execution import OfficialRunJob
from Thesis_ML.orchestration import campaign_engine


def _write_index(path: Path) -> None:
    df = pd.DataFrame(
        [
            {"sample_id": "s1", "subject": "sub-001", "session": "ses-01", "task": "emo", "modality": "audio"},
            {"sample_id": "s2", "subject": "sub-002", "session": "ses-01", "task": "emo", "modality": "video"},
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_registry(path: Path) -> None:
    payload = {
        "schema_version": "test",
        "experiments": [
            {
                "experiment_id": "E02",
                "title": "Task pooling experiment",
                "stage": "Stage 1 - Target lock",
                "decision_id": "D01",
                "manipulated_factor": "task",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e02_template",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                        },
                    }
                ],
            },
            {
                "experiment_id": "E03",
                "title": "Modality pooling experiment",
                "stage": "Stage 1 - Target lock",
                "decision_id": "D01",
                "manipulated_factor": "modality",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e03_template",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-002",
                        },
                    }
                ],
            },
            {
                "experiment_id": "E24",
                "title": "Reproducibility rerun audit",
                "stage": "Stage 6 - Robustness analysis",
                "decision_id": "D09",
                "manipulated_factor": "rerun",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e24_template",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                        },
                    }
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _success_payload_for_jobs(jobs: list[OfficialRunJob]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        run_id = str(job.run_id)
        rows.append(
            {
                "order_index": int(job.order_index),
                "run_id": run_id,
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "ended_at_utc": "2026-01-01T00:00:01+00:00",
                "watchdog_result": {
                    "status": "success",
                    "run_payload": {
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
                    },
                },
                "execution_error": None,
            }
        )
    return rows


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


def test_native_dispatch_uses_official_jobs_for_same_phase_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.json"
    index_csv = tmp_path / "index.csv"
    _write_registry(registry_path)
    _write_index(index_csv)

    captured: list[dict[str, Any]] = []

    def _fake_execute_official_jobs(*, jobs: list[OfficialRunJob], max_parallel_runs: int, run_experiment_fn=None):
        captured.append(
            {
                "jobs": jobs,
                "max_parallel_runs": int(max_parallel_runs),
                "run_ids": [str(job.run_id) for job in jobs],
                "experiment_ids": [str(job.run_identity.get("experiment_id")) for job in jobs],
            }
        )
        return _success_payload_for_jobs(jobs)

    monkeypatch.setattr(campaign_engine, "_execute_official_jobs", _fake_execute_official_jobs)

    campaign_engine.run_decision_support_campaign(
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
        max_parallel_runs=3,
        run_experiment_fn=_stub_run_experiment,
    )

    assert captured
    first_dispatch = captured[0]
    assert first_dispatch["max_parallel_runs"] == 3
    assert all(isinstance(job, OfficialRunJob) for job in first_dispatch["jobs"])
    assert {"E02", "E03"} <= set(first_dispatch["experiment_ids"])


def test_e24_dispatch_forces_serial_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.json"
    index_csv = tmp_path / "index.csv"
    _write_registry(registry_path)
    _write_index(index_csv)

    captured_parallelism: list[int] = []

    def _fake_execute_official_jobs(*, jobs: list[OfficialRunJob], max_parallel_runs: int, run_experiment_fn=None):
        if any(str(job.run_identity.get("experiment_id")) == "E24" for job in jobs):
            captured_parallelism.append(int(max_parallel_runs))
        return _success_payload_for_jobs(jobs)

    monkeypatch.setattr(campaign_engine, "_execute_official_jobs", _fake_execute_official_jobs)

    campaign_engine.run_decision_support_campaign(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "outputs",
        experiment_id="E24",
        stage=None,
        run_all=False,
        seed=42,
        n_permutations=0,
        dry_run=False,
        max_parallel_runs=4,
        run_experiment_fn=_stub_run_experiment,
    )

    assert captured_parallelism == [1]
