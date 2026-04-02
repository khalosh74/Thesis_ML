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
        "metrics": {
            "balanced_accuracy": 0.5,
            "macro_f1": 0.5,
            "accuracy": 0.5,
            "n_folds": 2,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_campaign_observability_files_exist_and_include_start_finish_events(tmp_path: Path) -> None:
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
    events_path = campaign_root / "execution_events.jsonl"
    live_path = campaign_root / "campaign_live_status.json"
    assert events_path.exists()
    assert live_path.exists()

    events = _read_jsonl(events_path)
    assert any(event["event_name"] == "campaign_started" for event in events)
    assert any(event["event_name"] == "campaign_finished" for event in events)

    live_payload = json.loads(live_path.read_text(encoding="utf-8"))
    assert live_payload["counts"]["runs_dry_run"] >= 1
    assert live_payload["counts"]["runs_blocked"] >= 1


def test_campaign_observability_reflects_completed_and_blocked_counts(tmp_path: Path) -> None:
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
    live_payload = json.loads(
        (Path(result["campaign_root"]) / "campaign_live_status.json").read_text(encoding="utf-8")
    )
    assert live_payload["counts"]["runs_completed"] >= 1
    assert live_payload["counts"]["runs_blocked"] >= 1


def test_experiment_finished_is_emitted_when_experiment_becomes_terminal_without_duplicates(
    tmp_path: Path,
) -> None:
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

    events = _read_jsonl(Path(result["campaign_root"]) / "execution_events.jsonl")
    exp_finished_indexes = [
        idx
        for idx, event in enumerate(events)
        if str(event.get("event_name")) == "experiment_finished"
        and str(event.get("experiment_id")) == "E01"
    ]
    assert len(exp_finished_indexes) == 1
    exp_finished_idx = exp_finished_indexes[0]

    terminal_run_indexes = [
        idx
        for idx, event in enumerate(events)
        if str(event.get("experiment_id")) == "E01"
        and str(event.get("event_name")) in {"run_finished", "run_failed", "run_blocked", "run_dry_run"}
    ]
    assert terminal_run_indexes
    assert exp_finished_idx > max(terminal_run_indexes)

    phase_finished_indexes = [
        idx
        for idx, event in enumerate(events)
        if str(event.get("event_name")) == "phase_finished"
    ]
    assert phase_finished_indexes
    assert exp_finished_idx < min(phase_finished_indexes)


def test_campaign_observability_writes_stage_evidence_summaries(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    index_csv = tmp_path / "index.csv"
    _write_registry(registry_path)
    _write_index(index_csv)

    def _stub_run_experiment_with_stage_evidence(**kwargs: object) -> dict[str, object]:
        run_id = str(kwargs.get("run_id"))
        reports_root = Path(str(kwargs.get("reports_root")))
        report_dir = reports_root / run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        config_path = report_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "stage_execution": {
                        "policy": {
                            "source": "run_level_compute_policy_bridge_v1",
                            "hardware_mode_requested": "cpu_only",
                            "hardware_mode_effective": "cpu_only",
                            "requested_backend_family": "sklearn_cpu",
                            "effective_backend_family": "sklearn_cpu",
                            "assigned_compute_lane": "cpu",
                            "deterministic_compute": False,
                        },
                        "assignments": [],
                        "telemetry": [
                            {
                                "stage": "model_fit",
                                "status": "executed",
                                "duration_seconds": 1.0,
                                "duration_source": "section_timing",
                                "resource_coverage": "partial",
                                "evidence_quality": "medium",
                                "fallback_used": False,
                                "planned_backend_family": "sklearn_cpu",
                                "observed_backend_family": "sklearn_cpu",
                                "planned_compute_lane": "cpu",
                                "observed_compute_lane": "cpu",
                                "backend_match": True,
                                "lane_match": True,
                                "executor_match": True,
                            }
                        ],
                    }
                },
                indent=2,
            )
            + "\\n",
            encoding="utf-8",
        )
        metrics_path = report_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "balanced_accuracy": 0.5,
                    "macro_f1": 0.5,
                    "accuracy": 0.5,
                    "n_folds": 2,
                },
                indent=2,
            )
            + "\\n",
            encoding="utf-8",
        )
        return {
            "run_id": run_id,
            "report_dir": str(report_dir),
            "config_path": str(config_path),
            "metrics_path": str(metrics_path),
            "fold_metrics_path": str(report_dir / "fold_metrics.csv"),
            "fold_splits_path": str(report_dir / "fold_splits.csv"),
            "predictions_path": str(report_dir / "predictions.csv"),
            "spatial_compatibility_report_path": str(report_dir / "spatial.json"),
            "metrics": {
                "balanced_accuracy": 0.5,
                "macro_f1": 0.5,
                "accuracy": 0.5,
                "n_folds": 2,
            },
        }

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
        run_experiment_fn=_stub_run_experiment_with_stage_evidence,
    )
    campaign_root = Path(result["campaign_root"])
    assert (campaign_root / "stage_execution_summary.json").exists()
    assert (campaign_root / "stage_resource_summary.csv").exists()
    assert (campaign_root / "backend_fallback_summary.json").exists()
    assert (campaign_root / "stage_lease_summary.json").exists()
    assert (campaign_root / "stage_queue_summary.csv").exists()
    assert (campaign_root / "gpu_stage_utilization_summary.json").exists()
