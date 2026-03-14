from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from Thesis_ML.artifacts.registry import ARTIFACT_TYPE_EXPERIMENT_REPORT, list_artifacts_for_run
from Thesis_ML.orchestration import decision_support as orchestrator


def _write_index_csv(path: Path) -> None:
    df = pd.DataFrame(
        [
            {"sample_id": "s1", "subject": "sub-001", "task": "passive", "modality": "audio"},
            {"sample_id": "s2", "subject": "sub-001", "task": "emo", "modality": "video"},
            {"sample_id": "s3", "subject": "sub-002", "task": "passive", "modality": "audio"},
            {"sample_id": "s4", "subject": "sub-002", "task": "emo", "modality": "video"},
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
                "title": "Target granularity experiment",
                "stage": "Stage 1 - Target lock",
                "decision_id": "D01",
                "manipulated_factor": "Target definition",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "target_compare",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                        },
                        "expand": {"subject": "subjects"},
                    }
                ],
            },
            {
                "experiment_id": "E07",
                "title": "Class weighting experiment",
                "stage": "Stage 3 - Model lock",
                "decision_id": "D04",
                "manipulated_factor": "Weighting strategy",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "weighting",
                        "supported": False,
                        "unsupported_reason": "weighting flag not implemented",
                    }
                ],
            },
        ],
    }
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _write_segment_registry(path: Path) -> None:
    payload = {
        "schema_version": "test",
        "experiments": [
            {
                "experiment_id": "E02",
                "title": "Segment execution experiment",
                "stage": "Stage 1 - Target lock",
                "decision_id": "D01",
                "manipulated_factor": "Section range",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "segment_variant",
                        "supported": True,
                        "start_section": "feature_matrix_load",
                        "end_section": "evaluation",
                        "base_artifact_id": "feature_cache_base_123",
                        "reuse_policy": "require_explicit_base",
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "loso_session",
                        },
                    }
                ],
            }
        ],
    }
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _stub_run_experiment(**kwargs: object) -> dict[str, object]:
    run_id = str(kwargs["run_id"])
    reports_root = Path(kwargs["reports_root"])
    report_dir = reports_root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": run_id,
        "report_dir": str(report_dir),
        "config_path": str(report_dir / "config.json"),
        "metrics_path": str(report_dir / "metrics.json"),
        "fold_metrics_path": str(report_dir / "fold_metrics.csv"),
        "fold_splits_path": str(report_dir / "fold_splits.csv"),
        "predictions_path": str(report_dir / "predictions.csv"),
        "spatial_compatibility_report_path": str(report_dir / "spatial_compatibility_report.json"),
        "metrics": {
            "accuracy": 0.61,
            "balanced_accuracy": 0.58,
            "macro_f1": 0.57,
            "n_folds": 3,
        },
    }


def test_expand_experiment_variants_ordered_pairs() -> None:
    experiment = {
        "experiment_id": "E05",
        "variant_templates": [
            {
                "template_id": "pairs",
                "supported": True,
                "params": {
                    "target": "coarse_affect",
                    "model": "ridge",
                    "cv": "frozen_cross_person_transfer",
                },
                "expand": {"train_test_pair": "ordered_subject_pairs"},
            }
        ],
    }
    dataset_scope = {
        "subjects": ["sub-001", "sub-002"],
        "tasks": ["passive"],
        "modalities": ["audio"],
        "ordered_subject_pairs": [("sub-001", "sub-002"), ("sub-002", "sub-001")],
        "models_linear": ["ridge", "logreg", "linearsvc"],
    }

    variants, warnings = orchestrator._expand_experiment_variants(
        experiment=experiment,
        dataset_scope=dataset_scope,
    )
    assert warnings == []
    assert len(variants) == 2
    pairs = {(row["params"]["train_subject"], row["params"]["test_subject"]) for row in variants}
    assert pairs == {("sub-001", "sub-002"), ("sub-002", "sub-001")}


def test_campaign_writes_exports_and_marks_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = tmp_path / "decision_support_registry.json"
    index_csv = tmp_path / "dataset_index.csv"
    _write_registry(registry_path)
    _write_index_csv(index_csv)
    monkeypatch.setattr(orchestrator, "run_experiment", _stub_run_experiment)

    result = orchestrator.run_decision_support_campaign(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "artifacts" / "decision_support",
        experiment_id=None,
        stage=None,
        run_all=True,
        seed=42,
        n_permutations=0,
        dry_run=False,
    )

    campaign_root = Path(result["campaign_root"])
    assert (campaign_root / "run_log_export.csv").exists()
    assert (campaign_root / "decision_support_summary.csv").exists()
    assert (campaign_root / "decision_recommendations.md").exists()
    assert (campaign_root / "study_review_summary.json").exists()
    assert (campaign_root / "stage1_target_lock_summary.csv").exists()
    assert (campaign_root / "stage3_model_lock_summary.csv").exists()

    summary_df = pd.read_csv(campaign_root / "decision_support_summary.csv")
    e01 = summary_df[summary_df["experiment_id"] == "E01"].iloc[0]
    e07 = summary_df[summary_df["experiment_id"] == "E07"].iloc[0]
    assert e01["status"] in {"completed", "partial"}
    assert e01["completed_variants"] > 0
    assert e07["status"] == "blocked"
    assert e07["blocked_variants"] > 0

    run_log_df = pd.read_csv(campaign_root / "run_log_export.csv")
    assert {"Run_ID", "Experiment_ID", "Result_Summary", "Primary_Metric_Value"} <= set(
        run_log_df.columns
    )
    assert set(run_log_df["Experiment_ID"].astype(str).tolist()) >= {"E01", "E07"}
    output_root = tmp_path / "artifacts" / "decision_support"
    registry_path = output_root / "artifact_registry.sqlite3"
    assert registry_path.exists()
    first_run_id = str(run_log_df.iloc[0]["Run_ID"])
    run_artifacts = list_artifacts_for_run(registry_path=registry_path, run_id=first_run_id)
    assert any(record.artifact_type == ARTIFACT_TYPE_EXPERIMENT_REPORT for record in run_artifacts)


def test_single_blocked_experiment_raises(tmp_path: Path) -> None:
    registry_path = tmp_path / "decision_support_registry.json"
    index_csv = tmp_path / "dataset_index.csv"
    _write_registry(registry_path)
    _write_index_csv(index_csv)

    with pytest.raises(RuntimeError, match="not executable"):
        orchestrator.run_decision_support_campaign(
            registry_path=registry_path,
            index_csv=index_csv,
            data_root=tmp_path / "Data",
            cache_dir=tmp_path / "cache",
            output_root=tmp_path / "artifacts" / "decision_support",
            experiment_id="E07",
            stage=None,
            run_all=False,
            seed=42,
            n_permutations=0,
            dry_run=False,
        )


def test_campaign_passes_segment_arguments_to_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = tmp_path / "decision_support_registry.json"
    index_csv = tmp_path / "dataset_index.csv"
    _write_segment_registry(registry_path)
    _write_index_csv(index_csv)

    seen_kwargs: list[dict[str, object]] = []

    def _capturing_stub(**kwargs: object) -> dict[str, object]:
        seen_kwargs.append(dict(kwargs))
        return _stub_run_experiment(**kwargs)

    monkeypatch.setattr(orchestrator, "run_experiment", _capturing_stub)

    orchestrator.run_decision_support_campaign(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "artifacts" / "decision_support",
        experiment_id=None,
        stage=None,
        run_all=True,
        seed=42,
        n_permutations=0,
        dry_run=False,
    )

    assert len(seen_kwargs) == 1
    call = seen_kwargs[0]
    assert call["start_section"] == "feature_matrix_load"
    assert call["end_section"] == "evaluation"
    assert call["base_artifact_id"] == "feature_cache_base_123"
    assert call["reuse_policy"] == "require_explicit_base"
