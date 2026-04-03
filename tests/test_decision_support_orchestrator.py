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


def _write_e01_e04_registry(path: Path) -> None:
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
                "executable_now": True,
                "execution_status": "executable",
                "blocked_reasons": [],
                "variant_templates": [
                    {
                        "template_id": "coarse_affect_within_subject",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                        },
                        "expand": {"subject": "subjects"},
                    },
                    {
                        "template_id": "emotion_within_subject",
                        "supported": True,
                        "params": {
                            "target": "emotion",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                        },
                        "expand": {"subject": "subjects"},
                    },
                    {
                        "template_id": "binary_valence_like",
                        "supported": True,
                        "params": {
                            "target": "binary_valence_like",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                        },
                        "expand": {"subject": "subjects"},
                    },
                ],
            },
            {
                "experiment_id": "E04",
                "title": "Split-strength stress test",
                "stage": "Stage 2 - Split lock",
                "decision_id": "D02",
                "manipulated_factor": "Split logic",
                "primary_metric": "balanced_accuracy",
                "executable_now": True,
                "execution_status": "executable",
                "blocked_reasons": [],
                "variant_templates": [
                    {
                        "template_id": "aux_grouped_loso_session",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "loso_session",
                        },
                    },
                    {
                        "template_id": "strict_within_subject",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                        },
                        "expand": {"subject": "subjects"},
                    },
                    {
                        "template_id": "record_random_split",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "record_random_split",
                        },
                    },
                ],
            },
        ],
    }
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _write_e07_e08_registry(path: Path) -> None:
    payload = {
        "schema_version": "test",
        "experiments": [
            {
                "experiment_id": "E07",
                "title": "Class weighting experiment",
                "stage": "Stage 3 - Model lock",
                "decision_id": "D04",
                "manipulated_factor": "Weighting strategy",
                "primary_metric": "balanced_accuracy",
                "executable_now": True,
                "execution_status": "executable",
                "blocked_reasons": [],
                "variant_templates": [
                    {
                        "template_id": "class_weight_none",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "methodology_policy_name": "fixed_baselines_only",
                            "class_weight_policy": "none",
                        },
                        "expand": {"subject": "subjects", "model": "models_linear"},
                    },
                    {
                        "template_id": "class_weight_balanced",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "methodology_policy_name": "fixed_baselines_only",
                            "class_weight_policy": "balanced",
                        },
                        "expand": {"subject": "subjects", "model": "models_linear"},
                    },
                ],
            },
            {
                "experiment_id": "E08",
                "title": "Hyperparameter strategy experiment",
                "stage": "Stage 3 - Model lock",
                "decision_id": "D05",
                "manipulated_factor": "Tuning strategy",
                "primary_metric": "balanced_accuracy",
                "executable_now": True,
                "execution_status": "executable",
                "blocked_reasons": [],
                "variant_templates": [
                    {
                        "template_id": "strategy_fixed_baselines",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "methodology_policy_name": "fixed_baselines_only",
                            "class_weight_policy": "none",
                        },
                        "expand": {"subject": "subjects", "model": "models_linear"},
                    },
                    {
                        "template_id": "strategy_grouped_nested",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "methodology_policy_name": "grouped_nested_tuning",
                            "class_weight_policy": "none",
                            "tuning_search_space_id": "official-linear-grouped-nested-v2",
                            "tuning_search_space_version": "2.0.0",
                            "tuning_inner_cv_scheme": "grouped_leave_one_group_out",
                            "tuning_inner_group_field": "session",
                        },
                        "expand": {"subject": "subjects", "model": "models_linear"},
                    },
                ],
            },
        ],
    }
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _write_e12_confirmatory_anchor_registry(path: Path) -> None:
    payload = {
        "schema_version": "test",
        "experiments": [
            {
                "experiment_id": "E12",
                "title": "Permutation test experiment",
                "stage": "Stage 6 - Robustness analysis",
                "decision_id": "D08",
                "manipulated_factor": "Label structure",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e12_template",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "anchor_experiment_id": "E16",
                        },
                    }
                ],
            },
            {
                "experiment_id": "E16",
                "title": "Final within-person confirmatory analysis",
                "stage": "Stage 5 - Confirmatory analysis",
                "decision_id": "CFM_LOCK_V1",
                "manipulated_factor": "none",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e16_anchor",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                            "feature_space": "whole_brain_masked",
                        },
                    }
                ],
            },
            {
                "experiment_id": "E17",
                "title": "Final within-person confirmatory analysis",
                "stage": "Stage 5 - Confirmatory analysis",
                "decision_id": "CFM_LOCK_V1",
                "manipulated_factor": "none",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e17_anchor",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-002",
                            "feature_space": "whole_brain_masked",
                        },
                    }
                ],
            },
            {
                "experiment_id": "E18",
                "title": "Final cross-person confirmatory analysis",
                "stage": "Stage 5 - Confirmatory analysis",
                "decision_id": "CFM_LOCK_V1",
                "manipulated_factor": "none",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e18_anchor",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "frozen_cross_person_transfer",
                            "train_subject": "sub-001",
                            "test_subject": "sub-002",
                            "feature_space": "whole_brain_masked",
                        },
                    }
                ],
            },
            {
                "experiment_id": "E19",
                "title": "Final cross-person confirmatory analysis",
                "stage": "Stage 5 - Confirmatory analysis",
                "decision_id": "CFM_LOCK_V1",
                "manipulated_factor": "none",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e19_anchor",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "frozen_cross_person_transfer",
                            "train_subject": "sub-002",
                            "test_subject": "sub-001",
                            "feature_space": "whole_brain_masked",
                        },
                    }
                ],
            },
        ],
    }
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _write_e13_confirmatory_anchor_registry(path: Path) -> None:
    payload = {
        "schema_version": "test",
        "experiments": [
            {
                "experiment_id": "E13",
                "title": "Trivial baseline experiment",
                "stage": "Stage 6 - Robustness analysis",
                "decision_id": "D08",
                "manipulated_factor": "Baseline type",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e13_template",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "dummy_or_majority",
                            "cv": "within_subject_loso_session",
                            "subject": "None",
                        },
                    }
                ],
            },
            {
                "experiment_id": "E16",
                "title": "Final within-person confirmatory analysis",
                "stage": "Stage 5 - Confirmatory analysis",
                "decision_id": "CFM_LOCK_V1",
                "manipulated_factor": "none",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e16_anchor",
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
                "experiment_id": "E17",
                "title": "Final cross-person confirmatory analysis",
                "stage": "Stage 5 - Confirmatory analysis",
                "decision_id": "CFM_LOCK_V1",
                "manipulated_factor": "none",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e17_anchor",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "frozen_cross_person_transfer",
                            "train_subject": "sub-001",
                            "test_subject": "sub-002",
                        },
                    }
                ],
            },
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


def _stub_run_experiment_with_permutation_metrics(**kwargs: object) -> dict[str, object]:
    run_id = str(kwargs["run_id"])
    reports_root = Path(kwargs["reports_root"])
    report_dir = reports_root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "metrics.json"
    n_permutations = int(kwargs.get("n_permutations", 0))
    observed = 0.58
    null_scores = [0.20 + (0.001 * idx) for idx in range(n_permutations)]
    ge_count = sum(float(value) >= float(observed) for value in null_scores)
    p_value = (float(ge_count) + 1.0) / (float(len(null_scores)) + 1.0)
    permutation_payload = {
        "n_permutations": int(n_permutations),
        "metric_name": "balanced_accuracy",
        "observed_score": float(observed),
        "observed_metric": float(observed),
        "p_value": float(p_value),
        "null_summary": {
            "mean": float(sum(null_scores) / len(null_scores)) if null_scores else 0.0,
            "std": 0.0,
            "min": float(min(null_scores)) if null_scores else 0.0,
            "max": float(max(null_scores)) if null_scores else 0.0,
            "q25": float(min(null_scores)) if null_scores else 0.0,
            "q50": float(min(null_scores)) if null_scores else 0.0,
            "q75": float(max(null_scores)) if null_scores else 0.0,
        },
        "null_scores": [float(value) for value in null_scores],
        "alpha": 0.05,
        "minimum_required": 100,
        "meets_minimum": bool(n_permutations >= 100),
        "passes_threshold": bool(p_value <= 0.05),
        "interpretation_status": ("passes_threshold" if p_value <= 0.05 else "fails_threshold"),
    }
    metrics_payload = {
        "primary_metric_name": "balanced_accuracy",
        "primary_metric_value": 0.58,
        "balanced_accuracy": 0.58,
        "macro_f1": 0.57,
        "accuracy": 0.61,
        "n_folds": 3,
        "permutation_test": permutation_payload,
    }
    metrics_path.write_text(f"{json.dumps(metrics_payload, indent=2)}\n", encoding="utf-8")
    return {
        "run_id": run_id,
        "report_dir": str(report_dir),
        "config_path": str(report_dir / "config.json"),
        "metrics_path": str(metrics_path),
        "fold_metrics_path": str(report_dir / "fold_metrics.csv"),
        "fold_splits_path": str(report_dir / "fold_splits.csv"),
        "predictions_path": str(report_dir / "predictions.csv"),
        "spatial_compatibility_report_path": str(report_dir / "spatial_compatibility_report.json"),
        "metrics": metrics_payload,
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


def test_campaign_forwards_methodology_kwargs_for_e07_e08(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = tmp_path / "decision_support_registry.json"
    index_csv = tmp_path / "dataset_index.csv"
    _write_e07_e08_registry(registry_path)
    _write_index_csv(index_csv)

    seen_kwargs: list[dict[str, object]] = []

    def _capturing_stub(**kwargs: object) -> dict[str, object]:
        seen_kwargs.append(dict(kwargs))
        return _stub_run_experiment(**kwargs)

    monkeypatch.setattr(orchestrator, "run_experiment", _capturing_stub)

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

    assert result["blocked_experiments"] == []
    assert len(seen_kwargs) == 24

    fixed_calls = [
        row for row in seen_kwargs if row.get("methodology_policy_name") == "fixed_baselines_only"
    ]
    grouped_calls = [
        row for row in seen_kwargs if row.get("methodology_policy_name") == "grouped_nested_tuning"
    ]

    assert fixed_calls
    assert grouped_calls
    assert len(grouped_calls) == 6
    assert {str(row.get("class_weight_policy")) for row in fixed_calls} == {"none", "balanced"}

    for row in fixed_calls:
        assert row["tuning_enabled"] is False
        assert row["tuning_search_space_id"] is None
        assert row["tuning_search_space_version"] is None
        assert row["tuning_inner_cv_scheme"] is None
        assert row["tuning_inner_group_field"] is None

    for row in grouped_calls:
        assert row["class_weight_policy"] == "none"
        assert row["tuning_enabled"] is True
        assert row["tuning_search_space_id"] == "official-linear-grouped-nested-v2"
        assert row["tuning_search_space_version"] == "2.0.0"
        assert row["tuning_inner_cv_scheme"] == "grouped_leave_one_group_out"
        assert row["tuning_inner_group_field"] == "session"


def test_campaign_dry_run_e07_e08_not_blocked(tmp_path: Path) -> None:
    registry_path = tmp_path / "decision_support_registry.json"
    index_csv = tmp_path / "dataset_index.csv"
    _write_e07_e08_registry(registry_path)
    _write_index_csv(index_csv)

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
        dry_run=True,
    )

    assert result["blocked_experiments"] == []

    manifest_path = Path(result["campaign_manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    blocked_ids = {
        str(row.get("experiment_id"))
        for row in payload.get("blocked_experiments", [])
        if isinstance(row, dict)
    }
    assert "E07" not in blocked_ids
    assert "E08" not in blocked_ids
    assert int(payload.get("status_counts", {}).get("blocked", 0)) == 0

    summary_df = pd.read_csv(result["decision_support_summary_path"])
    e07 = summary_df[summary_df["experiment_id"] == "E07"].iloc[0]
    e08 = summary_df[summary_df["experiment_id"] == "E08"].iloc[0]
    assert e07["status"] == "dry_run"
    assert e08["status"] == "dry_run"

    stage_df = pd.read_csv(Path(result["campaign_root"]) / "stage3_model_lock_summary.csv")
    labels = stage_df["variant_label"].astype(str).tolist()
    assert any("class_weight_policy=balanced" in label for label in labels)
    assert any("methodology_policy_name=grouped_nested_tuning" in label for label in labels)


def test_campaign_dry_run_e01_e04_not_blocked(tmp_path: Path) -> None:
    registry_path = tmp_path / "decision_support_registry.json"
    index_csv = tmp_path / "dataset_index.csv"
    _write_e01_e04_registry(registry_path)
    _write_index_csv(index_csv)

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
        dry_run=True,
    )

    assert result["blocked_experiments"] == []
    manifest_path = Path(result["campaign_manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    blocked_ids = {
        str(row.get("experiment_id"))
        for row in payload.get("blocked_experiments", [])
        if isinstance(row, dict)
    }
    assert "E01" not in blocked_ids
    assert "E04" not in blocked_ids
    assert int(payload.get("status_counts", {}).get("blocked", 0)) == 0

    summary_df = pd.read_csv(result["decision_support_summary_path"])
    e01 = summary_df[summary_df["experiment_id"] == "E01"].iloc[0]
    e04 = summary_df[summary_df["experiment_id"] == "E04"].iloc[0]
    assert e01["status"] == "dry_run"
    assert e04["status"] == "dry_run"


def test_e12_uses_anchor_subject_and_merged_summary_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = tmp_path / "decision_support_registry.json"
    index_csv = tmp_path / "dataset_index.csv"
    _write_e12_confirmatory_anchor_registry(registry_path)
    _write_index_csv(index_csv)

    captured_identities: list[tuple[str, str | None, str | None, str | None]] = []

    def _capturing_stub(**kwargs: object) -> dict[str, object]:
        captured_identities.append(
            (
                str(kwargs.get("cv")),
                (str(kwargs.get("subject")) if kwargs.get("subject") else None),
                (str(kwargs.get("train_subject")) if kwargs.get("train_subject") else None),
                (str(kwargs.get("test_subject")) if kwargs.get("test_subject") else None),
            )
        )
        if kwargs.get("cv") == "within_subject_loso_session" and not kwargs.get("subject"):
            raise ValueError("missing subject for within_subject_loso_session")
        return _stub_run_experiment_with_permutation_metrics(**kwargs)

    monkeypatch.setattr(orchestrator, "run_experiment", _capturing_stub)

    result = orchestrator.run_decision_support_campaign(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "artifacts" / "decision_support",
        experiment_id="E12",
        stage=None,
        run_all=False,
        seed=42,
        n_permutations=120,
        dry_run=False,
    )

    assert captured_identities
    within_subjects = {
        identity[1]
        for identity in captured_identities
        if identity[0] == "within_subject_loso_session"
    }
    transfer_pairs = {
        (identity[2], identity[3])
        for identity in captured_identities
        if identity[0] == "frozen_cross_person_transfer"
    }
    assert within_subjects == {"sub-001", "sub-002"}
    assert transfer_pairs == {("sub-001", "sub-002"), ("sub-002", "sub-001")}

    campaign_root = Path(result["campaign_root"])
    run_log_df = pd.read_csv(campaign_root / "run_log_export.csv")
    assert len(run_log_df[run_log_df["Experiment_ID"] == "E12"]) == 12

    summary_df = pd.read_csv(campaign_root / "decision_support_summary.csv")
    e12_row = summary_df[summary_df["experiment_id"] == "E12"].iloc[0]
    assert int(e12_row["total_variants"]) == 4
    assert int(e12_row["completed_variants"]) == 4
    assert str(e12_row["status"]) == "completed"

    aggregation_payload = json.loads((campaign_root / "result_aggregation.json").read_text())
    best_runs = list(aggregation_payload.get("best_full_pipeline_runs", []))
    assert len(best_runs) == 4
    assert all(str(row.get("run_id", "")).endswith("__perm_merged") for row in best_runs)

    summary_outputs_df = pd.read_csv(campaign_root / "summary_outputs_export.csv")
    merged_run_ids = [str(value) for value in summary_outputs_df["run_id"] if str(value).strip()]
    assert merged_run_ids
    assert all(value.endswith("__perm_merged") for value in merged_run_ids)
    decision_notes = (campaign_root / "decision_recommendations.md").read_text(encoding="utf-8")
    assert "__perm_merged" in decision_notes
    table_ready_path = (
        campaign_root / "special_aggregations" / "E12" / "e12_permutation_analysis_summary.csv"
    )
    assert table_ready_path.exists()
    table_ready_df = pd.read_csv(table_ready_path)
    assert len(table_ready_df) == 4


def test_e13_materializes_anchor_matched_dummy_baselines_and_writes_table_ready_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = tmp_path / "decision_support_registry.json"
    index_csv = tmp_path / "dataset_index.csv"
    _write_e13_confirmatory_anchor_registry(registry_path)
    _write_index_csv(index_csv)

    captured_identities: list[tuple[str, str | None, str | None, str | None, str]] = []

    def _capturing_stub(**kwargs: object) -> dict[str, object]:
        captured_identities.append(
            (
                str(kwargs.get("cv")),
                (str(kwargs.get("subject")) if kwargs.get("subject") else None),
                (str(kwargs.get("train_subject")) if kwargs.get("train_subject") else None),
                (str(kwargs.get("test_subject")) if kwargs.get("test_subject") else None),
                str(kwargs.get("model")),
            )
        )
        if str(kwargs.get("model")) != "dummy":
            raise ValueError("E13 baseline run must use model=dummy")
        if kwargs.get("subject") == "None":
            raise ValueError("placeholder subject should not survive E13 materialization")
        return _stub_run_experiment(**kwargs)

    monkeypatch.setattr(orchestrator, "run_experiment", _capturing_stub)

    result = orchestrator.run_decision_support_campaign(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "artifacts" / "decision_support",
        experiment_id="E13",
        stage=None,
        run_all=False,
        seed=42,
        n_permutations=0,
        dry_run=False,
    )

    assert len(captured_identities) == 2
    assert {identity[4] for identity in captured_identities} == {"dummy"}
    assert {
        identity[1]
        for identity in captured_identities
        if identity[0] == "within_subject_loso_session"
    } == {"sub-001"}
    assert {
        (identity[2], identity[3])
        for identity in captured_identities
        if identity[0] == "frozen_cross_person_transfer"
    } == {("sub-001", "sub-002")}

    campaign_root = Path(result["campaign_root"])
    run_log_df = pd.read_csv(campaign_root / "run_log_export.csv")
    e13_rows = run_log_df[run_log_df["Experiment_ID"] == "E13"]
    assert len(e13_rows) == 2
    assert set(e13_rows["Model"].astype(str).tolist()) == {"dummy"}
    assert "None" not in set(e13_rows["Data_Subset"].astype(str).tolist())

    summary_outputs_df = pd.read_csv(campaign_root / "summary_outputs_export.csv")
    run_ids = set(summary_outputs_df["run_id"].astype(str).tolist())
    assert len(run_ids) >= 1

    table_ready_path = (
        campaign_root / "special_aggregations" / "E13" / "e13_dummy_baseline_analysis_summary.csv"
    )
    assert table_ready_path.exists()
    table_ready_df = pd.read_csv(table_ready_path)
    assert len(table_ready_df) == 2
    assert set(table_ready_df["model"].astype(str).tolist()) == {"dummy"}
    assert set(table_ready_df["anchor_analysis_type"].astype(str).tolist()) == {
        "within_person_loso",
        "cross_person_transfer",
    }
