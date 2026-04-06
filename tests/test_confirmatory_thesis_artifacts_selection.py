from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.orchestration import campaign_engine


def test_confirmatory_thesis_artifacts_exclude_legacy_locked_core_mismatch(
    tmp_path: Path,
) -> None:
    runtime_anchor_rows = [
        {
            "analysis_label": "within_subject_loso_session:sub-001",
            "experiment_id": "E16",
            "template_id": "wb_row_014",
            "cv": "within_subject_loso_session",
            "subject": "sub-001",
            "target": "coarse_affect",
            "model": "ridge",
            "feature_space": "whole_brain_masked",
            "preprocessing_strategy": "none",
            "dimensionality_strategy": "none",
            "methodology_policy_name": "fixed_baselines_only",
            "class_weight_policy": "none",
        },
        {
            "analysis_label": "frozen_cross_person_transfer:sub-001->sub-002",
            "experiment_id": "E17",
            "template_id": "wb_row_015",
            "cv": "frozen_cross_person_transfer",
            "train_subject": "sub-001",
            "test_subject": "sub-002",
            "target": "coarse_affect",
            "model": "ridge",
            "feature_space": "whole_brain_masked",
            "preprocessing_strategy": "standardize_zscore",
            "dimensionality_strategy": "none",
            "methodology_policy_name": "fixed_baselines_only",
            "class_weight_policy": "none",
        },
    ]
    confirmatory_model_rows = [
        {
            "analysis_label": "within_subject_loso_session:sub-001",
            "experiment_id": "E16",
            "model": "ridge",
            "status": "completed",
            "run_id": "run_e16",
            "metrics_path": str((tmp_path / "e16_metrics.json").resolve()),
            "observed_score": 0.72,
            "target": "coarse_affect",
            "feature_space": "whole_brain_masked",
            "preprocessing_strategy": "none",
            "dimensionality_strategy": "none",
            "methodology_policy_name": "fixed_baselines_only",
            "class_weight_policy": "none",
        },
        {
            "analysis_label": "frozen_cross_person_transfer:sub-001->sub-002",
            "experiment_id": "E17",
            "model": "ridge",
            "status": "completed",
            "run_id": "run_e17",
            "metrics_path": str((tmp_path / "e17_metrics.json").resolve()),
            "observed_score": 0.44,
            "target": "coarse_affect",
            "feature_space": "whole_brain_masked",
            "preprocessing_strategy": "standardize_zscore",
            "dimensionality_strategy": "none",
            "methodology_policy_name": "fixed_baselines_only",
            "class_weight_policy": "none",
        },
    ]
    e12_table_rows = [
        {
            "analysis_label": "within_subject_loso_session:sub-001",
            "observed_balanced_accuracy": 0.72,
            "empirical_p": 0.001,
            "n_permutations": 1000,
            "meets_minimum": True,
        },
        {
            "analysis_label": "frozen_cross_person_transfer:sub-001->sub-002",
            "observed_balanced_accuracy": 0.44,
            "empirical_p": 0.01,
            "n_permutations": 1000,
            "meets_minimum": True,
        },
    ]
    e13_table_rows = [
        {
            "analysis_label": "within_subject_loso_session:sub-001",
            "model": "dummy",
            "metric_name": "balanced_accuracy",
            "observed_baseline_score": 0.3333,
            "status": "completed",
        },
        {
            "analysis_label": "frozen_cross_person_transfer:sub-001->sub-002",
            "model": "dummy",
            "metric_name": "balanced_accuracy",
            "observed_baseline_score": 0.3333,
            "status": "completed",
        },
    ]
    coverage_rows = [
        {
            "analysis_label": "within_subject_loso_session:sub-001",
            "e12_covered": True,
            "e13_covered": True,
        },
        {
            "analysis_label": "frozen_cross_person_transfer:sub-001->sub-002",
            "e12_covered": True,
            "e13_covered": True,
        },
    ]

    artifact_paths = campaign_engine._write_confirmatory_thesis_artifacts(
        campaign_root=tmp_path,
        runtime_anchor_rows=runtime_anchor_rows,
        confirmatory_model_rows=confirmatory_model_rows,
        e12_table_rows=e12_table_rows,
        e13_table_rows=e13_table_rows,
        coverage_rows=coverage_rows,
    )

    manifest_json = json.loads(
        Path(str(artifact_paths["confirmatory_anchor_manifest_json"])).read_text(encoding="utf-8")
    )
    by_label = {str(row["analysis_label"]): row for row in manifest_json}
    assert (
        by_label["within_subject_loso_session:sub-001"]["thesis_selection_status"] == "selected"
    )
    assert (
        by_label["frozen_cross_person_transfer:sub-001->sub-002"]["thesis_selection_status"]
        == "legacy_diagnostic_excluded"
    )
    assert (
        "locked_core_mismatch"
        in by_label["frozen_cross_person_transfer:sub-001->sub-002"]["legacy_reason"]
    )

    transfer_json = json.loads(
        Path(str(artifact_paths["thesis_e17_transfer_summary_json"])).read_text(encoding="utf-8")
    )
    assert transfer_json == []
