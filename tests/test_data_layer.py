from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.methodology import DataPolicy
from Thesis_ML.experiments.data_reporting import (
    evaluate_official_data_policy,
    write_official_data_artifacts,
)


def _index_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "task": "passive",
                "modality": "audio",
                "beta_path": "b1.nii",
                "mask_path": "m1.nii",
                "regressor_label": "run-1_passive_anger_audio",
                "emotion": "anger",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-02",
                "task": "passive",
                "modality": "audio",
                "beta_path": "b2.nii",
                "mask_path": "m1.nii",
                "regressor_label": "run-1_passive_happiness_audio",
                "emotion": "happiness",
                "coarse_affect": "positive",
            },
        ]
    )


def test_data_policy_rejects_invalid_threshold_order() -> None:
    with pytest.raises(ValueError, match="min_class_fraction_blocking"):
        DataPolicy.model_validate(
            {
                "class_balance": {
                    "min_class_fraction_warning": 0.05,
                    "min_class_fraction_blocking": 0.2,
                }
            }
        )


def test_evaluate_official_data_policy_flags_blocking_duplicate_sample_id(tmp_path: Path) -> None:
    frame = _index_frame().copy()
    frame.loc[1, "sample_id"] = "s1"
    index_csv = tmp_path / "dataset_index.csv"
    frame.to_csv(index_csv, index=False)

    assessment = evaluate_official_data_policy(
        framework_mode=FrameworkMode.CONFIRMATORY,
        index_csv=index_csv,
        data_root=tmp_path,
        cache_dir=tmp_path / "cache",
        full_index_df=frame,
        selected_index_df=frame,
        target_column="coarse_affect",
        cv_mode="within_subject_loso_session",
        subject="sub-001",
        train_subject=None,
        test_subject=None,
        filter_task=None,
        filter_modality=None,
        official_context={
            "data_policy": {
                "leakage": {
                    "enabled": True,
                    "fail_on_duplicate_sample_id": True,
                    "warn_on_duplicate_beta_path": True,
                    "fail_on_duplicate_beta_path": False,
                    "fail_on_subject_overlap_for_transfer": True,
                    "fail_on_cv_group_overlap": True,
                }
            }
        },
    )

    assert int(len(assessment["blocking_issues"])) >= 1
    assert any(
        issue["code"] == "leakage_duplicate_sample_id"
        for issue in assessment["blocking_issues"]
    )


def test_external_validation_required_dataset_missing_is_blocking(tmp_path: Path) -> None:
    frame = _index_frame()
    index_csv = tmp_path / "dataset_index.csv"
    frame.to_csv(index_csv, index=False)

    assessment = evaluate_official_data_policy(
        framework_mode=FrameworkMode.LOCKED_COMPARISON,
        index_csv=index_csv,
        data_root=tmp_path,
        cache_dir=tmp_path / "cache",
        full_index_df=frame,
        selected_index_df=frame,
        target_column="coarse_affect",
        cv_mode="within_subject_loso_session",
        subject="sub-001",
        train_subject=None,
        test_subject=None,
        filter_task=None,
        filter_modality=None,
        official_context={
            "data_policy": {
                "external_validation": {
                    "enabled": True,
                    "mode": "compatibility_only",
                    "require_compatible": False,
                    "require_for_official_runs": True,
                    "datasets": [
                        {
                            "dataset_id": "ext-1",
                            "index_csv": "missing_external_index.csv",
                            "required": True,
                        }
                    ],
                }
            }
        },
    )

    assert any(
        issue["code"] in {"external_dataset_incompatible", "external_validation_required_missing"}
        for issue in assessment["blocking_issues"]
    )


def test_write_official_data_artifacts_creates_expected_files(tmp_path: Path) -> None:
    frame = _index_frame()
    index_csv = tmp_path / "dataset_index.csv"
    frame.to_csv(index_csv, index=False)
    assessment = evaluate_official_data_policy(
        framework_mode=FrameworkMode.CONFIRMATORY,
        index_csv=index_csv,
        data_root=tmp_path,
        cache_dir=tmp_path / "cache",
        full_index_df=frame,
        selected_index_df=frame,
        target_column="coarse_affect",
        cv_mode="within_subject_loso_session",
        subject="sub-001",
        train_subject=None,
        test_subject=None,
        filter_task=None,
        filter_modality=None,
        official_context={},
        target_derivation_audit_df=pd.DataFrame(
            [
                {
                    "sample_id": "s3",
                    "subject": "sub-001",
                    "session": "ses-03",
                    "task": "passive",
                    "modality": "audio",
                    "emotion": "neutral",
                    "coarse_affect": "neutral",
                    "binary_valence_like": pd.NA,
                    "target_column": "binary_valence_like",
                    "source_column": "coarse_affect",
                    "source_value": "neutral",
                    "drop_category": "intended_target_exclusion",
                    "drop_reason": "coarse_affect='neutral' is intentionally excluded from binary_valence_like.",
                }
            ]
        ),
        selection_exclusion_manifest_df=pd.DataFrame(
            [
                {
                    "sample_id": "s3",
                    "subject": "sub-001",
                    "session": "ses-03",
                    "task": "emo",
                    "modality": "audio",
                    "target_column": "coarse_affect",
                    "target_value": pd.NA,
                    "exclusion_stage": "filter_task",
                    "exclusion_reason": "task_mismatch",
                    "cv_mode": "within_subject_loso_session",
                    "requested_subject": "sub-001",
                    "requested_train_subject": pd.NA,
                    "requested_test_subject": pd.NA,
                    "requested_filter_task": "passive",
                    "requested_filter_modality": pd.NA,
                }
            ]
        ),
    )

    report_dir = tmp_path / "report"
    payload = write_official_data_artifacts(
        report_dir=report_dir,
        assessment=assessment,
        framework_mode=FrameworkMode.CONFIRMATORY,
        index_csv=index_csv,
        data_root=tmp_path,
        cache_dir=tmp_path / "cache",
        target_column="coarse_affect",
        cv_mode="within_subject_loso_session",
        subject="sub-001",
        train_subject=None,
        test_subject=None,
        filter_task=None,
        filter_modality=None,
        sample_unit="beta_event",
        label_policy="affect_mapping_v1",
        target_mapping_version="affect_mapping_v1",
        target_mapping_hash="abc",
        dataset_fingerprint={"index_csv_sha256": "hash"},
    )

    assert isinstance(payload["data_policy_effective"], dict)
    assert (report_dir / "dataset_card.json").exists()
    assert (report_dir / "dataset_card.md").exists()
    assert (report_dir / "dataset_summary.json").exists()
    assert (report_dir / "dataset_summary.csv").exists()
    assert (report_dir / "data_quality_report.json").exists()
    assert (report_dir / "class_balance_report.csv").exists()
    assert (report_dir / "missingness_report.csv").exists()
    assert (report_dir / "leakage_audit.json").exists()
    assert (report_dir / "cv_split_audit.json").exists()
    assert (report_dir / "cv_split_audit.csv").exists()
    assert (report_dir / "external_validation_compatibility.json").exists()
    assert (report_dir / "target_derivation_audit.json").exists()
    assert (report_dir / "target_derivation_audit.csv").exists()

    split_payload = json.loads((report_dir / "cv_split_audit.json").read_text(encoding="utf-8"))
    assert split_payload["status"] == "pass"
    assert int(split_payload["n_folds"]) == 2
    assert (report_dir / "selection_exclusion_summary.json").exists()
    assert (report_dir / "selection_exclusion_manifest.csv").exists()
    


def test_evaluate_official_data_policy_builds_exact_within_subject_split_audit(
    tmp_path: Path,
) -> None:
    frame = _index_frame()
    index_csv = tmp_path / "dataset_index.csv"
    frame.to_csv(index_csv, index=False)

    assessment = evaluate_official_data_policy(
        framework_mode=FrameworkMode.CONFIRMATORY,
        index_csv=index_csv,
        data_root=tmp_path,
        cache_dir=tmp_path / "cache",
        full_index_df=frame,
        selected_index_df=frame,
        target_column="coarse_affect",
        cv_mode="within_subject_loso_session",
        subject="sub-001",
        train_subject=None,
        test_subject=None,
        filter_task=None,
        filter_modality=None,
        official_context={},
    )

    cv_split_audit = assessment["cv_split_audit"]
    assert cv_split_audit["status"] == "pass"
    assert int(cv_split_audit["n_folds"]) == 2
    assert int(cv_split_audit["expected_n_folds"]) == 2
    assert int(cv_split_audit["missing_expected_test_rows"]) == 0
    assert int(cv_split_audit["unexpected_test_rows"]) == 0
    assert int(cv_split_audit["duplicate_test_coverage_rows"]) == 0

    rows = assessment["cv_split_audit_rows"]
    assert len(rows) == 2
    assert {row["test_sessions"] for row in rows} == {"ses-01", "ses-02"}
    assert all(row["status"] == "pass" for row in rows)

def test_evaluate_official_data_policy_blocks_within_subject_subject_mix(
    tmp_path: Path,
) -> None:
    frame = _index_frame().copy()
    frame.loc[1, "subject"] = "sub-002"
    index_csv = tmp_path / "dataset_index.csv"
    frame.to_csv(index_csv, index=False)

    assessment = evaluate_official_data_policy(
        framework_mode=FrameworkMode.CONFIRMATORY,
        index_csv=index_csv,
        data_root=tmp_path,
        cache_dir=tmp_path / "cache",
        full_index_df=frame,
        selected_index_df=frame,
        target_column="coarse_affect",
        cv_mode="within_subject_loso_session",
        subject="sub-001",
        train_subject=None,
        test_subject=None,
        filter_task=None,
        filter_modality=None,
        official_context={
            "data_policy": {
                "leakage": {
                    "enabled": True,
                    "fail_on_duplicate_sample_id": True,
                    "warn_on_duplicate_beta_path": True,
                    "fail_on_duplicate_beta_path": False,
                    "fail_on_subject_overlap_for_transfer": True,
                    "fail_on_cv_group_overlap": True,
                }
            }
        },
    )

    assert any(
        issue["code"] == "leakage_cv_split_plan_invalid"
        for issue in assessment["blocking_issues"]
    )

    cv_split_audit = assessment["cv_split_audit"]
    assert cv_split_audit["status"] == "fail"
    assert "split_planner_error" in cv_split_audit["failure_codes"]
    assert (
        "within_subject_loso_session requires exactly one subject in the evaluated data."
        in cv_split_audit["planner_error"]
    )

def test_evaluate_official_data_policy_loso_session_allows_same_session_name_across_subjects(
    tmp_path: Path,
) -> None:
    
    frame = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "task": "passive",
                "modality": "audio",
                "emotion": "happiness",
                "coarse_affect": "positive",
                "beta_path": "sub-001/ses-01/BAS/beta_0001.nii",
                "mask_path": "sub-001/ses-01/BAS/mask.nii",
                "regressor_label": "run-1_passive_happiness_audio",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-02",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
                "beta_path": "sub-001/ses-02/BAS/beta_0001.nii",
                "mask_path": "sub-001/ses-02/BAS/mask.nii",
                "regressor_label": "run-1_passive_anger_audio",
            },
            {
                "sample_id": "s3",
                "subject": "sub-002",
                "session": "ses-01",
                "task": "passive",
                "modality": "audio",
                "emotion": "happiness",
                "coarse_affect": "positive",
                "beta_path": "sub-002/ses-01/BAS/beta_0001.nii",
                "mask_path": "sub-002/ses-01/BAS/mask.nii",
                "regressor_label": "run-1_passive_happiness_audio",
            },
            {
                "sample_id": "s4",
                "subject": "sub-002",
                "session": "ses-02",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
                "beta_path": "sub-002/ses-02/BAS/beta_0001.nii",
                "mask_path": "sub-002/ses-02/BAS/mask.nii",
                "regressor_label": "run-1_passive_anger_audio",
            },
        ]
    )

    index_csv = tmp_path / "dataset_index.csv"
    frame.to_csv(index_csv, index=False)

    assessment = evaluate_official_data_policy(
        framework_mode=FrameworkMode.CONFIRMATORY,
        index_csv=index_csv,
        data_root=tmp_path,
        cache_dir=tmp_path / "cache",
        full_index_df=frame,
        selected_index_df=frame,
        target_column="coarse_affect",
        cv_mode="loso_session",
        subject=None,
        train_subject=None,
        test_subject=None,
        filter_task=None,
        filter_modality=None,
        official_context={},
    )

    cv_split_audit = assessment["cv_split_audit"]
    assert cv_split_audit["status"] == "pass"
    assert "group_overlap" not in cv_split_audit["failure_codes"]

    rows = assessment["cv_split_audit_rows"]
    assert len(rows) == 4
    assert all(row["status"] == "pass" for row in rows)

def test_evaluate_official_data_policy_carries_selection_exclusion_summary(tmp_path: Path) -> None:
    frame = _index_frame()
    index_csv = tmp_path / "dataset_index.csv"
    frame.to_csv(index_csv, index=False)

    exclusion_manifest = pd.DataFrame(
        [
            {
                "sample_id": "s9",
                "subject": "sub-002",
                "session": "ses-01",
                "task": "emo",
                "modality": "audio",
                "target_column": "coarse_affect",
                "target_value": pd.NA,
                "exclusion_stage": "cv_scope",
                "exclusion_reason": "subject_mismatch_for_within_subject",
                "cv_mode": "within_subject_loso_session",
                "requested_subject": "sub-001",
                "requested_train_subject": pd.NA,
                "requested_test_subject": pd.NA,
                "requested_filter_task": pd.NA,
                "requested_filter_modality": pd.NA,
            }
        ]
    )

    assessment = evaluate_official_data_policy(
        framework_mode=FrameworkMode.CONFIRMATORY,
        index_csv=index_csv,
        data_root=tmp_path,
        cache_dir=tmp_path / "cache",
        full_index_df=frame,
        selected_index_df=frame,
        target_column="coarse_affect",
        cv_mode="within_subject_loso_session",
        subject="sub-001",
        train_subject=None,
        test_subject=None,
        filter_task=None,
        filter_modality=None,
        official_context={},
        selection_exclusion_manifest_df=exclusion_manifest,
    )

    summary = assessment["selection_exclusion_summary"]
    assert int(summary["n_rows"]) == 1
    assert summary["by_stage"]["cv_scope"] == 1
    assert summary["by_reason"]["subject_mismatch_for_within_subject"] == 1