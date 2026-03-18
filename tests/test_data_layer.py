from __future__ import annotations

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
    assert (report_dir / "external_validation_compatibility.json").exists()
