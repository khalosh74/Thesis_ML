from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.errors import (
    OfficialArtifactContractError,
    OfficialContractValidationError,
)
from Thesis_ML.experiments.execution_policy import read_run_status
from Thesis_ML.experiments.official_contracts import (
    validate_official_preflight,
    validate_run_artifact_contract,
)
from Thesis_ML.experiments.run_experiment import run_experiment
from Thesis_ML.experiments.segment_execution import SegmentExecutionResult


def _official_context() -> dict[str, object]:
    return {
        "artifact_requirements": ["config.json", "metrics.json"],
        "required_run_metadata_fields": [
            "framework_mode",
            "canonical_run",
            "methodology_policy_name",
        ],
    }


def _confirmatory_lock_context() -> dict[str, object]:
    context = _official_context()
    context["target_mapping_version"] = "affect_mapping_v1"
    context["controls"] = {
        "dummy_baseline_run": False,
    }
    context["confirmatory_lock"] = {
        "protocol_id": "thesis_confirmatory_v1",
        "analysis_status": "locked",
        "target_name": "coarse_affect",
        "target_source_column": "emotion",
        "target_mapping_version": "affect_mapping_v1",
        "split": "within_subject_loso_session",
        "primary_metric": "balanced_accuracy",
        "model_family": "ridge",
        "hyperparameter_policy": "fixed",
        "class_weight_policy": "none",
        "required_index_columns": [
            "sample_id",
            "subject",
            "session",
            "task",
            "modality",
            "beta_path",
            "mask_path",
            "regressor_label",
            "emotion",
            "coarse_affect",
        ],
        "minimum_subjects": 1,
        "minimum_sessions_per_subject": 2,
        "permutation_required": True,
        "minimum_permutations": 10,
        "allowed_subgroup_axes": ["task", "modality", "subject"],
        "subgroup_min_samples_per_group": 20,
        "subgroup_min_classes_per_group": 2,
        "subgroup_report_small_groups": False,
    }
    return context


def _write_index(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_official_preflight_rejects_missing_required_dataset_columns(tmp_path: Path) -> None:
    index_csv = tmp_path / "dataset_index.csv"
    _write_index(
        index_csv,
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "task": "passive",
                "beta_path": "b1.nii",
                "mask_path": "m1.nii",
                "regressor_label": "run-1_passive_anger_audio",
                "emotion": "anger",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-02",
                "task": "passive",
                "beta_path": "b2.nii",
                "mask_path": "m1.nii",
                "regressor_label": "run-1_passive_happiness_audio",
                "emotion": "happiness",
            },
        ],
    )

    with pytest.raises(OfficialContractValidationError, match="missing required columns"):
        validate_official_preflight(
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
            n_permutations=0,
            primary_metric_name="balanced_accuracy",
            permutation_metric_name="balanced_accuracy",
            methodology_policy_name="fixed_baselines_only",
            class_weight_policy="none",
            model="ridge",
            tuning_enabled=False,
            tuning_search_space_id=None,
            tuning_search_space_version=None,
            tuning_inner_group_field=None,
            subgroup_reporting_enabled=True,
            subgroup_dimensions=["task", "modality"],
            subgroup_min_samples_per_group=1,
            subgroup_min_classes_per_group=1,
            subgroup_report_small_groups=False,
            official_context=_official_context(),
        )


def test_official_preflight_rejects_permutation_metric_drift(tmp_path: Path) -> None:
    index_csv = tmp_path / "dataset_index.csv"
    _write_index(
        index_csv,
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
            },
        ],
    )

    with pytest.raises(OfficialContractValidationError, match="permutation metric"):
        validate_official_preflight(
            framework_mode=FrameworkMode.LOCKED_COMPARISON,
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
            n_permutations=10,
            primary_metric_name="balanced_accuracy",
            permutation_metric_name="accuracy",
            methodology_policy_name="fixed_baselines_only",
            class_weight_policy="none",
            model="ridge",
            tuning_enabled=False,
            tuning_search_space_id=None,
            tuning_search_space_version=None,
            tuning_inner_group_field=None,
            subgroup_reporting_enabled=False,
            subgroup_dimensions=[],
            subgroup_min_samples_per_group=1,
            subgroup_min_classes_per_group=1,
            subgroup_report_small_groups=False,
            official_context=_official_context(),
        )


def test_confirmatory_preflight_rejects_locked_model_drift_for_non_control_runs(
    tmp_path: Path,
) -> None:
    index_csv = tmp_path / "dataset_index.csv"
    _write_index(
        index_csv,
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
            },
        ],
    )

    with pytest.raises(OfficialContractValidationError, match="model differs"):
        validate_official_preflight(
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
            n_permutations=10,
            primary_metric_name="balanced_accuracy",
            permutation_metric_name="balanced_accuracy",
            methodology_policy_name="fixed_baselines_only",
            class_weight_policy="none",
            model="linearsvc",
            tuning_enabled=False,
            tuning_search_space_id=None,
            tuning_search_space_version=None,
            tuning_inner_group_field=None,
            subgroup_reporting_enabled=True,
            subgroup_dimensions=["task", "modality"],
            subgroup_min_samples_per_group=20,
            subgroup_min_classes_per_group=2,
            subgroup_report_small_groups=False,
            official_context=_confirmatory_lock_context(),
        )


def test_confirmatory_preflight_rejects_invalid_subgroup_policy_override(
    tmp_path: Path,
) -> None:
    index_csv = tmp_path / "dataset_index.csv"
    _write_index(
        index_csv,
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
            },
        ],
    )

    with pytest.raises(OfficialContractValidationError, match="small-group reporting differs"):
        validate_official_preflight(
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
            n_permutations=10,
            primary_metric_name="balanced_accuracy",
            permutation_metric_name="balanced_accuracy",
            methodology_policy_name="fixed_baselines_only",
            class_weight_policy="none",
            model="ridge",
            tuning_enabled=False,
            tuning_search_space_id=None,
            tuning_search_space_version=None,
            tuning_inner_group_field=None,
            subgroup_reporting_enabled=True,
            subgroup_dimensions=["task", "modality"],
            subgroup_min_samples_per_group=20,
            subgroup_min_classes_per_group=2,
            subgroup_report_small_groups=True,
            official_context=_confirmatory_lock_context(),
        )


def test_run_artifact_contract_validation_rejects_missing_metadata(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "metrics.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(OfficialArtifactContractError, match="missing required metadata keys"):
        validate_run_artifact_contract(
            report_dir=report_dir,
            required_run_artifacts=["config.json", "metrics.json"],
            required_run_metadata_fields=["framework_mode", "canonical_run"],
            framework_mode=FrameworkMode.CONFIRMATORY,
            canonical_run=True,
            config_payload={},
            metrics_payload={},
        )


def _failing_segment_stub(request) -> SegmentExecutionResult:
    raise RuntimeError("segment execution boom")


def test_run_experiment_failure_writes_structured_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "Thesis_ML.experiments.run_experiment.execute_section_segment",
        _failing_segment_stub,
    )

    with pytest.raises(RuntimeError, match="segment execution boom"):
        run_experiment(
            index_csv=tmp_path / "missing_index.csv",
            data_root=tmp_path / "Data",
            cache_dir=tmp_path / "cache",
            target="emotion",
            model="ridge",
            cv="loso_session",
            run_id="rc1_failure_status",
            reports_root=tmp_path / "reports",
        )

    status = read_run_status(tmp_path / "reports" / "rc1_failure_status")
    assert status is not None
    assert status["status"] == "failed"
    assert status["error_code"] == "unhandled_exception"
    assert status["error_type"] == "RuntimeError"
    assert status["failure_stage"] == "runtime"
    assert "stage_timings_seconds" in status
