from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_FEATURE_CACHE,
    ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE,
    ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    list_artifacts_for_run,
)
from Thesis_ML.data.index_dataset import build_dataset_index
from Thesis_ML.experiments.run_experiment import run_experiment
from Thesis_ML.experiments.segment_execution import plan_section_path
from Thesis_ML.experiments.stage_registry import (
    MODEL_FIT_CPU_EXECUTOR_ID,
    PERMUTATION_REFERENCE_EXECUTOR_ID,
    SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID,
    SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID,
)
from Thesis_ML.experiments.tuning_search_spaces import (
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
)


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.eye(4, dtype=np.float64) if affine is None else np.asarray(affine, dtype=np.float64)
    image = nib.Nifti1Image(data.astype(np.float32), affine=matrix)
    nib.save(image, str(path))


def _create_glm_session(
    glm_dir: Path,
    labels: list[str],
    class_signal: bool = False,
    shape: tuple[int, int, int] = (3, 3, 3),
) -> None:
    glm_dir.mkdir(parents=True, exist_ok=True)

    mask = np.zeros(shape, dtype=np.float32)
    mask[1:, 1:, 1:] = 1.0
    _write_nifti(glm_dir / "mask.nii", mask)
    pd.Series(labels).to_csv(glm_dir / "regressor_labels.csv", index=False, header=False)

    for idx, label in enumerate(labels, start=1):
        beta = np.full(shape, fill_value=float(idx), dtype=np.float32)
        if class_signal:
            if "_anger_" in label:
                beta[1:, 1:, 1:] += 5.0
            if "_happiness_" in label:
                beta[1:, 1:, 1:] -= 5.0
        _write_nifti(glm_dir / f"beta_{idx:04d}.nii", beta)


@pytest.fixture
def prepared_dataset(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
        "run-1_passive_anger_video",
        "run-1_passive_happiness_video",
    ]
    for subject in ("sub-001", "sub-002"):
        for session in ("ses-01", "ses-02"):
            _create_glm_session(
                glm_dir=data_root / subject / session / "BAS2",
                labels=labels,
                class_signal=True,
            )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)
    return {
        "data_root": data_root,
        "index_csv": index_csv,
        "cache_dir": tmp_path / "cache",
        "reports_root": tmp_path / "reports" / "experiments",
    }


def _base_run_kwargs(prepared_dataset: dict[str, Path]) -> dict[str, object]:
    return {
        "index_csv": prepared_dataset["index_csv"],
        "data_root": prepared_dataset["data_root"],
        "cache_dir": prepared_dataset["cache_dir"],
        "target": "emotion",
        "model": "ridge",
        "cv": "loso_session",
        "seed": 13,
        "reports_root": prepared_dataset["reports_root"],
    }


def test_plan_section_path_feature_matrix_to_evaluation() -> None:
    path = plan_section_path(start_section="feature_matrix_load", end_section="evaluation")
    assert [section.value for section in path] == [
        "feature_matrix_load",
        "spatial_validation",
        "model_fit",
        "interpretability",
        "evaluation",
    ]


def test_full_pipeline_execution_remains_supported(prepared_dataset: dict[str, Path]) -> None:
    result = run_experiment(
        **_base_run_kwargs(prepared_dataset),
        run_id="full_pipeline_default",
    )

    assert result["planned_sections"] == [
        "dataset_selection",
        "feature_cache_build",
        "feature_matrix_load",
        "spatial_validation",
        "model_fit",
        "interpretability",
        "evaluation",
    ]
    assert result["executed_sections"] == result["planned_sections"]
    assert result["metrics"]["n_folds"] >= 2
    assert Path(result["metrics_path"]).exists()


def test_full_pipeline_stage_execution_metadata_is_additive(
    prepared_dataset: dict[str, Path],
) -> None:
    result = run_experiment(
        **_base_run_kwargs(prepared_dataset),
        run_id="full_pipeline_stage_metadata",
    )

    assert result["planned_sections"] == [
        "dataset_selection",
        "feature_cache_build",
        "feature_matrix_load",
        "spatial_validation",
        "model_fit",
        "interpretability",
        "evaluation",
    ]
    stage_execution = result.get("stage_execution")
    assert isinstance(stage_execution, dict)
    assert isinstance(stage_execution.get("policy"), dict)
    assert isinstance(stage_execution.get("assignments"), list)
    assert isinstance(stage_execution.get("telemetry"), list)
    assert any(
        row.get("stage") == "model_fit" and row.get("status") == "executed"
        for row in stage_execution["telemetry"]
    )
    assert any(
        row.get("stage") == "tuning" and row.get("status") == "skipped"
        for row in stage_execution["telemetry"]
    )
    assignment_by_stage = {
        str(row.get("stage")): row for row in stage_execution["assignments"]
    }
    assert assignment_by_stage["model_fit"]["executor_id"] == MODEL_FIT_CPU_EXECUTOR_ID
    assert isinstance(assignment_by_stage["model_fit"]["reason"], str)

    config_payload = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
    metrics_payload = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    assert config_payload["planned_sections"] == result["planned_sections"]
    assert metrics_payload["framework_mode"] == result["framework_mode"]
    assert isinstance(config_payload.get("stage_execution"), dict)
    assert isinstance(metrics_payload.get("stage_execution"), dict)
    assert config_payload.get("preprocessing_kind") == "standard_scaler"
    assert metrics_payload.get("preprocessing_kind") == "standard_scaler"


def test_segment_execution_feature_matrix_to_evaluation(
    prepared_dataset: dict[str, Path],
) -> None:
    base_result = run_experiment(
        **_base_run_kwargs(prepared_dataset),
        run_id="segment_base_run",
    )
    base_feature_cache_id = base_result["artifact_ids"]["feature_cache"]

    segment_result = run_experiment(
        **_base_run_kwargs(prepared_dataset),
        run_id="segment_only_run",
        start_section="feature_matrix_load",
        end_section="evaluation",
        base_artifact_id=base_feature_cache_id,
        reuse_policy="require_explicit_base",
    )

    assert segment_result["executed_sections"] == [
        "feature_matrix_load",
        "spatial_validation",
        "model_fit",
        "interpretability",
        "evaluation",
    ]
    assert Path(segment_result["metrics_path"]).exists()
    assert segment_result["metrics"]["n_folds"] >= 2

    registry_path = Path(segment_result["artifact_registry_path"])
    run_records = list_artifacts_for_run(registry_path=registry_path, run_id="segment_only_run")
    record_types = {record.artifact_type for record in run_records}
    assert ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE in record_types
    assert ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE in record_types
    assert ARTIFACT_TYPE_METRICS_BUNDLE in record_types
    assert ARTIFACT_TYPE_FEATURE_CACHE not in record_types


def test_linearsvc_tuning_and_permutation_dispatch_through_stage_planner(
    prepared_dataset: dict[str, Path],
) -> None:
    run_kwargs = _base_run_kwargs(prepared_dataset)
    run_kwargs["model"] = "linearsvc"
    result = run_experiment(
        **run_kwargs,
        run_id="stage_planner_linearsvc_dispatch",
        methodology_policy_name="grouped_nested_tuning",
        tuning_enabled=True,
        tuning_search_space_id=LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
        tuning_search_space_version=LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
        tuning_inner_cv_scheme="grouped_leave_one_group_out",
        tuning_inner_group_field="session",
        n_permutations=4,
    )

    stage_execution = result.get("stage_execution")
    assert isinstance(stage_execution, dict)
    assignment_by_stage = {
        str(row.get("stage")): row for row in stage_execution["assignments"]
    }
    assert assignment_by_stage["model_fit"]["executor_id"] == MODEL_FIT_CPU_EXECUTOR_ID
    assert assignment_by_stage["tuning"]["executor_id"] == SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID
    assert assignment_by_stage["permutation"]["executor_id"] == PERMUTATION_REFERENCE_EXECUTOR_ID

    permutation_payload = result["metrics"].get("permutation_test")
    assert isinstance(permutation_payload, dict)
    assert permutation_payload.get("permutation_executor_id") == PERMUTATION_REFERENCE_EXECUTOR_ID

    tuning_summary = json.loads(Path(result["tuning_summary_path"]).read_text(encoding="utf-8"))
    assert SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID in tuning_summary["tuning_executor_ids"]


def test_logreg_tuning_dispatch_through_stage_planner_with_progress_telemetry(
    prepared_dataset: dict[str, Path],
) -> None:
    run_kwargs = _base_run_kwargs(prepared_dataset)
    run_kwargs["model"] = "logreg"
    result = run_experiment(
        **run_kwargs,
        run_id="stage_planner_logreg_dispatch",
        methodology_policy_name="grouped_nested_tuning",
        tuning_enabled=True,
        tuning_search_space_id=LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
        tuning_search_space_version=LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
        tuning_inner_cv_scheme="grouped_leave_one_group_out",
        tuning_inner_group_field="session",
        n_permutations=0,
    )

    stage_execution = result.get("stage_execution")
    assert isinstance(stage_execution, dict)
    assignment_by_stage = {
        str(row.get("stage")): row for row in stage_execution["assignments"]
    }
    assert assignment_by_stage["model_fit"]["executor_id"] == MODEL_FIT_CPU_EXECUTOR_ID
    assert assignment_by_stage["tuning"]["executor_id"] == SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID
    assert assignment_by_stage["tuning"]["fallback_used"] is False

    tuning_summary = json.loads(Path(result["tuning_summary_path"]).read_text(encoding="utf-8"))
    assert SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID in tuning_summary["tuning_executor_ids"]
    assert tuning_summary["specialized_logreg_tuning_used"] is True
    assert int(tuning_summary["tuning_progress_event_count_total"]) > 0
    assert int(tuning_summary["tuning_progress_total_units_total"]) > 0


def test_invalid_start_end_combination_raises(prepared_dataset: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="start_section must be before or equal to end_section"):
        run_experiment(
            **_base_run_kwargs(prepared_dataset),
            run_id="invalid_start_end",
            start_section="evaluation",
            end_section="feature_matrix_load",
        )


def test_incompatible_base_artifact_raises(prepared_dataset: dict[str, Path]) -> None:
    base_result = run_experiment(
        **_base_run_kwargs(prepared_dataset),
        run_id="incompatible_base_source",
    )
    metrics_artifact_id = base_result["artifact_ids"][ARTIFACT_TYPE_METRICS_BUNDLE]

    with pytest.raises(ValueError, match="Incompatible base artifact"):
        run_experiment(
            **_base_run_kwargs(prepared_dataset),
            run_id="incompatible_base_target",
            start_section="feature_matrix_load",
            end_section="evaluation",
            base_artifact_id=metrics_artifact_id,
            reuse_policy="require_explicit_base",
        )
