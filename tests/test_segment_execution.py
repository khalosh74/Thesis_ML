from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
from Thesis_ML.experiments import segment_execution as segment_execution_module
from Thesis_ML.experiments.run_experiment import run_experiment
from Thesis_ML.experiments.segment_execution import (
    SegmentExecutionRequest,
    execute_section_segment,
    plan_section_path,
)
from Thesis_ML.experiments.stage_execution import StageAssignment, StageBackendFamily, StageKey
from Thesis_ML.experiments.stage_lease_manager import StageLeaseHandle, StageLeaseReleaseResult
from Thesis_ML.experiments.stage_observability import StageBoundaryRecorder
from Thesis_ML.experiments.stage_planner import StageResourceContract
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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


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
    telemetry_by_stage = {str(row.get("stage")): row for row in stage_execution["telemetry"]}
    assert telemetry_by_stage["dataset_selection"]["details"]["duration_source"] == "section_timing"
    assert telemetry_by_stage["model_fit"]["details"]["duration_source"] in {
        "fit_timing_summary.totals_seconds.outer_fold",
        "section_timing",
    }
    assert telemetry_by_stage["tuning"]["details"]["duration_source"] in {
        "not_applicable",
        "unavailable_derived_stage",
        "unavailable",
    }
    assignment_by_stage = {str(row.get("stage")): row for row in stage_execution["assignments"]}
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


def test_segment_execution_emits_stage_boundary_artifacts(prepared_dataset: dict[str, Path]) -> None:
    result = run_experiment(
        **_base_run_kwargs(prepared_dataset),
        run_id="segment_stage_boundary_artifacts",
        n_permutations=0,
        tuning_enabled=False,
    )
    report_dir = Path(result["report_dir"])
    stage_events_path = report_dir / "stage_events.jsonl"
    stage_observed_path = report_dir / "stage_observed_evidence.json"
    assert stage_events_path.exists()
    assert stage_observed_path.exists()

    events = _read_jsonl(stage_events_path)
    assert any(str(event.get("event_type")) == "stage_started" for event in events)
    assert any(str(event.get("event_type")) == "stage_finished" for event in events)
    assert any(str(event.get("stage_key")) == "model_fit" for event in events)
    assert any(str(event.get("stage_key")) == "reporting" for event in events)

    observed_payload = json.loads(stage_observed_path.read_text(encoding="utf-8"))
    assert observed_payload["schema_version"] == "stage-observed-evidence-v1"
    rows = {
        str(row.get("stage_key")): row
        for row in observed_payload.get("stages", [])
        if isinstance(row, dict)
    }
    assert rows["model_fit"]["observed_status"] == "executed"
    assert rows["tuning"]["observed_status"] == "skipped"
    assert rows["permutation"]["observed_status"] == "skipped"
    assert rows["reporting"]["observed_status"] == "executed"


def test_segment_execution_reused_stage_is_reflected_in_observed_evidence(
    prepared_dataset: dict[str, Path],
) -> None:
    run_experiment(
        **_base_run_kwargs(prepared_dataset),
        run_id="segment_stage_reuse_base",
    )
    reused_result = run_experiment(
        **_base_run_kwargs(prepared_dataset),
        run_id="segment_stage_reuse_followup",
        start_section="feature_cache_build",
        end_section="feature_cache_build",
        reuse_completed_artifacts=True,
    )
    observed_payload = json.loads(
        (Path(reused_result["report_dir"]) / "stage_observed_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    observed_rows = {
        str(row.get("stage_key")): row
        for row in observed_payload.get("stages", [])
        if isinstance(row, dict)
    }
    assert observed_rows["feature_cache_build"]["observed_status"] in {"reused", "executed"}


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
    assignment_by_stage = {str(row.get("stage")): row for row in stage_execution["assignments"]}
    assert assignment_by_stage["model_fit"]["executor_id"] == MODEL_FIT_CPU_EXECUTOR_ID
    assert assignment_by_stage["tuning"]["executor_id"] == SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID
    assert assignment_by_stage["permutation"]["executor_id"] == PERMUTATION_REFERENCE_EXECUTOR_ID
    telemetry_by_stage = {str(row.get("stage")): row for row in stage_execution["telemetry"]}
    assert (
        telemetry_by_stage["tuning"]["details"]["duration_source"]
        == "tuning_summary.timing_totals_seconds.tuned_search_total"
    )
    assert (
        telemetry_by_stage["permutation"]["details"]["duration_source"]
        == "metrics.permutation_test.permutation_loop_seconds"
    )
    stage_timings = result.get("stage_timings_seconds")
    assert isinstance(stage_timings, dict)
    assert "model_fit" in stage_timings
    assert "tuning" in stage_timings
    assert "permutation" in stage_timings

    permutation_payload = result["metrics"].get("permutation_test")
    assert isinstance(permutation_payload, dict)
    assert permutation_payload.get("permutation_executor_id") == PERMUTATION_REFERENCE_EXECUTOR_ID
    assert permutation_payload.get("execution_mode") == "grouped_nested_tuning_reference"
    assert permutation_payload.get("tuning_reapplied_under_null") is True
    assert permutation_payload.get("null_matches_confirmatory_setup") is True
    assert (
        permutation_payload.get("null_tuning_search_space_id")
        == LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID
    )
    assert (
        permutation_payload.get("null_tuning_search_space_version")
        == LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION
    )
    assert permutation_payload.get("null_inner_cv_scheme") == "grouped_leave_one_group_out"
    assert permutation_payload.get("null_inner_group_field") == "session"

    tuning_summary = json.loads(Path(result["tuning_summary_path"]).read_text(encoding="utf-8"))
    assert SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID in tuning_summary["tuning_executor_ids"]


def test_segment_execution_threads_tuning_metadata_into_evaluation_input(
    prepared_dataset: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    original_evaluation = segment_execution_module.evaluation

    def _capture_evaluation(section_input: object) -> object:
        captured["section_input"] = section_input
        return original_evaluation(section_input)

    monkeypatch.setattr(segment_execution_module, "evaluation", _capture_evaluation)
    run_kwargs = _base_run_kwargs(prepared_dataset)
    run_kwargs["model"] = "linearsvc"
    run_experiment(
        **run_kwargs,
        run_id="stage_planner_capture_evaluation_input",
        methodology_policy_name="grouped_nested_tuning",
        tuning_enabled=True,
        tuning_search_space_id=LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
        tuning_search_space_version=LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
        tuning_inner_cv_scheme="grouped_leave_one_group_out",
        tuning_inner_group_field="session",
        n_permutations=0,
    )

    section_input = captured.get("section_input")
    assert section_input is not None
    assert section_input.tuning_enabled is True
    assert section_input.tuning_search_space_id == LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID
    assert section_input.tuning_search_space_version == LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION
    assert section_input.tuning_inner_cv_scheme == "grouped_leave_one_group_out"
    assert section_input.tuning_inner_group_field == "session"
    tuning_assignment = section_input.tuning_assignment
    assert tuning_assignment is not None
    assert tuning_assignment.executor_id == SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID


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
    assignment_by_stage = {str(row.get("stage")): row for row in stage_execution["assignments"]}
    assert assignment_by_stage["model_fit"]["executor_id"] == MODEL_FIT_CPU_EXECUTOR_ID
    assert assignment_by_stage["tuning"]["executor_id"] == SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID
    assert assignment_by_stage["tuning"]["fallback_used"] is False

    tuning_summary = json.loads(Path(result["tuning_summary_path"]).read_text(encoding="utf-8"))
    assert SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID in tuning_summary["tuning_executor_ids"]
    assert tuning_summary["specialized_logreg_tuning_used"] is True
    assert int(tuning_summary["tuning_progress_event_count_total"]) > 0
    assert int(tuning_summary["tuning_progress_total_units_total"]) > 0


class _FakeStageLeaseManager:
    def __init__(self) -> None:
        self.acquire_requests: list[object] = []
        self.release_handles: list[object] = []

    def acquire(self, request: object) -> StageLeaseHandle:
        self.acquire_requests.append(request)
        return StageLeaseHandle(
            lease_id="lease-test-001",
            lease_class="gpu",
            run_id="segment_lease_run",
            stage_key="dataset_selection",
            owner_identity="segment_lease_run:dataset_selection",
            acquired_at_utc="2026-01-01T00:00:00+00:00",
            wait_seconds=0.125,
            queue_depth_at_acquire=1,
            lease_path=None,
        )

    def release(self, handle: object) -> StageLeaseReleaseResult:
        self.release_handles.append(handle)
        return StageLeaseReleaseResult(
            lease_id="lease-test-001",
            lease_class="gpu",
            released=True,
            released_at_utc="2026-01-01T00:00:02+00:00",
            hold_seconds=2.0,
        )


def _minimal_dataset_only_request(
    *,
    tmp_path: Path,
    observer: StageBoundaryRecorder,
    lease_manager: object,
    stage_resource_contracts: tuple[StageResourceContract, ...],
) -> SegmentExecutionRequest:
    index_csv = tmp_path / "index.csv"
    index_csv.write_text("row_id\n0\n", encoding="utf-8")
    report_dir = tmp_path / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    return SegmentExecutionRequest(
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        target_column="emotion",
        cv_mode="loso_session",
        model="ridge",
        subject=None,
        train_subject=None,
        test_subject=None,
        filter_task=None,
        filter_modality=None,
        seed=7,
        n_permutations=0,
        run_id="segment_lease_run",
        config_filename="config.json",
        report_dir=report_dir,
        artifact_registry_path=tmp_path / "artifact_registry.sqlite3",
        code_ref=None,
        affine_atol=1e-6,
        fold_metrics_path=report_dir / "fold_metrics.csv",
        fold_splits_path=report_dir / "fold_splits.csv",
        predictions_path=report_dir / "predictions.csv",
        metrics_path=report_dir / "metrics.json",
        subgroup_metrics_json_path=report_dir / "subgroup_metrics.json",
        subgroup_metrics_csv_path=report_dir / "subgroup_metrics.csv",
        tuning_summary_path=report_dir / "tuning_summary.json",
        tuning_best_params_path=report_dir / "tuning_best_params.csv",
        fit_timing_summary_path=report_dir / "fit_timing_summary.json",
        spatial_report_path=report_dir / "spatial.json",
        calibration_summary_path=report_dir / "calibration_summary.json",
        calibration_table_path=report_dir / "calibration_table.csv",
        interpretability_summary_path=report_dir / "interpretability_summary.json",
        interpretability_fold_artifacts_path=report_dir / "interpretability_folds.csv",
        start_section="dataset_selection",
        end_section="dataset_selection",
        build_pipeline_fn=lambda **_: None,
        load_features_from_cache_fn=lambda **_: (np.zeros((0, 0)), pd.DataFrame(), {}),
        scores_for_predictions_fn=lambda **_: {},
        extract_linear_coefficients_fn=lambda **_: (np.zeros((0, 0)), np.zeros((0,)), []),
        compute_interpretability_stability_fn=lambda _: {},
        evaluate_permutations_fn=lambda **_: {},
        stage_assignments=(
            StageAssignment(
                stage=StageKey.DATASET_SELECTION,
                backend_family=StageBackendFamily.SKLEARN_CPU,
                compute_lane="cpu",
                source="stage_planner_v1",
                reason="unit_test",
                executor_id="dataset_selection_cpu_reference_v1",
                official_admitted=True,
            ),
        ),
        stage_fallback_executor_ids={},
        stage_observer=observer,
        stage_resource_contracts=stage_resource_contracts,
        stage_lease_manager=lease_manager,
    )


def test_segment_execution_acquires_and_releases_stage_lease_when_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = StageBoundaryRecorder(report_dir=tmp_path / "report", run_id="segment_lease_run")
    lease_manager = _FakeStageLeaseManager()
    contract = StageResourceContract(
        stage_key=StageKey.DATASET_SELECTION,
        requires_gpu_lease=True,
        preferred_compute_lane="gpu",
        lease_class="gpu",
        lease_reason="test_gpu_required",
        expected_backend_family=StageBackendFamily.TORCH_GPU,
        expected_executor_id="dataset_selection_gpu_test",
    )
    monkeypatch.setattr(
        segment_execution_module,
        "dataset_selection",
        lambda *_args, **_kwargs: SimpleNamespace(selected_index_df=pd.DataFrame()),
    )

    result = execute_section_segment(
        _minimal_dataset_only_request(
            tmp_path=tmp_path,
            observer=observer,
            lease_manager=lease_manager,
            stage_resource_contracts=(contract,),
        )
    )

    assert result.executed_sections == ["dataset_selection"]
    assert len(lease_manager.acquire_requests) == 1
    assert len(lease_manager.release_handles) == 1

    observed_payload = json.loads(
        (tmp_path / "report" / "stage_observed_evidence.json").read_text(encoding="utf-8")
    )
    rows = {
        str(row.get("stage_key")): row
        for row in observed_payload.get("stages", [])
        if isinstance(row, dict)
    }
    assert rows["dataset_selection"]["lease_required"] is True
    assert rows["dataset_selection"]["lease_acquired"] is True
    assert rows["dataset_selection"]["lease_released"] is True


def test_segment_execution_skips_stage_lease_when_not_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = StageBoundaryRecorder(report_dir=tmp_path / "report", run_id="segment_lease_run")
    lease_manager = _FakeStageLeaseManager()
    contract = StageResourceContract(
        stage_key=StageKey.DATASET_SELECTION,
        requires_gpu_lease=False,
        preferred_compute_lane="cpu",
        lease_class="cpu",
        lease_reason="test_cpu_only",
        expected_backend_family=StageBackendFamily.SKLEARN_CPU,
        expected_executor_id="dataset_selection_cpu_reference_v1",
    )
    monkeypatch.setattr(
        segment_execution_module,
        "dataset_selection",
        lambda *_args, **_kwargs: SimpleNamespace(selected_index_df=pd.DataFrame()),
    )

    result = execute_section_segment(
        _minimal_dataset_only_request(
            tmp_path=tmp_path,
            observer=observer,
            lease_manager=lease_manager,
            stage_resource_contracts=(contract,),
        )
    )

    assert result.executed_sections == ["dataset_selection"]
    assert lease_manager.acquire_requests == []
    assert lease_manager.release_handles == []


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
