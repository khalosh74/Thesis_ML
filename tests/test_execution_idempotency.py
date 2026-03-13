from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from Thesis_ML.experiments.execution_policy import read_run_status
from Thesis_ML.experiments.run_experiment import run_experiment
from Thesis_ML.experiments.segment_execution import SegmentExecutionResult
from Thesis_ML.orchestration.workbook_writeback import write_workbook_results
from Thesis_ML.workbook.builder import build_workbook


def _base_run_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "index_csv": tmp_path / "dataset_index.csv",
        "data_root": tmp_path / "Data",
        "cache_dir": tmp_path / "cache",
        "target": "emotion",
        "model": "ridge",
        "cv": "loso_session",
        "run_id": "idempotency_run",
        "reports_root": tmp_path / "reports" / "experiments",
    }


def _successful_segment_stub(request) -> SegmentExecutionResult:
    request.report_dir.mkdir(parents=True, exist_ok=True)
    request.metrics_path.write_text(
        json.dumps({"accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.5}) + "\n",
        encoding="utf-8",
    )
    request.interpretability_summary_path.write_text(
        json.dumps(
            {
                "enabled": False,
                "performed": False,
                "status": "not_applicable",
                "fold_artifacts_path": None,
                "stability": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    request.spatial_report_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "passed": True,
                "n_groups_checked": 1,
                "reference_group_id": "g1",
                "affine_atol": 1e-5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    request.fold_metrics_path.write_text("fold,score\n0,0.5\n", encoding="utf-8")
    request.fold_splits_path.write_text("fold,train,test\n0,a,b\n", encoding="utf-8")
    request.predictions_path.write_text("y_true,y_pred\nanger,anger\n", encoding="utf-8")
    return SegmentExecutionResult(
        planned_sections=["dataset_selection", "feature_cache_build", "feature_matrix_load"],
        executed_sections=["dataset_selection", "feature_cache_build", "feature_matrix_load"],
        reused_sections=[],
        artifact_ids={
            "feature_cache": "feature_cache_fake",
            "feature_matrix_bundle": "feature_matrix_fake",
            "metrics_bundle": "metrics_fake",
            "interpretability_bundle": "interpretability_fake",
        },
        metrics={"accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.5, "n_folds": 1},
        spatial_compatibility={
            "status": "passed",
            "passed": True,
            "n_groups_checked": 1,
            "reference_group_id": "g1",
            "affine_atol": 1e-5,
        },
        interpretability_summary={
            "enabled": False,
            "performed": False,
            "status": "not_applicable",
            "fold_artifacts_path": None,
            "stability": None,
        },
    )


def test_rerun_without_force_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "Thesis_ML.experiments.run_experiment.execute_section_segment",
        _successful_segment_stub,
    )
    kwargs = _base_run_kwargs(tmp_path)

    first = run_experiment(**kwargs)
    assert Path(first["report_dir"]).exists()

    with pytest.raises(FileExistsError, match="already completed"):
        run_experiment(**kwargs)


def test_force_and_resume_mutually_exclusive_raises(
    tmp_path: Path,
) -> None:
    kwargs = _base_run_kwargs(tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_experiment(**kwargs, force=True, resume=True)


def test_resume_requires_existing_run_directory(
    tmp_path: Path,
) -> None:
    kwargs = _base_run_kwargs(tmp_path)
    with pytest.raises(FileNotFoundError, match="Cannot resume run"):
        run_experiment(**kwargs, resume=True)


def test_rerun_with_force_replaces_existing_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "Thesis_ML.experiments.run_experiment.execute_section_segment",
        _successful_segment_stub,
    )
    kwargs = _base_run_kwargs(tmp_path)

    first = run_experiment(**kwargs)
    report_dir = Path(first["report_dir"])
    marker = report_dir / "old_marker.txt"
    marker.write_text("old\n", encoding="utf-8")
    assert marker.exists()

    second = run_experiment(**kwargs, force=True)
    assert Path(second["report_dir"]) == report_dir
    assert not marker.exists()
    assert second["run_mode"] == "forced_rerun"


def test_resume_after_partial_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_state = {"count": 0, "reuse_flag": None}

    def _flaky_stub(request) -> SegmentExecutionResult:
        call_state["count"] += 1
        if call_state["count"] == 1:
            raise RuntimeError("simulated partial failure")
        call_state["reuse_flag"] = bool(request.reuse_completed_artifacts)
        return _successful_segment_stub(request)

    monkeypatch.setattr(
        "Thesis_ML.experiments.run_experiment.execute_section_segment",
        _flaky_stub,
    )
    kwargs = _base_run_kwargs(tmp_path)

    with pytest.raises(RuntimeError, match="simulated partial failure"):
        run_experiment(**kwargs)

    report_dir = kwargs["reports_root"] / kwargs["run_id"]
    status = read_run_status(report_dir)
    assert status is not None
    assert status["status"] == "failed"

    with pytest.raises(RuntimeError, match="Use resume=True to continue"):
        run_experiment(**kwargs)

    resumed = run_experiment(**kwargs, resume=True)
    assert resumed["run_mode"] == "resume"
    assert call_state["reuse_flag"] is True
    status_after = read_run_status(report_dir)
    assert status_after is not None
    assert status_after["status"] == "completed"


def test_explicit_reuse_completed_artifacts_flag_is_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_state = {"reuse_flag": None}

    def _capturing_stub(request) -> SegmentExecutionResult:
        call_state["reuse_flag"] = bool(request.reuse_completed_artifacts)
        return _successful_segment_stub(request)

    monkeypatch.setattr(
        "Thesis_ML.experiments.run_experiment.execute_section_segment",
        _capturing_stub,
    )
    kwargs = _base_run_kwargs(tmp_path)
    result = run_experiment(**kwargs, reuse_completed_artifacts=True)
    assert result["run_mode"] == "fresh"
    assert call_state["reuse_flag"] is True


def test_output_directory_collision_without_status_requires_explicit_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "Thesis_ML.experiments.run_experiment.execute_section_segment",
        _successful_segment_stub,
    )
    kwargs = _base_run_kwargs(tmp_path)
    report_dir = kwargs["reports_root"] / kwargs["run_id"]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "orphaned.txt").write_text("orphaned\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Use resume=True to continue a partial run"):
        run_experiment(**kwargs)


def test_writeback_collision_is_safe_and_overwrite_is_explicit(tmp_path: Path) -> None:
    workbook_path = tmp_path / "thesis_experiment_program.xlsx"
    build_workbook().save(workbook_path)

    row_machine = {
        "machine_id": "machine_1",
        "hostname": "localhost",
        "environment_name": "test",
        "python_version": "3.12",
        "gpu": "none",
        "status": "Closed",
        "last_checked": "2026-01-01T00:00:00Z",
        "notes": "test",
    }
    row_trial = {
        "trial_id": "trial_1",
        "experiment_id": "E01",
        "run_id": "run_1",
        "status": "completed",
        "primary_metric_name": "balanced_accuracy",
        "primary_metric_value": 0.5,
        "report_path": "",
        "metrics_path": "",
        "artifact_bundle": "",
        "notes": "ok",
    }

    first = write_workbook_results(
        source_workbook_path=workbook_path,
        version_tag="collision_tag",
        machine_status_rows=[row_machine],
        trial_result_rows=[row_trial],
        summary_output_rows=[],
        run_log_rows=[],
        append_run_log=False,
        output_dir=tmp_path / "out",
    )
    assert first.exists()

    with pytest.raises(
        ValueError, match="Refusing to overwrite existing workbook write-back target"
    ):
        write_workbook_results(
            source_workbook_path=workbook_path,
            version_tag="collision_tag",
            machine_status_rows=[row_machine],
            trial_result_rows=[row_trial],
            summary_output_rows=[],
            run_log_rows=[],
            append_run_log=False,
            output_dir=tmp_path / "out",
        )

    replaced = write_workbook_results(
        source_workbook_path=workbook_path,
        version_tag="collision_tag",
        machine_status_rows=[row_machine],
        trial_result_rows=[row_trial],
        summary_output_rows=[],
        run_log_rows=[],
        append_run_log=False,
        output_dir=tmp_path / "out",
        overwrite_existing=True,
    )
    assert replaced == first
