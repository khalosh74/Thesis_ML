from __future__ import annotations

import json
from pathlib import Path

import pytest

from Thesis_ML.config.framework_mode import FrameworkMode, coerce_framework_mode
from Thesis_ML.config.paths import (
    DEFAULT_COMPARISON_REPORTS_ROOT,
    DEFAULT_CONFIRMATORY_REPORTS_ROOT,
    DEFAULT_EXPLORATORY_REPORTS_ROOT,
)
from Thesis_ML.experiments.execution_policy import read_run_status
from Thesis_ML.experiments.run_experiment import run_experiment
from Thesis_ML.experiments.segment_execution import SegmentExecutionResult


def _base_run_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "index_csv": tmp_path / "dataset_index.csv",
        "data_root": tmp_path / "Data",
        "cache_dir": tmp_path / "cache",
        "target": "emotion",
        "model": "ridge",
        "cv": "loso_session",
        "run_id": "framework_mode_smoke",
        "reports_root": tmp_path / "reports" / "exploratory",
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


def test_framework_mode_coercion() -> None:
    assert coerce_framework_mode("exploratory") == FrameworkMode.EXPLORATORY
    assert coerce_framework_mode("locked_comparison") == FrameworkMode.LOCKED_COMPARISON
    assert coerce_framework_mode("confirmatory") == FrameworkMode.CONFIRMATORY
    with pytest.raises(ValueError, match="Unsupported framework_mode"):
        coerce_framework_mode("invalid_mode")


def test_default_reports_roots_are_mode_separated() -> None:
    assert str(DEFAULT_EXPLORATORY_REPORTS_ROOT).replace("\\", "/").endswith(
        "outputs/reports/exploratory"
    )
    assert str(DEFAULT_COMPARISON_REPORTS_ROOT).replace("\\", "/").endswith(
        "outputs/reports/comparisons"
    )
    assert str(DEFAULT_CONFIRMATORY_REPORTS_ROOT).replace("\\", "/").endswith(
        "outputs/reports/confirmatory"
    )


def test_exploratory_runner_stamps_framework_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "Thesis_ML.experiments.run_experiment.execute_section_segment",
        _successful_segment_stub,
    )
    kwargs = _base_run_kwargs(tmp_path)
    result = run_experiment(**kwargs)

    report_dir = Path(result["report_dir"])
    config = json.loads((report_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    status = read_run_status(report_dir)

    assert config["framework_mode"] == FrameworkMode.EXPLORATORY.value
    assert config["canonical_run"] is False
    assert metrics["framework_mode"] == FrameworkMode.EXPLORATORY.value
    assert metrics["canonical_run"] is False
    assert result["framework_mode"] == FrameworkMode.EXPLORATORY.value
    assert status is not None and status["status"] == "completed"


def test_mode_boundary_rejects_illegal_context_crossing(tmp_path: Path) -> None:
    kwargs = _base_run_kwargs(tmp_path)

    with pytest.raises(ValueError, match="cannot accept protocol_context"):
        run_experiment(
            **kwargs,
            framework_mode=FrameworkMode.EXPLORATORY,
            protocol_context={
                "framework_mode": FrameworkMode.CONFIRMATORY.value,
                "canonical_run": True,
                "protocol_id": "x",
                "protocol_version": "1",
                "protocol_schema_version": "thesis-protocol-v1",
                "suite_id": "s",
                "claim_ids": ["c1"],
            },
        )

    with pytest.raises(ValueError, match="requires non-empty protocol_context"):
        run_experiment(
            **kwargs,
            framework_mode=FrameworkMode.CONFIRMATORY,
        )

    with pytest.raises(ValueError, match="requires non-empty comparison_context"):
        run_experiment(
            **kwargs,
            framework_mode=FrameworkMode.LOCKED_COMPARISON,
        )

