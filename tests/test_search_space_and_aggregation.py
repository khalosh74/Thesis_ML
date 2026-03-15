from __future__ import annotations

import json
from pathlib import Path

import pytest

from Thesis_ML.orchestration.contracts import SearchSpaceSpec
from Thesis_ML.orchestration.result_aggregation import (
    aggregate_variant_records,
    build_summary_output_rows,
)
from Thesis_ML.orchestration.search_space import expand_variant_search_space


def test_deterministic_search_space_expands_cartesian_grid() -> None:
    base_variant = {
        "template_id": "t1",
        "params": {
            "target": "coarse_affect",
            "cv": "within_subject_loso_session",
            "model": "ridge",
        },
        "supported": True,
        "start_section": "dataset_selection",
        "end_section": "evaluation",
    }
    search_space = SearchSpaceSpec.model_validate(
        {
            "search_space_id": "SS01",
            "enabled": True,
            "optimization_mode": "deterministic_grid",
            "objective_metric": "balanced_accuracy",
            "dimensions": [
                {"parameter_name": "model", "values": ["ridge", "logreg"]},
                {
                    "parameter_name": "start_section",
                    "values": ["dataset_selection", "feature_matrix_load"],
                },
            ],
        }
    )

    expanded = expand_variant_search_space(
        base_variant,
        search_space=search_space,
        seed=42,
        optuna_enabled=False,
    )

    assert len(expanded) == 4
    assignments = {(str(row["params"]["model"]), str(row["start_section"])) for row in expanded}
    assert assignments == {
        ("ridge", "dataset_selection"),
        ("ridge", "feature_matrix_load"),
        ("logreg", "dataset_selection"),
        ("logreg", "feature_matrix_load"),
    }


def test_optuna_search_space_requires_optuna_mode_flag() -> None:
    base_variant = {
        "template_id": "t1",
        "params": {
            "target": "coarse_affect",
            "cv": "within_subject_loso_session",
            "model": "ridge",
        },
        "supported": True,
    }
    search_space = SearchSpaceSpec.model_validate(
        {
            "search_space_id": "SS_OPTUNA",
            "enabled": True,
            "optimization_mode": "optuna",
            "objective_metric": "balanced_accuracy",
            "dimensions": [
                {"parameter_name": "model", "values": ["ridge", "logreg"]},
            ],
        }
    )

    try:
        expand_variant_search_space(
            base_variant,
            search_space=search_space,
            seed=42,
            optuna_enabled=False,
        )
    except ValueError as exc:
        assert "optuna mode is disabled" in str(exc)
    else:
        raise AssertionError("Expected ValueError when optuna search is used without optuna mode.")


def test_optuna_search_space_expands_when_enabled() -> None:
    pytest.importorskip("optuna")

    base_variant = {
        "template_id": "t1",
        "params": {
            "target": "coarse_affect",
            "cv": "within_subject_loso_session",
            "model": "ridge",
        },
        "supported": True,
    }
    search_space = SearchSpaceSpec.model_validate(
        {
            "search_space_id": "SS_OPTUNA_ENABLED",
            "enabled": True,
            "optimization_mode": "optuna",
            "objective_metric": "balanced_accuracy",
            "dimensions": [
                {"parameter_name": "model", "values": ["ridge", "logreg"]},
                {"parameter_name": "start_section", "values": ["dataset_selection"]},
            ],
            "max_trials": 4,
        }
    )

    expanded = expand_variant_search_space(
        base_variant,
        search_space=search_space,
        seed=42,
        optuna_enabled=True,
        optuna_trials=4,
    )

    assert expanded
    assert len(expanded) <= 4
    for row in expanded:
        assignment = row.get("search_assignment", {})
        assert str(assignment.get("model")) in {"ridge", "logreg"}
        assert str(row["params"]["model"]) in {"ridge", "logreg"}


def _write_metrics(path: Path, *, performed: bool, status: str) -> None:
    payload = {
        "balanced_accuracy": 0.6,
        "interpretability": {
            "performed": performed,
            "status": status,
        },
    }
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def test_aggregate_variant_records_summarizes_segments_and_xai(tmp_path: Path) -> None:
    metrics_full = tmp_path / "full_metrics.json"
    metrics_segment = tmp_path / "segment_metrics.json"
    _write_metrics(metrics_full, performed=True, status="performed")
    _write_metrics(metrics_segment, performed=False, status="not_applicable")

    records = [
        {
            "status": "completed",
            "experiment_id": "E16",
            "variant_id": "v001",
            "trial_id": "S01_cell_001_r001",
            "study_id": "S01",
            "cell_id": "S01_cell_001",
            "factor_settings": {"model": "ridge", "filter_task": "passive"},
            "run_id": "run_full_best",
            "model": "ridge",
            "cv": "within_subject_loso_session",
            "target": "coarse_affect",
            "start_section": None,
            "end_section": None,
            "primary_metric_name": "balanced_accuracy",
            "primary_metric_value": 0.64,
            "report_dir": str(tmp_path / "reports" / "full_best"),
            "metrics_path": str(metrics_full),
        },
        {
            "status": "completed",
            "experiment_id": "E16",
            "variant_id": "v002",
            "trial_id": "S01_cell_002_r001",
            "study_id": "S01",
            "cell_id": "S01_cell_002",
            "factor_settings": {"model": "ridge", "filter_task": "emo"},
            "run_id": "run_full_other",
            "model": "ridge",
            "cv": "within_subject_loso_session",
            "target": "coarse_affect",
            "start_section": "dataset_selection",
            "end_section": "evaluation",
            "primary_metric_name": "balanced_accuracy",
            "primary_metric_value": 0.58,
            "report_dir": str(tmp_path / "reports" / "full_other"),
            "metrics_path": str(metrics_full),
        },
        {
            "status": "completed",
            "experiment_id": "E17",
            "variant_id": "v003",
            "run_id": "run_segment_best",
            "model": "logreg",
            "cv": "frozen_cross_person_transfer",
            "target": "coarse_affect",
            "start_section": "feature_matrix_load",
            "end_section": "evaluation",
            "primary_metric_name": "balanced_accuracy",
            "primary_metric_value": 0.69,
            "report_dir": str(tmp_path / "reports" / "segment_best"),
            "metrics_path": str(metrics_segment),
        },
        {
            "status": "blocked",
            "experiment_id": "E18",
            "variant_id": "v004",
            "run_id": "run_blocked",
            "model": "ridge",
            "cv": "within_subject_loso_session",
            "target": "coarse_affect",
            "primary_metric_name": "balanced_accuracy",
            "primary_metric_value": None,
            "metrics_path": "",
        },
    ]

    aggregation = aggregate_variant_records(records, top_k=3)
    assert aggregation["completed_with_metric_count"] == 3
    assert aggregation["factorial"]["n_factorial_trials"] == 2
    assert aggregation["best_full_pipeline_runs"][0]["run_id"] == "run_full_best"
    assert aggregation["best_segment_runs"][0]["run_id"] == "run_segment_best"
    section_keys = {row["section_key"] for row in aggregation["section_level_effects"]}
    assert "feature_matrix_load->evaluation" in section_keys

    xai_methods = {row["xai_method"] for row in aggregation["xai"]["method_effects"]}
    assert "linear_coefficients_stability" in xai_methods

    rows = build_summary_output_rows(aggregation)
    summary_types = {row["summary_type"] for row in rows}
    assert "best_full_pipeline" in summary_types
    assert "best_segment_run" in summary_types
    assert "section_effect" in summary_types
    assert "xai_method_effect" in summary_types
    assert "factor_level_effect_descriptive" in summary_types
    metric_names = {
        str(row.get("primary_metric_name"))
        for row in rows
        if row.get("primary_metric_name") not in (None, "")
    }
    assert metric_names == {"balanced_accuracy"}
    assert "mean_primary_metric_value" not in metric_names
