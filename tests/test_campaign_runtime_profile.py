from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import Thesis_ML.verification.campaign_runtime_profile as runtime_profile
from Thesis_ML.experiments.run_experiment import (
    _build_pipeline,
    _extract_linear_coefficients,
    _scores_for_predictions,
)
from Thesis_ML.experiments.section_models import ModelFitInput
from Thesis_ML.experiments.sections_impl import execute_model_fit


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _demo_index_csv() -> Path:
    return _repo_root() / "demo_data" / "synthetic_v1" / "dataset_index.csv"


def _confirmatory_protocol() -> Path:
    return _repo_root() / "configs" / "protocols" / "thesis_canonical_v1.json"


def _comparison_specs() -> list[Path]:
    return [
        _repo_root() / "configs" / "comparisons" / "model_family_comparison_v1.json",
        _repo_root() / "configs" / "comparisons" / "model_family_grouped_nested_comparison_v1.json",
    ]


def test_campaign_runtime_profile_summary_shape_and_positive_eta(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        run_id = str(kwargs["run_id"])
        report_dir = Path(kwargs["reports_root"]) / run_id
        return {
            "run_id": run_id,
            "report_dir": str(report_dir),
            "run_status_path": str(report_dir / "run_status.json"),
            "stage_timings_seconds": {"total": 7.5},
            "metrics": {"n_folds": 1, "primary_metric_value": 0.5},
        }

    monkeypatch.setattr(runtime_profile, "run_experiment", _fake_run_experiment)

    summary = runtime_profile.verify_campaign_runtime_profile(
        index_csv=_demo_index_csv(),
        data_root=_repo_root() / "demo_data" / "synthetic_v1" / "data_root",
        cache_dir=_repo_root() / "demo_data" / "synthetic_v1" / "cache",
        confirmatory_protocol=_confirmatory_protocol(),
        comparison_specs=_comparison_specs(),
        profile_root=tmp_path / "runtime_profiles",
    )

    assert summary["passed"] is True
    assert int(summary["profiling_runs_executed"]) > 0
    assert int(summary["profiling_runs_executed"]) == int(summary["n_cohorts"])
    assert float(summary["estimated_total_wall_time_seconds"]) > 0.0
    assert isinstance(summary["cohort_estimates"], list)
    assert summary["cohort_estimates"]
    assert all(item["status"] == "passed" for item in summary["cohort_estimates"])
    assert "phase_estimates" in summary
    assert "model_estimates" in summary

    assert calls
    for call in calls:
        profiling_context = call.get("profiling_context")
        assert isinstance(profiling_context, dict)
        assert profiling_context.get("source") == "campaign_runtime_profile_precheck"
        assert profiling_context.get("profiling_only") is True
        assert profiling_context.get("precheck_only") is True
        assert int(profiling_context.get("max_outer_folds", 0)) == 1


def test_campaign_runtime_profile_emits_runtime_warnings_and_recommendations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _slow_fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
        profile_context = kwargs.get("profiling_context", {})
        source_phase = str(profile_context.get("source_phase", "comparison"))
        elapsed = 2 * 3600.0 if source_phase == "comparison" else 20 * 60.0
        run_id = str(kwargs["run_id"])
        report_dir = Path(kwargs["reports_root"]) / run_id
        return {
            "run_id": run_id,
            "report_dir": str(report_dir),
            "run_status_path": str(report_dir / "run_status.json"),
            "stage_timings_seconds": {"total": elapsed},
            "metrics": {"n_folds": 1, "primary_metric_value": 0.5},
        }

    monkeypatch.setattr(runtime_profile, "run_experiment", _slow_fake_run_experiment)

    summary = runtime_profile.verify_campaign_runtime_profile(
        index_csv=_demo_index_csv(),
        data_root=_repo_root() / "demo_data" / "synthetic_v1" / "data_root",
        cache_dir=_repo_root() / "demo_data" / "synthetic_v1" / "cache",
        confirmatory_protocol=_confirmatory_protocol(),
        comparison_specs=_comparison_specs(),
        profile_root=tmp_path / "runtime_profiles",
    )

    warning_codes = {str(item.get("code")) for item in summary.get("warnings", [])}
    recommendation_codes = {str(item.get("code")) for item in summary.get("recommendations", [])}
    assert "estimated_runtime_exceeds_8h" in warning_codes
    assert "run_confirmatory_before_comparison" in recommendation_codes


def test_campaign_runtime_profile_fails_when_a_cohort_profile_run_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _failing_fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
        profile_context = kwargs.get("profiling_context", {})
        source_run_id = str(profile_context.get("source_run_id", ""))
        if "linearsvc" in source_run_id:
            raise RuntimeError("synthetic profile failure")
        run_id = str(kwargs["run_id"])
        report_dir = Path(kwargs["reports_root"]) / run_id
        return {
            "run_id": run_id,
            "report_dir": str(report_dir),
            "run_status_path": str(report_dir / "run_status.json"),
            "stage_timings_seconds": {"total": 5.0},
            "metrics": {"n_folds": 1, "primary_metric_value": 0.5},
        }

    monkeypatch.setattr(runtime_profile, "run_experiment", _failing_fake_run_experiment)

    summary = runtime_profile.verify_campaign_runtime_profile(
        index_csv=_demo_index_csv(),
        data_root=_repo_root() / "demo_data" / "synthetic_v1" / "data_root",
        cache_dir=_repo_root() / "demo_data" / "synthetic_v1" / "cache",
        confirmatory_protocol=_confirmatory_protocol(),
        comparison_specs=_comparison_specs(),
        profile_root=tmp_path / "runtime_profiles",
    )

    assert summary["passed"] is False
    issue_codes = {str(item.get("code")) for item in summary.get("issues", [])}
    assert "profiling_run_failed" in issue_codes


def test_model_fit_outer_fold_cap_only_applies_when_profiling_flag_set(tmp_path: Path) -> None:
    metadata_df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "bas": "BAS2",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-01",
                "bas": "BAS2",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s3",
                "subject": "sub-001",
                "session": "ses-02",
                "bas": "BAS2",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s4",
                "subject": "sub-001",
                "session": "ses-02",
                "bas": "BAS2",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s5",
                "subject": "sub-001",
                "session": "ses-03",
                "bas": "BAS2",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s6",
                "subject": "sub-001",
                "session": "ses-03",
                "bas": "BAS2",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
                "coarse_affect": "positive",
            },
        ]
    )
    x_matrix = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.2, 1.1],
            [1.1, 0.3],
            [0.1, 1.3],
            [1.2, 0.1],
        ],
        dtype=np.float64,
    )

    common_kwargs = {
        "x_matrix": x_matrix,
        "metadata_df": metadata_df,
        "target_column": "coarse_affect",
        "cv_mode": "within_subject_loso_session",
        "model": "ridge",
        "subject": "sub-001",
        "seed": 42,
        "run_id": "profile_fold_cap_test",
        "config_filename": "config.json",
        "report_dir": tmp_path,
        "build_pipeline_fn": lambda model_name, seed: _build_pipeline(
            model_name=model_name,
            seed=seed,
            class_weight_policy="none",
        ),
        "scores_for_predictions_fn": _scores_for_predictions,
        "extract_linear_coefficients_fn": _extract_linear_coefficients,
    }

    full_output = execute_model_fit(ModelFitInput(**common_kwargs))
    capped_output = execute_model_fit(ModelFitInput(**common_kwargs, max_outer_folds=1))

    assert len(full_output["splits"]) >= 2
    assert len(capped_output["splits"]) == 1
