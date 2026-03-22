from __future__ import annotations

import json
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
from Thesis_ML.experiments.tuning_search_spaces import (
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
)


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
    assert int(summary["profiling_runs_executed"]) + int(
        summary.get("fallback_estimates_used", 0)
    ) == int(summary["n_cohorts"])
    assert float(summary["estimated_total_wall_time_seconds"]) > 0.0
    assert isinstance(summary["cohort_estimates"], list)
    assert summary["cohort_estimates"]
    assert all(item["status"] == "passed" for item in summary["cohort_estimates"])
    assert all(
        str(item.get("estimate_source")) in {"measured_profile", "conservative_fallback"}
        for item in summary["cohort_estimates"]
    )
    assert all(
        str(item.get("estimate_confidence")) in {"medium", "low"}
        for item in summary["cohort_estimates"]
    )
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


def test_campaign_runtime_profile_profile_permutations_override_and_extrapolation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        run_id = str(kwargs["run_id"])
        report_dir = Path(kwargs["reports_root"]) / run_id
        profiled_n_permutations = int(kwargs["n_permutations"])
        return {
            "run_id": run_id,
            "report_dir": str(report_dir),
            "run_status_path": str(report_dir / "run_status.json"),
            "stage_timings_seconds": {"total": 15.0},
            "metrics": {
                "n_folds": 1,
                "primary_metric_value": 0.5,
                "permutation_test": {
                    "permutation_loop_seconds": float(profiled_n_permutations),
                },
            },
        }

    monkeypatch.setattr(runtime_profile, "run_experiment", _fake_run_experiment)

    summary = runtime_profile.verify_campaign_runtime_profile(
        index_csv=_demo_index_csv(),
        data_root=_repo_root() / "demo_data" / "synthetic_v1" / "data_root",
        cache_dir=_repo_root() / "demo_data" / "synthetic_v1" / "cache",
        confirmatory_protocol=_confirmatory_protocol(),
        comparison_specs=_comparison_specs(),
        profile_root=tmp_path / "runtime_profiles",
        profile_permutations=5,
    )

    assert summary["passed"] is True
    assert summary["inputs"]["profile_permutations_override"] == 5
    assert calls
    assert any(int(call["n_permutations"]) > 0 for call in calls)

    for call in calls:
        profile_context = call.get("profiling_context")
        assert isinstance(profile_context, dict)
        configured = int(profile_context.get("configured_n_permutations", 0))
        profiled = int(profile_context.get("profiled_n_permutations", 0))
        if configured > 0:
            assert profiled == min(configured, 5)
            assert int(call["n_permutations"]) == profiled

    measured_rows = [
        row
        for row in summary["cohort_estimates"]
        if str(row.get("status")) == "passed"
        and str(row.get("estimate_source")) == "measured_profile"
        and int(row.get("configured_n_permutations", 0)) > 0
    ]
    assert measured_rows

    extrapolated_rows = [
        row
        for row in measured_rows
        if int(row.get("configured_n_permutations", 0)) > int(row.get("profiled_n_permutations", 0))
    ]
    assert extrapolated_rows
    for row in extrapolated_rows:
        assert bool(row["permutation_extrapolation_applied"]) is True
        measured_loop = float(row["permutation_loop_measured_seconds"])
        configured = int(row["configured_n_permutations"])
        profiled = int(row["profiled_n_permutations"])
        expected = measured_loop * float(configured) / float(profiled)
        assert float(row["estimated_full_permutation_seconds"]) == expected


def test_campaign_runtime_profile_uses_fallback_for_unprofileable_grouped_nested_cohorts(
    tmp_path: Path, monkeypatch
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
            "stage_timings_seconds": {"total": 3.0},
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
    assert int(summary.get("fallback_estimates_used", 0)) > 0
    warning_codes = {str(item.get("code")) for item in summary.get("warnings", [])}
    assert "profiling_fallback_used" in warning_codes
    fallback_rows = [
        row
        for row in summary["cohort_estimates"]
        if str(row.get("estimate_source")) == "conservative_fallback"
    ]
    assert fallback_rows
    assert all(str(row.get("estimate_confidence")) == "low" for row in fallback_rows)
    assert all(row.get("fallback_reason") for row in fallback_rows)


def test_campaign_runtime_profile_feature_matrix_memoization_reuses_exact_keys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    load_calls = 0
    cache_manifest = tmp_path / "cache" / "cache_manifest.csv"
    cache_manifest.parent.mkdir(parents=True, exist_ok=True)
    cache_manifest.write_text("cache_key,cache_path\n", encoding="utf-8")

    def _fake_load_features_from_cache(
        *,
        index_df: pd.DataFrame,
        cache_manifest_path: Path,
        spatial_report_path: Path | None = None,
        affine_atol: float,
    ) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
        del spatial_report_path
        del affine_atol
        assert Path(cache_manifest_path).resolve() == cache_manifest.resolve()
        nonlocal load_calls
        load_calls += 1
        x_matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        metadata_df = index_df.reset_index(drop=True).copy(deep=True)
        return x_matrix, metadata_df, {"status": "passed", "passed": True}

    def _fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
        loader = kwargs.get("load_features_from_cache_fn_override")
        assert callable(loader)
        loader(
            index_df=pd.DataFrame(
                [
                    {"sample_id": "sample_a", "beta_path": "beta_a.nii"},
                    {"sample_id": "sample_b", "beta_path": "beta_b.nii"},
                ]
            ),
            cache_manifest_path=cache_manifest,
            spatial_report_path=tmp_path / "spatial_report.json",
            affine_atol=1e-5,
        )
        run_id = str(kwargs["run_id"])
        report_dir = Path(kwargs["reports_root"]) / run_id
        return {
            "run_id": run_id,
            "report_dir": str(report_dir),
            "run_status_path": str(report_dir / "run_status.json"),
            "stage_timings_seconds": {"total": 3.0},
            "metrics": {"n_folds": 1, "primary_metric_value": 0.5},
        }

    monkeypatch.setattr(runtime_profile, "load_features_from_cache", _fake_load_features_from_cache)
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
    assert int(summary["profiling_runs_executed"]) > 1
    memo_summary = summary["feature_matrix_memoization"]
    assert memo_summary["enabled"] is True
    assert int(memo_summary["hits"]) >= 1
    assert int(memo_summary["misses"]) >= 1
    assert int(memo_summary["misses"]) == load_calls
    assert bool(memo_summary["reuse_happened"]) is True

    measured_rows = [
        row
        for row in summary["cohort_estimates"]
        if str(row.get("status")) == "passed"
        and str(row.get("estimate_source")) == "measured_profile"
    ]
    assert measured_rows
    assert any(bool(row.get("feature_matrix_cache_hit")) for row in measured_rows)


def test_grouped_nested_cohort_selects_valid_representative_when_available(
    tmp_path: Path,
) -> None:
    index_csv = tmp_path / "index.csv"
    index_rows: list[dict[str, Any]] = []

    def _add_rows(subject: str, session: str) -> None:
        index_rows.append(
            {
                "sample_id": f"{subject}_{session}_anger",
                "subject": subject,
                "session": session,
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
            }
        )
        index_rows.append(
            {
                "sample_id": f"{subject}_{session}_happy",
                "subject": subject,
                "session": session,
                "task": "passive",
                "modality": "audio",
                "emotion": "happiness",
                "coarse_affect": "positive",
            }
        )

    for session in ("ses-01", "ses-02"):
        _add_rows("sub-001", session)
    for session in ("ses-01", "ses-02", "ses-03"):
        _add_rows("sub-002", session)

    pd.DataFrame(index_rows).to_csv(index_csv, index=False)

    comparison = runtime_profile.load_comparison_spec(
        _repo_root() / "configs" / "comparisons" / "model_family_grouped_nested_comparison_v1.json"
    )
    manifest = runtime_profile.compile_comparison(comparison, index_csv=index_csv)
    candidate_runs = [
        run
        for run in manifest.runs
        if str(run.model) == "ridge"
        and str(run.cv_mode) == "within_subject_loso_session"
        and bool(run.tuning_enabled)
        and int(run.repeat_id) == 1
    ]
    assert candidate_runs

    planned_records = [
        runtime_profile._PlannedProfileRun(  # type: ignore[attr-defined]
            phase="comparison",
            source_id=manifest.comparison_id,
            source_version=manifest.comparison_version,
            run=run,
            evidence_policy=manifest.evidence_policy.model_dump(mode="json"),
            data_policy=manifest.data_policy.model_dump(mode="json"),
        )
        for run in candidate_runs
    ]
    representative, validity = runtime_profile._minimal_valid_profile_subset(  # type: ignore[attr-defined]
        records=planned_records,
        index_csv=index_csv,
    )
    assert validity.can_profile_measured is True
    assert str(representative.run.subject) == "sub-002"


def test_model_fit_writes_fold_level_timing_fields(tmp_path: Path) -> None:
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

    output = execute_model_fit(
        ModelFitInput(
            x_matrix=x_matrix,
            metadata_df=metadata_df,
            target_column="coarse_affect",
            cv_mode="within_subject_loso_session",
            model="ridge",
            subject="sub-001",
            seed=42,
            run_id="timing_fold_fields",
            config_filename="config.json",
            report_dir=tmp_path,
            build_pipeline_fn=lambda model_name, seed: _build_pipeline(
                model_name=model_name,
                seed=seed,
                class_weight_policy="none",
            ),
            scores_for_predictions_fn=_scores_for_predictions,
            extract_linear_coefficients_fn=_extract_linear_coefficients,
        )
    )

    assert output["fold_rows"]
    for row in output["fold_rows"]:
        assert float(row["outer_fold_elapsed_seconds"]) >= 0.0
        assert float(row["estimator_fit_elapsed_seconds"]) >= 0.0
        assert row["tuned_search_elapsed_seconds"] is None
        assert int(row["tuned_search_candidate_count"]) == 0
    fit_timing_summary_path = Path(str(output["fit_timing_summary_path"]))
    assert fit_timing_summary_path.exists()
    fit_timing_payload = json.loads(fit_timing_summary_path.read_text(encoding="utf-8"))
    assert fit_timing_payload["status"] == "captured"
    assert int(fit_timing_payload["n_folds"]) == len(output["fold_rows"])


def test_model_fit_tuned_rows_include_search_timing_summary(tmp_path: Path) -> None:
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

    output = execute_model_fit(
        ModelFitInput(
            x_matrix=x_matrix,
            metadata_df=metadata_df,
            target_column="coarse_affect",
            cv_mode="within_subject_loso_session",
            model="ridge",
            subject="sub-001",
            seed=42,
            primary_metric_name="balanced_accuracy",
            methodology_policy_name="grouped_nested_tuning",
            tuning_enabled=True,
            tuning_search_space_id=LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
            tuning_search_space_version=LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
            tuning_inner_cv_scheme="grouped_leave_one_group_out",
            tuning_inner_group_field="session",
            run_id="timing_tuned_fields",
            config_filename="config.json",
            report_dir=tmp_path,
            build_pipeline_fn=lambda model_name, seed: _build_pipeline(
                model_name=model_name,
                seed=seed,
                class_weight_policy="none",
            ),
            scores_for_predictions_fn=_scores_for_predictions,
            extract_linear_coefficients_fn=_extract_linear_coefficients,
        )
    )

    tuned_rows = [row for row in output["tuning_records"] if str(row.get("status")) == "tuned"]
    assert tuned_rows
    for row in tuned_rows:
        assert int(row["n_candidates"]) > 0
        assert float(row["tuned_search_elapsed_seconds"]) >= 0.0
        assert float(row["cv_mean_fit_time_seconds"]) >= 0.0
        assert float(row["cv_std_fit_time_seconds"]) >= 0.0
        assert float(row["cv_mean_score_time_seconds"]) >= 0.0
        assert float(row["cv_std_score_time_seconds"]) >= 0.0


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

def test_campaign_runtime_profile_emits_progress_events(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[object] = []
    calls: list[dict[str, Any]] = []

    def _fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        run_id = str(kwargs["run_id"])
        report_dir = Path(kwargs["reports_root"]) / run_id
        return {
            "run_id": run_id,
            "report_dir": str(report_dir),
            "run_status_path": str(report_dir / "run_status.json"),
            "stage_timings_seconds": {"total": 2.5},
            "metrics": {"n_folds": 1, "primary_metric_value": 0.5},
        }

    monkeypatch.setattr(runtime_profile, "run_experiment", _fake_run_experiment)

    def _capture(event: object) -> None:
        events.append(event)

    summary = runtime_profile.verify_campaign_runtime_profile(
        index_csv=_demo_index_csv(),
        data_root=_repo_root() / "demo_data" / "synthetic_v1" / "data_root",
        cache_dir=_repo_root() / "demo_data" / "synthetic_v1" / "cache",
        confirmatory_protocol=_confirmatory_protocol(),
        comparison_specs=_comparison_specs(),
        profile_root=tmp_path / "runtime_profiles",
        progress_callback=_capture,
    )

    assert summary["passed"] is True
    assert calls
    assert all(call.get("progress_callback") is _capture for call in calls)
    assert events
    stages = [getattr(event, "stage", None) for event in events]
    assert "campaign" in stages
