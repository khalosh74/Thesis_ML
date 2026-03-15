"""Leakage-safe experiment runner with grouped cross-validation."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import tracemalloc
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_EXPERIMENT_REPORT,
    ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    compute_config_hash,
    register_artifact,
)
from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.methodology import (
    MethodologyPolicy,
    MethodologyPolicyName,
    SubgroupReportingPolicy,
)
from Thesis_ML.config.paths import DEFAULT_EXPERIMENT_REPORTS_ROOT
from Thesis_ML.experiments.cache_loading import load_features_from_cache
from Thesis_ML.experiments.errors import ThesisMLError
from Thesis_ML.experiments.execution_policy import (
    prepare_report_dir,
    write_run_status,
)
from Thesis_ML.experiments.metrics import (
    compute_interpretability_stability,
    evaluate_permutations,
    extract_linear_coefficients,
    scores_for_predictions,
)
from Thesis_ML.experiments.model_factory import MODEL_NAMES, make_model
from Thesis_ML.experiments.official_contracts import (
    validate_official_preflight,
    validate_run_artifact_contract,
)
from Thesis_ML.experiments.provenance import (
    collect_dataset_fingerprint,
    collect_git_provenance,
)
from Thesis_ML.experiments.run_artifacts import (
    build_run_config_payload,
    build_run_result_payload,
    resolve_run_identity,
    stamp_metrics_artifact,
)
from Thesis_ML.experiments.runtime_policies import (
    resolve_framework_context as _runtime_resolve_framework_context,
)
from Thesis_ML.experiments.runtime_policies import (
    resolve_methodology_runtime as _runtime_resolve_methodology_runtime,
)
from Thesis_ML.experiments.runtime_policies import (
    resolve_metric_policy_runtime,
)
from Thesis_ML.experiments.segment_execution import (
    SegmentExecutionRequest,
    execute_section_segment,
)
from Thesis_ML.experiments.spatial_validation import SPATIAL_AFFINE_ATOL
from Thesis_ML.experiments.tuning_search_spaces import (
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
)

LOGGER = logging.getLogger(__name__)

_CV_MODES = ("loso_session", "within_subject_loso_session", "frozen_cross_person_transfer")
_TARGET_ALIASES = {
    "emotion": "emotion",
    "coarse_affect": "coarse_affect",
    "modality": "modality",
    "task": "task",
    "regressor_label": "regressor_label",
}


def _serialize_warning_records(
    records: list[warnings.WarningMessage],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for warning_record in records:
        payloads.append(
            {
                "category": warning_record.category.__name__,
                "message": str(warning_record.message),
                "filename": str(warning_record.filename),
                "lineno": int(warning_record.lineno),
            }
        )
    return payloads


def _warning_summary(warnings_payload: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    for payload in warnings_payload:
        category = str(payload.get("category", "UnknownWarning"))
        by_category[category] = int(by_category.get(category, 0)) + 1
    return {
        "warning_count": int(len(warnings_payload)),
        "by_category": by_category,
        "convergence_warning_count": int(by_category.get("ConvergenceWarning", 0)),
    }


def _failure_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ThesisMLError):
        return {
            "error_code": str(exc.code),
            "error_type": type(exc).__name__,
            "failure_stage": str(exc.stage),
            "error_details": dict(exc.details or {}),
        }
    return {
        "error_code": "unhandled_exception",
        "error_type": type(exc).__name__,
        "failure_stage": "runtime",
        "error_details": {},
    }


def _make_model(name: str, seed: int, class_weight_policy: str = "none") -> Any:
    return make_model(
        name=name,
        seed=seed,
        class_weight_policy=class_weight_policy,
    )


def _build_pipeline(model_name: str, seed: int, class_weight_policy: str = "none"):
    model = _make_model(
        name=model_name,
        seed=seed,
        class_weight_policy=class_weight_policy,
    )
    # fMRI voxel vectors are dense numeric arrays; centered scaling is appropriate.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", model),
        ]
    )


def _resolve_target_column(target: str) -> str:
    if target not in _TARGET_ALIASES:
        allowed = ", ".join(sorted(_TARGET_ALIASES))
        raise ValueError(f"Unsupported target '{target}'. Allowed values: {allowed}")
    return _TARGET_ALIASES[target]


def _resolve_cv_mode(cv: str) -> str:
    if cv not in _CV_MODES:
        allowed = ", ".join(_CV_MODES)
        raise ValueError(f"Unsupported cv '{cv}'. Allowed values: {allowed}")
    return cv


def _load_features_from_cache(
    index_df: pd.DataFrame,
    cache_manifest_path: Path,
    spatial_report_path: Path | None = None,
    affine_atol: float = SPATIAL_AFFINE_ATOL,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    return load_features_from_cache(
        index_df=index_df,
        cache_manifest_path=cache_manifest_path,
        spatial_report_path=spatial_report_path,
        affine_atol=affine_atol,
    )


def _scores_for_predictions(estimator, x_test: np.ndarray) -> dict[str, list[Any]]:
    return scores_for_predictions(estimator=estimator, x_test=x_test)


def _evaluate_permutations(
    pipeline_template,
    x_matrix: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_permutations: int,
    metric_name: str,
    observed_metric: float,
) -> dict[str, Any]:
    return evaluate_permutations(
        pipeline_template=pipeline_template,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=seed,
        n_permutations=n_permutations,
        metric_name=metric_name,
        observed_metric=observed_metric,
    )


def _extract_linear_coefficients(estimator) -> tuple[np.ndarray, np.ndarray, list[str]]:
    return extract_linear_coefficients(estimator=estimator)


def _compute_interpretability_stability(coef_vectors: list[np.ndarray]) -> dict[str, Any]:
    return compute_interpretability_stability(coef_vectors=coef_vectors)


def _resolve_framework_context(
    framework_mode: FrameworkMode | str,
    *,
    protocol_context: dict[str, Any] | None,
    comparison_context: dict[str, Any] | None,
) -> tuple[
    FrameworkMode,
    bool,
    dict[str, Any],
    dict[str, Any],
]:
    return _runtime_resolve_framework_context(
        framework_mode,
        protocol_context=protocol_context,
        comparison_context=comparison_context,
    )


def _resolve_methodology_runtime(
    *,
    framework_mode: FrameworkMode,
    methodology_policy_name: str,
    class_weight_policy: str,
    tuning_enabled: bool,
    tuning_search_space_id: str | None,
    tuning_search_space_version: str | None,
    tuning_inner_cv_scheme: str | None,
    tuning_inner_group_field: str | None,
    subgroup_reporting_enabled: bool,
    subgroup_dimensions: list[str] | None,
    subgroup_min_samples_per_group: int,
    protocol_context: dict[str, Any],
    comparison_context: dict[str, Any],
) -> tuple[MethodologyPolicy, SubgroupReportingPolicy]:
    return _runtime_resolve_methodology_runtime(
        framework_mode=framework_mode,
        methodology_policy_name=methodology_policy_name,
        class_weight_policy=class_weight_policy,
        tuning_enabled=tuning_enabled,
        tuning_search_space_id=tuning_search_space_id,
        tuning_search_space_version=tuning_search_space_version,
        tuning_inner_cv_scheme=tuning_inner_cv_scheme,
        tuning_inner_group_field=tuning_inner_group_field,
        subgroup_reporting_enabled=subgroup_reporting_enabled,
        subgroup_dimensions=subgroup_dimensions,
        subgroup_min_samples_per_group=subgroup_min_samples_per_group,
        protocol_context=protocol_context,
        comparison_context=comparison_context,
    )


def run_experiment(
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    target: str,
    model: str,
    cv: str | None = None,
    subject: str | None = None,
    train_subject: str | None = None,
    test_subject: str | None = None,
    seed: int = 42,
    filter_task: str | None = None,
    filter_modality: str | None = None,
    n_permutations: int = 0,
    primary_metric_name: str = "balanced_accuracy",
    permutation_metric_name: str | None = None,
    methodology_policy_name: str = MethodologyPolicyName.FIXED_BASELINES_ONLY.value,
    class_weight_policy: str = "none",
    tuning_enabled: bool = False,
    tuning_search_space_id: str | None = None,
    tuning_search_space_version: str | None = None,
    tuning_inner_cv_scheme: str | None = None,
    tuning_inner_group_field: str | None = None,
    subgroup_reporting_enabled: bool = True,
    subgroup_dimensions: list[str] | None = None,
    subgroup_min_samples_per_group: int = 1,
    interpretability_enabled_override: bool | None = None,
    framework_mode: FrameworkMode | str = FrameworkMode.EXPLORATORY,
    protocol_context: dict[str, Any] | None = None,
    comparison_context: dict[str, Any] | None = None,
    run_id: str | None = None,
    reports_root: Path | str = DEFAULT_EXPERIMENT_REPORTS_ROOT,
    start_section: str | None = None,
    end_section: str | None = None,
    base_artifact_id: str | None = None,
    reuse_policy: str | None = None,
    force: bool = False,
    resume: bool = False,
    reuse_completed_artifacts: bool = False,
) -> dict[str, Any]:
    """Run one leakage-safe grouped-CV experiment and write standardized artifacts."""
    index_csv = Path(index_csv)
    data_root = Path(data_root)
    cache_dir = Path(cache_dir)
    reports_root = Path(reports_root)
    overall_start = perf_counter()
    stage_timings: dict[str, float] = {}
    warnings_payload: list[dict[str, Any]] = []
    resource_summary: dict[str, Any] = {}
    dataset_fingerprint: dict[str, Any] | None = None
    required_run_artifacts: list[str] = []
    required_run_metadata_fields: list[str] = []

    if cv is None or not str(cv).strip():
        allowed = ", ".join(_CV_MODES)
        raise ValueError(
            f"run_experiment requires explicit cv mode selection. Provide one of: {allowed}"
        )
    cv_mode = _resolve_cv_mode(str(cv).strip())
    if cv_mode == "within_subject_loso_session":
        if subject is None or not str(subject).strip():
            raise ValueError("cv='within_subject_loso_session' requires a non-empty subject.")
        subject = str(subject).strip()
    if cv_mode == "frozen_cross_person_transfer":
        if train_subject is None or not str(train_subject).strip():
            raise ValueError(
                "cv='frozen_cross_person_transfer' requires a non-empty train_subject."
            )
        if test_subject is None or not str(test_subject).strip():
            raise ValueError("cv='frozen_cross_person_transfer' requires a non-empty test_subject.")
        train_subject = str(train_subject).strip()
        test_subject = str(test_subject).strip()
        if train_subject == test_subject:
            raise ValueError("train_subject and test_subject must be different.")

    target_column = _resolve_target_column(target)
    context_start = perf_counter()
    (
        resolved_framework_mode,
        canonical_run,
        resolved_protocol_context,
        resolved_comparison_context,
    ) = _resolve_framework_context(
        framework_mode,
        protocol_context=protocol_context,
        comparison_context=comparison_context,
    )
    official_context: dict[str, Any] = {}
    if resolved_framework_mode == FrameworkMode.CONFIRMATORY:
        official_context = dict(resolved_protocol_context)
    if resolved_framework_mode == FrameworkMode.LOCKED_COMPARISON:
        official_context = dict(resolved_comparison_context)
    stage_timings["context_resolution"] = float(perf_counter() - context_start)

    metric_policy_start = perf_counter()
    (
        resolved_primary_metric_name,
        resolved_permutation_metric_name,
        metric_policy_effective,
    ) = resolve_metric_policy_runtime(
        framework_mode=resolved_framework_mode,
        official_context=official_context,
        primary_metric_name=primary_metric_name,
        permutation_metric_name=permutation_metric_name,
        n_permutations=n_permutations,
        interpretability_enabled_override=interpretability_enabled_override,
    )
    stage_timings["metric_policy_resolution"] = float(perf_counter() - metric_policy_start)

    methodology_start = perf_counter()
    methodology_policy, subgroup_policy = _resolve_methodology_runtime(
        framework_mode=resolved_framework_mode,
        methodology_policy_name=methodology_policy_name,
        class_weight_policy=class_weight_policy,
        tuning_enabled=tuning_enabled,
        tuning_search_space_id=tuning_search_space_id,
        tuning_search_space_version=tuning_search_space_version,
        tuning_inner_cv_scheme=tuning_inner_cv_scheme,
        tuning_inner_group_field=tuning_inner_group_field,
        subgroup_reporting_enabled=subgroup_reporting_enabled,
        subgroup_dimensions=subgroup_dimensions,
        subgroup_min_samples_per_group=subgroup_min_samples_per_group,
        protocol_context=resolved_protocol_context,
        comparison_context=resolved_comparison_context,
    )
    stage_timings["methodology_resolution"] = float(perf_counter() - methodology_start)

    effective_tuning_space_id = methodology_policy.tuning_search_space_id
    effective_tuning_space_version = methodology_policy.tuning_search_space_version
    effective_tuning_inner_cv_scheme = methodology_policy.inner_cv_scheme
    effective_tuning_inner_group_field = methodology_policy.inner_group_field
    if methodology_policy.policy_name == MethodologyPolicyName.GROUPED_NESTED_TUNING:
        if effective_tuning_space_id is None:
            effective_tuning_space_id = LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID
        if effective_tuning_space_version is None:
            effective_tuning_space_version = LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION
        if effective_tuning_inner_cv_scheme is None:
            effective_tuning_inner_cv_scheme = "grouped_leave_one_group_out"
        if effective_tuning_inner_group_field is None:
            effective_tuning_inner_group_field = "session"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_run_id = run_id or f"{timestamp}_{model}_{target_column}"
    report_dir = reports_root / resolved_run_id
    should_reuse_completed_artifacts = bool(resume or reuse_completed_artifacts)
    artifact_registry_path = reports_root / "artifact_registry.sqlite3"
    git_provenance = collect_git_provenance()
    code_ref = str(git_provenance.get("git_commit") or "") or None

    if resolved_framework_mode in {
        FrameworkMode.CONFIRMATORY,
        FrameworkMode.LOCKED_COMPARISON,
    }:
        preflight_start = perf_counter()
        preflight = validate_official_preflight(
            framework_mode=resolved_framework_mode,
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=cache_dir,
            target_column=target_column,
            cv_mode=cv_mode,
            subject=subject,
            train_subject=train_subject,
            test_subject=test_subject,
            filter_task=filter_task,
            filter_modality=filter_modality,
            n_permutations=n_permutations,
            primary_metric_name=resolved_primary_metric_name,
            permutation_metric_name=resolved_permutation_metric_name,
            methodology_policy_name=methodology_policy.policy_name.value,
            model=model,
            tuning_enabled=bool(methodology_policy.tuning_enabled),
            tuning_search_space_id=effective_tuning_space_id,
            tuning_search_space_version=effective_tuning_space_version,
            tuning_inner_group_field=effective_tuning_inner_group_field,
            subgroup_reporting_enabled=bool(subgroup_policy.enabled),
            subgroup_dimensions=list(subgroup_policy.subgroup_dimensions),
            official_context=official_context,
        )
        dataset_fingerprint = collect_dataset_fingerprint(
            index_csv=index_csv,
            selected_index_df=preflight.selected_index_df,
            index_row_count=preflight.index_row_count,
            target_column=target_column,
            cv_mode=cv_mode,
            subject=subject,
            train_subject=train_subject,
            test_subject=test_subject,
            filter_task=filter_task,
            filter_modality=filter_modality,
        )
        required_run_artifacts = list(preflight.required_run_artifacts)
        required_run_metadata_fields = list(preflight.required_run_metadata_fields)
        stage_timings["preflight_validation"] = float(perf_counter() - preflight_start)

    run_mode = prepare_report_dir(
        report_dir,
        run_id=resolved_run_id,
        force=bool(force),
        resume=bool(resume),
    )

    fold_metrics_path = report_dir / "fold_metrics.csv"
    fold_splits_path = report_dir / "fold_splits.csv"
    predictions_path = report_dir / "predictions.csv"
    metrics_path = report_dir / "metrics.json"
    subgroup_metrics_json_path = report_dir / "subgroup_metrics.json"
    subgroup_metrics_csv_path = report_dir / "subgroup_metrics.csv"
    config_path = report_dir / "config.json"
    tuning_summary_path = report_dir / "tuning_summary.json"
    tuning_best_params_path = report_dir / "best_params_per_fold.csv"
    spatial_compatibility_report_path = report_dir / "spatial_compatibility_report.json"
    interpretability_summary_path = report_dir / "interpretability_summary.json"
    interpretability_fold_artifacts_path = report_dir / "interpretability_fold_explanations.csv"

    write_run_status(
        report_dir,
        run_id=resolved_run_id,
        status="running",
        message=f"run_mode={run_mode}",
        stage_timings_seconds=stage_timings,
    )
    try:
        execute_start = perf_counter()
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            tracemalloc.start()
            try:
                segment_result = execute_section_segment(
                    SegmentExecutionRequest(
                        index_csv=index_csv,
                        data_root=data_root,
                        cache_dir=cache_dir,
                        target_column=target_column,
                        cv_mode=cv_mode,
                        model=model,
                        subject=subject,
                        train_subject=train_subject,
                        test_subject=test_subject,
                        filter_task=filter_task,
                        filter_modality=filter_modality,
                        seed=seed,
                        n_permutations=n_permutations,
                        primary_metric_name=resolved_primary_metric_name,
                        permutation_metric_name=resolved_permutation_metric_name,
                        methodology_policy_name=methodology_policy.policy_name.value,
                        class_weight_policy=methodology_policy.class_weight_policy.value,
                        tuning_enabled=bool(methodology_policy.tuning_enabled),
                        tuning_search_space_id=effective_tuning_space_id,
                        tuning_search_space_version=effective_tuning_space_version,
                        tuning_inner_cv_scheme=effective_tuning_inner_cv_scheme,
                        tuning_inner_group_field=effective_tuning_inner_group_field,
                        subgroup_reporting_enabled=bool(subgroup_policy.enabled),
                        subgroup_dimensions=tuple(subgroup_policy.subgroup_dimensions),
                        subgroup_min_samples_per_group=int(subgroup_policy.min_samples_per_group),
                        interpretability_enabled_override=interpretability_enabled_override,
                        run_id=resolved_run_id,
                        config_filename=config_path.name,
                        report_dir=report_dir,
                        artifact_registry_path=artifact_registry_path,
                        code_ref=code_ref,
                        affine_atol=SPATIAL_AFFINE_ATOL,
                        fold_metrics_path=fold_metrics_path,
                        fold_splits_path=fold_splits_path,
                        predictions_path=predictions_path,
                        metrics_path=metrics_path,
                        subgroup_metrics_json_path=subgroup_metrics_json_path,
                        subgroup_metrics_csv_path=subgroup_metrics_csv_path,
                        tuning_summary_path=tuning_summary_path,
                        tuning_best_params_path=tuning_best_params_path,
                        spatial_report_path=spatial_compatibility_report_path,
                        interpretability_summary_path=interpretability_summary_path,
                        interpretability_fold_artifacts_path=interpretability_fold_artifacts_path,
                        start_section=start_section,
                        end_section=end_section,
                        base_artifact_id=base_artifact_id,
                        reuse_policy=reuse_policy,
                        reuse_completed_artifacts=should_reuse_completed_artifacts,
                        build_pipeline_fn=lambda model_name, seed: _build_pipeline(
                            model_name=model_name,
                            seed=seed,
                            class_weight_policy=methodology_policy.class_weight_policy.value,
                        ),
                        load_features_from_cache_fn=_load_features_from_cache,
                        scores_for_predictions_fn=_scores_for_predictions,
                        extract_linear_coefficients_fn=_extract_linear_coefficients,
                        compute_interpretability_stability_fn=_compute_interpretability_stability,
                        evaluate_permutations_fn=_evaluate_permutations,
                    )
                )
            finally:
                warnings_payload = _serialize_warning_records(list(warning_records))
                if tracemalloc.is_tracing():
                    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    resource_summary = {
                        "memory_current_mb": round(float(current_bytes) / (1024.0 * 1024.0), 6),
                        "memory_peak_mb": round(float(peak_bytes) / (1024.0 * 1024.0), 6),
                    }
        stage_timings["segment_execution"] = float(perf_counter() - execute_start)
    except Exception as exc:
        failure = _failure_payload(exc)
        stage_timings["total"] = float(perf_counter() - overall_start)
        write_run_status(
            report_dir,
            run_id=resolved_run_id,
            status="failed",
            error=str(exc),
            error_code=str(failure["error_code"]),
            error_type=str(failure["error_type"]),
            failure_stage=str(failure["failure_stage"]),
            error_details=dict(failure["error_details"]),
            warnings=warnings_payload,
            warning_summary=_warning_summary(warnings_payload),
            stage_timings_seconds=stage_timings,
            resource_summary=resource_summary,
        )
        raise
    artifact_ids = dict(segment_result.artifact_ids)
    metrics = dict(segment_result.metrics or {})
    spatial_compatibility = segment_result.spatial_compatibility
    interpretability_summary = segment_result.interpretability_summary
    identity = resolve_run_identity(
        protocol_context=resolved_protocol_context,
        comparison_context=resolved_comparison_context,
    )
    warning_summary = _warning_summary(warnings_payload)
    metrics_stamp_start = perf_counter()
    try:
        persisted_metrics = stamp_metrics_artifact(
            metrics_path=metrics_path,
            canonical_run=canonical_run,
            framework_mode=resolved_framework_mode.value,
            methodology_policy_name=methodology_policy.policy_name.value,
            class_weight_policy=methodology_policy.class_weight_policy.value,
            tuning_enabled=bool(methodology_policy.tuning_enabled),
            tuning_summary_path=tuning_summary_path,
            tuning_best_params_path=tuning_best_params_path,
            subgroup_metrics_json_path=subgroup_metrics_json_path,
            subgroup_metrics_csv_path=subgroup_metrics_csv_path,
            metric_policy_effective=metric_policy_effective,
            identity=identity,
            dataset_fingerprint=dataset_fingerprint,
            git_provenance=git_provenance,
            stage_timings_seconds=stage_timings,
            resource_summary=resource_summary,
            warning_summary=warning_summary,
        )
    except Exception as exc:
        failure = _failure_payload(exc)
        stage_timings["total"] = float(perf_counter() - overall_start)
        write_run_status(
            report_dir,
            run_id=resolved_run_id,
            status="failed",
            error=str(exc),
            error_code=str(failure["error_code"]),
            error_type=str(failure["error_type"]),
            failure_stage=str(failure["failure_stage"]),
            error_details=dict(failure["error_details"]),
            warnings=warnings_payload,
            warning_summary=warning_summary,
            stage_timings_seconds=stage_timings,
            resource_summary=resource_summary,
        )
        raise
    stage_timings["metrics_stamping"] = float(perf_counter() - metrics_stamp_start)
    if isinstance(persisted_metrics, dict):
        metrics = dict(persisted_metrics)

    spatial_status = str(spatial_compatibility["status"]) if spatial_compatibility else None
    spatial_passed = bool(spatial_compatibility["passed"]) if spatial_compatibility else None
    spatial_groups_checked = (
        int(spatial_compatibility["n_groups_checked"]) if spatial_compatibility else None
    )
    spatial_reference_group = (
        spatial_compatibility["reference_group_id"] if spatial_compatibility else None
    )
    spatial_affine_atol = (
        float(spatial_compatibility["affine_atol"]) if spatial_compatibility else None
    )

    interpretability_enabled = (
        bool(interpretability_summary["enabled"]) if interpretability_summary else None
    )
    interpretability_performed = (
        bool(interpretability_summary["performed"]) if interpretability_summary else None
    )
    interpretability_status = (
        str(interpretability_summary["status"]) if interpretability_summary else None
    )
    interpretability_fold_artifacts = (
        interpretability_summary.get("fold_artifacts_path") if interpretability_summary else None
    )

    config_build_start = perf_counter()
    try:
        config = build_run_config_payload(
            run_id=resolved_run_id,
            timestamp=timestamp,
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=cache_dir,
            target_column=target_column,
            model=model,
            cv_mode=cv_mode,
            subject=subject,
            train_subject=train_subject,
            test_subject=test_subject,
            seed=seed,
            primary_metric_name=resolved_primary_metric_name,
            permutation_metric_name=resolved_permutation_metric_name,
            metric_policy_effective=metric_policy_effective,
            methodology_policy_name=methodology_policy.policy_name.value,
            class_weight_policy=methodology_policy.class_weight_policy.value,
            tuning_enabled=bool(methodology_policy.tuning_enabled),
            tuning_search_space_id=effective_tuning_space_id,
            tuning_search_space_version=effective_tuning_space_version,
            tuning_inner_cv_scheme=effective_tuning_inner_cv_scheme,
            tuning_inner_group_field=effective_tuning_inner_group_field,
            tuning_summary_path=tuning_summary_path,
            tuning_best_params_path=tuning_best_params_path,
            subgroup_reporting_enabled=bool(subgroup_policy.enabled),
            subgroup_dimensions=list(subgroup_policy.subgroup_dimensions),
            subgroup_min_samples_per_group=int(subgroup_policy.min_samples_per_group),
            subgroup_metrics_json_path=subgroup_metrics_json_path,
            subgroup_metrics_csv_path=subgroup_metrics_csv_path,
            filter_task=filter_task,
            filter_modality=filter_modality,
            n_permutations=n_permutations,
            framework_mode=resolved_framework_mode.value,
            canonical_run=bool(canonical_run),
            identity=identity,
            protocol_context=resolved_protocol_context,
            comparison_context=resolved_comparison_context,
            start_section=start_section,
            end_section=end_section,
            base_artifact_id=base_artifact_id,
            reuse_policy=reuse_policy,
            force=bool(force),
            resume=bool(resume),
            reuse_completed_artifacts=bool(should_reuse_completed_artifacts),
            run_mode=run_mode,
            segment_result=segment_result,
            fold_splits_path=fold_splits_path,
            spatial_compatibility_status=spatial_status,
            spatial_compatibility_passed=spatial_passed,
            spatial_compatibility_n_groups_checked=spatial_groups_checked,
            spatial_compatibility_reference_group_id=spatial_reference_group,
            spatial_compatibility_affine_atol=spatial_affine_atol,
            spatial_compatibility_report_path=spatial_compatibility_report_path,
            interpretability_enabled=interpretability_enabled,
            interpretability_performed=interpretability_performed,
            interpretability_status=interpretability_status,
            interpretability_fold_artifacts_path=interpretability_fold_artifacts,
            interpretability_summary_path=interpretability_summary_path,
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            pandas_version=pd.__version__,
            sklearn_version=sklearn.__version__,
            nibabel_version=nib.__version__,
            git_commit=str(git_provenance.get("git_commit") or "") or None,
            git_branch=str(git_provenance.get("git_branch") or "") or None,
            git_dirty=bool(git_provenance.get("git_dirty", False)),
            dataset_fingerprint=dataset_fingerprint,
            stage_timings_seconds=stage_timings,
            resource_summary=resource_summary,
            warning_summary=warning_summary,
        )
        config_path.write_text(f"{json.dumps(config, indent=2)}\n", encoding="utf-8")
    except Exception as exc:
        failure = _failure_payload(exc)
        stage_timings["total"] = float(perf_counter() - overall_start)
        write_run_status(
            report_dir,
            run_id=resolved_run_id,
            status="failed",
            error=str(exc),
            error_code=str(failure["error_code"]),
            error_type=str(failure["error_type"]),
            failure_stage=str(failure["failure_stage"]),
            error_details=dict(failure["error_details"]),
            warnings=warnings_payload,
            warning_summary=warning_summary,
            stage_timings_seconds=stage_timings,
            resource_summary=resource_summary,
        )
        raise
    stage_timings["config_write"] = float(perf_counter() - config_build_start)

    registry_update_start = perf_counter()
    report_upstream_candidates = [
        artifact_ids.get(ARTIFACT_TYPE_METRICS_BUNDLE),
        artifact_ids.get(ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE),
    ]
    report_upstream = [
        artifact_id for artifact_id in report_upstream_candidates if isinstance(artifact_id, str)
    ]
    try:
        experiment_report_artifact = register_artifact(
            registry_path=artifact_registry_path,
            artifact_type=ARTIFACT_TYPE_EXPERIMENT_REPORT,
            run_id=resolved_run_id,
            upstream_artifact_ids=report_upstream,
            config_hash=compute_config_hash(
                {"run_id": resolved_run_id, "report_dir": str(report_dir)}
            ),
            code_ref=code_ref,
            path=report_dir,
            status="created",
        )
    except Exception as exc:
        failure = _failure_payload(exc)
        stage_timings["total"] = float(perf_counter() - overall_start)
        write_run_status(
            report_dir,
            run_id=resolved_run_id,
            status="failed",
            error=str(exc),
            error_code=str(failure["error_code"]),
            error_type=str(failure["error_type"]),
            failure_stage=str(failure["failure_stage"]),
            error_details=dict(failure["error_details"]),
            warnings=warnings_payload,
            warning_summary=warning_summary,
            stage_timings_seconds=stage_timings,
            resource_summary=resource_summary,
        )
        raise
    artifact_ids[ARTIFACT_TYPE_EXPERIMENT_REPORT] = experiment_report_artifact.artifact_id
    stage_timings["artifact_registry_update"] = float(perf_counter() - registry_update_start)

    if resolved_framework_mode in {
        FrameworkMode.CONFIRMATORY,
        FrameworkMode.LOCKED_COMPARISON,
    }:
        artifact_validation_start = perf_counter()
        try:
            validate_run_artifact_contract(
                report_dir=report_dir,
                required_run_artifacts=required_run_artifacts,
                required_run_metadata_fields=required_run_metadata_fields,
                framework_mode=resolved_framework_mode,
                canonical_run=bool(canonical_run),
                config_payload=config,
                metrics_payload=metrics,
            )
        except Exception as exc:
            failure = _failure_payload(exc)
            stage_timings["total"] = float(perf_counter() - overall_start)
            write_run_status(
                report_dir,
                run_id=resolved_run_id,
                status="failed",
                error=str(exc),
                error_code=str(failure["error_code"]),
                error_type=str(failure["error_type"]),
                failure_stage=str(failure["failure_stage"]),
                error_details=dict(failure["error_details"]),
                warnings=warnings_payload,
                warning_summary=warning_summary,
                stage_timings_seconds=stage_timings,
                resource_summary=resource_summary,
            )
            raise
        stage_timings["official_artifact_validation"] = float(
            perf_counter() - artifact_validation_start
        )

    stage_timings["total"] = float(perf_counter() - overall_start)
    run_status = write_run_status(
        report_dir,
        run_id=resolved_run_id,
        status="completed",
        executed_sections=segment_result.executed_sections,
        reused_sections=segment_result.reused_sections,
        warnings=warnings_payload,
        warning_summary=warning_summary,
        stage_timings_seconds=stage_timings,
        resource_summary=resource_summary,
    )

    return build_run_result_payload(
        run_id=resolved_run_id,
        report_dir=report_dir,
        config_path=config_path,
        metrics_path=metrics_path,
        subgroup_metrics_json_path=subgroup_metrics_json_path,
        subgroup_metrics_csv_path=subgroup_metrics_csv_path,
        tuning_summary_path=tuning_summary_path,
        tuning_best_params_path=tuning_best_params_path,
        fold_metrics_path=fold_metrics_path,
        fold_splits_path=fold_splits_path,
        predictions_path=predictions_path,
        spatial_compatibility_report_path=spatial_compatibility_report_path,
        interpretability_summary_path=interpretability_summary_path,
        interpretability_fold_artifacts_path=interpretability_fold_artifacts,
        artifact_registry_path=artifact_registry_path,
        segment_result=segment_result,
        artifact_ids=artifact_ids,
        metrics=metrics,
        run_status_path=run_status,
        run_mode=run_mode,
        framework_mode=resolved_framework_mode.value,
        canonical_run=bool(canonical_run),
        metric_policy_effective=metric_policy_effective,
        methodology_policy_name=methodology_policy.policy_name.value,
        class_weight_policy=methodology_policy.class_weight_policy.value,
        tuning_enabled=bool(methodology_policy.tuning_enabled),
        protocol_context=resolved_protocol_context,
        comparison_context=resolved_comparison_context,
        stage_timings_seconds=stage_timings,
        resource_summary=resource_summary,
        warning_summary=warning_summary,
        dataset_fingerprint=dataset_fingerprint,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe grouped-CV fMRI experiments (exploratory mode). "
            "For locked comparisons use thesisml-run-comparison; for confirmatory runs use thesisml-run-protocol."
        )
    )
    parser.add_argument("--index-csv", required=True, help="Dataset index CSV.")
    parser.add_argument("--data-root", required=True, help="Root directory for relative paths.")
    parser.add_argument("--cache-dir", required=True, help="Feature cache directory.")
    parser.add_argument(
        "--target", required=True, choices=sorted(_TARGET_ALIASES), help="Target label."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[*MODEL_NAMES, "all"],
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--cv",
        required=True,
        choices=list(_CV_MODES),
        help=(
            "Experiment mode. Required. within_subject_loso_session=primary thesis mode; "
            "frozen_cross_person_transfer=secondary transfer mode; "
            "loso_session=auxiliary grouped mode."
        ),
    )
    parser.add_argument(
        "--subject",
        default=None,
        help=(
            "Subject identifier (required for cv=within_subject_loso_session; for example sub-001)."
        ),
    )
    parser.add_argument(
        "--train-subject",
        default=None,
        help=("Training subject identifier (required for cv=frozen_cross_person_transfer)."),
    )
    parser.add_argument(
        "--test-subject",
        default=None,
        help=("Test subject identifier (required for cv=frozen_cross_person_transfer)."),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--filter-task", default=None, help="Optional task filter.")
    parser.add_argument("--filter-modality", default=None, help="Optional modality filter.")
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=0,
        help="Number of permutation rounds for optional significance testing.",
    )
    parser.add_argument(
        "--methodology-policy",
        default=MethodologyPolicyName.FIXED_BASELINES_ONLY.value,
        choices=[
            MethodologyPolicyName.FIXED_BASELINES_ONLY.value,
            MethodologyPolicyName.GROUPED_NESTED_TUNING.value,
        ],
        help=(
            "Methodology policy for exploratory runs. "
            "Official comparison/protocol runs always load this from spec/protocol."
        ),
    )
    parser.add_argument(
        "--class-weight-policy",
        default="none",
        choices=["none", "balanced"],
        help="Class-weight policy for exploratory runs.",
    )
    parser.add_argument(
        "--tuning-search-space-id",
        default=LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
        help="Search-space ID used when --methodology-policy grouped_nested_tuning.",
    )
    parser.add_argument(
        "--tuning-search-space-version",
        default=LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
        help="Search-space version used when --methodology-policy grouped_nested_tuning.",
    )
    parser.add_argument(
        "--subgroup-dimension",
        action="append",
        default=[],
        help=(
            "Subgroup dimension for subgroup reporting. Repeat to include multiple "
            "(label, task, modality, session, subject)."
        ),
    )
    parser.add_argument(
        "--subgroup-min-samples",
        type=int,
        default=1,
        help="Minimum samples per subgroup row.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier. If omitted, timestamp-based ID is used.",
    )
    parser.add_argument(
        "--reports-root",
        default=str(DEFAULT_EXPERIMENT_REPORTS_ROOT),
        help="Root directory for exploratory experiment reports.",
    )
    parser.add_argument(
        "--start-section",
        default=None,
        help="Optional first section to execute for segmented runs.",
    )
    parser.add_argument(
        "--end-section",
        default=None,
        help="Optional last section to execute for segmented runs.",
    )
    parser.add_argument(
        "--base-artifact-id",
        default=None,
        help=(
            "Optional artifact ID used to resume segmented runs when start-section "
            "is after feature_cache_build."
        ),
    )
    parser.add_argument(
        "--reuse-policy",
        default=None,
        help="Optional reuse policy for segmented runs (auto, require_explicit_base, disallow).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun: clear existing run output directory before execution.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partial run from existing output directory.",
    )
    parser.add_argument(
        "--reuse-completed-artifacts",
        action="store_true",
        help="Allow reusing completed same-run section artifacts when available.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    models = list(MODEL_NAMES) if args.model == "all" else [args.model]
    subgroup_dimensions = (
        list(args.subgroup_dimension) if list(args.subgroup_dimension) else None
    )
    results: list[dict[str, Any]] = []

    for model_name in models:
        model_run_id = args.run_id
        if args.model == "all":
            base = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            model_run_id = f"{base}_{model_name}_{args.target}"

        result = run_experiment(
            index_csv=Path(args.index_csv),
            data_root=Path(args.data_root),
            cache_dir=Path(args.cache_dir),
            target=args.target,
            model=model_name,
            cv=args.cv,
            subject=args.subject,
            train_subject=args.train_subject,
            test_subject=args.test_subject,
            seed=args.seed,
            filter_task=args.filter_task,
            filter_modality=args.filter_modality,
            n_permutations=args.n_permutations,
            methodology_policy_name=args.methodology_policy,
            class_weight_policy=args.class_weight_policy,
            tuning_enabled=(
                str(args.methodology_policy)
                == MethodologyPolicyName.GROUPED_NESTED_TUNING.value
            ),
            tuning_search_space_id=args.tuning_search_space_id,
            tuning_search_space_version=args.tuning_search_space_version,
            tuning_inner_cv_scheme=(
                "grouped_leave_one_group_out"
                if str(args.methodology_policy)
                == MethodologyPolicyName.GROUPED_NESTED_TUNING.value
                else None
            ),
            tuning_inner_group_field=(
                "session"
                if str(args.methodology_policy)
                == MethodologyPolicyName.GROUPED_NESTED_TUNING.value
                else None
            ),
            subgroup_reporting_enabled=True,
            subgroup_dimensions=subgroup_dimensions,
            subgroup_min_samples_per_group=args.subgroup_min_samples,
            run_id=model_run_id,
            reports_root=Path(args.reports_root),
            start_section=args.start_section,
            end_section=args.end_section,
            base_artifact_id=args.base_artifact_id,
            reuse_policy=args.reuse_policy,
            force=bool(args.force),
            resume=bool(args.resume),
            reuse_completed_artifacts=bool(args.reuse_completed_artifacts),
        )
        results.append(
            {
                "model": model_name,
                "run_id": result["run_id"],
                "report_dir": result["report_dir"],
                "accuracy": result["metrics"].get("accuracy"),
                "balanced_accuracy": result["metrics"].get("balanced_accuracy"),
                "macro_f1": result["metrics"].get("macro_f1"),
            }
        )

    print(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
