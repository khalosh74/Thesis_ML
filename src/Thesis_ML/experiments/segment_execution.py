from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_FEATURE_CACHE,
    ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE,
    ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    compute_config_hash,
    get_artifact,
)
from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy
from Thesis_ML.experiments.progress import ProgressCallback, emit_progress
from Thesis_ML.experiments.section_models import (
    DatasetSelectionInput,
    EvaluationInput,
    FeatureCacheBuildInput,
    FeatureMatrixLoadInput,
    InterpretabilityInput,
    ModelFitInput,
    SpatialValidationInput,
)
from Thesis_ML.experiments.sections import (
    dataset_selection,
    evaluation,
    feature_cache_build,
    feature_matrix_load,
    interpretability,
    model_fit,
    spatial_validation,
)
from Thesis_ML.experiments.segment_execution_helpers import (
    expected_base_artifact_type,
    find_reusable_run_artifact,
    is_after_or_equal,
    normalize_reuse_policy,
    plan_section_path,
    require_callable,
    resolve_base_artifact,
)
from Thesis_ML.experiments.stage_execution import StageAssignment, StageKey
from Thesis_ML.features.preprocessing import BASELINE_STANDARD_SCALER_RECIPE_ID
from Thesis_ML.orchestration.contracts import ReusePolicy, SectionName


@dataclass(frozen=True)
class SegmentExecutionRequest:
    index_csv: Path
    data_root: Path
    cache_dir: Path
    target_column: str
    cv_mode: str
    model: str
    subject: str | None
    train_subject: str | None
    test_subject: str | None
    filter_task: str | None
    filter_modality: str | None
    seed: int
    n_permutations: int
    run_id: str
    config_filename: str
    report_dir: Path
    artifact_registry_path: Path
    code_ref: str | None
    affine_atol: float
    fold_metrics_path: Path
    fold_splits_path: Path
    predictions_path: Path
    metrics_path: Path
    subgroup_metrics_json_path: Path
    subgroup_metrics_csv_path: Path
    tuning_summary_path: Path
    tuning_best_params_path: Path
    fit_timing_summary_path: Path
    spatial_report_path: Path
    calibration_summary_path: Path
    calibration_table_path: Path
    interpretability_summary_path: Path
    interpretability_fold_artifacts_path: Path
    primary_metric_name: str = "balanced_accuracy"
    permutation_metric_name: str | None = None
    permutation_alpha: float = 0.05
    permutation_minimum_required: int = 0
    permutation_require_pass_for_validity: bool = False
    methodology_policy_name: str = "fixed_baselines_only"
    evidence_run_role: str = "primary"
    repeat_id: int = 1
    repeat_count: int = 1
    base_run_id: str | None = None
    evidence_policy_effective: dict[str, Any] | None = None
    class_weight_policy: str = "none"
    feature_recipe_id: str = BASELINE_STANDARD_SCALER_RECIPE_ID
    emit_feature_qc_artifacts: bool = True
    feature_qc_summary_path: Path | None = None
    feature_qc_selected_samples_path: Path | None = None
    feature_quality_policy: dict[str, Any] | None = None
    tuning_enabled: bool = False
    tuning_search_space_id: str | None = None
    tuning_search_space_version: str | None = None
    tuning_inner_cv_scheme: str | None = None
    tuning_inner_group_field: str | None = None
    progress_callback: ProgressCallback | None = None
    subgroup_reporting_enabled: bool = True
    subgroup_dimensions: tuple[str, ...] = (
        "label",
        "task",
        "modality",
        "session",
        "subject",
    )
    subgroup_min_samples_per_group: int = 1
    subgroup_min_classes_per_group: int = 1
    subgroup_report_small_groups: bool = False
    confirmatory_guardrails_enabled: bool = False
    subgroup_evidence_role: str = "exploratory"
    subgroup_primary_evidence_allowed: bool = True
    calibration_enabled: bool = True
    calibration_n_bins: int = 10
    calibration_require_probabilities_for_validity: bool = False
    interpretability_enabled_override: bool | None = None
    max_outer_folds: int | None = None
    profiling_only: bool = False
    profile_inner_folds: int | None = None
    profile_tuning_candidates: int | None = None
    compute_policy: ResolvedComputePolicy | None = None
    start_section: str | SectionName | None = None
    end_section: str | SectionName | None = None
    base_artifact_id: str | None = None
    reuse_policy: str | ReusePolicy | None = None
    reuse_completed_artifacts: bool = False
    build_pipeline_fn: Callable[..., Any] | None = None
    load_features_from_cache_fn: (
        Callable[..., tuple[np.ndarray, pd.DataFrame, dict[str, Any]]] | None
    ) = None
    scores_for_predictions_fn: Callable[..., dict[str, list[Any]]] | None = None
    extract_linear_coefficients_fn: (
        Callable[..., tuple[np.ndarray, np.ndarray, list[str]]] | None
    ) = None
    compute_interpretability_stability_fn: Callable[[list[np.ndarray]], dict[str, Any]] | None = (
        None
    )
    evaluate_permutations_fn: Callable[..., dict[str, Any]] | None = None
    stage_assignments: tuple[StageAssignment, ...] | None = None
    stage_fallback_executor_ids: dict[str, str] | None = None


@dataclass(frozen=True)
class SegmentExecutionResult:
    planned_sections: list[str]
    executed_sections: list[str]
    reused_sections: list[str]
    artifact_ids: dict[str, str]
    metrics: dict[str, Any] | None
    spatial_compatibility: dict[str, Any] | None
    interpretability_summary: dict[str, Any] | None
    compute_runtime_metadata: dict[str, Any] | None = None
    section_timings_seconds: dict[str, float] | None = None
    stage_assignments: list[StageAssignment] | None = None


def execute_section_segment(request: SegmentExecutionRequest) -> SegmentExecutionResult:
    def _emit_section_event(
    *,
    section_name: str,
    status: str,
    completed_units: float | None,
    total_units: float | None,
    message: str,
    ) -> None:
        emit_progress(
            request.progress_callback,
            stage="section",
            message=message,
            completed_units=completed_units,
            total_units=total_units,
            metadata={
                "run_id": str(request.run_id),
                "model": str(request.model),
                "target": str(request.target_column),
                "cv_mode": str(request.cv_mode),
                "section": str(section_name),
                "status": str(status),
            },
        )
    planned_sections = plan_section_path(request.start_section, request.end_section)
    total_sections = int(len(planned_sections))
    start_section = planned_sections[0]
    reuse_policy = normalize_reuse_policy(request.reuse_policy)
    base_artifact = resolve_base_artifact(
        request=request,
        start_section=start_section,
        reuse_policy=reuse_policy,
    )

    build_pipeline_fn = require_callable("build_pipeline_fn", request.build_pipeline_fn)
    load_features_from_cache_fn = require_callable(
        "load_features_from_cache_fn", request.load_features_from_cache_fn
    )
    scores_for_predictions_fn = require_callable(
        "scores_for_predictions_fn", request.scores_for_predictions_fn
    )
    extract_linear_coefficients_fn = require_callable(
        "extract_linear_coefficients_fn", request.extract_linear_coefficients_fn
    )
    compute_interpretability_stability_fn = require_callable(
        "compute_interpretability_stability_fn",
        request.compute_interpretability_stability_fn,
    )
    evaluate_permutations_fn = require_callable(
        "evaluate_permutations_fn", request.evaluate_permutations_fn
    )

    executed_sections: list[str] = []
    reused_sections: list[str] = []
    artifact_ids: dict[str, str] = {}

    selected_index_df: pd.DataFrame | None = None
    cache_manifest_path: Path | None = None
    feature_cache_artifact_id: str | None = None
    x_matrix: np.ndarray | None = None
    metadata_df: pd.DataFrame | None = None
    spatial_compatibility: dict[str, Any] | None = None
    feature_matrix_artifact_id: str | None = None
    fit_output = None
    interpretability_summary: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    compute_runtime_metadata: dict[str, Any] | None = None
    section_timings_seconds: dict[str, float] = {}
    stage_assignment_map: dict[StageKey, StageAssignment] = {}
    for assignment in request.stage_assignments or ():
        normalized_assignment = (
            assignment
            if isinstance(assignment, StageAssignment)
            else StageAssignment.model_validate(dict(assignment))
        )
        stage_assignment_map[StageKey(str(normalized_assignment.stage))] = normalized_assignment
    stage_fallback_executor_ids: dict[StageKey, str] = {}
    if isinstance(request.stage_fallback_executor_ids, dict):
        for stage_key_raw, executor_id_raw in request.stage_fallback_executor_ids.items():
            if not isinstance(executor_id_raw, str) or not executor_id_raw.strip():
                continue
            stage_fallback_executor_ids[StageKey(str(stage_key_raw))] = executor_id_raw.strip()

    if SectionName.DATASET_SELECTION not in planned_sections and is_after_or_equal(
        planned_sections[-1], SectionName.FEATURE_MATRIX_LOAD
    ):
        selected_index_df = dataset_selection(
            DatasetSelectionInput(
                index_csv=request.index_csv,
                target_column=request.target_column,
                cv_mode=request.cv_mode,
                subject=request.subject,
                train_subject=request.train_subject,
                test_subject=request.test_subject,
                filter_task=request.filter_task,
                filter_modality=request.filter_modality,
            )
        ).selected_index_df

    if base_artifact is not None:
        expected_type = expected_base_artifact_type(start_section)
        if expected_type == ARTIFACT_TYPE_FEATURE_CACHE:
            cache_manifest_path = Path(base_artifact.path)
            feature_cache_artifact_id = base_artifact.artifact_id
            artifact_ids["feature_cache"] = base_artifact.artifact_id
        else:
            feature_matrix_artifact_id = base_artifact.artifact_id
            artifact_ids["feature_matrix_bundle"] = base_artifact.artifact_id
            if selected_index_df is None:
                raise ValueError("Segment state error: selected dataset rows were not initialized.")
            if not base_artifact.upstream_artifact_ids:
                raise ValueError(
                    f"Incompatible base artifact '{base_artifact.artifact_id}': missing upstream "
                    "feature_cache artifact reference."
                )
            upstream_feature_cache_id = str(base_artifact.upstream_artifact_ids[0])
            upstream_feature_cache = get_artifact(
                registry_path=request.artifact_registry_path,
                artifact_id=upstream_feature_cache_id,
            )
            if upstream_feature_cache is None:
                raise ValueError(
                    "Incompatible base artifact: upstream feature_cache artifact was not found."
                )
            if upstream_feature_cache.artifact_type != ARTIFACT_TYPE_FEATURE_CACHE:
                raise ValueError(
                    "Incompatible base artifact: expected upstream artifact_type "
                    f"'{ARTIFACT_TYPE_FEATURE_CACHE}', got "
                    f"'{upstream_feature_cache.artifact_type}'."
                )
            cache_manifest_path = Path(upstream_feature_cache.path)
            feature_cache_artifact_id = upstream_feature_cache.artifact_id
            artifact_ids["feature_cache"] = upstream_feature_cache.artifact_id
            x_matrix, metadata_df, spatial_compatibility = load_features_from_cache_fn(
                index_df=selected_index_df,
                cache_manifest_path=cache_manifest_path,
                spatial_report_path=request.spatial_report_path,
                affine_atol=request.affine_atol,
            )

    for section in planned_sections:
        section_number = int(len(executed_sections) + 1)
        section_start = perf_counter()
        _emit_section_event(
            section_name=section.value,
            status="starting",
            completed_units=float(section_number - 1),
            total_units=float(total_sections),
            message=f"starting section {section.value}",
        )
        if section == SectionName.DATASET_SELECTION:
            selection_output = dataset_selection(
                DatasetSelectionInput(
                    index_csv=request.index_csv,
                    target_column=request.target_column,
                    cv_mode=request.cv_mode,
                    subject=request.subject,
                    train_subject=request.train_subject,
                    test_subject=request.test_subject,
                    filter_task=request.filter_task,
                    filter_modality=request.filter_modality,
                )
            )
            selected_index_df = selection_output.selected_index_df
        elif section == SectionName.FEATURE_CACHE_BUILD:
            if request.reuse_completed_artifacts:
                reusable_feature_cache = find_reusable_run_artifact(
                    request,
                    artifact_type=ARTIFACT_TYPE_FEATURE_CACHE,
                    expected_config_hash=compute_config_hash(
                        {
                            "index_csv": str(request.index_csv.resolve()),
                            "data_root": str(request.data_root.resolve()),
                            "cache_dir": str(request.cache_dir.resolve()),
                            "group_key": "subject_session_bas",
                            "force": False,
                        }
                    ),
                )
                if reusable_feature_cache is not None:
                    cache_manifest_path = Path(reusable_feature_cache.path)
                    feature_cache_artifact_id = reusable_feature_cache.artifact_id
                    artifact_ids["feature_cache"] = reusable_feature_cache.artifact_id
                    reused_sections.append(section.value)
                    executed_sections.append(section.value)
                    section_timings_seconds[section.value] = float(perf_counter() - section_start)
                    _emit_section_event(
                        section_name=section.value,
                        status="reused",
                        completed_units=float(section_number),
                        total_units=float(total_sections),
                        message=f"reused section {section.value}",
                    )
                    continue

            cache_output = feature_cache_build(
                FeatureCacheBuildInput(
                    index_csv=request.index_csv,
                    data_root=request.data_root,
                    cache_dir=request.cache_dir,
                    group_key="subject_session_bas",
                    force=False,
                    run_id=request.run_id,
                    artifact_registry_path=request.artifact_registry_path,
                    code_ref=request.code_ref,
                )
            )
            cache_manifest_path = cache_output.cache_manifest_path
            feature_cache_artifact_id = cache_output.feature_cache_artifact_id
            artifact_ids["feature_cache"] = cache_output.feature_cache_artifact_id
        elif section == SectionName.FEATURE_MATRIX_LOAD:
            if selected_index_df is None:
                raise ValueError(
                    "feature_matrix_load requires selected dataset rows from dataset_selection."
                )
            if cache_manifest_path is None or feature_cache_artifact_id is None:
                raise ValueError(
                    "feature_matrix_load requires a feature_cache artifact. "
                    "Run feature_cache_build or provide a compatible base_artifact_id."
                )
            if request.reuse_completed_artifacts:
                reusable_feature_matrix = find_reusable_run_artifact(
                    request,
                    artifact_type=ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE,
                    expected_config_hash=compute_config_hash(
                        {
                            "target": request.target_column,
                            "cv_mode": request.cv_mode,
                            "subject": request.subject,
                            "train_subject": request.train_subject,
                            "test_subject": request.test_subject,
                            "filter_task": request.filter_task,
                            "filter_modality": request.filter_modality,
                            "cache_manifest_path": str(cache_manifest_path.resolve()),
                        }
                    ),
                )
                if reusable_feature_matrix is not None:
                    x_matrix, metadata_df, spatial_compatibility = load_features_from_cache_fn(
                        index_df=selected_index_df,
                        cache_manifest_path=cache_manifest_path,
                        spatial_report_path=request.spatial_report_path,
                        affine_atol=request.affine_atol,
                    )
                    feature_matrix_artifact_id = reusable_feature_matrix.artifact_id
                    artifact_ids["feature_matrix_bundle"] = reusable_feature_matrix.artifact_id
                    reused_sections.append(section.value)
                    executed_sections.append(section.value)
                    section_timings_seconds[section.value] = float(perf_counter() - section_start)
                    _emit_section_event(
                        section_name=section.value,
                        status="reused",
                        completed_units=float(section_number),
                        total_units=float(total_sections),
                        message=f"reused section {section.value}",
                    )
                    continue
            matrix_output = feature_matrix_load(
                FeatureMatrixLoadInput(
                    selected_index_df=selected_index_df,
                    cache_manifest_path=cache_manifest_path,
                    spatial_report_path=request.spatial_report_path,
                    affine_atol=request.affine_atol,
                    run_id=request.run_id,
                    artifact_registry_path=request.artifact_registry_path,
                    code_ref=request.code_ref,
                    upstream_feature_cache_artifact_id=feature_cache_artifact_id,
                    target_column=request.target_column,
                    cv_mode=request.cv_mode,
                    subject=request.subject,
                    train_subject=request.train_subject,
                    test_subject=request.test_subject,
                    filter_task=request.filter_task,
                    filter_modality=request.filter_modality,
                    load_features_from_cache_fn=load_features_from_cache_fn,
                )
            )
            x_matrix = matrix_output.x_matrix
            metadata_df = matrix_output.metadata_df
            spatial_compatibility = matrix_output.spatial_compatibility
            feature_matrix_artifact_id = matrix_output.feature_matrix_artifact_id
            artifact_ids["feature_matrix_bundle"] = matrix_output.feature_matrix_artifact_id
        elif section == SectionName.SPATIAL_VALIDATION:
            if spatial_compatibility is None:
                raise ValueError(
                    "spatial_validation requires feature_matrix_load outputs or "
                    "a compatible base_artifact_id."
                )
            spatial_validation(SpatialValidationInput(spatial_compatibility=spatial_compatibility))
        elif section == SectionName.MODEL_FIT:
            if x_matrix is None or metadata_df is None:
                raise ValueError(
                    "model_fit requires feature_matrix_load outputs or a compatible "
                    "feature_matrix base artifact."
                )
            fit_output = model_fit(
                ModelFitInput(
                    x_matrix=x_matrix,
                    metadata_df=metadata_df,
                    target_column=request.target_column,
                    cv_mode=request.cv_mode,
                    model=request.model,
                    subject=request.subject,
                    train_subject=request.train_subject,
                    test_subject=request.test_subject,
                    seed=request.seed,
                    primary_metric_name=request.primary_metric_name,
                    methodology_policy_name=request.methodology_policy_name,
                    class_weight_policy=request.class_weight_policy,
                    feature_recipe_id=request.feature_recipe_id,
                    tuning_enabled=request.tuning_enabled,
                    tuning_search_space_id=request.tuning_search_space_id,
                    tuning_search_space_version=request.tuning_search_space_version,
                    tuning_inner_cv_scheme=request.tuning_inner_cv_scheme,
                    tuning_inner_group_field=request.tuning_inner_group_field,
                    tuning_summary_path=request.tuning_summary_path,
                    tuning_best_params_path=request.tuning_best_params_path,
                    fit_timing_summary_path=request.fit_timing_summary_path,
                    interpretability_enabled=request.interpretability_enabled_override,
                    max_outer_folds=request.max_outer_folds,
                    profiling_only=bool(request.profiling_only),
                    profile_inner_folds=request.profile_inner_folds,
                    profile_tuning_candidates=request.profile_tuning_candidates,
                    compute_policy=request.compute_policy,
                    model_fit_assignment=stage_assignment_map.get(StageKey.MODEL_FIT),
                    tuning_assignment=stage_assignment_map.get(StageKey.TUNING),
                    tuning_fallback_executor_id=stage_fallback_executor_ids.get(StageKey.TUNING),
                    run_id=request.run_id,
                    config_filename=request.config_filename,
                    report_dir=request.report_dir,
                    build_pipeline_fn=build_pipeline_fn,
                    scores_for_predictions_fn=scores_for_predictions_fn,
                    extract_linear_coefficients_fn=extract_linear_coefficients_fn,
                    progress_callback=request.progress_callback,
                )
            )
            compute_runtime_metadata = (
                dict(fit_output.compute_runtime_metadata)
                if isinstance(fit_output.compute_runtime_metadata, dict)
                else None
            )
            tuning_assignment = stage_assignment_map.get(StageKey.TUNING)
            if tuning_assignment is not None:
                tuning_fallback_reason: str | None = None
                for row in fit_output.tuning_records:
                    fallback_reason = row.get("tuning_executor_fallback_reason")
                    if isinstance(fallback_reason, str) and fallback_reason.strip():
                        tuning_fallback_reason = fallback_reason.strip()
                        break
                if tuning_fallback_reason is not None:
                    stage_assignment_map[StageKey.TUNING] = tuning_assignment.model_copy(
                        update={
                            "fallback_used": True,
                            "fallback_reason": tuning_fallback_reason,
                        }
                    )
        elif section == SectionName.INTERPRETABILITY:
            if fit_output is None:
                raise ValueError(
                    "interpretability requires model_fit outputs. Include model_fit in the "
                    "requested section path."
                )
            if feature_matrix_artifact_id is None:
                raise ValueError("interpretability requires a feature_matrix artifact reference.")
            if request.reuse_completed_artifacts and request.interpretability_summary_path.exists():
                reusable_interpretability = find_reusable_run_artifact(
                    request,
                    artifact_type=ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
                    expected_config_hash=None,
                )
                if reusable_interpretability is not None:
                    interpretability_summary = json.loads(
                        request.interpretability_summary_path.read_text(encoding="utf-8")
                    )
                    artifact_ids[ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE] = (
                        reusable_interpretability.artifact_id
                    )
                    reused_sections.append(section.value)
                    executed_sections.append(section.value)
                    section_timings_seconds[section.value] = float(perf_counter() - section_start)
                    _emit_section_event(
                        section_name=section.value,
                        status="reused",
                        completed_units=float(section_number),
                        total_units=float(total_sections),
                        message=f"reused section {section.value}",
                    )
                    continue
            interpretability_output = interpretability(
                InterpretabilityInput(
                    interpretability_enabled=fit_output.interpretability_enabled,
                    interpretability_fold_rows=fit_output.interpretability_fold_rows,
                    interpretability_vectors=fit_output.interpretability_vectors,
                    fold_artifacts_path=request.interpretability_fold_artifacts_path,
                    summary_path=request.interpretability_summary_path,
                    compute_interpretability_stability_fn=compute_interpretability_stability_fn,
                    run_id=request.run_id,
                    artifact_registry_path=request.artifact_registry_path,
                    code_ref=request.code_ref,
                    upstream_feature_matrix_artifact_id=feature_matrix_artifact_id,
                    cv_mode=request.cv_mode,
                    model=request.model,
                    target_column=request.target_column,
                    subject=request.subject,
                )
            )
            interpretability_summary = interpretability_output.interpretability_summary
            artifact_ids[ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE] = (
                interpretability_output.interpretability_artifact_id
            )
        elif section == SectionName.EVALUATION:
            if fit_output is None:
                raise ValueError(
                    "evaluation requires model_fit outputs. Include model_fit in the requested "
                    "section path."
                )
            if x_matrix is None:
                raise ValueError("evaluation requires the feature matrix in memory.")
            if spatial_compatibility is None:
                raise ValueError(
                    "evaluation requires spatial compatibility outputs from feature_matrix_load."
                )
            if feature_matrix_artifact_id is None:
                raise ValueError("evaluation requires a feature_matrix artifact reference.")
            if interpretability_summary is None:
                raise ValueError(
                    "evaluation requires interpretability summary. Include interpretability in "
                    "the section path."
                )
            if request.reuse_completed_artifacts and request.metrics_path.exists():
                reusable_metrics = find_reusable_run_artifact(
                    request,
                    artifact_type=ARTIFACT_TYPE_METRICS_BUNDLE,
                    expected_config_hash=compute_config_hash(
                        {
                            "run_id": request.run_id,
                            "target": request.target_column,
                            "model": request.model,
                            "cv": request.cv_mode,
                            "seed": int(request.seed),
                        }
                    ),
                )
                if reusable_metrics is not None:
                    metrics = json.loads(request.metrics_path.read_text(encoding="utf-8"))
                    artifact_ids[ARTIFACT_TYPE_METRICS_BUNDLE] = reusable_metrics.artifact_id
                    reused_sections.append(section.value)
                    executed_sections.append(section.value)
                    section_timings_seconds[section.value] = float(perf_counter() - section_start)
                    _emit_section_event(
                        section_name=section.value,
                        status="reused",
                        completed_units=float(section_number),
                        total_units=float(total_sections),
                        message=f"reused section {section.value}",
                    )
                    continue
            evaluation_output = evaluation(
                EvaluationInput(
                    x_matrix=x_matrix,
                    metadata_df=metadata_df,
                    y=fit_output.y,
                    splits=fit_output.splits,
                    fold_rows=fit_output.fold_rows,
                    split_rows=fit_output.split_rows,
                    prediction_rows=fit_output.prediction_rows,
                    y_true_all=fit_output.y_true_all,
                    y_pred_all=fit_output.y_pred_all,
                    subject=request.subject,
                    train_subject=request.train_subject,
                    test_subject=request.test_subject,
                    n_permutations=request.n_permutations,
                    primary_metric_name=request.primary_metric_name,
                    feature_recipe_id=request.feature_recipe_id,
                    emit_feature_qc_artifacts=bool(request.emit_feature_qc_artifacts),
                    feature_qc_summary_path=request.feature_qc_summary_path,
                    feature_qc_selected_samples_path=request.feature_qc_selected_samples_path,
                    feature_quality_policy=(
                        dict(request.feature_quality_policy)
                        if isinstance(request.feature_quality_policy, dict)
                        else None
                    ),
                    permutation_metric_name=request.permutation_metric_name,
                    permutation_alpha=float(request.permutation_alpha),
                    permutation_minimum_required=int(request.permutation_minimum_required),
                    permutation_require_pass_for_validity=bool(
                        request.permutation_require_pass_for_validity
                    ),
                    permutation_assignment=stage_assignment_map.get(StageKey.PERMUTATION),
                    methodology_policy_name=request.methodology_policy_name,
                    evidence_run_role=str(request.evidence_run_role),
                    repeat_id=int(request.repeat_id),
                    repeat_count=int(request.repeat_count),
                    base_run_id=request.base_run_id,
                    evidence_policy_effective=(
                        dict(request.evidence_policy_effective)
                        if isinstance(request.evidence_policy_effective, dict)
                        else None
                    ),
                    subgroup_reporting_enabled=request.subgroup_reporting_enabled,
                    subgroup_dimensions=list(request.subgroup_dimensions),
                    subgroup_min_samples_per_group=request.subgroup_min_samples_per_group,
                    subgroup_min_classes_per_group=request.subgroup_min_classes_per_group,
                    subgroup_report_small_groups=request.subgroup_report_small_groups,
                    confirmatory_guardrails_enabled=bool(request.confirmatory_guardrails_enabled),
                    subgroup_evidence_role=str(request.subgroup_evidence_role),
                    subgroup_primary_evidence_allowed=bool(
                        request.subgroup_primary_evidence_allowed
                    ),
                    subgroup_metrics_json_path=request.subgroup_metrics_json_path,
                    subgroup_metrics_csv_path=request.subgroup_metrics_csv_path,
                    tuning_summary_path=request.tuning_summary_path,
                    tuning_best_params_path=request.tuning_best_params_path,
                    calibration_enabled=bool(request.calibration_enabled),
                    calibration_n_bins=int(request.calibration_n_bins),
                    calibration_require_probabilities_for_validity=bool(
                        request.calibration_require_probabilities_for_validity
                    ),
                    calibration_summary_path=request.calibration_summary_path,
                    calibration_table_path=request.calibration_table_path,
                    spatial_compatibility=spatial_compatibility,
                    spatial_report_path=request.spatial_report_path,
                    interpretability_summary=interpretability_summary,
                    interpretability_summary_path=request.interpretability_summary_path,
                    fold_metrics_path=request.fold_metrics_path,
                    fold_splits_path=request.fold_splits_path,
                    predictions_path=request.predictions_path,
                    config_filename=request.config_filename,
                    build_pipeline_fn=build_pipeline_fn,
                    evaluate_permutations_fn=evaluate_permutations_fn,
                    run_id=request.run_id,
                    artifact_registry_path=request.artifact_registry_path,
                    code_ref=request.code_ref,
                    upstream_feature_matrix_artifact_id=feature_matrix_artifact_id,
                    metrics_path=request.metrics_path,
                    model=request.model,
                    target_column=request.target_column,
                    cv_mode=request.cv_mode,
                    seed=request.seed,
                    progress_callback=request.progress_callback,
                )
            )
            metrics = evaluation_output.metrics
            permutation_assignment = stage_assignment_map.get(StageKey.PERMUTATION)
            permutation_payload = metrics.get("permutation_test")
            if permutation_assignment is not None and isinstance(permutation_payload, dict):
                fallback_reason = permutation_payload.get("permutation_executor_fallback_reason")
                if isinstance(fallback_reason, str) and fallback_reason.strip():
                    stage_assignment_map[StageKey.PERMUTATION] = permutation_assignment.model_copy(
                        update={
                            "fallback_used": True,
                            "fallback_reason": fallback_reason.strip(),
                        }
                    )
            artifact_ids[ARTIFACT_TYPE_METRICS_BUNDLE] = evaluation_output.metrics_artifact_id
        else:
            raise ValueError(f"Unsupported section encountered in execution plan: {section.value}")

        executed_sections.append(section.value)
        section_timings_seconds[section.value] = float(perf_counter() - section_start)
        _emit_section_event(
            section_name=section.value,
            status="finished",
            completed_units=float(section_number),
            total_units=float(total_sections),
            message=f"finished section {section.value}",
        )

    return SegmentExecutionResult(
        planned_sections=[section.value for section in planned_sections],
        executed_sections=executed_sections,
        reused_sections=reused_sections,
        artifact_ids=artifact_ids,
        metrics=metrics,
        spatial_compatibility=spatial_compatibility,
        interpretability_summary=interpretability_summary,
        compute_runtime_metadata=compute_runtime_metadata,
        section_timings_seconds=section_timings_seconds,
        stage_assignments=[stage_assignment_map[stage] for stage in StageKey if stage in stage_assignment_map],
    )
