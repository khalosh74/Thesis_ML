from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_FEATURE_CACHE,
    ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE,
    ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    ArtifactRecord,
    compute_config_hash,
    find_latest_compatible_artifact,
    get_artifact,
)
from Thesis_ML.experiments.sections import (
    DatasetSelectionInput,
    EvaluationInput,
    FeatureCacheBuildInput,
    FeatureMatrixLoadInput,
    InterpretabilityInput,
    ModelFitInput,
    SpatialValidationInput,
    dataset_selection,
    evaluation,
    feature_cache_build,
    feature_matrix_load,
    interpretability,
    model_fit,
    spatial_validation,
)
from Thesis_ML.orchestration.contracts import ReusePolicy, SectionName

EXECUTION_SECTION_ORDER: tuple[SectionName, ...] = (
    SectionName.DATASET_SELECTION,
    SectionName.FEATURE_CACHE_BUILD,
    SectionName.FEATURE_MATRIX_LOAD,
    SectionName.SPATIAL_VALIDATION,
    SectionName.MODEL_FIT,
    SectionName.INTERPRETABILITY,
    SectionName.EVALUATION,
)

_SECTION_TO_INDEX = {name: idx for idx, name in enumerate(EXECUTION_SECTION_ORDER)}
_EARLIEST_SECTION = EXECUTION_SECTION_ORDER[0]
_LATEST_SECTION = EXECUTION_SECTION_ORDER[-1]


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
    spatial_report_path: Path
    interpretability_summary_path: Path
    interpretability_fold_artifacts_path: Path
    start_section: str | SectionName | None = None
    end_section: str | SectionName | None = None
    base_artifact_id: str | None = None
    reuse_policy: str | ReusePolicy | None = None
    build_pipeline_fn: Callable[..., Any] | None = None
    load_features_from_cache_fn: Callable[..., tuple[np.ndarray, pd.DataFrame, dict[str, Any]]] | None = None
    scores_for_predictions_fn: Callable[..., dict[str, list[Any]]] | None = None
    extract_linear_coefficients_fn: Callable[..., tuple[np.ndarray, np.ndarray, list[str]]] | None = None
    compute_interpretability_stability_fn: Callable[[list[np.ndarray]], dict[str, Any]] | None = None
    evaluate_permutations_fn: Callable[..., dict[str, Any]] | None = None


@dataclass(frozen=True)
class SegmentExecutionResult:
    planned_sections: list[str]
    executed_sections: list[str]
    artifact_ids: dict[str, str]
    metrics: dict[str, Any] | None
    spatial_compatibility: dict[str, Any] | None
    interpretability_summary: dict[str, Any] | None


def _normalize_section_name(
    value: str | SectionName | None,
    *,
    field_name: str,
    default: SectionName,
) -> SectionName:
    if value is None:
        return default
    if isinstance(value, SectionName):
        return value
    try:
        return SectionName(str(value))
    except ValueError as exc:
        allowed = ", ".join(section.value for section in EXECUTION_SECTION_ORDER)
        raise ValueError(
            f"Invalid {field_name}='{value}'. Allowed values: {allowed}"
        ) from exc


def _normalize_reuse_policy(value: str | ReusePolicy | None) -> ReusePolicy:
    if value is None:
        return ReusePolicy.AUTO
    if isinstance(value, ReusePolicy):
        return value
    try:
        return ReusePolicy(str(value))
    except ValueError as exc:
        allowed = ", ".join(policy.value for policy in ReusePolicy)
        raise ValueError(f"Invalid reuse_policy='{value}'. Allowed values: {allowed}") from exc


def plan_section_path(
    start_section: str | SectionName | None = None,
    end_section: str | SectionName | None = None,
) -> list[SectionName]:
    start = _normalize_section_name(
        start_section,
        field_name="start_section",
        default=_EARLIEST_SECTION,
    )
    end = _normalize_section_name(
        end_section,
        field_name="end_section",
        default=_LATEST_SECTION,
    )
    start_idx = _SECTION_TO_INDEX[start]
    end_idx = _SECTION_TO_INDEX[end]
    if start_idx > end_idx:
        raise ValueError(
            "Invalid section range: start_section must be before or equal to end_section."
        )
    return list(EXECUTION_SECTION_ORDER[start_idx : end_idx + 1])


def _requires_base_artifact(start: SectionName) -> bool:
    return _SECTION_TO_INDEX[start] >= _SECTION_TO_INDEX[SectionName.FEATURE_MATRIX_LOAD]


def _expected_base_artifact_type(start: SectionName) -> str:
    if start == SectionName.FEATURE_MATRIX_LOAD:
        return ARTIFACT_TYPE_FEATURE_CACHE
    return ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE


def _resolve_base_artifact(
    request: SegmentExecutionRequest,
    start_section: SectionName,
    reuse_policy: ReusePolicy,
) -> ArtifactRecord | None:
    if not _requires_base_artifact(start_section):
        if request.base_artifact_id:
            raise ValueError(
                "base_artifact_id is only valid when start_section is at or after feature_matrix_load."
            )
        return None

    if reuse_policy == ReusePolicy.DISALLOW:
        raise ValueError(
            "reuse_policy='disallow' is incompatible with segmented execution that starts after "
            "feature_cache_build."
        )

    expected_type = _expected_base_artifact_type(start_section)
    if request.base_artifact_id:
        record = get_artifact(
            registry_path=request.artifact_registry_path,
            artifact_id=request.base_artifact_id,
        )
        if record is None:
            raise ValueError(
                f"Base artifact '{request.base_artifact_id}' was not found in registry."
            )
        if record.artifact_type != expected_type:
            raise ValueError(
                f"Incompatible base artifact '{record.artifact_id}': expected artifact_type "
                f"'{expected_type}', got '{record.artifact_type}'."
            )
        return record

    if reuse_policy == ReusePolicy.REQUIRE_EXPLICIT_BASE:
        raise ValueError(
            "reuse_policy='require_explicit_base' requires base_artifact_id for segmented runs."
        )

    if start_section == SectionName.FEATURE_MATRIX_LOAD:
        feature_cache_config_hash = compute_config_hash(
            {
                "index_csv": str(request.index_csv.resolve()),
                "data_root": str(request.data_root.resolve()),
                "cache_dir": str(request.cache_dir.resolve()),
                "group_key": "subject_session_bas",
                "force": False,
            }
        )
        cached = find_latest_compatible_artifact(
            registry_path=request.artifact_registry_path,
            artifact_type=ARTIFACT_TYPE_FEATURE_CACHE,
            config_hash=feature_cache_config_hash,
            code_ref=request.code_ref,
        )
        if cached is not None:
            return cached

    raise ValueError(
        f"start_section='{start_section.value}' requires a compatible base artifact. "
        "Provide base_artifact_id explicitly."
    )


def _is_after_or_equal(left: SectionName, right: SectionName) -> bool:
    return _SECTION_TO_INDEX[left] >= _SECTION_TO_INDEX[right]


def _require_callable(name: str, value: Callable[..., Any] | None) -> Callable[..., Any]:
    if value is None:
        raise ValueError(f"Missing required callable for segment execution: {name}")
    return value


def execute_section_segment(request: SegmentExecutionRequest) -> SegmentExecutionResult:
    planned_sections = plan_section_path(request.start_section, request.end_section)
    start_section = planned_sections[0]
    reuse_policy = _normalize_reuse_policy(request.reuse_policy)
    base_artifact = _resolve_base_artifact(
        request=request,
        start_section=start_section,
        reuse_policy=reuse_policy,
    )

    build_pipeline_fn = _require_callable("build_pipeline_fn", request.build_pipeline_fn)
    load_features_from_cache_fn = _require_callable(
        "load_features_from_cache_fn", request.load_features_from_cache_fn
    )
    scores_for_predictions_fn = _require_callable(
        "scores_for_predictions_fn", request.scores_for_predictions_fn
    )
    extract_linear_coefficients_fn = _require_callable(
        "extract_linear_coefficients_fn", request.extract_linear_coefficients_fn
    )
    compute_interpretability_stability_fn = _require_callable(
        "compute_interpretability_stability_fn",
        request.compute_interpretability_stability_fn,
    )
    evaluate_permutations_fn = _require_callable(
        "evaluate_permutations_fn", request.evaluate_permutations_fn
    )

    executed_sections: list[str] = []
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

    if (
        SectionName.DATASET_SELECTION not in planned_sections
        and _is_after_or_equal(planned_sections[-1], SectionName.FEATURE_MATRIX_LOAD)
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
        expected_type = _expected_base_artifact_type(start_section)
        if expected_type == ARTIFACT_TYPE_FEATURE_CACHE:
            cache_manifest_path = Path(base_artifact.path)
            feature_cache_artifact_id = base_artifact.artifact_id
            artifact_ids["feature_cache"] = base_artifact.artifact_id
        else:
            feature_matrix_artifact_id = base_artifact.artifact_id
            artifact_ids["feature_matrix_bundle"] = base_artifact.artifact_id
            if selected_index_df is None:
                raise ValueError(
                    "Segment state error: selected dataset rows were not initialized."
                )
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
                    run_id=request.run_id,
                    config_filename=request.config_filename,
                    report_dir=request.report_dir,
                    build_pipeline_fn=build_pipeline_fn,
                    scores_for_predictions_fn=scores_for_predictions_fn,
                    extract_linear_coefficients_fn=extract_linear_coefficients_fn,
                )
            )
        elif section == SectionName.INTERPRETABILITY:
            if fit_output is None:
                raise ValueError(
                    "interpretability requires model_fit outputs. Include model_fit in the "
                    "requested section path."
                )
            if feature_matrix_artifact_id is None:
                raise ValueError(
                    "interpretability requires a feature_matrix artifact reference."
                )
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
            evaluation_output = evaluation(
                EvaluationInput(
                    x_matrix=x_matrix,
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
                )
            )
            metrics = evaluation_output.metrics
            artifact_ids[ARTIFACT_TYPE_METRICS_BUNDLE] = evaluation_output.metrics_artifact_id
        else:
            raise ValueError(f"Unsupported section encountered in execution plan: {section.value}")

        executed_sections.append(section.value)

    return SegmentExecutionResult(
        planned_sections=[section.value for section in planned_sections],
        executed_sections=executed_sections,
        artifact_ids=artifact_ids,
        metrics=metrics,
        spatial_compatibility=spatial_compatibility,
        interpretability_summary=interpretability_summary,
    )
