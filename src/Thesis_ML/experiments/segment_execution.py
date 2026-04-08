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
    ARTIFACT_TYPE_MODEL_BUNDLE,
    ARTIFACT_TYPE_MODEL_REFIT_BUNDLE,
    compute_config_hash,
    get_artifact,
)
from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy
from Thesis_ML.experiments.feature_space_loading import load_feature_matrix
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
from Thesis_ML.experiments.stage_lease_manager import (
    StageLeaseHandle,
    StageLeaseManager,
    StageLeaseReleaseResult,
    StageLeaseRequest,
)
from Thesis_ML.experiments.stage_observability import StageBoundaryRecorder
from Thesis_ML.experiments.stage_planner import StageResourceContract
from Thesis_ML.features.preprocessing import BASELINE_STANDARD_SCALER_RECIPE_ID
from Thesis_ML.experiments.contracts import ReusePolicy, SectionName


@dataclass(frozen=True, kw_only=True)
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
    release_scope_enforcement: bool = False
    compiled_scope_selected_samples_path: Path | None = None
    compiled_scope_manifest_path: Path | None = None
    feature_space: str = "whole_brain_masked"
    roi_spec_path: Path | None = None
    preprocessing_strategy: str | None = None
    dimensionality_strategy: str = "none"
    pca_n_components: int | None = None
    pca_variance_ratio: float | None = None
    seed: int
    n_permutations: int
    run_id: str
    config_filename: str
    report_dir: Path
    artifact_registry_path: Path
    artifact_registry_fallback_paths: tuple[Path, ...] = ()
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
    primary_metric_aggregation: str = "mean_fold_scores"
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
    persist_models: bool = False
    persist_fold_models: bool = True
    persist_final_refit_model: bool = False
    experiment_id: str | None = None
    variant_id: str | None = None
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
    stage_observer: StageBoundaryRecorder | None = None
    stage_resource_contracts: tuple[StageResourceContract, ...] | None = None
    stage_lease_manager: StageLeaseManager | None = None


@dataclass(frozen=True)
class SegmentExecutionResult:
    planned_sections: list[str]
    executed_sections: list[str]
    reused_sections: list[str]
    artifact_ids: dict[str, str]
    metrics: dict[str, Any] | None
    spatial_compatibility: dict[str, Any] | None
    interpretability_summary: dict[str, Any] | None
    model_persistence: dict[str, Any] | None = None
    compute_runtime_metadata: dict[str, Any] | None = None
    section_timings_seconds: dict[str, float] | None = None
    stage_timings_seconds: dict[str, float] | None = None
    stage_timing_metadata: dict[str, dict[str, Any]] | None = None
    stage_assignments: list[StageAssignment] | None = None


_SECTION_TO_STAGE_KEY: dict[SectionName, StageKey] = {
    SectionName.DATASET_SELECTION: StageKey.DATASET_SELECTION,
    SectionName.FEATURE_CACHE_BUILD: StageKey.FEATURE_CACHE_BUILD,
    SectionName.FEATURE_MATRIX_LOAD: StageKey.FEATURE_MATRIX_LOAD,
    SectionName.SPATIAL_VALIDATION: StageKey.SPATIAL_VALIDATION,
    SectionName.MODEL_FIT: StageKey.MODEL_FIT,
    SectionName.EVALUATION: StageKey.EVALUATION,
}

_SECTION_TO_STAGE_KEYS_FOR_LEASING: dict[SectionName, tuple[StageKey, ...]] = {
    SectionName.DATASET_SELECTION: (StageKey.DATASET_SELECTION,),
    SectionName.FEATURE_CACHE_BUILD: (StageKey.FEATURE_CACHE_BUILD,),
    SectionName.FEATURE_MATRIX_LOAD: (StageKey.FEATURE_MATRIX_LOAD,),
    SectionName.SPATIAL_VALIDATION: (StageKey.SPATIAL_VALIDATION,),
    SectionName.MODEL_FIT: (StageKey.PREPROCESS, StageKey.MODEL_FIT, StageKey.TUNING),
    SectionName.EVALUATION: (StageKey.PERMUTATION, StageKey.EVALUATION),
}


def _section_stage_keys_for_leasing(section_name: SectionName) -> tuple[StageKey, ...]:
    if section_name in _SECTION_TO_STAGE_KEYS_FOR_LEASING:
        return _SECTION_TO_STAGE_KEYS_FOR_LEASING[section_name]
    mapped = _SECTION_TO_STAGE_KEY.get(section_name)
    if mapped is None:
        return ()
    return (mapped,)


def _build_section_stage_lease_plan(
    *,
    section_name: SectionName,
    run_id: str,
    stage_resource_contract_map: dict[StageKey, StageResourceContract],
) -> dict[str, Any]:
    relevant_stage_keys = _section_stage_keys_for_leasing(section_name)
    relevant_contracts = [
        stage_resource_contract_map[stage_key]
        for stage_key in relevant_stage_keys
        if stage_key in stage_resource_contract_map
    ]
    gpu_contracts = [
        contract for contract in relevant_contracts if bool(contract.requires_gpu_lease)
    ]
    lease_required = bool(gpu_contracts)
    selected_contract = (
        gpu_contracts[0]
        if gpu_contracts
        else (relevant_contracts[0] if relevant_contracts else None)
    )
    lease_owner_identity = f"{str(run_id)}:{section_name.value}"
    lease_reason = (
        ";".join(
            f"{contract.stage_key.value}:{contract.lease_reason}" for contract in gpu_contracts
        )
        if gpu_contracts
        else "section_cpu_only"
    )
    return {
        "lease_required": bool(lease_required),
        "lease_class": "gpu" if lease_required else "cpu",
        "lease_reason": str(lease_reason),
        "lease_owner_identity": str(lease_owner_identity),
        "lease_expected_stage_keys": [stage_key.value for stage_key in relevant_stage_keys],
        "lease_expected_backend_family": (
            str(selected_contract.expected_backend_family)
            if selected_contract is not None
            and selected_contract.expected_backend_family is not None
            else None
        ),
        "lease_expected_executor_id": (
            str(selected_contract.expected_executor_id)
            if selected_contract is not None and selected_contract.expected_executor_id is not None
            else None
        ),
        "lease_acquired": False,
        "lease_wait_seconds": 0.0,
        "lease_queue_depth_at_acquire": 0,
        "lease_acquired_at_utc": None,
        "lease_released_at_utc": None,
        "lease_held_seconds": None,
        "lease_released": None,
        "lease_id": None,
    }


def _stage_assignment_metadata(assignment: StageAssignment | None) -> dict[str, Any]:
    if assignment is None:
        return {}
    payload: dict[str, Any] = {
        "planned_backend_family": str(assignment.backend_family),
        "planned_compute_lane": (
            str(assignment.compute_lane) if assignment.compute_lane is not None else None
        ),
        "planned_executor_id": (
            str(assignment.executor_id) if assignment.executor_id is not None else None
        ),
        "official_admitted": assignment.official_admitted,
        "assignment_source": str(assignment.source),
        "fallback_expected": bool(assignment.fallback_used),
    }
    if assignment.fallback_reason is not None:
        payload["fallback_reason"] = str(assignment.fallback_reason)
    return payload


def _stage_observed_metadata(
    *,
    assignment: StageAssignment | None,
    fallback_reason: str | None = None,
    fallback_used: bool | None = None,
    execution_mode: str | None = None,
    observed_backend_family: str | None = None,
    observed_compute_lane: str | None = None,
    observed_executor_id: str | None = None,
    stage_lease_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _stage_assignment_metadata(assignment)
    payload["observed_backend_family"] = (
        str(observed_backend_family)
        if observed_backend_family is not None
        else (
            str(assignment.backend_family)
            if assignment is not None and assignment.backend_family is not None
            else None
        )
    )
    payload["observed_compute_lane"] = (
        str(observed_compute_lane)
        if observed_compute_lane is not None
        else (
            str(assignment.compute_lane)
            if assignment is not None and assignment.compute_lane is not None
            else None
        )
    )
    payload["observed_executor_id"] = (
        str(observed_executor_id)
        if observed_executor_id is not None
        else (
            str(assignment.executor_id)
            if assignment is not None and assignment.executor_id is not None
            else None
        )
    )
    payload["fallback_used"] = (
        bool(fallback_used)
        if fallback_used is not None
        else (bool(assignment.fallback_used) if assignment is not None else False)
    )
    if fallback_reason is not None:
        payload["fallback_reason"] = str(fallback_reason)
    elif assignment is not None and assignment.fallback_reason is not None:
        payload["fallback_reason"] = str(assignment.fallback_reason)
    if execution_mode is not None:
        payload["execution_mode"] = str(execution_mode)
    if isinstance(stage_lease_metadata, dict):
        for key, value in stage_lease_metadata.items():
            payload[str(key)] = value
    return payload


def _release_section_stage_lease(
    *,
    stage_lease_handle: StageLeaseHandle | None,
    stage_lease_manager: StageLeaseManager | None,
    section_stage_lease_metadata: dict[str, Any],
    stage_observer: StageBoundaryRecorder | None,
    stage_key: StageKey | None,
) -> StageLeaseReleaseResult | None:
    if stage_lease_handle is None or stage_lease_manager is None:
        return None
    release_result = stage_lease_manager.release(stage_lease_handle)
    section_stage_lease_metadata["lease_released"] = bool(release_result.released)
    section_stage_lease_metadata["lease_released_at_utc"] = str(release_result.released_at_utc)
    section_stage_lease_metadata["lease_held_seconds"] = (
        float(release_result.hold_seconds) if release_result.hold_seconds is not None else None
    )
    if stage_observer is not None and stage_key is not None:
        stage_observer.update_stage_context(
            stage_key,
            metadata=dict(section_stage_lease_metadata),
        )
    return release_result


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
    model_persistence_summary: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    compute_runtime_metadata: dict[str, Any] | None = None
    section_timings_seconds: dict[str, float] = {}
    stage_timings_seconds: dict[str, float] = {}
    stage_timing_metadata: dict[str, dict[str, Any]] = {}
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
    stage_observer = request.stage_observer
    stage_resource_contract_map: dict[StageKey, StageResourceContract] = {}
    for contract in request.stage_resource_contracts or ():
        normalized_contract = (
            contract
            if isinstance(contract, StageResourceContract)
            else StageResourceContract(**dict(contract))
        )
        stage_resource_contract_map[StageKey(str(normalized_contract.stage_key))] = (
            normalized_contract
        )
    stage_lease_manager = request.stage_lease_manager

    def _section_stage_key(section_name: SectionName) -> StageKey | None:
        return _SECTION_TO_STAGE_KEY.get(section_name)

    def _primary_artifacts_for_section(section_name: SectionName) -> list[str]:
        if section_name == SectionName.FEATURE_CACHE_BUILD:
            artifact_id = artifact_ids.get("feature_cache")
            return [str(artifact_id)] if isinstance(artifact_id, str) and artifact_id else []
        if section_name == SectionName.FEATURE_MATRIX_LOAD:
            artifact_id = artifact_ids.get("feature_matrix_bundle")
            return [str(artifact_id)] if isinstance(artifact_id, str) and artifact_id else []
        if section_name == SectionName.EVALUATION:
            artifact_id = artifact_ids.get(ARTIFACT_TYPE_METRICS_BUNDLE)
            return [str(artifact_id)] if isinstance(artifact_id, str) and artifact_id else []
        return []

    def _record_section_duration(section_name: str, duration_seconds: float) -> None:
        section_timings_seconds[section_name] = float(duration_seconds)
        if section_name in stage_timings_seconds:
            return
        stage_timings_seconds[section_name] = float(duration_seconds)
        stage_timing_metadata[section_name] = {
            "duration_source": "section_timing",
            "derived_from": str(section_name),
        }

    def _set_stage_timing(
        *,
        stage_name: str,
        duration_seconds: float | None,
        duration_source: str,
        derived_from: str | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        metadata_payload: dict[str, Any] = {
            "duration_source": str(duration_source),
        }
        if derived_from is not None:
            metadata_payload["derived_from"] = str(derived_from)
        if fallback_reason is not None:
            metadata_payload["fallback_reason"] = str(fallback_reason)
        if duration_seconds is not None:
            stage_timings_seconds[stage_name] = float(duration_seconds)
        stage_timing_metadata[stage_name] = metadata_payload

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
                release_scope_enforcement=bool(request.release_scope_enforcement),
                compiled_scope_selected_samples_path=request.compiled_scope_selected_samples_path,
                compiled_scope_manifest_path=request.compiled_scope_manifest_path,
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
            x_matrix, metadata_df, spatial_compatibility = load_feature_matrix(
                selected_index_df=selected_index_df,
                data_root=request.data_root,
                cache_manifest_path=cache_manifest_path,
                spatial_report_path=request.spatial_report_path,
                affine_atol=request.affine_atol,
                feature_space=request.feature_space,
                roi_spec_path=request.roi_spec_path,
                load_features_from_cache_fn=load_features_from_cache_fn,
            )

    for section in planned_sections:
        section_number = int(len(executed_sections) + 1)
        section_start = perf_counter()
        stage_key = _section_stage_key(section)
        stage_assignment = (
            stage_assignment_map.get(stage_key) if isinstance(stage_key, StageKey) else None
        )
        section_stage_lease_metadata = _build_section_stage_lease_plan(
            section_name=section,
            run_id=str(request.run_id),
            stage_resource_contract_map=stage_resource_contract_map,
        )
        stage_lease_handle: StageLeaseHandle | None = None

        if bool(section_stage_lease_metadata.get("lease_required", False)):
            if stage_lease_manager is None:
                raise ValueError("gpu_stage_requires_stage_lease_manager")
            stage_lease_request = StageLeaseRequest(
                run_id=str(request.run_id),
                stage_key=str(
                    stage_key.value if isinstance(stage_key, StageKey) else section.value
                ),
                owner_identity=str(section_stage_lease_metadata["lease_owner_identity"]),
                lease_class="gpu",
                lease_reason=str(section_stage_lease_metadata.get("lease_reason") or ""),
                expected_backend_family=(
                    str(section_stage_lease_metadata.get("lease_expected_backend_family"))
                    if section_stage_lease_metadata.get("lease_expected_backend_family") is not None
                    else None
                ),
                expected_executor_id=(
                    str(section_stage_lease_metadata.get("lease_expected_executor_id"))
                    if section_stage_lease_metadata.get("lease_expected_executor_id") is not None
                    else None
                ),
                expected_compute_lane=(
                    str(stage_assignment.compute_lane)
                    if stage_assignment is not None and stage_assignment.compute_lane is not None
                    else None
                ),
            )
            stage_lease_handle = stage_lease_manager.acquire(stage_lease_request)
            section_stage_lease_metadata["lease_id"] = str(stage_lease_handle.lease_id)
            section_stage_lease_metadata["lease_acquired"] = True
            section_stage_lease_metadata["lease_wait_seconds"] = float(
                stage_lease_handle.wait_seconds
            )
            section_stage_lease_metadata["lease_queue_depth_at_acquire"] = int(
                stage_lease_handle.queue_depth_at_acquire
            )
            section_stage_lease_metadata["lease_acquired_at_utc"] = str(
                stage_lease_handle.acquired_at_utc
            )

        if stage_observer is not None and isinstance(stage_key, StageKey):
            stage_observer.stage_started(
                stage_key,
                metadata={
                    **_stage_assignment_metadata(stage_assignment),
                    **dict(section_stage_lease_metadata),
                },
            )
            stage_observer.update_stage_context(
                stage_key,
                metadata=dict(section_stage_lease_metadata),
            )
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
                    release_scope_enforcement=bool(request.release_scope_enforcement),
                    compiled_scope_selected_samples_path=request.compiled_scope_selected_samples_path,
                    compiled_scope_manifest_path=request.compiled_scope_manifest_path,
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
                    _record_section_duration(
                        section_name=section.value,
                        duration_seconds=float(perf_counter() - section_start),
                    )
                    _emit_section_event(
                        section_name=section.value,
                        status="reused",
                        completed_units=float(section_number),
                        total_units=float(total_sections),
                        message=f"reused section {section.value}",
                    )
                    if stage_observer is not None and isinstance(stage_key, StageKey):
                        stage_observer.stage_reused(
                            stage_key,
                            metadata={
                                **_stage_observed_metadata(
                                    assignment=stage_assignment,
                                    stage_lease_metadata=section_stage_lease_metadata,
                                ),
                                "primary_artifacts": _primary_artifacts_for_section(section),
                            },
                        )
                    _release_section_stage_lease(
                        stage_lease_handle=stage_lease_handle,
                        stage_lease_manager=stage_lease_manager,
                        section_stage_lease_metadata=section_stage_lease_metadata,
                        stage_observer=stage_observer,
                        stage_key=(stage_key if isinstance(stage_key, StageKey) else None),
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
                            "feature_space": request.feature_space,
                            "roi_spec_path": (
                                str(request.roi_spec_path.resolve())
                                if request.roi_spec_path is not None
                                else None
                            ),
                            "cache_manifest_path": str(cache_manifest_path.resolve()),
                        }
                    ),
                )
                if reusable_feature_matrix is not None:
                    x_matrix, metadata_df, spatial_compatibility = load_feature_matrix(
                        selected_index_df=selected_index_df,
                        data_root=request.data_root,
                        cache_manifest_path=cache_manifest_path,
                        spatial_report_path=request.spatial_report_path,
                        affine_atol=request.affine_atol,
                        feature_space=request.feature_space,
                        roi_spec_path=request.roi_spec_path,
                        load_features_from_cache_fn=load_features_from_cache_fn,
                    )
                    feature_matrix_artifact_id = reusable_feature_matrix.artifact_id
                    artifact_ids["feature_matrix_bundle"] = reusable_feature_matrix.artifact_id
                    reused_sections.append(section.value)
                    executed_sections.append(section.value)
                    _record_section_duration(
                        section_name=section.value,
                        duration_seconds=float(perf_counter() - section_start),
                    )
                    _emit_section_event(
                        section_name=section.value,
                        status="reused",
                        completed_units=float(section_number),
                        total_units=float(total_sections),
                        message=f"reused section {section.value}",
                    )
                    if stage_observer is not None and isinstance(stage_key, StageKey):
                        stage_observer.stage_reused(
                            stage_key,
                            metadata={
                                **_stage_observed_metadata(
                                    assignment=stage_assignment,
                                    stage_lease_metadata=section_stage_lease_metadata,
                                ),
                                "primary_artifacts": _primary_artifacts_for_section(section),
                            },
                        )
                    _release_section_stage_lease(
                        stage_lease_handle=stage_lease_handle,
                        stage_lease_manager=stage_lease_manager,
                        section_stage_lease_metadata=section_stage_lease_metadata,
                        stage_observer=stage_observer,
                        stage_key=(stage_key if isinstance(stage_key, StageKey) else None),
                    )
                    continue
            matrix_output = feature_matrix_load(
                FeatureMatrixLoadInput(
                    selected_index_df=selected_index_df,
                    data_root=request.data_root,
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
                    feature_space=request.feature_space,
                    roi_spec_path=request.roi_spec_path,
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
                    preprocessing_strategy=request.preprocessing_strategy,
                    dimensionality_strategy=request.dimensionality_strategy,
                    pca_n_components=request.pca_n_components,
                    pca_variance_ratio=request.pca_variance_ratio,
                    feature_recipe_id=request.feature_recipe_id,
                    persist_models=bool(request.persist_models),
                    persist_fold_models=bool(request.persist_fold_models),
                    persist_final_refit_model=bool(request.persist_final_refit_model),
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
                    artifact_registry_path=request.artifact_registry_path,
                    code_ref=request.code_ref,
                    upstream_feature_matrix_artifact_id=feature_matrix_artifact_id,
                    experiment_id=request.experiment_id,
                    variant_id=request.variant_id,
                    feature_space=request.feature_space,
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
            if fit_output.model_bundle_artifact_id is not None:
                artifact_ids[ARTIFACT_TYPE_MODEL_BUNDLE] = str(fit_output.model_bundle_artifact_id)
            if fit_output.final_refit_artifact_id is not None:
                artifact_ids[ARTIFACT_TYPE_MODEL_REFIT_BUNDLE] = str(
                    fit_output.final_refit_artifact_id
                )
            model_persistence_summary = {
                "enabled": bool(request.persist_models),
                "fold_models_saved": bool(request.persist_models and request.persist_fold_models),
                "n_fold_models": int(
                    sum(
                        1
                        for row in fit_output.saved_model_rows
                        if str(row.get("artifact_role")) == "fold_model"
                    )
                ),
                "model_summary_path": (
                    str(fit_output.model_summary_path.resolve())
                    if fit_output.model_summary_path is not None
                    else None
                ),
                "model_artifacts_csv_path": (
                    str(fit_output.model_artifacts_csv_path.resolve())
                    if fit_output.model_artifacts_csv_path is not None
                    else None
                ),
                "final_refit_saved": fit_output.final_refit_model_path is not None,
                "final_refit_model_path": (
                    str(fit_output.final_refit_model_path.resolve())
                    if fit_output.final_refit_model_path is not None
                    else None
                ),
                "final_refit_metadata_path": (
                    str(fit_output.final_refit_metadata_path.resolve())
                    if fit_output.final_refit_metadata_path is not None
                    else None
                ),
                "artifact_ids": {
                    ARTIFACT_TYPE_MODEL_BUNDLE: fit_output.model_bundle_artifact_id,
                    ARTIFACT_TYPE_MODEL_REFIT_BUNDLE: fit_output.final_refit_artifact_id,
                },
            }
            tuning_assignment = stage_assignment_map.get(StageKey.TUNING)
            tuning_fallback_reason: str | None = None
            if tuning_assignment is not None:
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
            fit_timing_totals = (
                fit_output.fit_timing_summary.get("totals_seconds")
                if isinstance(fit_output.fit_timing_summary, dict)
                else None
            )
            model_fit_duration_raw = (
                fit_timing_totals.get("outer_fold") if isinstance(fit_timing_totals, dict) else None
            )
            model_fit_duration = (
                float(model_fit_duration_raw)
                if isinstance(model_fit_duration_raw, (int, float))
                else None
            )
            if model_fit_duration is not None:
                _set_stage_timing(
                    stage_name=StageKey.MODEL_FIT.value,
                    duration_seconds=model_fit_duration,
                    duration_source="fit_timing_summary.totals_seconds.outer_fold",
                    derived_from="fit_timing_summary",
                )
            else:
                _set_stage_timing(
                    stage_name=StageKey.MODEL_FIT.value,
                    duration_seconds=stage_timings_seconds.get(StageKey.MODEL_FIT.value),
                    duration_source="section_timing",
                    derived_from=SectionName.MODEL_FIT.value,
                    fallback_reason="fit_timing_summary_missing_outer_fold",
                )

            tuning_timing_totals = (
                fit_output.tuning_summary.get("timing_totals_seconds")
                if isinstance(fit_output.tuning_summary, dict)
                else None
            )
            tuning_duration_raw = (
                tuning_timing_totals.get("tuned_search_total")
                if isinstance(tuning_timing_totals, dict)
                else None
            )
            tuning_duration = (
                float(tuning_duration_raw)
                if isinstance(tuning_duration_raw, (int, float))
                else None
            )
            if bool(request.tuning_enabled) and tuning_duration is not None:
                _set_stage_timing(
                    stage_name=StageKey.TUNING.value,
                    duration_seconds=tuning_duration,
                    duration_source="tuning_summary.timing_totals_seconds.tuned_search_total",
                    derived_from="tuning_summary",
                )
            elif bool(request.tuning_enabled):
                _set_stage_timing(
                    stage_name=StageKey.TUNING.value,
                    duration_seconds=None,
                    duration_source="unavailable",
                    derived_from="tuning_summary",
                    fallback_reason="tuning_duration_not_measured",
                )
            else:
                _set_stage_timing(
                    stage_name=StageKey.TUNING.value,
                    duration_seconds=None,
                    duration_source="not_applicable",
                    derived_from="tuning_policy",
                    fallback_reason="tuning_disabled",
                )
            if stage_observer is not None:
                preprocess_assignment = stage_assignment_map.get(StageKey.PREPROCESS)
                stage_observer.stage_finished(
                    StageKey.PREPROCESS,
                    metadata={
                        **_stage_observed_metadata(
                            assignment=preprocess_assignment,
                            stage_lease_metadata=section_stage_lease_metadata,
                        ),
                        "derived_from_stage": StageKey.MODEL_FIT.value,
                        "status_source": "derived_from_model_fit",
                    },
                    status="executed",
                )
                tuning_assignment_effective = stage_assignment_map.get(StageKey.TUNING)
                if bool(request.tuning_enabled):
                    stage_observer.stage_finished(
                        StageKey.TUNING,
                        metadata={
                            **_stage_observed_metadata(
                                assignment=tuning_assignment_effective,
                                fallback_reason=tuning_fallback_reason,
                                fallback_used=(
                                    bool(tuning_assignment_effective.fallback_used)
                                    if tuning_assignment_effective is not None
                                    else None
                                ),
                                stage_lease_metadata=section_stage_lease_metadata,
                            ),
                            "derived_from_stage": StageKey.MODEL_FIT.value,
                            "status_source": "derived_from_model_fit",
                        },
                        status="executed",
                    )
                else:
                    stage_observer.stage_skipped(
                        StageKey.TUNING,
                        metadata={
                            **_stage_observed_metadata(
                                assignment=tuning_assignment_effective,
                                fallback_used=False,
                                stage_lease_metadata=section_stage_lease_metadata,
                            ),
                            "derived_from_stage": StageKey.MODEL_FIT.value,
                            "status_source": "derived_from_model_fit",
                        },
                        reason="tuning_disabled",
                        derived_from_stage=StageKey.MODEL_FIT,
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
                    _record_section_duration(
                        section_name=section.value,
                        duration_seconds=float(perf_counter() - section_start),
                    )
                    _emit_section_event(
                        section_name=section.value,
                        status="reused",
                        completed_units=float(section_number),
                        total_units=float(total_sections),
                        message=f"reused section {section.value}",
                    )
                    if stage_observer is not None and isinstance(stage_key, StageKey):
                        stage_observer.stage_reused(
                            stage_key,
                            metadata={
                                **_stage_observed_metadata(
                                    assignment=stage_assignment,
                                    stage_lease_metadata=section_stage_lease_metadata,
                                ),
                                "primary_artifacts": _primary_artifacts_for_section(section),
                            },
                        )
                    _release_section_stage_lease(
                        stage_lease_handle=stage_lease_handle,
                        stage_lease_manager=stage_lease_manager,
                        section_stage_lease_metadata=section_stage_lease_metadata,
                        stage_observer=stage_observer,
                        stage_key=(stage_key if isinstance(stage_key, StageKey) else None),
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
                            "preprocessing_strategy": request.preprocessing_strategy,
                            "dimensionality_strategy": request.dimensionality_strategy,
                            "pca_n_components": request.pca_n_components,
                            "pca_variance_ratio": request.pca_variance_ratio,
                        }
                    ),
                )
                if reusable_metrics is not None:
                    metrics = json.loads(request.metrics_path.read_text(encoding="utf-8"))
                    artifact_ids[ARTIFACT_TYPE_METRICS_BUNDLE] = reusable_metrics.artifact_id
                    reused_sections.append(section.value)
                    executed_sections.append(section.value)
                    _record_section_duration(
                        section_name=section.value,
                        duration_seconds=float(perf_counter() - section_start),
                    )
                    _emit_section_event(
                        section_name=section.value,
                        status="reused",
                        completed_units=float(section_number),
                        total_units=float(total_sections),
                        message=f"reused section {section.value}",
                    )
                    if stage_observer is not None and isinstance(stage_key, StageKey):
                        stage_observer.stage_reused(
                            stage_key,
                            metadata={
                                **_stage_observed_metadata(
                                    assignment=stage_assignment,
                                    stage_lease_metadata=section_stage_lease_metadata,
                                ),
                                "primary_artifacts": _primary_artifacts_for_section(section),
                            },
                        )
                        permutation_assignment_effective = stage_assignment_map.get(
                            StageKey.PERMUTATION
                        )
                        if int(request.n_permutations) > 0:
                            permutation_payload = (
                                metrics.get("permutation_test")
                                if isinstance(metrics, dict)
                                else None
                            )
                            permutation_execution_mode = (
                                str(permutation_payload.get("execution_mode"))
                                if isinstance(permutation_payload, dict)
                                and isinstance(permutation_payload.get("execution_mode"), str)
                                and str(permutation_payload.get("execution_mode")).strip()
                                else None
                            )
                            permutation_backend_family = (
                                str(permutation_payload.get("backend_family"))
                                if isinstance(permutation_payload, dict)
                                and isinstance(permutation_payload.get("backend_family"), str)
                                and str(permutation_payload.get("backend_family")).strip()
                                else None
                            )
                            permutation_executor_id = (
                                str(permutation_payload.get("permutation_executor_id"))
                                if isinstance(permutation_payload, dict)
                                and isinstance(
                                    permutation_payload.get("permutation_executor_id"), str
                                )
                                and str(permutation_payload.get("permutation_executor_id")).strip()
                                else None
                            )
                            stage_observer.stage_reused(
                                StageKey.PERMUTATION,
                                metadata={
                                    **_stage_observed_metadata(
                                        assignment=permutation_assignment_effective,
                                        execution_mode=permutation_execution_mode,
                                        observed_backend_family=permutation_backend_family,
                                        observed_executor_id=permutation_executor_id,
                                        stage_lease_metadata=section_stage_lease_metadata,
                                    ),
                                    "derived_from_stage": StageKey.EVALUATION.value,
                                    "status_source": "derived_from_evaluation",
                                },
                                derived_from_stage=StageKey.EVALUATION,
                            )
                        else:
                            stage_observer.stage_skipped(
                                StageKey.PERMUTATION,
                                metadata={
                                    **_stage_observed_metadata(
                                        assignment=permutation_assignment_effective,
                                        fallback_used=False,
                                        stage_lease_metadata=section_stage_lease_metadata,
                                    ),
                                    "derived_from_stage": StageKey.EVALUATION.value,
                                    "status_source": "derived_from_evaluation",
                                },
                                reason="permutations_disabled",
                                derived_from_stage=StageKey.EVALUATION,
                            )
                    _release_section_stage_lease(
                        stage_lease_handle=stage_lease_handle,
                        stage_lease_manager=stage_lease_manager,
                        section_stage_lease_metadata=section_stage_lease_metadata,
                        stage_observer=stage_observer,
                        stage_key=(stage_key if isinstance(stage_key, StageKey) else None),
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
                    primary_metric_aggregation=request.primary_metric_aggregation,
                    preprocessing_strategy=request.preprocessing_strategy,
                    dimensionality_strategy=request.dimensionality_strategy,
                    pca_n_components=request.pca_n_components,
                    pca_variance_ratio=request.pca_variance_ratio,
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
                    tuning_enabled=bool(request.tuning_enabled),
                    tuning_search_space_id=request.tuning_search_space_id,
                    tuning_search_space_version=request.tuning_search_space_version,
                    tuning_inner_cv_scheme=request.tuning_inner_cv_scheme,
                    tuning_inner_group_field=request.tuning_inner_group_field,
                    tuning_assignment=stage_assignment_map.get(StageKey.TUNING),
                    tuning_fallback_executor_id=stage_fallback_executor_ids.get(StageKey.TUNING),
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
            if isinstance(permutation_payload, dict) and isinstance(
                permutation_payload.get("permutation_loop_seconds"), (int, float)
            ):
                permutation_loop_seconds_raw = permutation_payload.get("permutation_loop_seconds")
                _set_stage_timing(
                    stage_name=StageKey.PERMUTATION.value,
                    duration_seconds=(
                        float(permutation_loop_seconds_raw)
                        if isinstance(permutation_loop_seconds_raw, (int, float))
                        else None
                    ),
                    duration_source="metrics.permutation_test.permutation_loop_seconds",
                    derived_from="metrics.permutation_test",
                )
            elif int(request.n_permutations) > 0:
                _set_stage_timing(
                    stage_name=StageKey.PERMUTATION.value,
                    duration_seconds=None,
                    duration_source="unavailable",
                    derived_from="metrics.permutation_test",
                    fallback_reason="permutation_loop_timing_not_measured",
                )
            else:
                _set_stage_timing(
                    stage_name=StageKey.PERMUTATION.value,
                    duration_seconds=None,
                    duration_source="not_applicable",
                    derived_from="permutation_policy",
                    fallback_reason="permutations_disabled",
                )
            if stage_observer is not None:
                permutation_assignment_effective = stage_assignment_map.get(StageKey.PERMUTATION)
                permutation_execution_mode = (
                    str(permutation_payload.get("execution_mode"))
                    if isinstance(permutation_payload, dict)
                    and isinstance(permutation_payload.get("execution_mode"), str)
                    and str(permutation_payload.get("execution_mode")).strip()
                    else None
                )
                permutation_backend_family = (
                    str(permutation_payload.get("backend_family"))
                    if isinstance(permutation_payload, dict)
                    and isinstance(permutation_payload.get("backend_family"), str)
                    and str(permutation_payload.get("backend_family")).strip()
                    else None
                )
                permutation_executor_id = (
                    str(permutation_payload.get("permutation_executor_id"))
                    if isinstance(permutation_payload, dict)
                    and isinstance(permutation_payload.get("permutation_executor_id"), str)
                    and str(permutation_payload.get("permutation_executor_id")).strip()
                    else None
                )
                permutation_fallback_reason = (
                    str(permutation_payload.get("permutation_executor_fallback_reason"))
                    if isinstance(permutation_payload, dict)
                    and isinstance(
                        permutation_payload.get("permutation_executor_fallback_reason"),
                        str,
                    )
                    and str(permutation_payload.get("permutation_executor_fallback_reason")).strip()
                    else None
                )
                if int(request.n_permutations) > 0:
                    stage_observer.stage_finished(
                        StageKey.PERMUTATION,
                        metadata={
                            **_stage_observed_metadata(
                                assignment=permutation_assignment_effective,
                                fallback_reason=permutation_fallback_reason,
                                fallback_used=(
                                    bool(permutation_assignment_effective.fallback_used)
                                    if permutation_assignment_effective is not None
                                    else None
                                ),
                                execution_mode=permutation_execution_mode,
                                observed_backend_family=permutation_backend_family,
                                observed_executor_id=permutation_executor_id,
                                stage_lease_metadata=section_stage_lease_metadata,
                            ),
                            "derived_from_stage": StageKey.EVALUATION.value,
                            "status_source": "derived_from_evaluation",
                        },
                        status="executed",
                    )
                else:
                    stage_observer.stage_skipped(
                        StageKey.PERMUTATION,
                        metadata={
                            **_stage_observed_metadata(
                                assignment=permutation_assignment_effective,
                                fallback_used=False,
                                stage_lease_metadata=section_stage_lease_metadata,
                            ),
                            "derived_from_stage": StageKey.EVALUATION.value,
                            "status_source": "derived_from_evaluation",
                        },
                        reason="permutations_disabled",
                        derived_from_stage=StageKey.EVALUATION,
                    )
            artifact_ids[ARTIFACT_TYPE_METRICS_BUNDLE] = evaluation_output.metrics_artifact_id
        else:
            raise ValueError(f"Unsupported section encountered in execution plan: {section.value}")

        _release_section_stage_lease(
            stage_lease_handle=stage_lease_handle,
            stage_lease_manager=stage_lease_manager,
            section_stage_lease_metadata=section_stage_lease_metadata,
            stage_observer=stage_observer,
            stage_key=(stage_key if isinstance(stage_key, StageKey) else None),
        )

        executed_sections.append(section.value)
        _record_section_duration(
            section_name=section.value,
            duration_seconds=float(perf_counter() - section_start),
        )
        _emit_section_event(
            section_name=section.value,
            status="finished",
            completed_units=float(section_number),
            total_units=float(total_sections),
            message=f"finished section {section.value}",
        )
        if stage_observer is not None and isinstance(stage_key, StageKey):
            observed_backend_family: str | None = None
            observed_executor_id: str | None = None
            if section == SectionName.MODEL_FIT and isinstance(compute_runtime_metadata, dict):
                backend_family_candidate = compute_runtime_metadata.get(
                    "actual_estimator_backend_family"
                ) or compute_runtime_metadata.get("backend_family")
                if isinstance(backend_family_candidate, str) and backend_family_candidate.strip():
                    observed_backend_family = backend_family_candidate.strip()
                backend_id_candidate = compute_runtime_metadata.get("actual_estimator_backend_id")
                if isinstance(backend_id_candidate, str) and backend_id_candidate.strip():
                    observed_executor_id = backend_id_candidate.strip()
            stage_assignment = stage_assignment_map.get(stage_key)
            stage_observer.stage_finished(
                stage_key,
                metadata={
                    **_stage_observed_metadata(
                        assignment=stage_assignment,
                        observed_backend_family=observed_backend_family,
                        observed_executor_id=observed_executor_id,
                        stage_lease_metadata=section_stage_lease_metadata,
                    ),
                    "primary_artifacts": _primary_artifacts_for_section(section),
                },
                status="executed",
            )

    return SegmentExecutionResult(
        planned_sections=[section.value for section in planned_sections],
        executed_sections=executed_sections,
        reused_sections=reused_sections,
        artifact_ids=artifact_ids,
        metrics=metrics,
        spatial_compatibility=spatial_compatibility,
        interpretability_summary=interpretability_summary,
        model_persistence=model_persistence_summary,
        compute_runtime_metadata=compute_runtime_metadata,
        section_timings_seconds=section_timings_seconds,
        stage_timings_seconds=stage_timings_seconds,
        stage_timing_metadata=stage_timing_metadata,
        stage_assignments=[
            stage_assignment_map[stage] for stage in StageKey if stage in stage_assignment_map
        ],
    )
