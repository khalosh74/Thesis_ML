"""Leakage-safe experiment runner with grouped cross-validation."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import tracemalloc
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import sklearn

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_EXPERIMENT_REPORT,
    ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    compute_config_hash,
    register_artifact,
)
from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.methodology import (
    EvidencePolicy,
    MethodologyPolicy,
    MethodologyPolicyName,
    SubgroupReportingPolicy,
)
from Thesis_ML.config.paths import DEFAULT_EXPERIMENT_REPORTS_ROOT
from Thesis_ML.experiments.cache_loading import load_features_from_cache
from Thesis_ML.experiments.compute_policy import (
    HARDWARE_MODE_CHOICES,
    resolve_compute_policy,
)
from Thesis_ML.experiments.compute_scheduler import (
    ComputeRunAssignment,
    ComputeRunRequest,
    materialize_scheduled_compute_policy,
    plan_compute_schedule,
)
from Thesis_ML.experiments.data_reporting import write_official_data_artifacts
from Thesis_ML.experiments.errors import exception_failure_payload
from Thesis_ML.experiments.execution_policy import (
    prepare_report_dir,
    write_run_status,
)
from Thesis_ML.experiments.feature_space_loading import (
    FEATURE_SPACE_ROI_MEAN_PREDEFINED,
    FEATURE_SPACE_WHOLE_BRAIN_MASKED,
    SUPPORTED_FEATURE_SPACES,
    normalize_feature_space,
)
from Thesis_ML.experiments.metrics import (
    compute_interpretability_stability,
    evaluate_permutations,
    extract_linear_coefficients,
    scores_for_predictions,
)
from Thesis_ML.experiments.model_catalog import (
    get_model_cost_entry,
)
from Thesis_ML.experiments.model_catalog import (
    projected_runtime_seconds as resolve_projected_runtime_seconds,
)
from Thesis_ML.experiments.model_factory import (
    DEFAULT_BATCH_MODEL_NAMES,
    MODEL_NAMES,
    SUPPORTED_FEATURE_RECIPE_IDS,
    make_model,
    model_preprocess_kind,
    resolve_preprocessing_recipe,
)
from Thesis_ML.experiments.model_factory import (
    build_pipeline as build_model_pipeline,
)
from Thesis_ML.experiments.official_contracts import (
    validate_official_preflight,
    validate_run_artifact_contract,
)
from Thesis_ML.experiments.progress import ProgressCallback, emit_progress
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
from Thesis_ML.experiments.run_states import RUN_STATUS_SUCCESS
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
from Thesis_ML.experiments.stage_execution import build_stage_execution_result
from Thesis_ML.experiments.stage_planner import (
    StagePlanningResult,
    plan_stage_execution,
)
from Thesis_ML.experiments.tuning_search_spaces import (
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
)
from Thesis_ML.features.dimensionality import (
    SUPPORTED_DIMENSIONALITY_STRATEGIES,
    ResolvedDimensionalityConfig,
    apply_dimensionality_to_pipeline,
    resolve_dimensionality_config,
)
from Thesis_ML.features.preprocessing import (
    BASELINE_STANDARD_SCALER_RECIPE_ID,
    SUPPORTED_PREPROCESSING_STRATEGIES,
    apply_preprocessing_to_pipeline,
    resolve_preprocessing_strategy,
)

LOGGER = logging.getLogger(__name__)

_CV_MODES = (
    "loso_session",
    "within_subject_loso_session",
    "frozen_cross_person_transfer",
    "record_random_split",
)
_TARGET_ALIASES = {
    "emotion": "emotion",
    "coarse_affect": "coarse_affect",
    "binary_valence_like": "binary_valence_like",
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
    return exception_failure_payload(exc, default_stage="runtime")


def _resolve_backend_family_from_backend_id(
    backend_id: str | None,
    *,
    fallback_family: str,
) -> str:
    if not isinstance(backend_id, str):
        return str(fallback_family)
    normalized = backend_id.strip().lower()
    if not normalized:
        return str(fallback_family)
    if "xgboost" in normalized:
        return "xgboost_gpu" if "gpu" in normalized else "xgboost_cpu"
    if normalized == "cpu_reference":
        return "sklearn_cpu"
    if "torch" in normalized or "gpu" in normalized:
        return "torch_gpu"
    return "sklearn_cpu"


def _make_model(
    name: str,
    seed: int,
    class_weight_policy: str = "none",
    compute_policy=None,
) -> Any:
    return make_model(
        name=name,
        seed=seed,
        class_weight_policy=class_weight_policy,
        compute_policy=compute_policy,
    )


def _build_pipeline(
    model_name: str,
    seed: int,
    class_weight_policy: str = "none",
    compute_policy=None,
    feature_recipe_id: str = BASELINE_STANDARD_SCALER_RECIPE_ID,
    preprocessing_strategy: str | None = None,
    dimensionality_config: ResolvedDimensionalityConfig | None = None,
):
    pipeline = build_model_pipeline(
        model_name=model_name,
        seed=seed,
        class_weight_policy=class_weight_policy,
        compute_policy=compute_policy,
        feature_recipe_id=feature_recipe_id,
    )
    pipeline = apply_preprocessing_to_pipeline(
        pipeline=pipeline,
        preprocessing_strategy=preprocessing_strategy,
    )
    if dimensionality_config is None:
        return pipeline
    return apply_dimensionality_to_pipeline(
        pipeline=pipeline,
        config=dimensionality_config,
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
    primary_metric_aggregation: str = "pooled_held_out_predictions",
    progress_callback=None,
    progress_metadata: dict[str, Any] | None = None,
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
        primary_metric_aggregation=primary_metric_aggregation,
        progress_callback=progress_callback,
        progress_metadata=progress_metadata,
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
    evidence_run_role: str | None,
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
        evidence_run_role=evidence_run_role,
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
    feature_space: str = FEATURE_SPACE_WHOLE_BRAIN_MASKED,
    roi_spec_path: Path | str | None = None,
    preprocessing_strategy: str | None = None,
    dimensionality_strategy: str = "none",
    pca_n_components: int | None = None,
    pca_variance_ratio: float | None = None,
    n_permutations: int = 0,
    primary_metric_name: str = "balanced_accuracy",
    primary_metric_aggregation: str = "mean_fold_scores",
    permutation_metric_name: str | None = None,
    repeat_id: int = 1,
    repeat_count: int = 1,
    base_run_id: str | None = None,
    evidence_run_role: str = "primary",
    evidence_policy: dict[str, Any] | None = None,
    data_policy: dict[str, Any] | None = None,
    methodology_policy_name: str = MethodologyPolicyName.FIXED_BASELINES_ONLY.value,
    class_weight_policy: str = "none",
    tuning_enabled: bool = False,
    tuning_search_space_id: str | None = None,
    tuning_search_space_version: str | None = None,
    tuning_inner_cv_scheme: str | None = None,
    tuning_inner_group_field: str | None = None,
    feature_recipe_id: str = BASELINE_STANDARD_SCALER_RECIPE_ID,
    emit_feature_qc_artifacts: bool = True,
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
    model_cost_tier: str | None = None,
    projected_runtime_seconds: int | None = None,
    timeout_policy_effective: dict[str, Any] | None = None,
    profiling_context: dict[str, Any] | None = None,
    hardware_mode: str = "cpu_only",
    gpu_device_id: int | None = None,
    deterministic_compute: bool = False,
    allow_backend_fallback: bool = False,
    progress_callback: ProgressCallback | None = None,
    max_parallel_runs: int = 1,
    max_parallel_gpu_runs: int = 1,
    scheduled_compute_assignment: dict[str, Any] | None = None,
    load_features_from_cache_fn_override: (
        Callable[..., tuple[np.ndarray, pd.DataFrame, dict[str, Any]]] | None
    ) = None,
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
    resolved_compute_policy = None
    resolved_profiling_context: dict[str, Any] | None = None
    profiling_max_outer_folds: int | None = None
    profiling_inner_fold_cap: int | None = None
    profiling_tuning_candidate_cap: int | None = None
    dataset_fingerprint: dict[str, Any] | None = None
    data_policy_effective: dict[str, Any] = {}
    data_assessment: dict[str, Any] = {}
    data_artifact_info: dict[str, Any] = {}
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

    if profiling_context is not None:
        if not isinstance(profiling_context, dict):
            raise ValueError("profiling_context must be an object when provided.")
        profile_source = str(profiling_context.get("source", "")).strip()
        if profile_source != "campaign_runtime_profile_precheck":
            raise ValueError(
                "profiling_context.source must be 'campaign_runtime_profile_precheck'."
            )
        if not bool(profiling_context.get("profiling_only", False)):
            raise ValueError("profiling_context.profiling_only must be true.")
        if not bool(profiling_context.get("precheck_only", False)):
            raise ValueError("profiling_context.precheck_only must be true.")
        profiling_max_outer_folds = int(profiling_context.get("max_outer_folds", 1))
        if profiling_max_outer_folds <= 0:
            raise ValueError("profiling_context.max_outer_folds must be > 0.")
        profile_inner_folds_raw = profiling_context.get("profile_inner_folds")
        if profile_inner_folds_raw is not None:
            profiling_inner_fold_cap = int(profile_inner_folds_raw)
            if profiling_inner_fold_cap <= 0:
                raise ValueError("profiling_context.profile_inner_folds must be > 0.")
        profile_tuning_candidates_raw = profiling_context.get("profile_tuning_candidates")
        if profile_tuning_candidates_raw is not None:
            profiling_tuning_candidate_cap = int(profile_tuning_candidates_raw)
            if profiling_tuning_candidate_cap <= 0:
                raise ValueError("profiling_context.profile_tuning_candidates must be > 0.")
        resolved_profiling_context = dict(profiling_context)
        resolved_profiling_context["source"] = profile_source
        resolved_profiling_context["profiling_only"] = True
        resolved_profiling_context["precheck_only"] = True
        resolved_profiling_context["max_outer_folds"] = int(profiling_max_outer_folds)
        resolved_profiling_context["profile_inner_folds"] = (
            int(profiling_inner_fold_cap) if profiling_inner_fold_cap is not None else None
        )
        resolved_profiling_context["profile_tuning_candidates"] = (
            int(profiling_tuning_candidate_cap)
            if profiling_tuning_candidate_cap is not None
            else None
        )

    target_column = _resolve_target_column(target)
    resolved_feature_space = normalize_feature_space(feature_space)
    resolved_roi_spec_path = (
        Path(roi_spec_path).resolve()
        if roi_spec_path is not None and str(roi_spec_path).strip()
        else None
    )
    if (
        resolved_feature_space == FEATURE_SPACE_ROI_MEAN_PREDEFINED
        and resolved_roi_spec_path is None
    ):
        raise ValueError("feature_space='roi_mean_predefined' requires a non-empty roi_spec_path.")
    resolved_dimensionality_config = resolve_dimensionality_config(
        dimensionality_strategy=dimensionality_strategy,
        pca_n_components=pca_n_components,
        pca_variance_ratio=pca_variance_ratio,
    )
    resolved_preprocessing_config = resolve_preprocessing_strategy(preprocessing_strategy)
    resolved_preprocessing_kind = model_preprocess_kind(model)
    resolved_feature_recipe_id = resolve_preprocessing_recipe(
        recipe_id=feature_recipe_id,
        model_name=model,
    )
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
    context_feature_recipe_id = official_context.get("feature_recipe_id")
    if context_feature_recipe_id is not None:
        context_recipe_id = resolve_preprocessing_recipe(
            recipe_id=str(context_feature_recipe_id),
            model_name=model,
        )
        if context_recipe_id != resolved_feature_recipe_id:
            raise ValueError(
                "Illegal override for official run key 'feature_recipe_id'. "
                "Use protocol/comparison spec values only."
            )
        resolved_feature_recipe_id = context_recipe_id
    context_preprocessing_strategy = official_context.get("preprocessing_strategy")
    if context_preprocessing_strategy is not None:
        context_preprocessing_config = resolve_preprocessing_strategy(
            str(context_preprocessing_strategy),
        )
        if context_preprocessing_config.strategy != resolved_preprocessing_config.strategy:
            raise ValueError(
                "Illegal override for official run key 'preprocessing_strategy'. "
                "Use protocol/comparison spec values only."
            )
        resolved_preprocessing_config = context_preprocessing_config
    context_emit_feature_qc_artifacts = official_context.get("emit_feature_qc_artifacts")
    if context_emit_feature_qc_artifacts is not None and bool(
        context_emit_feature_qc_artifacts
    ) != bool(emit_feature_qc_artifacts):
        raise ValueError(
            "Illegal override for official run key 'emit_feature_qc_artifacts'. "
            "Use protocol/comparison spec values only."
        )
    resolved_emit_feature_qc_artifacts = bool(
        context_emit_feature_qc_artifacts
        if context_emit_feature_qc_artifacts is not None
        else emit_feature_qc_artifacts
    )
    context_primary_metric_aggregation = official_context.get("primary_metric_aggregation")
    if (
        context_primary_metric_aggregation is not None
        and str(context_primary_metric_aggregation).strip()
        != str(primary_metric_aggregation).strip()
    ):
        raise ValueError(
            "Illegal override for official run key 'primary_metric_aggregation'. "
            "Use protocol/comparison spec values only."
        )
    resolved_primary_metric_aggregation = str(
        context_primary_metric_aggregation
        if context_primary_metric_aggregation is not None
        else primary_metric_aggregation
    ).strip()
    if resolved_primary_metric_aggregation not in {
        "mean_fold_scores",
        "pooled_held_out_predictions",
    }:
        raise ValueError(
            "primary_metric_aggregation must be 'mean_fold_scores' or "
            "'pooled_held_out_predictions'."
        )
    confirmatory_lock_candidate = official_context.get("confirmatory_lock")
    confirmatory_lock_payload = (
        dict(confirmatory_lock_candidate) if isinstance(confirmatory_lock_candidate, dict) else {}
    )
    confirmatory_subgroup_min_classes = int(
        confirmatory_lock_payload.get("subgroup_min_classes_per_group", 1)
    )
    confirmatory_subgroup_report_small_groups = bool(
        confirmatory_lock_payload.get("subgroup_report_small_groups", False)
    )
    confirmatory_guardrails_enabled = bool(
        resolved_framework_mode == FrameworkMode.CONFIRMATORY
        and str(confirmatory_lock_payload.get("protocol_id", "")).strip()
        == "thesis_confirmatory_v1"
    )
    subgroup_evidence_role = (
        str(confirmatory_lock_payload.get("subgroup_interpretation", "descriptive_only"))
        if confirmatory_guardrails_enabled
        else "exploratory"
    )
    subgroup_primary_evidence_allowed = bool(not confirmatory_guardrails_enabled)
    context_repeat_id = official_context.get("repeat_id")
    context_repeat_count = official_context.get("repeat_count")
    context_base_run_id = official_context.get("base_run_id")
    context_evidence_run_role = official_context.get("evidence_run_role")
    if context_repeat_id is not None and int(context_repeat_id) != int(repeat_id):
        raise ValueError(
            "Illegal override for official run key 'repeat_id'. "
            "Use protocol/comparison spec values only."
        )
    if context_repeat_count is not None and int(context_repeat_count) != int(repeat_count):
        raise ValueError(
            "Illegal override for official run key 'repeat_count'. "
            "Use protocol/comparison spec values only."
        )
    if context_base_run_id is not None and str(context_base_run_id) != str(base_run_id):
        raise ValueError(
            "Illegal override for official run key 'base_run_id'. "
            "Use protocol/comparison spec values only."
        )
    if context_evidence_run_role is not None and str(context_evidence_run_role) != str(
        evidence_run_role
    ):
        raise ValueError(
            "Illegal override for official run key 'evidence_run_role'. "
            "Use protocol/comparison spec values only."
        )
    resolved_repeat_id = int(context_repeat_id if context_repeat_id is not None else repeat_id)
    resolved_repeat_count = int(
        context_repeat_count if context_repeat_count is not None else repeat_count
    )
    resolved_base_run_id = (
        str(context_base_run_id)
        if context_base_run_id is not None
        else (str(base_run_id) if base_run_id is not None else None)
    )
    resolved_evidence_run_role = str(
        context_evidence_run_role if context_evidence_run_role is not None else evidence_run_role
    )
    official_evidence_policy_candidate = official_context.get("evidence_policy")
    resolved_evidence_policy_payload = (
        dict(official_evidence_policy_candidate)
        if isinstance(official_evidence_policy_candidate, dict)
        else dict(evidence_policy)
        if isinstance(evidence_policy, dict)
        else EvidencePolicy().model_dump(mode="json")
    )
    official_data_policy_candidate = official_context.get("data_policy")
    resolved_data_policy_payload = (
        dict(official_data_policy_candidate)
        if isinstance(official_data_policy_candidate, dict)
        else dict(data_policy)
        if isinstance(data_policy, dict)
        else {}
    )
    context_model_cost_tier = official_context.get("model_cost_tier")
    context_projected_runtime_seconds = official_context.get("projected_runtime_seconds")
    if (
        resolved_framework_mode in {FrameworkMode.CONFIRMATORY, FrameworkMode.LOCKED_COMPARISON}
        and context_model_cost_tier is None
        and model_cost_tier is not None
    ):
        raise ValueError(
            "Illegal override for official run key 'model_cost_tier'. "
            "Use protocol/comparison spec values only."
        )
    if (
        resolved_framework_mode in {FrameworkMode.CONFIRMATORY, FrameworkMode.LOCKED_COMPARISON}
        and context_projected_runtime_seconds is None
        and projected_runtime_seconds is not None
    ):
        raise ValueError(
            "Illegal override for official run key 'projected_runtime_seconds'. "
            "Use protocol/comparison spec values only."
        )
    if (
        context_model_cost_tier is not None
        and model_cost_tier is not None
        and str(context_model_cost_tier).strip().lower() != str(model_cost_tier).strip().lower()
    ):
        raise ValueError(
            "Illegal override for official run key 'model_cost_tier'. "
            "Use protocol/comparison spec values only."
        )
    if (
        context_projected_runtime_seconds is not None
        and projected_runtime_seconds is not None
        and int(context_projected_runtime_seconds) != int(projected_runtime_seconds)
    ):
        raise ValueError(
            "Illegal override for official run key 'projected_runtime_seconds'. "
            "Use protocol/comparison spec values only."
        )
    if (
        resolved_framework_mode in {FrameworkMode.CONFIRMATORY, FrameworkMode.LOCKED_COMPARISON}
        and "data_policy" not in official_context
    ):
        official_context["data_policy"] = dict(resolved_data_policy_payload)
    evidence_policy_model = EvidencePolicy.model_validate(resolved_evidence_policy_payload)
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
        evidence_run_role=resolved_evidence_run_role,
        protocol_context=resolved_protocol_context,
        comparison_context=resolved_comparison_context,
    )
    stage_timings["methodology_resolution"] = float(perf_counter() - methodology_start)
    methodology_policy_payload = methodology_policy.model_dump(mode="json")
    feature_quality_raw = methodology_policy_payload.get("feature_quality")
    feature_quality_policy_payload = (
        dict(feature_quality_raw) if isinstance(feature_quality_raw, dict) else None
    )

    compute_policy_start = perf_counter()
    resolved_compute_policy = resolve_compute_policy(
        framework_mode=resolved_framework_mode,
        hardware_mode=hardware_mode,
        gpu_device_id=gpu_device_id,
        deterministic_compute=deterministic_compute,
        allow_backend_fallback=allow_backend_fallback,
    )
    stage_timings["compute_policy_resolution"] = float(perf_counter() - compute_policy_start)

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

    catalog_cost_entry = get_model_cost_entry(model)
    catalog_tier = catalog_cost_entry.cost_tier.value
    resolved_model_cost_tier = (
        str(context_model_cost_tier).strip()
        if context_model_cost_tier is not None
        else str(model_cost_tier).strip()
        if model_cost_tier is not None
        else str(catalog_tier)
    )
    if resolved_model_cost_tier != str(catalog_tier):
        raise ValueError(
            "Model cost tier mismatch for run model. "
            f"model='{model}' expected_tier='{catalog_tier}' "
            f"but received '{resolved_model_cost_tier}'."
        )

    computed_projected_runtime_seconds = int(
        resolve_projected_runtime_seconds(
            model_name=model,
            framework_mode=resolved_framework_mode,
            methodology_policy=methodology_policy.policy_name,
            tuning_enabled=bool(methodology_policy.tuning_enabled),
        )
    )
    resolved_projected_runtime_seconds = (
        int(context_projected_runtime_seconds)
        if context_projected_runtime_seconds is not None
        else int(projected_runtime_seconds)
        if projected_runtime_seconds is not None
        else int(computed_projected_runtime_seconds)
    )
    if int(resolved_projected_runtime_seconds) <= 0:
        raise ValueError("projected_runtime_seconds must be > 0.")
    if resolved_framework_mode in {
        FrameworkMode.CONFIRMATORY,
        FrameworkMode.LOCKED_COMPARISON,
    } and int(resolved_projected_runtime_seconds) != int(computed_projected_runtime_seconds):
        raise ValueError(
            "Illegal override for official run key 'projected_runtime_seconds'. "
            "Use protocol/comparison spec values only."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_run_id = run_id or f"{timestamp}_{model}_{target_column}"
    if resolved_base_run_id is None:
        resolved_base_run_id = str(resolved_run_id)
    report_dir = reports_root / resolved_run_id
    should_reuse_completed_artifacts = bool(resume or reuse_completed_artifacts)
    artifact_registry_path = reports_root / "artifact_registry.sqlite3"
    git_provenance = collect_git_provenance()
    code_ref = str(git_provenance.get("git_commit") or "") or None

    if (
        resolved_framework_mode == FrameworkMode.EXPLORATORY
        or scheduled_compute_assignment is not None
    ):
        compute_schedule_start = perf_counter()
        if scheduled_compute_assignment is not None:
            scheduler_assignment = ComputeRunAssignment.from_payload(
                scheduled_compute_assignment,
                default_order_index=0,
                default_run_id=str(resolved_run_id),
                default_model_name=str(model),
            )
            if str(scheduler_assignment.run_id) != str(resolved_run_id):
                raise ValueError(
                    "scheduled_compute_assignment.run_id does not match run_experiment run_id."
                )
            if str(scheduler_assignment.model_name).strip().lower() != str(model).strip().lower():
                raise ValueError(
                    "scheduled_compute_assignment.model_name does not match run_experiment model."
                )
        else:
            [scheduler_assignment] = plan_compute_schedule(
                run_requests=[
                    ComputeRunRequest(
                        order_index=0,
                        run_id=str(resolved_run_id),
                        model_name=str(model),
                    )
                ],
                base_compute_policy=resolved_compute_policy,
                max_parallel_runs=int(max_parallel_runs),
                max_parallel_gpu_runs=int(max_parallel_gpu_runs),
            )
        resolved_compute_policy = materialize_scheduled_compute_policy(
            base_compute_policy=resolved_compute_policy,
            assignment=scheduler_assignment,
        )
        stage_timings["compute_schedule_resolution"] = float(
            perf_counter() - compute_schedule_start
        )

    emit_progress(
        progress_callback,
        stage="run",
        message="starting run execution",
        metadata={
            "run_id": str(resolved_run_id),
            "model": str(model),
            "target": str(target_column),
            "cv_mode": str(cv_mode),
            "framework_mode": str(resolved_framework_mode.value),
            "hardware_mode_requested": str(resolved_compute_policy.hardware_mode_requested),
            "hardware_mode_effective": str(resolved_compute_policy.hardware_mode_effective),
            "requested_backend_family": str(resolved_compute_policy.requested_backend_family),
            "effective_backend_family": str(resolved_compute_policy.effective_backend_family),
            "assigned_compute_lane": (
                str(resolved_compute_policy.assigned_compute_lane)
                if resolved_compute_policy.assigned_compute_lane is not None
                else None
            ),
            "assigned_backend_family": (
                str(resolved_compute_policy.assigned_backend_family)
                if resolved_compute_policy.assigned_backend_family is not None
                else None
            ),
            "lane_assignment_reason": resolved_compute_policy.lane_assignment_reason,
            "gpu_device_id": resolved_compute_policy.gpu_device_id,
            "gpu_device_name": resolved_compute_policy.gpu_device_name,
        },
    )

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
            feature_space=resolved_feature_space,
            roi_spec_path=resolved_roi_spec_path,
            dimensionality_strategy=resolved_dimensionality_config.strategy,
            pca_n_components=resolved_dimensionality_config.pca_n_components,
            pca_variance_ratio=resolved_dimensionality_config.pca_variance_ratio,
            n_permutations=n_permutations,
            primary_metric_name=resolved_primary_metric_name,
            permutation_metric_name=resolved_permutation_metric_name,
            methodology_policy_name=methodology_policy.policy_name.value,
            class_weight_policy=methodology_policy.class_weight_policy.value,
            model=model,
            tuning_enabled=bool(methodology_policy.tuning_enabled),
            tuning_search_space_id=effective_tuning_space_id,
            tuning_search_space_version=effective_tuning_space_version,
            tuning_inner_group_field=effective_tuning_inner_group_field,
            subgroup_reporting_enabled=bool(subgroup_policy.enabled),
            subgroup_dimensions=list(subgroup_policy.subgroup_dimensions),
            subgroup_min_samples_per_group=int(subgroup_policy.min_samples_per_group),
            subgroup_min_classes_per_group=int(confirmatory_subgroup_min_classes),
            subgroup_report_small_groups=bool(confirmatory_subgroup_report_small_groups),
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
            selected_beta_path_sha256=preflight.data_assessment.get("selected_beta_path_sha256"),
            cv_split_manifest_sha256=preflight.data_assessment.get("cv_split_manifest_sha256"),
        )
        required_run_artifacts = list(preflight.required_run_artifacts)
        required_run_metadata_fields = list(preflight.required_run_metadata_fields)
        data_policy_effective = dict(preflight.data_policy_effective)
        data_assessment = dict(preflight.data_assessment)
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
    dataset_card_json_path = report_dir / "dataset_card.json"
    dataset_card_md_path = report_dir / "dataset_card.md"
    dataset_summary_json_path = report_dir / "dataset_summary.json"
    dataset_summary_csv_path = report_dir / "dataset_summary.csv"
    data_quality_report_path = report_dir / "data_quality_report.json"
    class_balance_report_path = report_dir / "class_balance_report.csv"
    missingness_report_path = report_dir / "missingness_report.csv"
    leakage_audit_path = report_dir / "leakage_audit.json"
    external_dataset_card_path = report_dir / "external_dataset_card.json"
    external_dataset_summary_path = report_dir / "external_dataset_summary.json"
    external_validation_compatibility_path = report_dir / "external_validation_compatibility.json"
    metrics_path = report_dir / "metrics.json"
    subgroup_metrics_json_path = report_dir / "subgroup_metrics.json"
    subgroup_metrics_csv_path = report_dir / "subgroup_metrics.csv"
    config_path = report_dir / "config.json"
    tuning_summary_path = report_dir / "tuning_summary.json"
    tuning_best_params_path = report_dir / "best_params_per_fold.csv"
    fit_timing_summary_path = report_dir / "fit_timing_summary.json"
    spatial_compatibility_report_path = report_dir / "spatial_compatibility_report.json"
    calibration_summary_path = report_dir / "calibration_summary.json"
    calibration_table_path = report_dir / "calibration_table.csv"
    feature_qc_summary_path = report_dir / "feature_qc_summary.json"
    feature_qc_selected_samples_path = report_dir / "feature_qc_selected_samples.csv"
    interpretability_summary_path = report_dir / "interpretability_summary.json"
    interpretability_fold_artifacts_path = report_dir / "interpretability_fold_explanations.csv"

    write_run_status(
        report_dir,
        run_id=resolved_run_id,
        status="running",
        message=f"run_mode={run_mode}",
        stage_timings_seconds=stage_timings,
    )
    if resolved_framework_mode in {
        FrameworkMode.CONFIRMATORY,
        FrameworkMode.LOCKED_COMPARISON,
    }:
        data_artifacts_start = perf_counter()
        try:
            data_artifact_info = write_official_data_artifacts(
                report_dir=report_dir,
                assessment=data_assessment,
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
                sample_unit=(
                    str(official_context.get("sample_unit"))
                    if official_context.get("sample_unit") is not None
                    else None
                ),
                label_policy=(
                    str(official_context.get("label_policy"))
                    if official_context.get("label_policy") is not None
                    else None
                ),
                target_mapping_version=(
                    str(confirmatory_lock_payload.get("target_mapping_version"))
                    if confirmatory_lock_payload
                    else None
                ),
                target_mapping_hash=(
                    str(confirmatory_lock_payload.get("target_mapping_hash"))
                    if confirmatory_lock_payload
                    else None
                ),
                dataset_fingerprint=dataset_fingerprint,
            )
            if not data_policy_effective:
                data_policy_effective = dict(data_artifact_info.get("data_policy_effective", {}))
            data_fingerprint_payload = data_artifact_info.get("dataset_fingerprint")
            if isinstance(data_fingerprint_payload, dict):
                dataset_fingerprint = dict(data_fingerprint_payload)
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
        stage_timings["data_artifact_generation"] = float(perf_counter() - data_artifacts_start)

    stage_planning_result: StagePlanningResult | None = None
    try:
        stage_planning_start = perf_counter()
        stage_planning_result = plan_stage_execution(
            framework_mode=resolved_framework_mode,
            compute_policy=resolved_compute_policy,
            model_name=model,
            methodology_policy_name=methodology_policy.policy_name.value,
            tuning_enabled=bool(methodology_policy.tuning_enabled),
            n_permutations=int(n_permutations),
        )
        stage_timings["stage_planning"] = float(perf_counter() - stage_planning_start)
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
                        feature_space=resolved_feature_space,
                        roi_spec_path=resolved_roi_spec_path,
                        preprocessing_strategy=resolved_preprocessing_config.strategy,
                        dimensionality_strategy=resolved_dimensionality_config.strategy,
                        pca_n_components=resolved_dimensionality_config.pca_n_components,
                        pca_variance_ratio=resolved_dimensionality_config.pca_variance_ratio,
                        seed=seed,
                        n_permutations=n_permutations,
                        primary_metric_name=resolved_primary_metric_name,
                        primary_metric_aggregation=resolved_primary_metric_aggregation,
                        permutation_metric_name=resolved_permutation_metric_name,
                        permutation_alpha=float(evidence_policy_model.permutation.alpha),
                        permutation_minimum_required=int(
                            evidence_policy_model.permutation.minimum_permutations
                        ),
                        permutation_require_pass_for_validity=bool(
                            evidence_policy_model.permutation.require_pass_for_validity
                        ),
                        repeat_id=int(resolved_repeat_id),
                        repeat_count=int(resolved_repeat_count),
                        base_run_id=str(resolved_base_run_id),
                        evidence_run_role=str(resolved_evidence_run_role),
                        evidence_policy_effective=evidence_policy_model.model_dump(mode="json"),
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
                        subgroup_min_classes_per_group=int(confirmatory_subgroup_min_classes),
                        subgroup_report_small_groups=bool(
                            confirmatory_subgroup_report_small_groups
                        ),
                        confirmatory_guardrails_enabled=bool(confirmatory_guardrails_enabled),
                        subgroup_evidence_role=str(subgroup_evidence_role),
                        subgroup_primary_evidence_allowed=bool(subgroup_primary_evidence_allowed),
                        calibration_enabled=bool(evidence_policy_model.calibration.enabled),
                        calibration_n_bins=int(evidence_policy_model.calibration.n_bins),
                        calibration_require_probabilities_for_validity=bool(
                            evidence_policy_model.calibration.require_probabilities_for_validity
                        ),
                        interpretability_enabled_override=interpretability_enabled_override,
                        max_outer_folds=profiling_max_outer_folds,
                        profiling_only=bool(resolved_profiling_context is not None),
                        profile_inner_folds=profiling_inner_fold_cap,
                        profile_tuning_candidates=profiling_tuning_candidate_cap,
                        compute_policy=resolved_compute_policy,
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
                        fit_timing_summary_path=fit_timing_summary_path,
                        spatial_report_path=spatial_compatibility_report_path,
                        calibration_summary_path=calibration_summary_path,
                        calibration_table_path=calibration_table_path,
                        feature_qc_summary_path=feature_qc_summary_path,
                        feature_qc_selected_samples_path=feature_qc_selected_samples_path,
                        feature_quality_policy=feature_quality_policy_payload,
                        emit_feature_qc_artifacts=resolved_emit_feature_qc_artifacts,
                        interpretability_summary_path=interpretability_summary_path,
                        interpretability_fold_artifacts_path=interpretability_fold_artifacts_path,
                        start_section=start_section,
                        end_section=end_section,
                        base_artifact_id=base_artifact_id,
                        reuse_policy=reuse_policy,
                        reuse_completed_artifacts=should_reuse_completed_artifacts,
                        feature_recipe_id=resolved_feature_recipe_id,
                        build_pipeline_fn=lambda model_name, seed, feature_recipe_id=None, preprocessing_strategy=None: (
                            _build_pipeline(
                                model_name=model_name,
                                seed=seed,
                                class_weight_policy=methodology_policy.class_weight_policy.value,
                                compute_policy=resolved_compute_policy,
                                feature_recipe_id=(
                                    feature_recipe_id
                                    if feature_recipe_id is not None
                                    else resolved_feature_recipe_id
                                ),
                                preprocessing_strategy=(
                                    preprocessing_strategy
                                    if preprocessing_strategy is not None
                                    else resolved_preprocessing_config.strategy
                                ),
                                dimensionality_config=resolved_dimensionality_config,
                            )
                        ),
                        progress_callback=progress_callback,
                        load_features_from_cache_fn=(
                            load_features_from_cache_fn_override
                            if load_features_from_cache_fn_override is not None
                            else _load_features_from_cache
                        ),
                        scores_for_predictions_fn=_scores_for_predictions,
                        extract_linear_coefficients_fn=_extract_linear_coefficients,
                        compute_interpretability_stability_fn=_compute_interpretability_stability,
                        evaluate_permutations_fn=_evaluate_permutations,
                        stage_assignments=tuple(stage_planning_result.assignments),
                        stage_fallback_executor_ids=dict(
                            stage_planning_result.runtime_fallback_executor_ids
                        ),
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
    compute_runtime_metadata = (
        dict(segment_result.compute_runtime_metadata)
        if isinstance(segment_result.compute_runtime_metadata, dict)
        else None
    )
    actual_estimator_backend_id: str | None = None
    actual_estimator_backend_family: str | None = None
    if compute_runtime_metadata is not None:
        backend_id_candidate = compute_runtime_metadata.get("backend_id")
        if isinstance(backend_id_candidate, str) and backend_id_candidate.strip():
            actual_estimator_backend_id = backend_id_candidate.strip()
            actual_estimator_backend_family = _resolve_backend_family_from_backend_id(
                actual_estimator_backend_id,
                fallback_family=str(resolved_compute_policy.effective_backend_family),
            )
        else:
            actual_estimator_backend_family = str(resolved_compute_policy.effective_backend_family)
        compute_runtime_metadata["actual_estimator_backend_id"] = actual_estimator_backend_id
        compute_runtime_metadata["actual_estimator_backend_family"] = (
            actual_estimator_backend_family
        )
    else:
        actual_estimator_backend_family = str(resolved_compute_policy.effective_backend_family)
    if stage_planning_result is None:
        raise ValueError("Stage planning did not produce assignments.")
    planned_stage_assignments = (
        list(segment_result.stage_assignments)
        if isinstance(segment_result.stage_assignments, list)
        and bool(segment_result.stage_assignments)
        else list(stage_planning_result.assignments)
    )
    stage_execution = build_stage_execution_result(
        compute_policy=resolved_compute_policy,
        planned_sections=segment_result.planned_sections,
        executed_sections=segment_result.executed_sections,
        reused_sections=segment_result.reused_sections,
        tuning_enabled=bool(methodology_policy.tuning_enabled),
        n_permutations=int(n_permutations),
        section_timings_seconds=(
            segment_result.section_timings_seconds
            if isinstance(segment_result.section_timings_seconds, dict)
            else None
        ),
        stage_timings_seconds=stage_timings,
        reporting_status="planned",
        actual_estimator_backend_family=actual_estimator_backend_family,
        planned_assignments=planned_stage_assignments,
    )
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
            repeat_id=int(resolved_repeat_id),
            repeat_count=int(resolved_repeat_count),
            base_run_id=str(resolved_base_run_id),
            evidence_run_role=str(resolved_evidence_run_role),
            evidence_policy_effective=evidence_policy_model.model_dump(mode="json"),
            methodology_policy_name=methodology_policy.policy_name.value,
            class_weight_policy=methodology_policy.class_weight_policy.value,
            tuning_enabled=bool(methodology_policy.tuning_enabled),
            model_cost_tier=str(resolved_model_cost_tier),
            projected_runtime_seconds=int(resolved_projected_runtime_seconds),
            preprocessing_kind=resolved_preprocessing_kind,
            preprocessing_strategy=resolved_preprocessing_config.strategy,
            feature_recipe_id=resolved_feature_recipe_id,
            primary_metric_aggregation=resolved_primary_metric_aggregation,
            tuning_summary_path=tuning_summary_path,
            tuning_best_params_path=tuning_best_params_path,
            fit_timing_summary_path=fit_timing_summary_path,
            subgroup_metrics_json_path=subgroup_metrics_json_path,
            subgroup_metrics_csv_path=subgroup_metrics_csv_path,
            feature_qc_summary_path=feature_qc_summary_path,
            feature_qc_selected_samples_path=feature_qc_selected_samples_path,
            metric_policy_effective=metric_policy_effective,
            data_policy_effective=(
                data_policy_effective
                if data_policy_effective
                else dict(data_artifact_info.get("data_policy_effective", {}))
            ),
            data_artifacts=(
                dict(data_artifact_info.get("data_artifacts", {}))
                if isinstance(data_artifact_info, dict)
                else None
            ),
            identity=identity,
            dataset_fingerprint=dataset_fingerprint,
            git_provenance=git_provenance,
            stage_timings_seconds=stage_timings,
            resource_summary=resource_summary,
            warning_summary=warning_summary,
            timeout_policy_effective=timeout_policy_effective,
            profiling_context=resolved_profiling_context,
            compute_policy=resolved_compute_policy,
            compute_runtime_metadata=compute_runtime_metadata,
            stage_execution=stage_execution,
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
            repeat_id=int(resolved_repeat_id),
            repeat_count=int(resolved_repeat_count),
            base_run_id=str(resolved_base_run_id),
            evidence_run_role=str(resolved_evidence_run_role),
            evidence_policy_effective=evidence_policy_model.model_dump(mode="json"),
            data_policy_effective=(
                data_policy_effective
                if data_policy_effective
                else dict(data_artifact_info.get("data_policy_effective", {}))
            ),
            primary_metric_name=resolved_primary_metric_name,
            primary_metric_aggregation=resolved_primary_metric_aggregation,
            permutation_metric_name=resolved_permutation_metric_name,
            metric_policy_effective=metric_policy_effective,
            methodology_policy_name=methodology_policy.policy_name.value,
            class_weight_policy=methodology_policy.class_weight_policy.value,
            tuning_enabled=bool(methodology_policy.tuning_enabled),
            model_cost_tier=str(resolved_model_cost_tier),
            projected_runtime_seconds=int(resolved_projected_runtime_seconds),
            preprocessing_kind=resolved_preprocessing_kind,
            preprocessing_strategy=resolved_preprocessing_config.strategy,
            feature_recipe_id=resolved_feature_recipe_id,
            tuning_search_space_id=effective_tuning_space_id,
            tuning_search_space_version=effective_tuning_space_version,
            tuning_inner_cv_scheme=effective_tuning_inner_cv_scheme,
            tuning_inner_group_field=effective_tuning_inner_group_field,
            tuning_summary_path=tuning_summary_path,
            tuning_best_params_path=tuning_best_params_path,
            fit_timing_summary_path=fit_timing_summary_path,
            calibration_summary_path=calibration_summary_path,
            calibration_table_path=calibration_table_path,
            feature_qc_summary_path=feature_qc_summary_path,
            feature_qc_selected_samples_path=feature_qc_selected_samples_path,
            subgroup_reporting_enabled=bool(subgroup_policy.enabled),
            subgroup_dimensions=list(subgroup_policy.subgroup_dimensions),
            subgroup_min_samples_per_group=int(subgroup_policy.min_samples_per_group),
            subgroup_metrics_json_path=subgroup_metrics_json_path,
            subgroup_metrics_csv_path=subgroup_metrics_csv_path,
            dataset_card_json_path=dataset_card_json_path,
            dataset_card_md_path=dataset_card_md_path,
            dataset_summary_json_path=dataset_summary_json_path,
            dataset_summary_csv_path=dataset_summary_csv_path,
            data_quality_report_path=data_quality_report_path,
            class_balance_report_path=class_balance_report_path,
            missingness_report_path=missingness_report_path,
            leakage_audit_path=leakage_audit_path,
            external_dataset_card_path=external_dataset_card_path,
            external_dataset_summary_path=external_dataset_summary_path,
            external_validation_compatibility_path=external_validation_compatibility_path,
            data_artifacts=(
                dict(data_artifact_info.get("data_artifacts", {}))
                if isinstance(data_artifact_info, dict)
                else None
            ),
            filter_task=filter_task,
            filter_modality=filter_modality,
            feature_space=resolved_feature_space,
            roi_spec_path=resolved_roi_spec_path,
            dimensionality_strategy=resolved_dimensionality_config.strategy,
            pca_n_components=resolved_dimensionality_config.pca_n_components,
            pca_variance_ratio=resolved_dimensionality_config.pca_variance_ratio,
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
            timeout_policy_effective=timeout_policy_effective,
            profiling_context=resolved_profiling_context,
            compute_policy=resolved_compute_policy,
            compute_runtime_metadata=compute_runtime_metadata,
            stage_execution=stage_execution,
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
        status=RUN_STATUS_SUCCESS,
        executed_sections=segment_result.executed_sections,
        reused_sections=segment_result.reused_sections,
        warnings=warnings_payload,
        warning_summary=warning_summary,
        stage_timings_seconds=stage_timings,
        resource_summary=resource_summary,
    )
    emit_progress(
        progress_callback,
        stage="run",
        message="finished run execution",
        metadata={
            "run_id": str(resolved_run_id),
            "model": str(model),
            "target": str(target_column),
            "cv_mode": str(cv_mode),
            "framework_mode": str(resolved_framework_mode.value),
            "hardware_mode_requested": str(resolved_compute_policy.hardware_mode_requested),
            "hardware_mode_effective": str(resolved_compute_policy.hardware_mode_effective),
            "requested_backend_family": str(resolved_compute_policy.requested_backend_family),
            "effective_backend_family": str(resolved_compute_policy.effective_backend_family),
            "assigned_compute_lane": (
                str(resolved_compute_policy.assigned_compute_lane)
                if resolved_compute_policy.assigned_compute_lane is not None
                else None
            ),
            "assigned_backend_family": (
                str(resolved_compute_policy.assigned_backend_family)
                if resolved_compute_policy.assigned_backend_family is not None
                else None
            ),
            "actual_estimator_backend_id": actual_estimator_backend_id,
            "actual_estimator_backend_family": actual_estimator_backend_family,
            "report_dir": str(report_dir.resolve()),
        },
    )

    return build_run_result_payload(
        run_id=resolved_run_id,
        report_dir=report_dir,
        config_path=config_path,
        metrics_path=metrics_path,
        subgroup_metrics_json_path=subgroup_metrics_json_path,
        subgroup_metrics_csv_path=subgroup_metrics_csv_path,
        dataset_card_json_path=dataset_card_json_path,
        dataset_card_md_path=dataset_card_md_path,
        dataset_summary_json_path=dataset_summary_json_path,
        dataset_summary_csv_path=dataset_summary_csv_path,
        data_quality_report_path=data_quality_report_path,
        class_balance_report_path=class_balance_report_path,
        missingness_report_path=missingness_report_path,
        leakage_audit_path=leakage_audit_path,
        external_dataset_card_path=external_dataset_card_path,
        external_dataset_summary_path=external_dataset_summary_path,
        external_validation_compatibility_path=external_validation_compatibility_path,
        tuning_summary_path=tuning_summary_path,
        tuning_best_params_path=tuning_best_params_path,
        fit_timing_summary_path=fit_timing_summary_path,
        calibration_summary_path=calibration_summary_path,
        calibration_table_path=calibration_table_path,
        feature_qc_summary_path=feature_qc_summary_path,
        feature_qc_selected_samples_path=feature_qc_selected_samples_path,
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
        repeat_id=int(resolved_repeat_id),
        repeat_count=int(resolved_repeat_count),
        base_run_id=str(resolved_base_run_id),
        evidence_run_role=str(resolved_evidence_run_role),
        evidence_policy_effective=evidence_policy_model.model_dump(mode="json"),
        data_policy_effective=(
            data_policy_effective
            if data_policy_effective
            else dict(data_artifact_info.get("data_policy_effective", {}))
        ),
        data_artifacts=(
            dict(data_artifact_info.get("data_artifacts", {}))
            if isinstance(data_artifact_info, dict)
            else None
        ),
        metric_policy_effective=metric_policy_effective,
        methodology_policy_name=methodology_policy.policy_name.value,
        class_weight_policy=methodology_policy.class_weight_policy.value,
        tuning_enabled=bool(methodology_policy.tuning_enabled),
        preprocessing_strategy=resolved_preprocessing_config.strategy,
        feature_recipe_id=resolved_feature_recipe_id,
        feature_space=resolved_feature_space,
        roi_spec_path=resolved_roi_spec_path,
        dimensionality_strategy=resolved_dimensionality_config.strategy,
        pca_n_components=resolved_dimensionality_config.pca_n_components,
        pca_variance_ratio=resolved_dimensionality_config.pca_variance_ratio,
        model_cost_tier=str(resolved_model_cost_tier),
        projected_runtime_seconds=int(resolved_projected_runtime_seconds),
        protocol_context=resolved_protocol_context,
        comparison_context=resolved_comparison_context,
        stage_timings_seconds=stage_timings,
        resource_summary=resource_summary,
        warning_summary=warning_summary,
        dataset_fingerprint=dataset_fingerprint,
        timeout_policy_effective=timeout_policy_effective,
        profiling_context=resolved_profiling_context,
        compute_policy=resolved_compute_policy,
        compute_runtime_metadata=compute_runtime_metadata,
        stage_execution=stage_execution,
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
        help=(
            "Model to evaluate. 'all' runs the conservative default batch "
            "(ridge, logreg, linearsvc) and excludes exploratory-only models."
        ),
    )
    parser.add_argument(
        "--cv",
        required=True,
        choices=list(_CV_MODES),
        help=(
            "Experiment mode. Required. within_subject_loso_session=primary thesis mode; "
            "frozen_cross_person_transfer=secondary transfer mode; "
            "loso_session=auxiliary grouped mode; "
            "record_random_split=record-wise random split stress-test mode."
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
        "--feature-space",
        default=FEATURE_SPACE_WHOLE_BRAIN_MASKED,
        choices=list(SUPPORTED_FEATURE_SPACES),
        help="Feature representation path: whole-brain masked voxels or predefined ROI means.",
    )
    parser.add_argument(
        "--roi-spec-path",
        default=None,
        help=(
            "Path to ROI feature-space spec JSON. Required when --feature-space "
            "roi_mean_predefined."
        ),
    )
    parser.add_argument(
        "--preprocessing-strategy",
        default=None,
        choices=list(SUPPORTED_PREPROCESSING_STRATEGIES),
        help=(
            "Optional fold-local preprocessing override for lock experiments "
            "(none or standardize_zscore)."
        ),
    )
    parser.add_argument(
        "--dimensionality-strategy",
        default="none",
        choices=list(SUPPORTED_DIMENSIONALITY_STRATEGIES),
        help="Optional post-feature dimensionality strategy applied inside each fold.",
    )
    parser.add_argument(
        "--pca-n-components",
        type=int,
        default=None,
        help="Optional PCA component count when --dimensionality-strategy pca.",
    )
    parser.add_argument(
        "--pca-variance-ratio",
        type=float,
        default=None,
        help=(
            "Optional PCA explained-variance ratio in (0, 1] when --dimensionality-strategy pca."
        ),
    )
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
        "--feature-recipe",
        default=BASELINE_STANDARD_SCALER_RECIPE_ID,
        choices=list(SUPPORTED_FEATURE_RECIPE_IDS),
        help=(
            "Fold-local feature preprocessing recipe. "
            "Official confirmatory runs are locked to baseline_standard_scaler_v1."
        ),
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
    parser.add_argument(
        "--max-parallel-runs",
        type=int,
        default=1,
        help=(
            "Deterministic scheduling window used for exploratory run-level lane planning. "
            "Does not change scientific semantics."
        ),
    )
    parser.add_argument(
        "--max-parallel-gpu-runs",
        type=int,
        default=1,
        help=(
            "Maximum GPU-lane slots per exploratory scheduling window when hardware-mode is "
            "gpu_only or max_both."
        ),
    )
    parser.add_argument(
        "--hardware-mode",
        default="cpu_only",
        choices=list(HARDWARE_MODE_CHOICES),
        help=(
            "Operational compute policy only. "
            "PR 5 adds exploratory max_both run-level lane scheduling. "
            "Official paths remain CPU-only."
        ),
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=None,
        help="Optional GPU device ID for gpu_only or max_both operational modes.",
    )
    parser.add_argument(
        "--deterministic-compute",
        action="store_true",
        help="Record deterministic compute intent in compute metadata.",
    )
    parser.add_argument(
        "--allow-backend-fallback",
        action="store_true",
        help=(
            "Allow exploratory gpu_only requests to fall back to the CPU reference backend when "
            "GPU capability is unavailable. Official paths reject this flag."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    models = list(DEFAULT_BATCH_MODEL_NAMES) if args.model == "all" else [args.model]
    subgroup_dimensions = list(args.subgroup_dimension) if list(args.subgroup_dimension) else None
    results: list[dict[str, Any]] = []
    run_batch: list[dict[str, Any]] = []
    if args.model == "all":
        base_run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        for order_index, model_name in enumerate(models):
            run_batch.append(
                {
                    "order_index": int(order_index),
                    "model_name": str(model_name),
                    "run_id": f"{base_run_id}_{model_name}_{args.target}",
                }
            )
    else:
        run_batch.append(
            {
                "order_index": 0,
                "model_name": str(models[0]),
                "run_id": args.run_id,
            }
        )

    scheduled_assignments_by_run_id: dict[str, dict[str, Any]] = {}
    if len(run_batch) > 1:
        base_compute_policy = resolve_compute_policy(
            framework_mode=FrameworkMode.EXPLORATORY,
            hardware_mode=args.hardware_mode,
            gpu_device_id=args.gpu_device_id,
            deterministic_compute=bool(args.deterministic_compute),
            allow_backend_fallback=bool(args.allow_backend_fallback),
        )
        schedule = plan_compute_schedule(
            run_requests=[
                ComputeRunRequest(
                    order_index=int(run_spec["order_index"]),
                    run_id=str(run_spec["run_id"]),
                    model_name=str(run_spec["model_name"]),
                )
                for run_spec in run_batch
            ],
            base_compute_policy=base_compute_policy,
            max_parallel_runs=int(args.max_parallel_runs),
            max_parallel_gpu_runs=int(args.max_parallel_gpu_runs),
        )
        scheduled_assignments_by_run_id = {
            str(assignment.run_id): assignment.to_payload() for assignment in schedule
        }

    for run_spec in run_batch:
        model_name = str(run_spec["model_name"])
        model_run_id_raw = run_spec.get("run_id")
        model_run_id = str(model_run_id_raw) if model_run_id_raw is not None else None
        scheduled_assignment = (
            scheduled_assignments_by_run_id.get(model_run_id, None)
            if model_run_id is not None
            else None
        )

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
            feature_space=args.feature_space,
            roi_spec_path=args.roi_spec_path,
            preprocessing_strategy=args.preprocessing_strategy,
            dimensionality_strategy=args.dimensionality_strategy,
            pca_n_components=args.pca_n_components,
            pca_variance_ratio=args.pca_variance_ratio,
            n_permutations=args.n_permutations,
            methodology_policy_name=args.methodology_policy,
            class_weight_policy=args.class_weight_policy,
            feature_recipe_id=args.feature_recipe,
            tuning_enabled=(
                str(args.methodology_policy) == MethodologyPolicyName.GROUPED_NESTED_TUNING.value
            ),
            tuning_search_space_id=args.tuning_search_space_id,
            tuning_search_space_version=args.tuning_search_space_version,
            tuning_inner_cv_scheme=(
                "grouped_leave_one_group_out"
                if str(args.methodology_policy) == MethodologyPolicyName.GROUPED_NESTED_TUNING.value
                else None
            ),
            tuning_inner_group_field=(
                "session"
                if str(args.methodology_policy) == MethodologyPolicyName.GROUPED_NESTED_TUNING.value
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
            hardware_mode=args.hardware_mode,
            gpu_device_id=args.gpu_device_id,
            deterministic_compute=bool(args.deterministic_compute),
            allow_backend_fallback=bool(args.allow_backend_fallback),
            max_parallel_runs=int(args.max_parallel_runs),
            max_parallel_gpu_runs=int(args.max_parallel_gpu_runs),
            scheduled_compute_assignment=scheduled_assignment,
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
