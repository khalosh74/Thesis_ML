from __future__ import annotations

import inspect
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid

from Thesis_ML.config.methodology import FeatureQualityPolicy
from Thesis_ML.config.metric_policy import metric_bundle
from Thesis_ML.experiments.cv_split_plan import build_cv_split_plan
from Thesis_ML.experiments.evidence_statistics import build_calibration_outputs
from Thesis_ML.experiments.metrics import classification_metric_score
from Thesis_ML.experiments.model_factory import model_supports_linear_interpretability
from Thesis_ML.experiments.progress import emit_progress
from Thesis_ML.experiments.stage_execution import StageAssignment
from Thesis_ML.experiments.stage_registry import (
    MODEL_FIT_CPU_EXECUTOR_ID,
    PERMUTATION_REFERENCE_EXECUTOR_ID,
    SPECIALIZED_RIDGE_TUNING_EXECUTOR_ID,
    SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID,
    SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID,
    TUNING_GENERIC_EXECUTOR_ID,
    TUNING_SKIPPED_CONTROL_EXECUTOR_ID,
    run_model_fit_executor,
    run_permutation_executor,
    run_tuning_executor,
)
from Thesis_ML.experiments.tuning_search_spaces import resolve_tuning_search_space_for_model
from Thesis_ML.features.dimensionality import (
    resolve_dimensionality_config,
    validate_dimensionality_for_training_data,
)
from Thesis_ML.features.preprocessing import validate_preprocessing_for_training_data
from Thesis_ML.features.feature_qc import (
    FEATURE_QC_SAMPLE_FIELDS,
    compute_sample_feature_qc,
    summarize_group_feature_qc,
)

if TYPE_CHECKING:
    from Thesis_ML.experiments.section_models import (
        EvaluationInput,
        InterpretabilityInput,
        ModelFitInput,
    )


def _extract_fold_compute_runtime_metadata(estimator: Any) -> dict[str, Any] | None:
    model = getattr(estimator, "named_steps", {}).get("model")
    if model is None:
        return None
    metadata_candidate: Any = None
    if hasattr(model, "get_backend_runtime_metadata"):
        metadata_candidate = model.get_backend_runtime_metadata()
    elif hasattr(model, "backend_runtime_metadata_"):
        metadata_candidate = model.backend_runtime_metadata_
    if not isinstance(metadata_candidate, dict):
        return None
    return dict(metadata_candidate)


def _aggregate_compute_runtime_metadata(
    fold_metadata_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not fold_metadata_rows:
        return None

    peak_values: list[float] = []
    transfer_values: list[float] = []
    deterministic_values: list[bool] = []
    deterministic_limitations: list[str] = []
    backend_ids: list[str] = []

    for row in fold_metadata_rows:
        peak_value = row.get("gpu_memory_peak_mb")
        if isinstance(peak_value, (int, float)):
            peak_values.append(float(peak_value))

        transfer_value = row.get("device_transfer_seconds")
        if isinstance(transfer_value, (int, float)):
            transfer_values.append(float(transfer_value))

        deterministic_value = row.get("torch_deterministic_enforced")
        if isinstance(deterministic_value, bool):
            deterministic_values.append(bool(deterministic_value))

        deterministic_limit = row.get("torch_deterministic_limitations")
        if isinstance(deterministic_limit, str) and deterministic_limit.strip():
            deterministic_limitations.append(deterministic_limit.strip())

        backend_id = row.get("backend_id")
        if isinstance(backend_id, str) and backend_id.strip():
            backend_ids.append(backend_id.strip())

    payload: dict[str, Any] = {
        "backend_id": (backend_ids[0] if backend_ids else None),
        "backend_diagnostics_folds": int(len(fold_metadata_rows)),
        "gpu_memory_peak_mb": (max(peak_values) if peak_values else None),
        "device_transfer_seconds": (sum(transfer_values) if transfer_values else None),
        "torch_deterministic_enforced": (
            bool(all(deterministic_values)) if deterministic_values else None
        ),
        "torch_deterministic_limitations": (
            ";".join(sorted(set(deterministic_limitations))) if deterministic_limitations else None
        ),
    }
    if len(set(backend_ids)) > 1:
        payload["backend_id"] = "mixed"
    return payload


def _assignment_executor_id(
    assignment: StageAssignment | None,
    *,
    default_executor_id: str,
) -> str:
    if assignment is None:
        return str(default_executor_id)
    executor_id = assignment.executor_id
    if isinstance(executor_id, str) and executor_id.strip():
        return executor_id.strip()
    return str(default_executor_id)


def _build_pipeline_template(
    *,
    build_pipeline_fn: Any,
    model_name: str,
    seed: int,
    feature_recipe_id: str,
    preprocessing_strategy: str | None,
) -> Any:
    try:
        signature = inspect.signature(build_pipeline_fn)
    except (TypeError, ValueError):
        signature = None

    pipeline_kwargs: dict[str, Any] = {
        "model_name": model_name,
        "seed": seed,
    }
    if signature is None:
        return build_pipeline_fn(**pipeline_kwargs)

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if "feature_recipe_id" in signature.parameters or accepts_kwargs:
        pipeline_kwargs["feature_recipe_id"] = feature_recipe_id
    if "preprocessing_strategy" in signature.parameters or accepts_kwargs:
        pipeline_kwargs["preprocessing_strategy"] = preprocessing_strategy
    return build_pipeline_fn(**pipeline_kwargs)


def execute_model_fit(section_input: ModelFitInput) -> dict[str, Any]:
    y = section_input.metadata_df[section_input.target_column].astype(str).to_numpy()

    split_plan = build_cv_split_plan(
        metadata_df=section_input.metadata_df,
        target_column=section_input.target_column,
        cv_mode=section_input.cv_mode,
        subject=section_input.subject,
        train_subject=section_input.train_subject,
        test_subject=section_input.test_subject,
        seed=section_input.seed,
    )

    groups = split_plan.groups
    splits = [(fold.train_idx, fold.test_idx) for fold in split_plan.folds]

    planned_outer_folds = int(len(splits))
    max_outer_folds = section_input.max_outer_folds
    if max_outer_folds is not None:
        resolved_max_outer_folds = int(max_outer_folds)
        if resolved_max_outer_folds <= 0:
            raise ValueError("max_outer_folds must be > 0 when provided.")
        if planned_outer_folds > resolved_max_outer_folds:
            splits = splits[:resolved_max_outer_folds]

    pipeline_template = _build_pipeline_template(
        build_pipeline_fn=section_input.build_pipeline_fn,
        model_name=section_input.model,
        seed=section_input.seed,
        feature_recipe_id=section_input.feature_recipe_id,
        preprocessing_strategy=section_input.preprocessing_strategy,
    )
    tuning_summary_path = (
        section_input.tuning_summary_path
        if section_input.tuning_summary_path is not None
        else section_input.report_dir / "tuning_summary.json"
    )
    tuning_best_params_path = (
        section_input.tuning_best_params_path
        if section_input.tuning_best_params_path is not None
        else section_input.report_dir / "best_params_per_fold.csv"
    )
    fit_timing_summary_path = (
        section_input.fit_timing_summary_path
        if section_input.fit_timing_summary_path is not None
        else section_input.report_dir / "fit_timing_summary.json"
    )
    methodology_policy_name = str(section_input.methodology_policy_name).strip()
    tuning_enabled = bool(section_input.tuning_enabled)
    if methodology_policy_name == "fixed_baselines_only" and tuning_enabled:
        raise ValueError(
            "methodology_policy_name='fixed_baselines_only' forbids tuning_enabled=true."
        )
    if methodology_policy_name not in {"fixed_baselines_only", "grouped_nested_tuning"}:
        raise ValueError(
            "Unsupported methodology_policy_name. "
            "Allowed values: fixed_baselines_only, grouped_nested_tuning."
        )
    model_fit_executor_id = _assignment_executor_id(
        section_input.model_fit_assignment,
        default_executor_id=MODEL_FIT_CPU_EXECUTOR_ID,
    )
    default_tuning_executor_id = (
        TUNING_SKIPPED_CONTROL_EXECUTOR_ID
        if str(section_input.model).strip().lower() == "dummy"
        else TUNING_GENERIC_EXECUTOR_ID
    )
    if (
        str(section_input.model).strip().lower() == "linearsvc"
        and bool(tuning_enabled)
        and methodology_policy_name == "grouped_nested_tuning"
    ):
        default_tuning_executor_id = SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID
    if (
        str(section_input.model).strip().lower() == "ridge"
        and bool(tuning_enabled)
        and methodology_policy_name == "grouped_nested_tuning"
    ):
        default_tuning_executor_id = SPECIALIZED_RIDGE_TUNING_EXECUTOR_ID
    tuning_executor_id = _assignment_executor_id(
        section_input.tuning_assignment,
        default_executor_id=default_tuning_executor_id,
    )
    tuning_fallback_executor_id = (
        str(section_input.tuning_fallback_executor_id).strip()
        if isinstance(section_input.tuning_fallback_executor_id, str)
        and str(section_input.tuning_fallback_executor_id).strip()
        else (
            TUNING_GENERIC_EXECUTOR_ID
            if tuning_executor_id
            in {
                SPECIALIZED_RIDGE_TUNING_EXECUTOR_ID,
                SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID,
                SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID,
            }
            else None
        )
    )
    resolved_tuning_search_space_id = section_input.tuning_search_space_id
    resolved_tuning_search_space_version = section_input.tuning_search_space_version
    resolved_param_grid: dict[str, list[Any]] | None = None
    if (
        tuning_enabled
        and methodology_policy_name == "grouped_nested_tuning"
        and str(section_input.model).strip().lower() != "dummy"
    ):
        (
            resolved_tuning_search_space_id,
            resolved_tuning_search_space_version,
            resolved_param_grid,
        ) = resolve_tuning_search_space_for_model(
            model_name=section_input.model,
            search_space_id=section_input.tuning_search_space_id,
            search_space_version=section_input.tuning_search_space_version,
        )

    supports_linear_interpretability = model_supports_linear_interpretability(section_input.model)
    if section_input.interpretability_enabled is None:
        interpretability_enabled = (
            section_input.cv_mode == "within_subject_loso_session"
            and supports_linear_interpretability
        )
    else:
        interpretability_enabled = bool(section_input.interpretability_enabled)
    if interpretability_enabled and not supports_linear_interpretability:
        interpretability_enabled = False
    interpretability_fold_rows: list[dict[str, Any]] = []
    interpretability_vectors: list[np.ndarray] = []
    interpretability_dir: Path | None = None
    if interpretability_enabled:
        interpretability_dir = section_input.report_dir / "interpretability"
        interpretability_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    fold_compute_runtime_metadata: list[dict[str, Any]] = []
    total_outer_folds = int(len(splits))
    emit_progress(
        section_input.progress_callback,
        stage="model_fit",
        message=f"starting model fit with {total_outer_folds} outer folds",
        completed_units=0.0,
        total_units=float(total_outer_folds),
        metadata={
            "run_id": str(section_input.run_id),
            "model": str(section_input.model),
            "target": str(section_input.target_column),
            "cv_mode": str(section_input.cv_mode),
            "total_folds": int(total_outer_folds),
        },
    )

    for fold_index, (train_idx, test_idx) in enumerate(splits):
        outer_fold_start = perf_counter()
        emit_progress(
            section_input.progress_callback,
            stage="fold",
            message=f"starting outer fold {fold_index + 1}/{total_outer_folds}",
            completed_units=float(fold_index),
            total_units=float(total_outer_folds),
            metadata={
                "run_id": str(section_input.run_id),
                "model": str(section_input.model),
                "target": str(section_input.target_column),
                "cv_mode": str(section_input.cv_mode),
                "fold_index": int(fold_index + 1),
                "total_folds": int(total_outer_folds),
            },
        )
        train_meta = section_input.metadata_df.iloc[train_idx].reset_index(drop=True)
        test_meta = section_input.metadata_df.iloc[test_idx].reset_index(drop=True)
        train_subjects = sorted(train_meta["subject"].astype(str).unique().tolist())
        test_subjects = sorted(test_meta["subject"].astype(str).unique().tolist())
        train_sessions = sorted(train_meta["session"].astype(str).unique().tolist())
        test_sessions = sorted(test_meta["session"].astype(str).unique().tolist())

        if section_input.cv_mode == "within_subject_loso_session":
            expected_subjects = [str(section_input.subject)]
            if train_subjects != expected_subjects or test_subjects != expected_subjects:
                raise ValueError(
                    "within_subject_loso_session produced fold(s) with unexpected "
                    "subject membership."
                )
            if set(train_sessions) & set(test_sessions):
                raise ValueError(
                    "within_subject_loso_session produced overlapping train/test sessions."
                )
        if section_input.cv_mode == "frozen_cross_person_transfer":
            expected_train = [str(section_input.train_subject)]
            expected_test = [str(section_input.test_subject)]
            if train_subjects != expected_train or test_subjects != expected_test:
                raise ValueError(
                    "frozen_cross_person_transfer produced unexpected train/test subject "
                    "membership."
                )

        x_train_fold = np.asarray(section_input.x_matrix[train_idx])
        validate_preprocessing_for_training_data(
            preprocessing_strategy=section_input.preprocessing_strategy,
            x_train=x_train_fold,
        )

        dimensionality_config = resolve_dimensionality_config(
            dimensionality_strategy=section_input.dimensionality_strategy,
            pca_n_components=section_input.pca_n_components,
            pca_variance_ratio=section_input.pca_variance_ratio,
        )
        validate_dimensionality_for_training_data(
            config=dimensionality_config,
            x_train=x_train_fold,
        )

        estimator = clone(pipeline_template)
        estimator_fit_elapsed_seconds: float | None = None
        tuned_search_elapsed_seconds: float | None = None
        tuned_search_candidate_count = 0
        tuned_search_configured_candidate_count = 0
        tuned_search_profiled_candidate_count = 0
        tuned_search_configured_inner_fold_count = 0
        tuned_search_profiled_inner_fold_count = 0
        tuning_extrapolation_applied = False
        estimated_full_tuned_search_seconds: float | None = None
        measured_inner_tuning_seconds: float | None = None
        estimated_full_inner_tuning_seconds: float | None = None
        tuning_split_scale_seconds: float | None = None
        tuning_candidate_fit_seconds: float | None = None
        tuning_candidate_predict_seconds: float | None = None
        tuning_refit_elapsed_seconds: float | None = None
        tuning_executor = "none"
        tuning_executor_fallback_reason: str | None = None
        specialized_linearsvc_tuning_used = False
        specialized_logreg_tuning_used = False
        tuning_progress_event_count: int | None = None
        tuning_progress_total_units: int | None = None
        cv_mean_fit_time_seconds: float | None = None
        cv_std_fit_time_seconds: float | None = None
        cv_mean_score_time_seconds: float | None = None
        cv_std_score_time_seconds: float | None = None
        if tuning_enabled and methodology_policy_name == "grouped_nested_tuning":
            if section_input.model == "dummy":
                fit_payload = run_model_fit_executor(
                    executor_id=model_fit_executor_id,
                    estimator=estimator,
                    x_train=section_input.x_matrix[train_idx],
                    y_train=y[train_idx],
                )
                estimator = fit_payload["estimator"]
                estimator_fit_elapsed_seconds = float(fit_payload["estimator_fit_elapsed_seconds"])
                tuning_result = run_tuning_executor(
                    executor_id=tuning_executor_id,
                    fallback_executor_id=tuning_fallback_executor_id,
                    pipeline_template=pipeline_template,
                    x_outer_train=section_input.x_matrix[train_idx],
                    y_outer_train=y[train_idx],
                    inner_groups=np.asarray(groups[train_idx]).astype(str, copy=False),
                    inner_splits=[],
                    param_grid={},
                    candidate_params=[],
                    configured_candidate_count=0,
                    configured_inner_fold_count=0,
                    profiled_candidate_count=0,
                    profiled_inner_fold_count=0,
                    primary_metric_name=section_input.primary_metric_name,
                    progress_callback=section_input.progress_callback,
                    progress_metadata={
                        "run_id": str(section_input.run_id),
                        "model": str(section_input.model),
                        "target": str(section_input.target_column),
                        "cv_mode": str(section_input.cv_mode),
                        "fold_index": int(fold_index + 1),
                        "total_folds": int(total_outer_folds),
                    },
                )
                tuning_executor = str(tuning_result.get("tuning_executor", "skipped_control_model"))
                tuning_executor_fallback_reason = (
                    str(tuning_result["tuning_executor_fallback_reason"])
                    if isinstance(tuning_result.get("tuning_executor_fallback_reason"), str)
                    and str(tuning_result.get("tuning_executor_fallback_reason")).strip()
                    else None
                )
                specialized_linearsvc_tuning_used = bool(
                    tuning_result.get("specialized_linearsvc_tuning_used", False)
                )
                specialized_logreg_tuning_used = bool(
                    tuning_result.get("specialized_logreg_tuning_used", False)
                )
                tuning_progress_event_count = (
                    int(tuning_result["tuning_progress_event_count"])
                    if tuning_result.get("tuning_progress_event_count") is not None
                    else None
                )
                tuning_progress_total_units = (
                    int(tuning_result["tuning_progress_total_units"])
                    if tuning_result.get("tuning_progress_total_units") is not None
                    else None
                )
                tuning_rows.append(
                    {
                        "fold": int(fold_index),
                        "status": "skipped_control_model",
                        "model": section_input.model,
                        "best_score": pd.NA,
                        "best_params_json": "{}",
                        "n_candidates": 0,
                        "search_space_id": resolved_tuning_search_space_id,
                        "search_space_version": resolved_tuning_search_space_version,
                        "primary_metric_name": section_input.primary_metric_name,
                        "inner_group_field": section_input.tuning_inner_group_field,
                        "estimator_fit_elapsed_seconds": estimator_fit_elapsed_seconds,
                        "tuned_search_elapsed_seconds": pd.NA,
                        "configured_candidate_count": 0,
                        "profiled_candidate_count": 0,
                        "configured_inner_fold_count": 0,
                        "profiled_inner_fold_count": 0,
                        "tuning_extrapolation_applied": False,
                        "measured_inner_tuning_seconds": pd.NA,
                        "estimated_full_inner_tuning_seconds": pd.NA,
                        "estimated_full_tuned_search_seconds": pd.NA,
                        "tuning_executor": tuning_executor,
                        "tuning_executor_fallback_reason": tuning_executor_fallback_reason,
                        "specialized_linearsvc_tuning_used": bool(
                            specialized_linearsvc_tuning_used
                        ),
                        "specialized_logreg_tuning_used": bool(specialized_logreg_tuning_used),
                        "tuning_progress_event_count": (
                            int(tuning_progress_event_count)
                            if tuning_progress_event_count is not None
                            else pd.NA
                        ),
                        "tuning_progress_total_units": (
                            int(tuning_progress_total_units)
                            if tuning_progress_total_units is not None
                            else pd.NA
                        ),
                        "tuning_split_scale_seconds": pd.NA,
                        "tuning_candidate_fit_seconds": pd.NA,
                        "tuning_candidate_predict_seconds": pd.NA,
                        "tuning_refit_elapsed_seconds": pd.NA,
                        "cv_mean_fit_time_seconds": pd.NA,
                        "cv_std_fit_time_seconds": pd.NA,
                        "cv_mean_score_time_seconds": pd.NA,
                        "cv_std_score_time_seconds": pd.NA,
                    }
                )
            else:
                if resolved_param_grid is None:
                    raise ValueError(
                        "grouped_nested_tuning requires a resolved tuning search-space param grid."
                    )
                param_grid = dict(resolved_param_grid)
                candidate_params = list(ParameterGrid(param_grid))
                candidate_count = int(len(candidate_params))
                x_outer_train = section_input.x_matrix[train_idx]
                y_outer_train = y[train_idx]
                inner_groups = np.asarray(groups[train_idx]).astype(str, copy=False)
                inner_splits = list(
                    LeaveOneGroupOut().split(x_outer_train, y_outer_train, inner_groups)
                )
                inner_group_count = int(len(inner_splits))
                if inner_group_count < 2:
                    raise ValueError(
                        "Grouped nested tuning requires at least two inner groups in training data."
                    )
                configured_candidate_count = int(candidate_count)
                configured_inner_fold_count = int(inner_group_count)
                profiled_candidate_count = int(candidate_count)
                profiled_inner_fold_count = int(inner_group_count)
                if bool(section_input.profiling_only):
                    if section_input.profile_tuning_candidates is not None:
                        profiled_candidate_count = int(
                            min(
                                int(candidate_count),
                                int(section_input.profile_tuning_candidates),
                            )
                        )
                        if profiled_candidate_count <= 0:
                            raise ValueError("profile_tuning_candidates must be > 0 when provided.")
                    if section_input.profile_inner_folds is not None:
                        profiled_inner_fold_count = int(
                            min(
                                int(inner_group_count),
                                int(section_input.profile_inner_folds),
                            )
                        )
                        if profiled_inner_fold_count <= 0:
                            raise ValueError("profile_inner_folds must be > 0 when provided.")
                emit_progress(
                    section_input.progress_callback,
                    stage="tuning",
                    message=f"running nested tuning on fold {fold_index + 1}/{total_outer_folds}",
                    metadata={
                        "run_id": str(section_input.run_id),
                        "model": str(section_input.model),
                        "target": str(section_input.target_column),
                        "fold_index": int(fold_index + 1),
                        "total_folds": int(total_outer_folds),
                        "candidate_count": int(configured_candidate_count),
                        "inner_group_count": int(configured_inner_fold_count),
                        "profiled_candidate_count": int(profiled_candidate_count),
                        "profiled_inner_fold_count": int(profiled_inner_fold_count),
                    },
                )
                tuning_result = run_tuning_executor(
                    executor_id=tuning_executor_id,
                    fallback_executor_id=tuning_fallback_executor_id,
                    pipeline_template=pipeline_template,
                    x_outer_train=x_outer_train,
                    y_outer_train=y_outer_train,
                    inner_groups=inner_groups,
                    inner_splits=inner_splits,
                    param_grid=param_grid,
                    candidate_params=candidate_params,
                    configured_candidate_count=configured_candidate_count,
                    configured_inner_fold_count=configured_inner_fold_count,
                    profiled_candidate_count=profiled_candidate_count,
                    profiled_inner_fold_count=profiled_inner_fold_count,
                    primary_metric_name=section_input.primary_metric_name,
                    progress_callback=section_input.progress_callback,
                    progress_metadata={
                        "run_id": str(section_input.run_id),
                        "model": str(section_input.model),
                        "target": str(section_input.target_column),
                        "cv_mode": str(section_input.cv_mode),
                        "fold_index": int(fold_index + 1),
                        "total_folds": int(total_outer_folds),
                    },
                )
                estimator = tuning_result["estimator"]
                tuned_search_elapsed_seconds = (
                    float(tuning_result["tuned_search_elapsed_seconds"])
                    if tuning_result.get("tuned_search_elapsed_seconds") is not None
                    else None
                )
                tuned_search_candidate_count = int(tuning_result["tuned_search_candidate_count"])
                tuned_search_configured_candidate_count = int(
                    tuning_result["tuned_search_configured_candidate_count"]
                )
                tuned_search_profiled_candidate_count = int(
                    tuning_result["tuned_search_profiled_candidate_count"]
                )
                tuned_search_configured_inner_fold_count = int(
                    tuning_result["tuned_search_configured_inner_fold_count"]
                )
                tuned_search_profiled_inner_fold_count = int(
                    tuning_result["tuned_search_profiled_inner_fold_count"]
                )
                tuning_extrapolation_applied = bool(tuning_result["tuning_extrapolation_applied"])
                measured_inner_tuning_seconds = (
                    float(tuning_result["measured_inner_tuning_seconds"])
                    if tuning_result.get("measured_inner_tuning_seconds") is not None
                    else None
                )
                estimated_full_inner_tuning_seconds = (
                    float(tuning_result["estimated_full_inner_tuning_seconds"])
                    if tuning_result.get("estimated_full_inner_tuning_seconds") is not None
                    else None
                )
                estimated_full_tuned_search_seconds = (
                    float(tuning_result["estimated_full_tuned_search_seconds"])
                    if tuning_result.get("estimated_full_tuned_search_seconds") is not None
                    else None
                )
                tuning_split_scale_seconds = (
                    float(tuning_result["tuning_split_scale_seconds"])
                    if tuning_result.get("tuning_split_scale_seconds") is not None
                    else None
                )
                tuning_candidate_fit_seconds = (
                    float(tuning_result["tuning_candidate_fit_seconds"])
                    if tuning_result.get("tuning_candidate_fit_seconds") is not None
                    else None
                )
                tuning_candidate_predict_seconds = (
                    float(tuning_result["tuning_candidate_predict_seconds"])
                    if tuning_result.get("tuning_candidate_predict_seconds") is not None
                    else None
                )
                tuning_refit_elapsed_seconds = (
                    float(tuning_result["tuning_refit_elapsed_seconds"])
                    if tuning_result.get("tuning_refit_elapsed_seconds") is not None
                    else None
                )
                cv_mean_fit_time_seconds = (
                    float(tuning_result["cv_mean_fit_time_seconds"])
                    if tuning_result.get("cv_mean_fit_time_seconds") is not None
                    else None
                )
                cv_std_fit_time_seconds = (
                    float(tuning_result["cv_std_fit_time_seconds"])
                    if tuning_result.get("cv_std_fit_time_seconds") is not None
                    else None
                )
                cv_mean_score_time_seconds = (
                    float(tuning_result["cv_mean_score_time_seconds"])
                    if tuning_result.get("cv_mean_score_time_seconds") is not None
                    else None
                )
                cv_std_score_time_seconds = (
                    float(tuning_result["cv_std_score_time_seconds"])
                    if tuning_result.get("cv_std_score_time_seconds") is not None
                    else None
                )
                best_score = float(tuning_result["best_score"])
                best_params_json = str(tuning_result["best_params_json"])
                tuning_executor = str(tuning_result["tuning_executor"])
                tuning_executor_fallback_reason = (
                    str(tuning_result["tuning_executor_fallback_reason"])
                    if isinstance(tuning_result.get("tuning_executor_fallback_reason"), str)
                    and str(tuning_result.get("tuning_executor_fallback_reason")).strip()
                    else None
                )
                specialized_linearsvc_tuning_used = bool(
                    tuning_result.get("specialized_linearsvc_tuning_used", False)
                )
                specialized_logreg_tuning_used = bool(
                    tuning_result.get("specialized_logreg_tuning_used", False)
                )
                tuning_progress_event_count = (
                    int(tuning_result["tuning_progress_event_count"])
                    if tuning_result.get("tuning_progress_event_count") is not None
                    else None
                )
                tuning_progress_total_units = (
                    int(tuning_result["tuning_progress_total_units"])
                    if tuning_result.get("tuning_progress_total_units") is not None
                    else None
                )
                tuning_rows.append(
                    {
                        "fold": int(fold_index),
                        "status": "tuned",
                        "model": section_input.model,
                        "best_score": float(best_score),
                        "best_params_json": best_params_json,
                        "n_candidates": int(tuned_search_candidate_count),
                        "configured_candidate_count": int(tuned_search_configured_candidate_count),
                        "profiled_candidate_count": int(tuned_search_profiled_candidate_count),
                        "configured_inner_fold_count": int(
                            tuned_search_configured_inner_fold_count
                        ),
                        "profiled_inner_fold_count": int(tuned_search_profiled_inner_fold_count),
                        "tuning_extrapolation_applied": bool(tuning_extrapolation_applied),
                        "measured_inner_tuning_seconds": measured_inner_tuning_seconds,
                        "estimated_full_inner_tuning_seconds": estimated_full_inner_tuning_seconds,
                        "estimated_full_tuned_search_seconds": estimated_full_tuned_search_seconds,
                        "tuning_executor": str(tuning_executor),
                        "tuning_executor_fallback_reason": tuning_executor_fallback_reason,
                        "specialized_linearsvc_tuning_used": bool(
                            specialized_linearsvc_tuning_used
                        ),
                        "specialized_logreg_tuning_used": bool(specialized_logreg_tuning_used),
                        "tuning_progress_event_count": (
                            int(tuning_progress_event_count)
                            if tuning_progress_event_count is not None
                            else pd.NA
                        ),
                        "tuning_progress_total_units": (
                            int(tuning_progress_total_units)
                            if tuning_progress_total_units is not None
                            else pd.NA
                        ),
                        "tuning_split_scale_seconds": tuning_split_scale_seconds,
                        "tuning_candidate_fit_seconds": tuning_candidate_fit_seconds,
                        "tuning_candidate_predict_seconds": tuning_candidate_predict_seconds,
                        "tuning_refit_elapsed_seconds": tuning_refit_elapsed_seconds,
                        "search_space_id": resolved_tuning_search_space_id,
                        "search_space_version": resolved_tuning_search_space_version,
                        "primary_metric_name": section_input.primary_metric_name,
                        "inner_group_field": section_input.tuning_inner_group_field,
                        "estimator_fit_elapsed_seconds": pd.NA,
                        "tuned_search_elapsed_seconds": tuned_search_elapsed_seconds,
                        "cv_mean_fit_time_seconds": cv_mean_fit_time_seconds,
                        "cv_std_fit_time_seconds": cv_std_fit_time_seconds,
                        "cv_mean_score_time_seconds": cv_mean_score_time_seconds,
                        "cv_std_score_time_seconds": cv_std_score_time_seconds,
                    }
                )
        else:
            fit_payload = run_model_fit_executor(
                executor_id=model_fit_executor_id,
                estimator=estimator,
                x_train=section_input.x_matrix[train_idx],
                y_train=y[train_idx],
            )
            estimator = fit_payload["estimator"]
            estimator_fit_elapsed_seconds = float(fit_payload["estimator_fit_elapsed_seconds"])
        fold_backend_metadata = _extract_fold_compute_runtime_metadata(estimator)
        if fold_backend_metadata is not None:
            fold_compute_runtime_metadata.append(fold_backend_metadata)

        if interpretability_enabled:
            if interpretability_dir is None:
                raise ValueError("Interpretability directory was not initialized.")
            coef_array, intercept_array, class_labels = (
                section_input.extract_linear_coefficients_fn(estimator)
            )
            coef_path = interpretability_dir / f"fold_{fold_index:03d}_coefficients.npz"
            np.savez_compressed(
                coef_path,
                coefficients=coef_array.astype(np.float32, copy=False),
                intercept=intercept_array.astype(np.float32, copy=False),
                class_labels=np.asarray(class_labels, dtype=np.str_),
                feature_index=np.arange(coef_array.shape[1], dtype=np.int32),
            )
            interpretability_fold_rows.append(
                {
                    "fold": fold_index,
                    "experiment_mode": section_input.cv_mode,
                    "subject": str(section_input.subject),
                    "held_out_test_sessions": "|".join(test_sessions),
                    "model": section_input.model,
                    "target": section_input.target_column,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "coef_rows": int(coef_array.shape[0]),
                    "n_features": int(coef_array.shape[1]),
                    "coef_shape": json.dumps(list(coef_array.shape)),
                    "intercept_shape": json.dumps(list(intercept_array.shape)),
                    "class_labels": json.dumps(class_labels),
                    "coefficient_file": str(coef_path.resolve()),
                    "seed": int(section_input.seed),
                    "run_id": section_input.run_id,
                    "config_file": section_input.config_filename,
                }
            )
            interpretability_vectors.append(coef_array.reshape(-1).astype(np.float64))

        y_pred = estimator.predict(section_input.x_matrix[test_idx])
        y_true = y[test_idx]
        score_payload = section_input.scores_for_predictions_fn(
            estimator=estimator, x_test=section_input.x_matrix[test_idx]
        )

        fold_rows.append(
            {
                "fold": fold_index,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_groups": "|".join(sorted(np.unique(groups[test_idx]).tolist())),
                "experiment_mode": section_input.cv_mode,
                "subject": (
                    str(section_input.subject)
                    if section_input.cv_mode == "within_subject_loso_session"
                    else pd.NA
                ),
                "train_subject": (
                    str(section_input.train_subject)
                    if section_input.cv_mode == "frozen_cross_person_transfer"
                    else pd.NA
                ),
                "test_subject": (
                    str(section_input.test_subject)
                    if section_input.cv_mode == "frozen_cross_person_transfer"
                    else pd.NA
                ),
                "train_sessions": "|".join(train_sessions),
                "test_sessions": "|".join(test_sessions),
                "target": section_input.target_column,
                "model": section_input.model,
                "seed": int(section_input.seed),
                "estimator_fit_elapsed_seconds": estimator_fit_elapsed_seconds,
                "tuned_search_elapsed_seconds": tuned_search_elapsed_seconds,
                "tuned_search_candidate_count": int(tuned_search_candidate_count),
                "configured_candidate_count": int(tuned_search_configured_candidate_count),
                "profiled_candidate_count": int(tuned_search_profiled_candidate_count),
                "configured_inner_fold_count": int(tuned_search_configured_inner_fold_count),
                "profiled_inner_fold_count": int(tuned_search_profiled_inner_fold_count),
                "tuning_extrapolation_applied": bool(tuning_extrapolation_applied),
                "measured_inner_tuning_seconds": measured_inner_tuning_seconds,
                "estimated_full_inner_tuning_seconds": estimated_full_inner_tuning_seconds,
                "estimated_full_tuned_search_seconds": estimated_full_tuned_search_seconds,
                "tuning_executor": str(tuning_executor),
                "tuning_executor_fallback_reason": tuning_executor_fallback_reason,
                "specialized_linearsvc_tuning_used": bool(specialized_linearsvc_tuning_used),
                "specialized_logreg_tuning_used": bool(specialized_logreg_tuning_used),
                "tuning_progress_event_count": tuning_progress_event_count,
                "tuning_progress_total_units": tuning_progress_total_units,
                "tuning_split_scale_seconds": tuning_split_scale_seconds,
                "tuning_candidate_fit_seconds": tuning_candidate_fit_seconds,
                "tuning_candidate_predict_seconds": tuning_candidate_predict_seconds,
                "tuning_refit_elapsed_seconds": tuning_refit_elapsed_seconds,
                "cv_mean_fit_time_seconds": cv_mean_fit_time_seconds,
                "cv_std_fit_time_seconds": cv_std_fit_time_seconds,
                "cv_mean_score_time_seconds": cv_mean_score_time_seconds,
                "cv_std_score_time_seconds": cv_std_score_time_seconds,
                "accuracy": classification_metric_score(
                    y_true=y_true,
                    y_pred=y_pred,
                    metric_name="accuracy",
                ),
                "balanced_accuracy": classification_metric_score(
                    y_true=y_true,
                    y_pred=y_pred,
                    metric_name="balanced_accuracy",
                ),
                "macro_f1": classification_metric_score(
                    y_true=y_true,
                    y_pred=y_pred,
                    metric_name="macro_f1",
                ),
            }
        )

        split_rows.append(
            {
                "fold": fold_index,
                "experiment_mode": section_input.cv_mode,
                "subject": (
                    str(section_input.subject)
                    if section_input.cv_mode == "within_subject_loso_session"
                    else pd.NA
                ),
                "train_subject": (
                    str(section_input.train_subject)
                    if section_input.cv_mode == "frozen_cross_person_transfer"
                    else pd.NA
                ),
                "test_subject": (
                    str(section_input.test_subject)
                    if section_input.cv_mode == "frozen_cross_person_transfer"
                    else pd.NA
                ),
                "train_subjects": "|".join(train_subjects),
                "test_subjects": "|".join(test_subjects),
                "train_sessions": "|".join(train_sessions),
                "test_sessions": "|".join(test_sessions),
                "train_sample_count": int(len(train_idx)),
                "test_sample_count": int(len(test_idx)),
                "target": section_input.target_column,
                "model": section_input.model,
                "seed": int(section_input.seed),
            }
        )

        for row_idx in range(len(test_meta)):
            fold_row = test_meta.loc[row_idx]
            prediction_rows.append(
                {
                    "fold": fold_index,
                    "sample_id": str(fold_row["sample_id"]),
                    "y_true": str(y_true[row_idx]),
                    "y_pred": str(y_pred[row_idx]),
                    "decision_value": score_payload["decision_value"][row_idx],
                    "decision_vector": score_payload["decision_vector"][row_idx],
                    "proba_value": score_payload["proba_value"][row_idx],
                    "proba_vector": score_payload["proba_vector"][row_idx],
                    "subject": fold_row["subject"],
                    "session": fold_row["session"],
                    "bas": fold_row["bas"],
                    "task": fold_row["task"],
                    "modality": fold_row["modality"],
                    "emotion": fold_row.get("emotion", pd.NA),
                    "coarse_affect": fold_row.get("coarse_affect", pd.NA),
                    "experiment_mode": section_input.cv_mode,
                    "train_subject": (
                        str(section_input.train_subject)
                        if section_input.cv_mode == "frozen_cross_person_transfer"
                        else pd.NA
                    ),
                    "test_subject": (
                        str(section_input.test_subject)
                        if section_input.cv_mode == "frozen_cross_person_transfer"
                        else pd.NA
                    ),
                }
            )

        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())
        fold_rows[-1]["outer_fold_elapsed_seconds"] = float(perf_counter() - outer_fold_start)
        emit_progress(
            section_input.progress_callback,
            stage="fold",
            message=f"finished outer fold {fold_index + 1}/{total_outer_folds}",
            completed_units=float(fold_index + 1),
            total_units=float(total_outer_folds),
            metadata={
                "run_id": str(section_input.run_id),
                "model": str(section_input.model),
                "target": str(section_input.target_column),
                "cv_mode": str(section_input.cv_mode),
                "fold_index": int(fold_index + 1),
                "total_folds": int(total_outer_folds),
                "outer_fold_elapsed_seconds": float(fold_rows[-1]["outer_fold_elapsed_seconds"]),
            },
        )

    tuned_rows = [row for row in tuning_rows if str(row.get("status")) == "tuned"]
    tuning_summary = {
        "methodology_policy_name": methodology_policy_name,
        "tuning_enabled": bool(tuning_enabled),
        "status": (
            "performed"
            if tuning_enabled and any(row["status"] == "tuned" for row in tuning_rows)
            else "disabled"
        ),
        "primary_metric_name": section_input.primary_metric_name,
        "search_space_id": resolved_tuning_search_space_id,
        "search_space_version": resolved_tuning_search_space_version,
        "inner_cv_scheme": section_input.tuning_inner_cv_scheme,
        "inner_group_field": section_input.tuning_inner_group_field,
        "total_outer_folds": int(len(splits)),
        "planned_outer_folds": int(planned_outer_folds),
        "executed_outer_folds": int(len(splits)),
        "profiling_outer_fold_cap": (int(max_outer_folds) if max_outer_folds is not None else None),
        "profiling_only": bool(section_input.profiling_only),
        "profile_inner_folds_override": (
            int(section_input.profile_inner_folds)
            if section_input.profile_inner_folds is not None
            else None
        ),
        "profile_tuning_candidates_override": (
            int(section_input.profile_tuning_candidates)
            if section_input.profile_tuning_candidates is not None
            else None
        ),
        "n_tuned_folds": int(sum(row["status"] == "tuned" for row in tuning_rows)),
        "n_skipped_folds": int(sum(row["status"] != "tuned" for row in tuning_rows)),
        "specialized_linearsvc_tuning_used": bool(
            any(bool(row.get("specialized_linearsvc_tuning_used")) for row in tuned_rows)
        ),
        "specialized_logreg_tuning_used": bool(
            any(bool(row.get("specialized_logreg_tuning_used")) for row in tuned_rows)
        ),
        "tuning_executor_ids": sorted(
            {
                str(row.get("tuning_executor"))
                for row in tuned_rows
                if isinstance(row.get("tuning_executor"), str) and str(row.get("tuning_executor"))
            }
        ),
        "tuning_progress_event_count_total": int(
            sum(
                int(row.get("tuning_progress_event_count"))
                for row in tuned_rows
                if row.get("tuning_progress_event_count") is not None
                and not pd.isna(row.get("tuning_progress_event_count"))
            )
        ),
        "tuning_progress_total_units_total": int(
            sum(
                int(row.get("tuning_progress_total_units"))
                for row in tuned_rows
                if row.get("tuning_progress_total_units") is not None
                and not pd.isna(row.get("tuning_progress_total_units"))
            )
        ),
        "configured_inner_fold_count_total": int(
            sum(int(row.get("configured_inner_fold_count") or 0) for row in tuned_rows)
        ),
        "profiled_inner_fold_count_total": int(
            sum(int(row.get("profiled_inner_fold_count") or 0) for row in tuned_rows)
        ),
        "configured_candidate_count_total": int(
            sum(int(row.get("configured_candidate_count") or 0) for row in tuned_rows)
        ),
        "profiled_candidate_count_total": int(
            sum(int(row.get("profiled_candidate_count") or 0) for row in tuned_rows)
        ),
        "tuning_extrapolation_applied": bool(
            any(bool(row.get("tuning_extrapolation_applied")) for row in tuned_rows)
        ),
        "best_params_path": str(tuning_best_params_path.resolve()),
        "timing_totals_seconds": {
            "outer_fold_total": float(
                sum(float(row.get("outer_fold_elapsed_seconds") or 0.0) for row in fold_rows)
            ),
            "estimator_fit_total": float(
                sum(float(row.get("estimator_fit_elapsed_seconds") or 0.0) for row in fold_rows)
            ),
            "tuned_search_total": float(
                sum(float(row.get("tuned_search_elapsed_seconds") or 0.0) for row in fold_rows)
            ),
            "measured_inner_tuning_total": float(
                sum(float(row.get("measured_inner_tuning_seconds") or 0.0) for row in tuned_rows)
            ),
            "estimated_full_tuned_search_total": float(
                sum(
                    float(row.get("estimated_full_tuned_search_seconds") or 0.0)
                    for row in tuned_rows
                )
            ),
            "tuning_split_scale_total": float(
                sum(float(row.get("tuning_split_scale_seconds") or 0.0) for row in tuned_rows)
            ),
            "tuning_candidate_fit_total": float(
                sum(float(row.get("tuning_candidate_fit_seconds") or 0.0) for row in tuned_rows)
            ),
            "tuning_candidate_predict_total": float(
                sum(float(row.get("tuning_candidate_predict_seconds") or 0.0) for row in tuned_rows)
            ),
        },
    }
    tuning_summary_path.write_text(
        f"{json.dumps(tuning_summary, indent=2)}\n",
        encoding="utf-8",
    )
    tuning_frame = pd.DataFrame(tuning_rows)
    if tuning_frame.empty:
        tuning_frame = pd.DataFrame(
            columns=[
                "fold",
                "status",
                "model",
                "best_score",
                "best_params_json",
                "n_candidates",
                "configured_candidate_count",
                "profiled_candidate_count",
                "configured_inner_fold_count",
                "profiled_inner_fold_count",
                "tuning_extrapolation_applied",
                "measured_inner_tuning_seconds",
                "estimated_full_inner_tuning_seconds",
                "estimated_full_tuned_search_seconds",
                "tuning_executor",
                "tuning_executor_fallback_reason",
                "specialized_linearsvc_tuning_used",
                "specialized_logreg_tuning_used",
                "tuning_progress_event_count",
                "tuning_progress_total_units",
                "tuning_split_scale_seconds",
                "tuning_candidate_fit_seconds",
                "tuning_candidate_predict_seconds",
                "tuning_refit_elapsed_seconds",
                "search_space_id",
                "search_space_version",
                "primary_metric_name",
                "inner_group_field",
                "estimator_fit_elapsed_seconds",
                "tuned_search_elapsed_seconds",
                "cv_mean_fit_time_seconds",
                "cv_std_fit_time_seconds",
                "cv_mean_score_time_seconds",
                "cv_std_score_time_seconds",
            ]
        )
    tuning_frame.to_csv(tuning_best_params_path, index=False)

    fit_timing_summary = {
        "status": "captured",
        "methodology_policy_name": methodology_policy_name,
        "model": section_input.model,
        "cv_mode": section_input.cv_mode,
        "n_folds": int(len(fold_rows)),
        "n_tuned_folds": int(sum(row["status"] == "tuned" for row in tuning_rows)),
        "n_untuned_or_skipped_folds": int(sum(row["status"] != "tuned" for row in tuning_rows)),
        "totals_seconds": {
            "outer_fold": float(
                sum(float(row.get("outer_fold_elapsed_seconds") or 0.0) for row in fold_rows)
            ),
            "estimator_fit": float(
                sum(float(row.get("estimator_fit_elapsed_seconds") or 0.0) for row in fold_rows)
            ),
            "tuned_search": float(
                sum(float(row.get("tuned_search_elapsed_seconds") or 0.0) for row in fold_rows)
            ),
            "measured_inner_tuning": float(
                sum(float(row.get("measured_inner_tuning_seconds") or 0.0) for row in fold_rows)
            ),
            "estimated_full_tuned_search": float(
                sum(
                    float(row.get("estimated_full_tuned_search_seconds") or 0.0)
                    for row in fold_rows
                )
            ),
        },
        "fold_timings": [
            {
                "fold": int(row.get("fold", -1)),
                "outer_fold_elapsed_seconds": float(row.get("outer_fold_elapsed_seconds") or 0.0),
                "estimator_fit_elapsed_seconds": (
                    float(row.get("estimator_fit_elapsed_seconds"))
                    if row.get("estimator_fit_elapsed_seconds") is not None
                    else None
                ),
                "tuned_search_elapsed_seconds": (
                    float(row.get("tuned_search_elapsed_seconds"))
                    if row.get("tuned_search_elapsed_seconds") is not None
                    else None
                ),
                "tuned_search_candidate_count": int(row.get("tuned_search_candidate_count") or 0),
                "configured_candidate_count": int(row.get("configured_candidate_count") or 0),
                "profiled_candidate_count": int(row.get("profiled_candidate_count") or 0),
                "configured_inner_fold_count": int(row.get("configured_inner_fold_count") or 0),
                "profiled_inner_fold_count": int(row.get("profiled_inner_fold_count") or 0),
                "tuning_extrapolation_applied": bool(row.get("tuning_extrapolation_applied")),
                "measured_inner_tuning_seconds": (
                    float(row.get("measured_inner_tuning_seconds"))
                    if row.get("measured_inner_tuning_seconds") is not None
                    else None
                ),
                "estimated_full_inner_tuning_seconds": (
                    float(row.get("estimated_full_inner_tuning_seconds"))
                    if row.get("estimated_full_inner_tuning_seconds") is not None
                    else None
                ),
                "estimated_full_tuned_search_seconds": (
                    float(row.get("estimated_full_tuned_search_seconds"))
                    if row.get("estimated_full_tuned_search_seconds") is not None
                    else None
                ),
                "tuning_executor": (
                    str(row.get("tuning_executor"))
                    if row.get("tuning_executor") is not None
                    else None
                ),
                "tuning_executor_fallback_reason": (
                    str(row.get("tuning_executor_fallback_reason"))
                    if row.get("tuning_executor_fallback_reason") is not None
                    else None
                ),
                "specialized_linearsvc_tuning_used": bool(
                    row.get("specialized_linearsvc_tuning_used")
                ),
                "specialized_logreg_tuning_used": bool(row.get("specialized_logreg_tuning_used")),
                "tuning_progress_event_count": (
                    int(row.get("tuning_progress_event_count"))
                    if row.get("tuning_progress_event_count") is not None
                    else None
                ),
                "tuning_progress_total_units": (
                    int(row.get("tuning_progress_total_units"))
                    if row.get("tuning_progress_total_units") is not None
                    else None
                ),
                "tuning_split_scale_seconds": (
                    float(row.get("tuning_split_scale_seconds"))
                    if row.get("tuning_split_scale_seconds") is not None
                    else None
                ),
                "tuning_candidate_fit_seconds": (
                    float(row.get("tuning_candidate_fit_seconds"))
                    if row.get("tuning_candidate_fit_seconds") is not None
                    else None
                ),
                "tuning_candidate_predict_seconds": (
                    float(row.get("tuning_candidate_predict_seconds"))
                    if row.get("tuning_candidate_predict_seconds") is not None
                    else None
                ),
                "tuning_refit_elapsed_seconds": (
                    float(row.get("tuning_refit_elapsed_seconds"))
                    if row.get("tuning_refit_elapsed_seconds") is not None
                    else None
                ),
                "cv_mean_fit_time_seconds": (
                    float(row.get("cv_mean_fit_time_seconds"))
                    if row.get("cv_mean_fit_time_seconds") is not None
                    else None
                ),
                "cv_std_fit_time_seconds": (
                    float(row.get("cv_std_fit_time_seconds"))
                    if row.get("cv_std_fit_time_seconds") is not None
                    else None
                ),
                "cv_mean_score_time_seconds": (
                    float(row.get("cv_mean_score_time_seconds"))
                    if row.get("cv_mean_score_time_seconds") is not None
                    else None
                ),
                "cv_std_score_time_seconds": (
                    float(row.get("cv_std_score_time_seconds"))
                    if row.get("cv_std_score_time_seconds") is not None
                    else None
                ),
            }
            for row in fold_rows
        ],
    }
    fit_timing_summary_path.write_text(
        f"{json.dumps(fit_timing_summary, indent=2)}\n",
        encoding="utf-8",
    )
    emit_progress(
        section_input.progress_callback,
        stage="model_fit",
        message="finished model fit",
        completed_units=float(total_outer_folds),
        total_units=float(total_outer_folds),
        metadata={
            "run_id": str(section_input.run_id),
            "model": str(section_input.model),
            "target": str(section_input.target_column),
            "cv_mode": str(section_input.cv_mode),
            "total_folds": int(total_outer_folds),
        },
    )

    return {
        "y": y,
        "splits": splits,
        "fold_rows": fold_rows,
        "split_rows": split_rows,
        "prediction_rows": prediction_rows,
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
        "interpretability_enabled": interpretability_enabled,
        "interpretability_fold_rows": interpretability_fold_rows,
        "interpretability_vectors": interpretability_vectors,
        "interpretability_fold_artifacts_path": section_input.report_dir
        / "interpretability_fold_explanations.csv",
        "interpretability_summary_path": section_input.report_dir / "interpretability_summary.json",
        "tuning_summary": tuning_summary,
        "tuning_records": tuning_rows,
        "tuning_summary_path": tuning_summary_path,
        "tuning_best_params_path": tuning_best_params_path,
        "fit_timing_summary": fit_timing_summary,
        "fit_timing_summary_path": fit_timing_summary_path,
        "compute_runtime_metadata": _aggregate_compute_runtime_metadata(
            fold_compute_runtime_metadata
        ),
    }


def execute_interpretability(section_input: InterpretabilityInput) -> dict[str, Any]:
    caution_text = (
        "Linear coefficients are reported as model-behavior evidence only and must not be "
        "interpreted as direct neural localization."
    )
    if section_input.interpretability_enabled:
        pd.DataFrame(section_input.interpretability_fold_rows).to_csv(
            section_input.fold_artifacts_path, index=False
        )
        interpretability_summary: dict[str, Any] = {
            "enabled": True,
            "performed": True,
            "status": "performed",
            "reason": None,
            "caution": caution_text,
            "experiment_mode": section_input.cv_mode,
            "subject": str(section_input.subject),
            "model": section_input.model,
            "target": section_input.target_column,
            "n_fold_artifacts": int(len(section_input.interpretability_fold_rows)),
            "fold_artifacts_path": str(section_input.fold_artifacts_path.resolve()),
            "stability": section_input.compute_interpretability_stability_fn(
                section_input.interpretability_vectors
            ),
        }
    else:
        not_applicable_reason = (
            "Interpretability export is enabled only for within_subject_loso_session."
        )
        if (
            section_input.cv_mode == "within_subject_loso_session"
            and not model_supports_linear_interpretability(section_input.model)
        ):
            not_applicable_reason = (
                "Interpretability export currently supports linear coefficient models only."
            )
        interpretability_summary = {
            "enabled": False,
            "performed": False,
            "status": "not_applicable",
            "reason": not_applicable_reason,
            "caution": caution_text,
            "experiment_mode": section_input.cv_mode,
            "subject": None,
            "model": section_input.model,
            "target": section_input.target_column,
            "n_fold_artifacts": 0,
            "fold_artifacts_path": None,
            "stability": None,
        }
    section_input.summary_path.write_text(
        f"{json.dumps(interpretability_summary, indent=2)}\n",
        encoding="utf-8",
    )
    return interpretability_summary


def _compute_subgroup_metrics(
    prediction_rows: list[dict[str, Any]],
    *,
    subgroup_dimensions: list[str],
    min_samples_per_group: int,
    min_classes_per_group: int,
    report_small_groups: bool,
    strict_insufficient_data_rows: bool,
) -> list[dict[str, Any]]:
    if not prediction_rows:
        return []
    frame = pd.DataFrame(prediction_rows)
    subgroup_rows: list[dict[str, Any]] = []
    for requested_dimension in subgroup_dimensions:
        if requested_dimension == "label":
            group_column = "y_true"
        else:
            group_column = requested_dimension
        if group_column not in frame.columns:
            continue
        for group_value, subset in frame.groupby(group_column, dropna=False):
            n_samples = int(len(subset))
            y_true = subset["y_true"].astype(str).tolist()
            y_pred = subset["y_pred"].astype(str).tolist()
            n_classes = int(pd.Series(y_true).nunique(dropna=False))
            class_distribution = (
                pd.Series(y_true).value_counts(dropna=False).sort_index().astype(int).to_dict()
            )
            interpretable = n_samples >= int(min_samples_per_group) and n_classes >= int(
                min_classes_per_group
            )
            insufficient_reasons: list[str] = []
            if n_samples < int(min_samples_per_group):
                insufficient_reasons.append("min_samples")
            if n_classes < int(min_classes_per_group):
                insufficient_reasons.append("min_classes")
            if not interpretable and not (
                bool(report_small_groups) or bool(strict_insufficient_data_rows)
            ):
                continue
            # For single-class or undersized slices, classification scores are not
            # scientifically stable. Emit descriptive rows with null metrics.
            can_compute_metrics = bool(interpretable)
            subgroup_rows.append(
                {
                    "subgroup_key": requested_dimension,
                    "subgroup_value": str(group_value),
                    "n_samples": n_samples,
                    "n_classes": n_classes,
                    "class_distribution_json": json.dumps(class_distribution, sort_keys=True),
                    "insufficient_data_reasons": "|".join(insufficient_reasons),
                    "status": "ok" if interpretable else "insufficient_data",
                    "interpretable": bool(interpretable),
                    "balanced_accuracy": (
                        classification_metric_score(
                            y_true=y_true,
                            y_pred=y_pred,
                            metric_name="balanced_accuracy",
                        )
                        if can_compute_metrics
                        else None
                    ),
                    "macro_f1": (
                        classification_metric_score(
                            y_true=y_true,
                            y_pred=y_pred,
                            metric_name="macro_f1",
                        )
                        if can_compute_metrics
                        else None
                    ),
                    "accuracy": (
                        classification_metric_score(
                            y_true=y_true,
                            y_pred=y_pred,
                            metric_name="accuracy",
                        )
                        if can_compute_metrics
                        else None
                    ),
                }
            )
    return subgroup_rows


def _fallback_feature_qc_rows_from_matrix(
    x_matrix: np.ndarray,
    *,
    sample_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    matrix = np.asarray(x_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("x_matrix must be a 2D matrix for feature QC fallback.")
    rows: list[dict[str, Any]] = []
    for row_index in range(int(matrix.shape[0])):
        vector = np.asarray(matrix[row_index], dtype=np.float64)
        sample_id = (
            str(sample_ids[row_index]).strip()
            if sample_ids is not None and row_index < len(sample_ids)
            else ""
        )
        if not sample_id:
            sample_id = f"selected_{row_index:06d}"
        qc_payload = compute_sample_feature_qc(
            vector_before_repair=vector,
            vector_after_repair=vector,
        )
        rows.append(
            {
                "group_id": "selected_subset",
                "sample_id": sample_id,
                **qc_payload,
            }
        )
    return rows


def _selected_feature_qc_rows(
    metadata_df: pd.DataFrame,
    *,
    x_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    if metadata_df.empty:
        return _fallback_feature_qc_rows_from_matrix(x_matrix)
    sample_ids_from_metadata = [
        str(value).strip()
        for value in metadata_df.get("sample_id", pd.Series(dtype=str)).tolist()
    ]
    missing_columns = [
        field_name
        for field_name in FEATURE_QC_SAMPLE_FIELDS
        if field_name not in metadata_df.columns
    ]
    if missing_columns:
        # ROI and other non-cache feature spaces can provide valid matrices
        # without cache-derived per-sample QC columns; derive QC directly.
        return _fallback_feature_qc_rows_from_matrix(
            x_matrix,
            sample_ids=sample_ids_from_metadata,
        )
    rows: list[dict[str, Any]] = []
    for _, row in metadata_df.iterrows():
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(
                "Selected metadata contains blank sample_id while building feature QC."
            )
        payload: dict[str, Any] = {
            "group_id": "selected_subset",
            "sample_id": sample_id,
        }
        for field_name in FEATURE_QC_SAMPLE_FIELDS:
            payload[field_name] = row[field_name]
        rows.append(payload)
    if not rows:
        raise ValueError("Selected metadata is empty while building feature QC summary.")
    return rows


def _build_feature_qc_summary(
    *,
    metadata_df: pd.DataFrame,
    x_matrix: np.ndarray,
    feature_recipe_id: str,
) -> dict[str, Any]:
    summary = summarize_group_feature_qc(
        _selected_feature_qc_rows(
            metadata_df,
            x_matrix=x_matrix,
        )
    )
    return {
        "feature_recipe_id": str(feature_recipe_id),
        "n_selected_samples": int(summary.get("n_samples", 0)),
        "n_samples_with_any_repair": int(summary.get("n_samples_with_any_repair", 0)),
        "max_repair_fraction": float(summary.get("max_repair_fraction", 0.0)),
        "mean_repair_fraction": float(summary.get("mean_repair_fraction", 0.0)),
        "n_all_zero_vectors": int(summary.get("n_all_zero_vectors", 0)),
        "n_constant_vectors": int(summary.get("n_constant_vectors", 0)),
        "mean_vector_std_after_repair": float(summary.get("mean_vector_std_after_repair", 0.0)),
        "min_vector_std_after_repair": float(summary.get("min_vector_std_after_repair", 0.0)),
    }


def _evaluate_feature_quality_policy(
    *,
    summary: dict[str, Any],
    policy: FeatureQualityPolicy,
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    blocking: list[dict[str, Any]] = []

    n_samples_with_any_repair = int(summary.get("n_samples_with_any_repair", 0))
    n_all_zero_vectors = int(summary.get("n_all_zero_vectors", 0))
    n_constant_vectors = int(summary.get("n_constant_vectors", 0))

    if policy.warn_on_any_nonfinite_repair and n_samples_with_any_repair > 0:
        warnings.append(
            {
                "code": "feature_qc_nonfinite_repair_detected",
                "count": int(n_samples_with_any_repair),
            }
        )
    if policy.warn_on_any_all_zero_vector and n_all_zero_vectors > 0:
        warnings.append(
            {
                "code": "feature_qc_all_zero_vectors_detected",
                "count": int(n_all_zero_vectors),
            }
        )
    if policy.warn_on_any_constant_vector and n_constant_vectors > 0:
        warnings.append(
            {
                "code": "feature_qc_constant_vectors_detected",
                "count": int(n_constant_vectors),
            }
        )
    if policy.fail_on_any_all_zero_vector and n_all_zero_vectors > 0:
        blocking.append(
            {
                "code": "feature_qc_all_zero_vectors_blocking",
                "count": int(n_all_zero_vectors),
            }
        )
    if policy.fail_on_any_constant_vector and n_constant_vectors > 0:
        blocking.append(
            {
                "code": "feature_qc_constant_vectors_blocking",
                "count": int(n_constant_vectors),
            }
        )
    return {
        "status": ("fail" if blocking else "warning" if warnings else "pass"),
        "warnings": warnings,
        "blocking_issues": blocking,
    }


def _resolve_grouped_nested_tuning_executor(
    section_input: EvaluationInput,
) -> tuple[str, str | None]:
    model_name = str(section_input.model).strip().lower()
    default_tuning_executor_id = (
        TUNING_SKIPPED_CONTROL_EXECUTOR_ID if model_name == "dummy" else TUNING_GENERIC_EXECUTOR_ID
    )
    if model_name == "linearsvc":
        default_tuning_executor_id = SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID
    if model_name == "ridge":
        default_tuning_executor_id = SPECIALIZED_RIDGE_TUNING_EXECUTOR_ID
    tuning_executor_id = _assignment_executor_id(
        section_input.tuning_assignment,
        default_executor_id=default_tuning_executor_id,
    )
    fallback_executor_id = (
        str(section_input.tuning_fallback_executor_id).strip()
        if isinstance(section_input.tuning_fallback_executor_id, str)
        and str(section_input.tuning_fallback_executor_id).strip()
        else (
            TUNING_GENERIC_EXECUTOR_ID
            if tuning_executor_id
            in {
                SPECIALIZED_RIDGE_TUNING_EXECUTOR_ID,
                SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID,
                SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID,
            }
            else None
        )
    )
    return tuning_executor_id, fallback_executor_id


def _resolve_primary_metric_aggregation(aggregation: str) -> str:
    resolved = str(aggregation).strip()
    if resolved not in {"mean_fold_scores", "pooled_held_out_predictions"}:
        raise ValueError(
            "primary_metric_aggregation must be 'mean_fold_scores' or "
            "'pooled_held_out_predictions'."
        )
    return resolved


def _mean_fold_metric_value(
    *,
    fold_rows: list[dict[str, Any]],
    metric_name: str,
) -> float:
    metric_values: list[float] = []
    for fold_index, fold_row in enumerate(fold_rows):
        raw_value = fold_row.get(metric_name)
        if not isinstance(raw_value, (int, float, np.floating)):
            raise ValueError(
                "mean_fold_scores aggregation requires per-fold metric values in fold_rows; "
                f"missing metric='{metric_name}' at fold index {fold_index}."
            )
        metric_values.append(float(raw_value))
    if not metric_values:
        raise ValueError(
            f"mean_fold_scores aggregation requires at least one fold row for metric '{metric_name}'."
        )
    return float(np.mean(np.asarray(metric_values, dtype=np.float64)))


def _evaluate_grouped_nested_tuning_permutations(
    *,
    section_input: EvaluationInput,
    metric_name: str,
    observed_metric: float,
    primary_metric_aggregation: str,
    permutation_executor_id: str,
) -> dict[str, Any]:
    if section_input.metadata_df.empty:
        raise ValueError(
            "Grouped nested tuned permutation testing requires metadata_df to rebuild inner groups."
        )
    tuning_inner_cv_scheme = str(section_input.tuning_inner_cv_scheme or "").strip()
    if tuning_inner_cv_scheme != "grouped_leave_one_group_out":
        raise ValueError(
            "Grouped nested tuned permutation testing requires "
            "tuning_inner_cv_scheme='grouped_leave_one_group_out'."
        )
    tuning_inner_group_field = str(section_input.tuning_inner_group_field or "").strip()
    if not tuning_inner_group_field:
        raise ValueError(
            "Grouped nested tuned permutation testing requires tuning_inner_group_field."
        )
    if tuning_inner_group_field not in section_input.metadata_df.columns:
        raise ValueError(
            "Grouped nested tuned permutation testing inner group field is missing from metadata."
        )

    (
        resolved_tuning_search_space_id,
        resolved_tuning_search_space_version,
        resolved_param_grid,
    ) = resolve_tuning_search_space_for_model(
        model_name=section_input.model,
        search_space_id=section_input.tuning_search_space_id,
        search_space_version=section_input.tuning_search_space_version,
    )
    param_grid = dict(resolved_param_grid)
    candidate_params = list(ParameterGrid(param_grid))
    configured_candidate_count = int(len(candidate_params))
    if configured_candidate_count <= 0:
        raise ValueError("Grouped nested tuned permutation testing resolved an empty search space.")

    pipeline_template = _build_pipeline_template(
        build_pipeline_fn=section_input.build_pipeline_fn,
        model_name=section_input.model,
        seed=section_input.seed,
        feature_recipe_id=section_input.feature_recipe_id,
        preprocessing_strategy=section_input.preprocessing_strategy,
    )
    tuning_executor_id, fallback_executor_id = _resolve_grouped_nested_tuning_executor(section_input)
    permutation_count = int(section_input.n_permutations)
    rng = np.random.default_rng(int(section_input.seed))
    progress_base = {
        "run_id": str(section_input.run_id),
        "model": str(section_input.model),
        "target": str(section_input.target_column),
        "cv_mode": str(section_input.cv_mode),
    }
    emit_progress(
        section_input.progress_callback,
        stage="permutation",
        message=f"starting tuned permutation test with {permutation_count} permutations",
        completed_units=0.0,
        total_units=float(permutation_count),
        metadata=progress_base,
    )
    permutation_loop_start = perf_counter()
    permutation_scores: list[float] = []
    for permutation_index in range(permutation_count):
        emit_progress(
            section_input.progress_callback,
            stage="permutation",
            message=f"running permutation {permutation_index + 1}/{permutation_count}",
            completed_units=float(permutation_index),
            total_units=float(permutation_count),
            metadata={
                **progress_base,
                "permutation_index": int(permutation_index + 1),
                "n_permutations": permutation_count,
            },
        )
        y_true_all: list[str] = []
        y_pred_all: list[str] = []
        fold_metric_scores: list[float] = []
        for fold_index, (train_idx, test_idx) in enumerate(section_input.splits):
            train_idx_array = np.asarray(train_idx, dtype=np.int64)
            test_idx_array = np.asarray(test_idx, dtype=np.int64)
            x_outer_train = section_input.x_matrix[train_idx_array]
            y_outer_train = np.asarray(section_input.y[train_idx_array]).astype(str, copy=True)
            rng.shuffle(y_outer_train)
            inner_groups = (
                section_input.metadata_df.iloc[train_idx_array][tuning_inner_group_field]
                .astype(str)
                .to_numpy()
            )
            inner_splits = list(LeaveOneGroupOut().split(x_outer_train, y_outer_train, inner_groups))
            configured_inner_fold_count = int(len(inner_splits))
            if configured_inner_fold_count < 2:
                raise ValueError(
                    "Grouped nested tuned permutation testing requires at least two inner groups."
                )
            tuning_result = run_tuning_executor(
                executor_id=tuning_executor_id,
                fallback_executor_id=fallback_executor_id,
                pipeline_template=pipeline_template,
                x_outer_train=x_outer_train,
                y_outer_train=y_outer_train,
                inner_groups=inner_groups,
                inner_splits=inner_splits,
                param_grid=param_grid,
                candidate_params=candidate_params,
                configured_candidate_count=configured_candidate_count,
                configured_inner_fold_count=configured_inner_fold_count,
                profiled_candidate_count=configured_candidate_count,
                profiled_inner_fold_count=configured_inner_fold_count,
                primary_metric_name=section_input.primary_metric_name,
                progress_callback=section_input.progress_callback,
                progress_metadata={
                    **progress_base,
                    "permutation_index": int(permutation_index + 1),
                    "n_permutations": permutation_count,
                    "fold_index": int(fold_index + 1),
                    "total_folds": int(len(section_input.splits)),
                    "null_tuning": True,
                },
            )
            estimator = tuning_result["estimator"]
            y_true_fold = np.asarray(section_input.y[test_idx_array]).astype(str, copy=False)
            y_pred_fold = np.asarray(estimator.predict(section_input.x_matrix[test_idx_array])).astype(
                str, copy=False
            )
            if primary_metric_aggregation == "mean_fold_scores":
                fold_metric_scores.append(
                    float(
                        classification_metric_score(
                            y_true_fold,
                            y_pred_fold,
                            metric_name=metric_name,
                        )
                    )
                )
            else:
                y_true_all.extend(y_true_fold.tolist())
                y_pred_all.extend(y_pred_fold.tolist())
        permutation_scores.append(
            float(np.mean(np.asarray(fold_metric_scores, dtype=np.float64)))
            if primary_metric_aggregation == "mean_fold_scores"
            else float(classification_metric_score(y_true_all, y_pred_all, metric_name=metric_name))
        )
        emit_progress(
            section_input.progress_callback,
            stage="permutation",
            message=f"finished permutation {permutation_index + 1}/{permutation_count}",
            completed_units=float(permutation_index + 1),
            total_units=float(permutation_count),
            metadata={
                **progress_base,
                "permutation_index": int(permutation_index + 1),
                "n_permutations": permutation_count,
            },
        )
    permutation_loop_seconds = float(perf_counter() - permutation_loop_start)
    scores_array = np.asarray(permutation_scores, dtype=np.float64)
    p_value = float((np.count_nonzero(scores_array >= float(observed_metric)) + 1) / (len(scores_array) + 1))
    payload: dict[str, Any] = {
        "metric_name": str(metric_name),
        "primary_metric_aggregation": str(primary_metric_aggregation),
        "n_permutations": int(permutation_count),
        "observed_score": float(observed_metric),
        "permutation_seed": int(section_input.seed),
        "p_value": float(p_value),
        "null_summary": {
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "q25": float(np.quantile(scores_array, 0.25)),
            "q50": float(np.quantile(scores_array, 0.50)),
            "q75": float(np.quantile(scores_array, 0.75)),
        },
        "null_scores": [float(value) for value in permutation_scores],
        "observed_metric": float(observed_metric),
        "permutation_metric_mean": float(np.mean(scores_array)),
        "permutation_metric_std": float(np.std(scores_array)),
        "permutation_p_value": float(p_value),
        "execution_mode": "grouped_nested_tuning_reference",
        "backend_family": "sklearn_cpu",
        "estimator_name": "Pipeline",
        "shortcut_applied": False,
        "shortcut_reason": None,
        "shortcut_strategy": None,
        "n_folds": int(len(section_input.splits)),
        "fold_cache_build_seconds": 0.0,
        "permutation_loop_seconds": float(permutation_loop_seconds),
        "specialized_ridge_gpu_path_used": False,
        "specialized_ridge_gpu_fallback_reason": None,
        "permutation_batch_size": None,
        "fold_gpu_state_build_seconds": None,
        "fold_factorization_seconds": None,
        "batched_solve_seconds": None,
        "batched_predict_seconds": None,
        "permutation_executor_id": str(permutation_executor_id),
        "permutation_executor_fallback_reason": None,
        "tuning_reapplied_under_null": True,
        "null_matches_confirmatory_setup": True,
        "null_tuning_search_space_id": str(resolved_tuning_search_space_id),
        "null_tuning_search_space_version": str(resolved_tuning_search_space_version),
        "null_inner_cv_scheme": str(tuning_inner_cv_scheme),
        "null_inner_group_field": str(tuning_inner_group_field),
        "null_tuning_executor_id": str(tuning_executor_id),
        "null_tuning_fallback_executor_id": (
            str(fallback_executor_id) if isinstance(fallback_executor_id, str) else None
        ),
    }
    emit_progress(
        section_input.progress_callback,
        stage="permutation",
        message="finished tuned permutation test",
        completed_units=float(permutation_count),
        total_units=float(permutation_count),
        metadata=progress_base,
    )
    return payload


def execute_evaluation(section_input: EvaluationInput) -> dict[str, Any]:
    report_dir = section_input.metrics_path.parent
    subgroup_metrics_json_path = (
        section_input.subgroup_metrics_json_path
        if section_input.subgroup_metrics_json_path is not None
        else report_dir / "subgroup_metrics.json"
    )
    subgroup_metrics_csv_path = (
        section_input.subgroup_metrics_csv_path
        if section_input.subgroup_metrics_csv_path is not None
        else report_dir / "subgroup_metrics.csv"
    )
    tuning_summary_path = (
        section_input.tuning_summary_path
        if section_input.tuning_summary_path is not None
        else report_dir / "tuning_summary.json"
    )
    tuning_best_params_path = (
        section_input.tuning_best_params_path
        if section_input.tuning_best_params_path is not None
        else report_dir / "best_params_per_fold.csv"
    )
    calibration_summary_path = (
        section_input.calibration_summary_path
        if section_input.calibration_summary_path is not None
        else report_dir / "calibration_summary.json"
    )
    calibration_table_path = (
        section_input.calibration_table_path
        if section_input.calibration_table_path is not None
        else report_dir / "calibration_table.csv"
    )
    feature_qc_summary_path = (
        section_input.feature_qc_summary_path
        if section_input.feature_qc_summary_path is not None
        else report_dir / "feature_qc_summary.json"
    )
    feature_qc_selected_samples_path = (
        section_input.feature_qc_selected_samples_path
        if section_input.feature_qc_selected_samples_path is not None
        else report_dir / "feature_qc_selected_samples.csv"
    )
    feature_qc_summary = _build_feature_qc_summary(
        metadata_df=section_input.metadata_df,
        x_matrix=section_input.x_matrix,
        feature_recipe_id=section_input.feature_recipe_id,
    )
    feature_quality_policy = FeatureQualityPolicy.model_validate(
        section_input.feature_quality_policy or {}
    )
    feature_quality_status = _evaluate_feature_quality_policy(
        summary=feature_qc_summary,
        policy=feature_quality_policy,
    )
    if bool(section_input.emit_feature_qc_artifacts):
        feature_qc_summary_path.write_text(
            f"{json.dumps(feature_qc_summary, indent=2)}\n",
            encoding="utf-8",
        )
        if section_input.metadata_df.empty:
            fallback_rows = _fallback_feature_qc_rows_from_matrix(section_input.x_matrix)
            pd.DataFrame.from_records(fallback_rows).loc[
                :,
                ["sample_id", *FEATURE_QC_SAMPLE_FIELDS],
            ].to_csv(feature_qc_selected_samples_path, index=False)
        else:
            selected_qc_columns = [
                column_name
                for column_name in (
                    "sample_id",
                    "subject",
                    "session",
                    "bas",
                    "task",
                    "modality",
                    *FEATURE_QC_SAMPLE_FIELDS,
                )
                if column_name in section_input.metadata_df.columns
            ]
            section_input.metadata_df.loc[:, selected_qc_columns].to_csv(
                feature_qc_selected_samples_path,
                index=False,
            )
    if feature_quality_status["status"] == "fail":
        raise ValueError(
            "Feature quality policy blocking violations detected. "
            f"blocking_issues={feature_quality_status['blocking_issues']}"
        )
    metric_values = metric_bundle(
        section_input.y_true_all,
        section_input.y_pred_all,
        metric_names=("accuracy", "balanced_accuracy", "macro_f1"),
    )
    overall_accuracy = float(metric_values["accuracy"])
    overall_balanced = float(metric_values["balanced_accuracy"])
    overall_macro_f1 = float(metric_values["macro_f1"])
    primary_metric_name = str(section_input.primary_metric_name)
    primary_metric_aggregation = _resolve_primary_metric_aggregation(
        section_input.primary_metric_aggregation
    )
    primary_metric_value_pooled = classification_metric_score(
        section_input.y_true_all,
        section_input.y_pred_all,
        metric_name=primary_metric_name,
    )
    primary_metric_value_mean_fold = _mean_fold_metric_value(
        fold_rows=section_input.fold_rows,
        metric_name=primary_metric_name,
    )
    primary_metric_value = (
        float(primary_metric_value_mean_fold)
        if primary_metric_aggregation == "mean_fold_scores"
        else float(primary_metric_value_pooled)
    )
    labels_sorted = sorted(
        np.unique(
            np.concatenate(
                [np.asarray(section_input.y_true_all), np.asarray(section_input.y_pred_all)]
            )
        ).tolist()
    )
    cmatrix = confusion_matrix(
        section_input.y_true_all,
        section_input.y_pred_all,
        labels=labels_sorted,
    )

    metrics: dict[str, Any] = {
        "model": section_input.model,
        "target": section_input.target_column,
        "feature_recipe_id": str(section_input.feature_recipe_id),
        "preprocessing_strategy": (
            str(section_input.preprocessing_strategy)
            if section_input.preprocessing_strategy is not None
            else None
        ),
        "dimensionality_strategy": str(section_input.dimensionality_strategy),
        "pca_n_components": (
            int(section_input.pca_n_components)
            if section_input.pca_n_components is not None
            else None
        ),
        "pca_variance_ratio": (
            float(section_input.pca_variance_ratio)
            if section_input.pca_variance_ratio is not None
            else None
        ),
        "cv": section_input.cv_mode,
        "experiment_mode": section_input.cv_mode,
        "subject": (
            str(section_input.subject)
            if section_input.cv_mode == "within_subject_loso_session"
            else None
        ),
        "train_subject": (
            str(section_input.train_subject)
            if section_input.cv_mode == "frozen_cross_person_transfer"
            else None
        ),
        "test_subject": (
            str(section_input.test_subject)
            if section_input.cv_mode == "frozen_cross_person_transfer"
            else None
        ),
        "n_samples": int(len(section_input.y_true_all)),
        "n_features": int(section_input.x_matrix.shape[1]),
        "n_folds": int(len(section_input.fold_rows)),
        "accuracy": overall_accuracy,
        "balanced_accuracy": overall_balanced,
        "macro_f1": overall_macro_f1,
        "methodology_policy_name": section_input.methodology_policy_name,
        "evidence_run_role": str(section_input.evidence_run_role),
        "repeat_id": int(section_input.repeat_id),
        "repeat_count": int(section_input.repeat_count),
        "base_run_id": (
            str(section_input.base_run_id)
            if section_input.base_run_id is not None
            else str(section_input.run_id)
        ),
        "primary_metric_name": primary_metric_name,
        "primary_metric_aggregation": primary_metric_aggregation,
        "primary_metric_value": float(primary_metric_value),
        "primary_metric_value_mean_fold": float(primary_metric_value_mean_fold),
        "primary_metric_value_pooled": float(primary_metric_value_pooled),
        "tuning_enabled": bool(section_input.tuning_enabled),
        "tuning_summary_path": str(tuning_summary_path.resolve()),
        "tuning_best_params_path": str(tuning_best_params_path.resolve()),
        "labels": labels_sorted,
        "confusion_matrix": cmatrix.tolist(),
        "spatial_compatibility": {
            "status": str(section_input.spatial_compatibility["status"]),
            "passed": bool(section_input.spatial_compatibility["passed"]),
            "n_groups_checked": int(section_input.spatial_compatibility["n_groups_checked"]),
            "reference_group_id": section_input.spatial_compatibility["reference_group_id"],
            "affine_atol": float(section_input.spatial_compatibility["affine_atol"]),
            "report_path": str(section_input.spatial_report_path.resolve()),
        },
        "feature_engineering": {
            "feature_recipe_id": str(section_input.feature_recipe_id),
            "feature_qc_summary_path": str(feature_qc_summary_path.resolve()),
            "feature_qc_selected_samples_path": str(feature_qc_selected_samples_path.resolve()),
            "feature_qc_summary": feature_qc_summary,
            "feature_quality_policy": feature_quality_policy.model_dump(mode="json"),
            "feature_quality_status": feature_quality_status,
        },
    }

    if section_input.n_permutations > 0:
        permutation_metric_name = str(
            section_input.permutation_metric_name or section_input.primary_metric_name
        )
        permutation_executor_id = _assignment_executor_id(
            section_input.permutation_assignment,
            default_executor_id=PERMUTATION_REFERENCE_EXECUTOR_ID,
        )
        observed_permutation_metric = (
            _mean_fold_metric_value(
                fold_rows=section_input.fold_rows,
                metric_name=permutation_metric_name,
            )
            if primary_metric_aggregation == "mean_fold_scores"
            else float(
                classification_metric_score(
                    section_input.y_true_all,
                    section_input.y_pred_all,
                    metric_name=permutation_metric_name,
                )
            )
        )
        use_tuned_permutation = (
            bool(section_input.tuning_enabled)
            and str(section_input.methodology_policy_name).strip() == "grouped_nested_tuning"
            and str(section_input.evidence_run_role).strip() != "untuned_baseline"
            and str(section_input.model).strip().lower() != "dummy"
        )
        if use_tuned_permutation:
            permutation_payload = _evaluate_grouped_nested_tuning_permutations(
                section_input=section_input,
                metric_name=permutation_metric_name,
                observed_metric=observed_permutation_metric,
                primary_metric_aggregation=primary_metric_aggregation,
                permutation_executor_id=permutation_executor_id,
            )
        else:
            permutation_payload = run_permutation_executor(
                executor_id=permutation_executor_id,
                evaluate_permutations_fn=section_input.evaluate_permutations_fn,
                build_pipeline_fn=section_input.build_pipeline_fn,
                model_name=section_input.model,
                seed=section_input.seed,
                x_matrix=section_input.x_matrix,
                y=section_input.y,
                splits=section_input.splits,
                n_permutations=section_input.n_permutations,
                metric_name=permutation_metric_name,
                observed_metric=observed_permutation_metric,
                primary_metric_aggregation=primary_metric_aggregation,
                progress_callback=section_input.progress_callback,
                progress_metadata={
                    "run_id": str(section_input.run_id),
                    "model": str(section_input.model),
                    "target": str(section_input.target_column),
                    "cv_mode": str(section_input.cv_mode),
                },
            )
        permutation_payload["alpha"] = float(section_input.permutation_alpha)
        permutation_payload["minimum_required"] = int(section_input.permutation_minimum_required)
        permutation_payload["meets_minimum"] = bool(
            int(permutation_payload.get("n_permutations", 0))
            >= int(section_input.permutation_minimum_required)
        )
        p_value = permutation_payload.get("p_value")
        passes_threshold = isinstance(p_value, (int, float)) and float(p_value) <= float(
            section_input.permutation_alpha
        )
        permutation_payload["passes_threshold"] = bool(passes_threshold)
        permutation_payload["require_pass_for_validity"] = bool(
            section_input.permutation_require_pass_for_validity
        )
        permutation_payload["interpretation_status"] = (
            "passes_threshold" if bool(passes_threshold) else "fails_threshold"
        )
        metrics["permutation_test"] = permutation_payload

    metrics["interpretability"] = {
        "enabled": bool(section_input.interpretability_summary["enabled"]),
        "performed": bool(section_input.interpretability_summary["performed"]),
        "status": str(section_input.interpretability_summary["status"]),
        "summary_path": str(section_input.interpretability_summary_path.resolve()),
        "fold_artifacts_path": section_input.interpretability_summary["fold_artifacts_path"],
        "stability": section_input.interpretability_summary["stability"],
    }

    subgroup_rows = (
        _compute_subgroup_metrics(
            prediction_rows=section_input.prediction_rows,
            subgroup_dimensions=list(section_input.subgroup_dimensions),
            min_samples_per_group=int(section_input.subgroup_min_samples_per_group),
            min_classes_per_group=int(section_input.subgroup_min_classes_per_group),
            report_small_groups=bool(section_input.subgroup_report_small_groups),
            strict_insufficient_data_rows=bool(section_input.confirmatory_guardrails_enabled),
        )
        if section_input.subgroup_reporting_enabled
        else []
    )
    subgroup_payload = {
        "enabled": bool(section_input.subgroup_reporting_enabled),
        "confirmatory_guardrails_enabled": bool(section_input.confirmatory_guardrails_enabled),
        "evidence_role": str(section_input.subgroup_evidence_role),
        "primary_evidence_substitution_allowed": bool(
            section_input.subgroup_primary_evidence_allowed
        ),
        "dimensions_requested": list(section_input.subgroup_dimensions),
        "min_samples_per_group": int(section_input.subgroup_min_samples_per_group),
        "min_classes_per_group": int(section_input.subgroup_min_classes_per_group),
        "report_small_groups": bool(section_input.subgroup_report_small_groups),
        "generated": bool(subgroup_rows),
        "n_rows": int(len(subgroup_rows)),
        "json_path": str(subgroup_metrics_json_path.resolve()),
        "csv_path": str(subgroup_metrics_csv_path.resolve()),
        "rows": subgroup_rows,
    }
    subgroup_metrics_json_path.write_text(
        f"{json.dumps(subgroup_payload, indent=2)}\n",
        encoding="utf-8",
    )
    subgroup_frame = pd.DataFrame(subgroup_rows)
    if subgroup_frame.empty:
        subgroup_frame = pd.DataFrame(
            columns=[
                "subgroup_key",
                "subgroup_value",
                "n_samples",
                "n_classes",
                "class_distribution_json",
                "insufficient_data_reasons",
                "status",
                "interpretable",
                "balanced_accuracy",
                "macro_f1",
                "accuracy",
            ]
        )
    subgroup_frame.to_csv(subgroup_metrics_csv_path, index=False)
    metrics["subgroup_reporting"] = {
        key: value for key, value in subgroup_payload.items() if key != "rows"
    }

    if section_input.calibration_enabled:
        calibration_summary, calibration_table = build_calibration_outputs(
            section_input.prediction_rows,
            n_bins=int(section_input.calibration_n_bins),
        )
        if str(calibration_summary.get("status")) == "not_applicable" and bool(
            section_input.calibration_require_probabilities_for_validity
        ):
            calibration_summary["status"] = "failed"
            calibration_summary["reason"] = "probabilities_required_but_missing"
            calibration_summary["performed"] = False
    else:
        calibration_summary = {
            "status": "not_applicable",
            "reason": "calibration_disabled_by_policy",
            "performed": False,
            "n_bins": int(section_input.calibration_n_bins),
            "n_samples": int(len(section_input.prediction_rows)),
            "ece": None,
            "brier_score": None,
        }
        calibration_table = pd.DataFrame(
            columns=[
                "bin_index",
                "bin_lower",
                "bin_upper",
                "n_samples",
                "mean_confidence",
                "empirical_accuracy",
            ]
        )
    calibration_summary["enabled"] = bool(section_input.calibration_enabled)
    calibration_summary["require_probabilities_for_validity"] = bool(
        section_input.calibration_require_probabilities_for_validity
    )
    calibration_summary["policy_status"] = (
        "probabilities_required_for_validity"
        if bool(section_input.calibration_require_probabilities_for_validity)
        else "required_if_probabilities_available"
    )
    calibration_summary["probability_support_detected"] = bool(
        str(calibration_summary.get("status")) == "performed"
    )
    calibration_summary["summary_path"] = str(calibration_summary_path.resolve())
    calibration_summary["table_path"] = str(calibration_table_path.resolve())
    calibration_summary_path.write_text(
        f"{json.dumps(calibration_summary, indent=2)}\n",
        encoding="utf-8",
    )
    if calibration_table.empty:
        calibration_table = pd.DataFrame(
            columns=[
                "bin_index",
                "bin_lower",
                "bin_upper",
                "n_samples",
                "mean_confidence",
                "empirical_accuracy",
            ]
        )
    calibration_table.to_csv(calibration_table_path, index=False)
    metrics["calibration"] = calibration_summary

    if isinstance(section_input.evidence_policy_effective, dict):
        metrics["evidence_policy_effective"] = dict(section_input.evidence_policy_effective)

    fold_rows = [dict(row) for row in section_input.fold_rows]
    split_rows = [dict(row) for row in section_input.split_rows]
    prediction_rows = [dict(row) for row in section_input.prediction_rows]
    for row in fold_rows:
        row["run_id"] = section_input.run_id
        row["config_file"] = section_input.config_filename
    for row in split_rows:
        row["run_id"] = section_input.run_id
        row["config_file"] = section_input.config_filename

    pd.DataFrame(fold_rows).to_csv(section_input.fold_metrics_path, index=False)
    pd.DataFrame(split_rows).to_csv(section_input.fold_splits_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(section_input.predictions_path, index=False)
    section_input.metrics_path.write_text(f"{json.dumps(metrics, indent=2)}\n", encoding="utf-8")
    return metrics
