from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _SectionModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class DatasetSelectionInput(_SectionModel):
    index_csv: Path
    target_column: str = Field(min_length=1)
    cv_mode: str = Field(min_length=1)
    subject: str | None = None
    train_subject: str | None = None
    test_subject: str | None = None
    filter_task: str | None = None
    filter_modality: str | None = None

    @model_validator(mode="after")
    def _validate_mode_requirements(self) -> DatasetSelectionInput:
        if self.cv_mode == "within_subject_loso_session":
            if self.subject is None or not str(self.subject).strip():
                raise ValueError("cv='within_subject_loso_session' requires a non-empty subject.")
        if self.cv_mode == "frozen_cross_person_transfer":
            if self.train_subject is None or not str(self.train_subject).strip():
                raise ValueError(
                    "cv='frozen_cross_person_transfer' requires a non-empty train_subject."
                )
            if self.test_subject is None or not str(self.test_subject).strip():
                raise ValueError(
                    "cv='frozen_cross_person_transfer' requires a non-empty test_subject."
                )
        return self


class DatasetSelectionOutput(_SectionModel):
    selected_index_df: pd.DataFrame


class FeatureCacheBuildInput(_SectionModel):
    index_csv: Path
    data_root: Path
    cache_dir: Path
    group_key: str = "subject_session_bas"
    force: bool = False
    run_id: str = Field(min_length=1)
    artifact_registry_path: Path
    code_ref: str | None = None


class FeatureCacheBuildOutput(_SectionModel):
    cache_manifest_path: Path
    feature_cache_artifact_id: str


class FeatureMatrixLoadInput(_SectionModel):
    selected_index_df: pd.DataFrame
    cache_manifest_path: Path
    spatial_report_path: Path
    affine_atol: float
    run_id: str = Field(min_length=1)
    artifact_registry_path: Path
    code_ref: str | None = None
    upstream_feature_cache_artifact_id: str
    target_column: str = Field(min_length=1)
    cv_mode: str = Field(min_length=1)
    subject: str | None = None
    train_subject: str | None = None
    test_subject: str | None = None
    filter_task: str | None = None
    filter_modality: str | None = None
    load_features_from_cache_fn: Callable[..., tuple[np.ndarray, pd.DataFrame, dict[str, Any]]]


class FeatureMatrixLoadOutput(_SectionModel):
    x_matrix: np.ndarray
    metadata_df: pd.DataFrame
    spatial_compatibility: dict[str, Any]
    feature_matrix_artifact_id: str


class SpatialValidationInput(_SectionModel):
    spatial_compatibility: dict[str, Any]


class SpatialValidationOutput(_SectionModel):
    passed: bool
    report: dict[str, Any]


class ModelFitInput(_SectionModel):
    x_matrix: np.ndarray
    metadata_df: pd.DataFrame
    target_column: str = Field(min_length=1)
    cv_mode: str = Field(min_length=1)
    model: str = Field(min_length=1)
    subject: str | None = None
    train_subject: str | None = None
    test_subject: str | None = None
    seed: int
    primary_metric_name: str = "balanced_accuracy"
    methodology_policy_name: str = "fixed_baselines_only"
    class_weight_policy: str = "none"
    tuning_enabled: bool = False
    tuning_search_space_id: str | None = None
    tuning_search_space_version: str | None = None
    tuning_inner_cv_scheme: str | None = None
    tuning_inner_group_field: str | None = None
    tuning_summary_path: Path | None = None
    tuning_best_params_path: Path | None = None
    interpretability_enabled: bool | None = None
    run_id: str = Field(min_length=1)
    config_filename: str = Field(min_length=1)
    report_dir: Path
    build_pipeline_fn: Callable[..., Any]
    scores_for_predictions_fn: Callable[..., dict[str, list[Any]]]
    extract_linear_coefficients_fn: Callable[..., tuple[np.ndarray, np.ndarray, list[str]]]


class ModelFitOutput(_SectionModel):
    y: np.ndarray
    splits: list[tuple[np.ndarray, np.ndarray]]
    fold_rows: list[dict[str, Any]]
    split_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]
    y_true_all: list[str]
    y_pred_all: list[str]
    interpretability_enabled: bool
    interpretability_fold_rows: list[dict[str, Any]]
    interpretability_vectors: list[np.ndarray]
    interpretability_fold_artifacts_path: Path
    interpretability_summary_path: Path
    tuning_summary: dict[str, Any]
    tuning_records: list[dict[str, Any]]
    tuning_summary_path: Path
    tuning_best_params_path: Path


class InterpretabilityInput(_SectionModel):
    interpretability_enabled: bool
    interpretability_fold_rows: list[dict[str, Any]]
    interpretability_vectors: list[np.ndarray]
    fold_artifacts_path: Path
    summary_path: Path
    compute_interpretability_stability_fn: Callable[[list[np.ndarray]], dict[str, Any]]
    run_id: str = Field(min_length=1)
    artifact_registry_path: Path
    code_ref: str | None = None
    upstream_feature_matrix_artifact_id: str
    cv_mode: str = Field(min_length=1)
    model: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    subject: str | None = None


class InterpretabilityOutput(_SectionModel):
    interpretability_summary: dict[str, Any]
    interpretability_artifact_id: str


class EvaluationInput(_SectionModel):
    x_matrix: np.ndarray
    y: np.ndarray
    splits: list[tuple[np.ndarray, np.ndarray]]
    fold_rows: list[dict[str, Any]]
    split_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]
    y_true_all: list[str]
    y_pred_all: list[str]
    subject: str | None = None
    train_subject: str | None = None
    test_subject: str | None = None
    n_permutations: int = 0
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
    subgroup_reporting_enabled: bool = True
    subgroup_dimensions: list[str] = Field(
        default_factory=lambda: ["label", "task", "modality", "session", "subject"]
    )
    subgroup_min_samples_per_group: int = 1
    subgroup_min_classes_per_group: int = 1
    subgroup_report_small_groups: bool = False
    confirmatory_guardrails_enabled: bool = False
    subgroup_evidence_role: str = "exploratory"
    subgroup_primary_evidence_allowed: bool = True
    subgroup_metrics_json_path: Path | None = None
    subgroup_metrics_csv_path: Path | None = None
    tuning_summary_path: Path | None = None
    tuning_best_params_path: Path | None = None
    calibration_enabled: bool = True
    calibration_n_bins: int = 10
    calibration_require_probabilities_for_validity: bool = False
    calibration_summary_path: Path | None = None
    calibration_table_path: Path | None = None
    spatial_compatibility: dict[str, Any]
    spatial_report_path: Path
    interpretability_summary: dict[str, Any]
    interpretability_summary_path: Path
    fold_metrics_path: Path
    fold_splits_path: Path
    predictions_path: Path
    config_filename: str = Field(min_length=1)
    build_pipeline_fn: Callable[..., Any]
    evaluate_permutations_fn: Callable[..., dict[str, Any]]
    run_id: str = Field(min_length=1)
    artifact_registry_path: Path
    code_ref: str | None = None
    upstream_feature_matrix_artifact_id: str
    metrics_path: Path
    model: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    cv_mode: str = Field(min_length=1)
    seed: int


class EvaluationOutput(_SectionModel):
    metrics: dict[str, Any]
    metrics_artifact_id: str


__all__ = [
    "DatasetSelectionInput",
    "DatasetSelectionOutput",
    "FeatureCacheBuildInput",
    "FeatureCacheBuildOutput",
    "FeatureMatrixLoadInput",
    "FeatureMatrixLoadOutput",
    "SpatialValidationInput",
    "SpatialValidationOutput",
    "ModelFitInput",
    "ModelFitOutput",
    "InterpretabilityInput",
    "InterpretabilityOutput",
    "EvaluationInput",
    "EvaluationOutput",
]
