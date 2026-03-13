from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_FEATURE_CACHE,
    ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE,
    ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    compute_config_hash,
    register_artifact,
)
from Thesis_ML.data.affect_labels import with_coarse_affect
from Thesis_ML.features.nifti_features import build_feature_cache


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
    summary_path: Path
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


def dataset_selection(section_input: DatasetSelectionInput) -> DatasetSelectionOutput:
    index_df = pd.read_csv(section_input.index_csv)
    if index_df.empty:
        raise ValueError(f"Dataset index is empty: {section_input.index_csv}")
    index_df = with_coarse_affect(index_df, emotion_column="emotion", coarse_column="coarse_affect")

    for required in (
        "sample_id",
        "subject",
        "session",
        "task",
        "modality",
        section_input.target_column,
    ):
        if required not in index_df.columns:
            raise ValueError(f"Dataset index missing required column: {required}")

    if section_input.filter_task is not None:
        index_df = index_df[index_df["task"] == section_input.filter_task].copy()
    if section_input.filter_modality is not None:
        index_df = index_df[index_df["modality"] == section_input.filter_modality].copy()

    if section_input.cv_mode == "within_subject_loso_session":
        index_df = index_df[index_df["subject"].astype(str) == str(section_input.subject)].copy()
        if index_df.empty:
            raise ValueError(
                f"No samples found for subject '{section_input.subject}' after filtering."
            )

    if section_input.cv_mode == "frozen_cross_person_transfer":
        selected = {str(section_input.train_subject), str(section_input.test_subject)}
        index_df = index_df[index_df["subject"].astype(str).isin(selected)].copy()
        if index_df.empty:
            raise ValueError(
                "No samples found for frozen_cross_person_transfer after subject filtering."
            )

    index_df = index_df.dropna(subset=[section_input.target_column]).copy()
    index_df[section_input.target_column] = index_df[section_input.target_column].astype(str)
    if index_df.empty:
        raise ValueError("No samples left after filtering and target cleanup.")

    if section_input.cv_mode == "frozen_cross_person_transfer":
        subjects_after_target = set(index_df["subject"].astype(str).unique().tolist())
        if str(section_input.train_subject) not in subjects_after_target:
            raise ValueError(f"No samples found for train_subject '{section_input.train_subject}'.")
        if str(section_input.test_subject) not in subjects_after_target:
            raise ValueError(f"No samples found for test_subject '{section_input.test_subject}'.")

    return DatasetSelectionOutput(selected_index_df=index_df)


def feature_cache_build(section_input: FeatureCacheBuildInput) -> FeatureCacheBuildOutput:
    manifest_path = build_feature_cache(
        index_csv=section_input.index_csv,
        data_root=section_input.data_root,
        cache_dir=section_input.cache_dir,
        group_key=section_input.group_key,
        force=section_input.force,
    )
    artifact = register_artifact(
        registry_path=section_input.artifact_registry_path,
        artifact_type=ARTIFACT_TYPE_FEATURE_CACHE,
        run_id=section_input.run_id,
        upstream_artifact_ids=[],
        config_hash=compute_config_hash(
            {
                "index_csv": str(section_input.index_csv.resolve()),
                "data_root": str(section_input.data_root.resolve()),
                "cache_dir": str(section_input.cache_dir.resolve()),
                "group_key": section_input.group_key,
                "force": bool(section_input.force),
            }
        ),
        code_ref=section_input.code_ref,
        path=manifest_path,
        status="created",
    )
    return FeatureCacheBuildOutput(
        cache_manifest_path=manifest_path,
        feature_cache_artifact_id=artifact.artifact_id,
    )


def feature_matrix_load(section_input: FeatureMatrixLoadInput) -> FeatureMatrixLoadOutput:
    x_matrix, metadata_df, spatial_compatibility = section_input.load_features_from_cache_fn(
        index_df=section_input.selected_index_df,
        cache_manifest_path=section_input.cache_manifest_path,
        spatial_report_path=section_input.spatial_report_path,
        affine_atol=section_input.affine_atol,
    )
    metadata_df = with_coarse_affect(
        metadata_df, emotion_column="emotion", coarse_column="coarse_affect"
    )

    artifact = register_artifact(
        registry_path=section_input.artifact_registry_path,
        artifact_type=ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE,
        run_id=section_input.run_id,
        upstream_artifact_ids=[section_input.upstream_feature_cache_artifact_id],
        config_hash=compute_config_hash(
            {
                "target": section_input.target_column,
                "cv_mode": section_input.cv_mode,
                "subject": section_input.subject,
                "train_subject": section_input.train_subject,
                "test_subject": section_input.test_subject,
                "filter_task": section_input.filter_task,
                "filter_modality": section_input.filter_modality,
                "cache_manifest_path": str(section_input.cache_manifest_path.resolve()),
            }
        ),
        code_ref=section_input.code_ref,
        path=section_input.spatial_report_path,
        status="created",
    )

    return FeatureMatrixLoadOutput(
        x_matrix=x_matrix,
        metadata_df=metadata_df,
        spatial_compatibility=spatial_compatibility,
        feature_matrix_artifact_id=artifact.artifact_id,
    )


def spatial_validation(section_input: SpatialValidationInput) -> SpatialValidationOutput:
    required = {"status", "passed", "n_groups_checked"}
    missing = [key for key in required if key not in section_input.spatial_compatibility]
    if missing:
        raise ValueError(
            f"Spatial compatibility payload missing required keys: {', '.join(sorted(missing))}"
        )
    return SpatialValidationOutput(
        passed=bool(section_input.spatial_compatibility["passed"]),
        report=section_input.spatial_compatibility,
    )


def model_fit(section_input: ModelFitInput) -> ModelFitOutput:
    from Thesis_ML.experiments.sections_impl import execute_model_fit

    output_payload = execute_model_fit(section_input)
    return ModelFitOutput.model_validate(output_payload)


def interpretability(section_input: InterpretabilityInput) -> InterpretabilityOutput:
    from Thesis_ML.experiments.sections_impl import execute_interpretability

    summary = execute_interpretability(section_input)
    artifact = register_artifact(
        registry_path=section_input.artifact_registry_path,
        artifact_type=ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
        run_id=section_input.run_id,
        upstream_artifact_ids=[section_input.upstream_feature_matrix_artifact_id],
        config_hash=compute_config_hash(
            {
                "run_id": section_input.run_id,
                "cv": section_input.cv_mode,
                "model": section_input.model,
                "target": section_input.target_column,
                "subject": section_input.subject,
                "performed": bool(summary.get("performed", False)),
            }
        ),
        code_ref=section_input.code_ref,
        path=section_input.summary_path,
        status=str(summary.get("status", "unknown")),
    )
    return InterpretabilityOutput(
        interpretability_summary=summary,
        interpretability_artifact_id=artifact.artifact_id,
    )


def evaluation(section_input: EvaluationInput) -> EvaluationOutput:
    from Thesis_ML.experiments.sections_impl import execute_evaluation

    metrics = execute_evaluation(section_input)
    artifact = register_artifact(
        registry_path=section_input.artifact_registry_path,
        artifact_type=ARTIFACT_TYPE_METRICS_BUNDLE,
        run_id=section_input.run_id,
        upstream_artifact_ids=[section_input.upstream_feature_matrix_artifact_id],
        config_hash=compute_config_hash(
            {
                "run_id": section_input.run_id,
                "target": section_input.target_column,
                "model": section_input.model,
                "cv": section_input.cv_mode,
                "seed": int(section_input.seed),
            }
        ),
        code_ref=section_input.code_ref,
        path=section_input.metrics_path,
        status="created",
    )
    return EvaluationOutput(metrics=metrics, metrics_artifact_id=artifact.artifact_id)
