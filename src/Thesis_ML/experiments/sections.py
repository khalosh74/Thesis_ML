from __future__ import annotations

from pathlib import Path

import pandas as pd

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_FEATURE_CACHE,
    ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE,
    ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    ARTIFACT_TYPE_MODEL_BUNDLE,
    ARTIFACT_TYPE_MODEL_REFIT_BUNDLE,
    compute_config_hash,
    register_artifact,
)
from Thesis_ML.data.affect_labels import (
    blocking_derived_label_inconsistency_rows,
    blocking_target_derivation_audit_rows,
    build_derived_label_inconsistency_audit,
    build_target_derivation_audit,
    summarize_derived_label_inconsistency_audit,
    summarize_target_derivation_audit,
    with_binary_valence_like,
    with_coarse_affect,
)
from Thesis_ML.experiments.feature_space_loading import load_feature_matrix
from Thesis_ML.experiments.section_models import (
    DatasetSelectionInput,
    DatasetSelectionOutput,
    EvaluationInput,
    EvaluationOutput,
    FeatureCacheBuildInput,
    FeatureCacheBuildOutput,
    FeatureMatrixLoadInput,
    FeatureMatrixLoadOutput,
    InterpretabilityInput,
    InterpretabilityOutput,
    ModelFitInput,
    ModelFitOutput,
    SpatialValidationInput,
    SpatialValidationOutput,
)
from Thesis_ML.experiments.selection_manifest import apply_dataset_selection_filters
from Thesis_ML.features.nifti_features import build_feature_cache


def dataset_selection(section_input: DatasetSelectionInput) -> DatasetSelectionOutput:
    index_df = pd.read_csv(section_input.index_csv)
    if index_df.empty:
        raise ValueError(f"Dataset index is empty: {section_input.index_csv}")

    stored_labels_frame = index_df.copy()
    derived_label_inconsistency_audit_df = build_derived_label_inconsistency_audit(
        stored_labels_frame,
        emotion_column="emotion",
        coarse_column="coarse_affect",
        binary_column="binary_valence_like",
    )
    blocking_inconsistency_df = blocking_derived_label_inconsistency_rows(
        derived_label_inconsistency_audit_df
    )
    if not blocking_inconsistency_df.empty:
        summary = summarize_derived_label_inconsistency_audit(blocking_inconsistency_df)
        raise ValueError(
            "Dataset selection found inconsistent stored derived labels. "
            f"n_rows={summary['n_rows']}, "
            f"by_category={summary['by_category']}, "
            f"sample_ids_head={summary['sample_ids_head']}"
        )

    index_df = with_coarse_affect(
        index_df,
        emotion_column="emotion",
        coarse_column="coarse_affect",
        strict_recompute=True,
        attach_mapping_metadata=True,
    )
    index_df = with_binary_valence_like(
        index_df,
        coarse_column="coarse_affect",
        binary_column="binary_valence_like",
        strict_recompute=True,
        attach_mapping_metadata=True,
    )

    target_derivation_audit_df = build_target_derivation_audit(
        index_df,
        target_column=section_input.target_column,
    )
    blocking_target_audit_df = blocking_target_derivation_audit_rows(target_derivation_audit_df)

    if not blocking_target_audit_df.empty:
        summary = summarize_target_derivation_audit(blocking_target_audit_df)
        raise ValueError(
            "Dataset selection found unsupported or missing source labels for derived target "
            f"'{section_input.target_column}'. "
            f"n_rows={summary['n_rows']}, "
            f"by_category={summary['by_category']}, "
            f"sample_ids_head={summary['sample_ids_head']}"
        )

    selection_result = apply_dataset_selection_filters(
        index_df,
        target_column=section_input.target_column,
        cv_mode=section_input.cv_mode,
        subject=section_input.subject,
        train_subject=section_input.train_subject,
        test_subject=section_input.test_subject,
        filter_task=section_input.filter_task,
        filter_modality=section_input.filter_modality,
    )

    index_df = selection_result.selected_index_df
    if index_df.empty:
        if (
            section_input.cv_mode == "within_subject_loso_session"
            and section_input.subject is not None
        ):
            raise ValueError(f"No samples found for subject '{section_input.subject}'")

        if (
            section_input.cv_mode == "frozen_cross_person_transfer"
            and section_input.train_subject is not None
            and section_input.test_subject is not None
        ):
            raise ValueError(
                "No samples left for frozen cross-person transfer pair "
                f"train_subject='{section_input.train_subject}', "
                f"test_subject='{section_input.test_subject}'."
            )

        raise ValueError("No samples left after filtering and target cleanup.")

    if section_input.cv_mode == "frozen_cross_person_transfer":
        subjects_after_target = set(index_df["subject"].astype(str).unique().tolist())
        if str(section_input.train_subject) not in subjects_after_target:
            raise ValueError(f"No samples found for train_subject '{section_input.train_subject}'.")
        if str(section_input.test_subject) not in subjects_after_target:
            raise ValueError(f"No samples found for test_subject '{section_input.test_subject}'.")

    return DatasetSelectionOutput(
        selected_index_df=index_df,
        target_derivation_audit_df=target_derivation_audit_df,
        selection_exclusion_manifest_df=selection_result.exclusion_manifest_df,
        selection_summary=selection_result.selection_summary,
    )


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
    resolved_roi_spec_path = (
        str(section_input.roi_spec_path.resolve())
        if section_input.roi_spec_path is not None
        else None
    )
    x_matrix, metadata_df, spatial_compatibility = load_feature_matrix(
        selected_index_df=section_input.selected_index_df,
        data_root=section_input.data_root,
        cache_manifest_path=section_input.cache_manifest_path,
        spatial_report_path=section_input.spatial_report_path,
        affine_atol=section_input.affine_atol,
        feature_space=section_input.feature_space,
        roi_spec_path=section_input.roi_spec_path,
        load_features_from_cache_fn=section_input.load_features_from_cache_fn,
    )
    metadata_df = with_coarse_affect(
        metadata_df,
        emotion_column="emotion",
        coarse_column="coarse_affect",
        strict_recompute=True,
        attach_mapping_metadata=True,
    )
    metadata_df = with_binary_valence_like(
        metadata_df,
        coarse_column="coarse_affect",
        binary_column="binary_valence_like",
        strict_recompute=True,
        attach_mapping_metadata=True,
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
                "feature_space": section_input.feature_space,
                "roi_spec_path": resolved_roi_spec_path,
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
    if (
        section_input.artifact_registry_path is not None
        and section_input.upstream_feature_matrix_artifact_id is not None
        and output_payload.get("model_summary_path") is not None
    ):
        model_summary_path = Path(output_payload["model_summary_path"])
        model_bundle = register_artifact(
            registry_path=section_input.artifact_registry_path,
            artifact_type=ARTIFACT_TYPE_MODEL_BUNDLE,
            run_id=section_input.run_id,
            upstream_artifact_ids=[section_input.upstream_feature_matrix_artifact_id],
            config_hash=compute_config_hash(
                {
                    "run_id": section_input.run_id,
                    "model": section_input.model,
                    "target": section_input.target_column,
                    "cv": section_input.cv_mode,
                    "persist_models": bool(section_input.persist_models),
                    "persist_fold_models": bool(section_input.persist_fold_models),
                    "persist_final_refit_model": bool(section_input.persist_final_refit_model),
                }
            ),
            code_ref=section_input.code_ref,
            path=model_summary_path,
            status="created",
        )
        output_payload["model_bundle_artifact_id"] = model_bundle.artifact_id

        final_refit_model_path = output_payload.get("final_refit_model_path")
        if final_refit_model_path is not None:
            final_refit_bundle = register_artifact(
                registry_path=section_input.artifact_registry_path,
                artifact_type=ARTIFACT_TYPE_MODEL_REFIT_BUNDLE,
                run_id=section_input.run_id,
                upstream_artifact_ids=[model_bundle.artifact_id],
                config_hash=compute_config_hash(
                    {
                        "run_id": section_input.run_id,
                        "model": section_input.model,
                        "target": section_input.target_column,
                        "cv": section_input.cv_mode,
                        "artifact_role": "final_refit_after_confirmatory_evaluation",
                    }
                ),
                code_ref=section_input.code_ref,
                path=model_summary_path,
                status="created",
            )
            output_payload["final_refit_artifact_id"] = final_refit_bundle.artifact_id
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
                "preprocessing_strategy": section_input.preprocessing_strategy,
                "dimensionality_strategy": section_input.dimensionality_strategy,
                "pca_n_components": section_input.pca_n_components,
                "pca_variance_ratio": section_input.pca_variance_ratio,
            }
        ),
        code_ref=section_input.code_ref,
        path=section_input.metrics_path,
        status="created",
    )
    return EvaluationOutput(metrics=metrics, metrics_artifact_id=artifact.artifact_id)


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
    "dataset_selection",
    "feature_cache_build",
    "feature_matrix_load",
    "spatial_validation",
    "model_fit",
    "interpretability",
    "evaluation",
]
