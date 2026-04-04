from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_MODEL_BUNDLE,
    ARTIFACT_TYPE_MODEL_REFIT_BUNDLE,
    list_artifacts_for_run,
)
from Thesis_ML.experiments.run_experiment import (
    _build_pipeline,
    _extract_linear_coefficients,
    _scores_for_predictions,
)
from Thesis_ML.experiments.section_models import ModelFitInput
from Thesis_ML.experiments.sections import model_fit
from Thesis_ML.features.dimensionality import resolve_dimensionality_config


def _build_pipeline_for_test(
    model_name: str,
    seed: int,
    feature_recipe_id: str | None = None,
    preprocessing_strategy: str | None = None,
):
    return _build_pipeline(
        model_name=model_name,
        seed=seed,
        class_weight_policy="none",
        compute_policy=None,
        feature_recipe_id=feature_recipe_id,
        preprocessing_strategy=preprocessing_strategy,
        dimensionality_config=resolve_dimensionality_config(
            dimensionality_strategy="none",
            pca_n_components=None,
            pca_variance_ratio=None,
        ),
    )


def test_model_bundle_and_refit_bundle_are_registered(tmp_path: Path) -> None:
    x_matrix = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.1, 0.1],
            [-1.1, -0.1],
        ],
        dtype=np.float64,
    )
    metadata_df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "bas": "BAS1",
                "task": "emo",
                "modality": "audiovisual",
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-01",
                "bas": "BAS1",
                "task": "emo",
                "modality": "audiovisual",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s3",
                "subject": "sub-001",
                "session": "ses-02",
                "bas": "BAS1",
                "task": "recog",
                "modality": "audiovisual",
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s4",
                "subject": "sub-001",
                "session": "ses-02",
                "bas": "BAS1",
                "task": "recog",
                "modality": "audiovisual",
                "coarse_affect": "negative",
            },
        ]
    )

    registry_path = tmp_path / "artifact_registry.sqlite3"
    fit_output = model_fit(
        ModelFitInput(
            x_matrix=x_matrix,
            metadata_df=metadata_df,
            target_column="coarse_affect",
            cv_mode="within_subject_loso_session",
            model="ridge",
            subject="sub-001",
            seed=11,
            run_id="run_model_bundle",
            config_filename="config.json",
            report_dir=tmp_path / "report",
            artifact_registry_path=registry_path,
            upstream_feature_matrix_artifact_id="feature_matrix_bundle_test",
            build_pipeline_fn=_build_pipeline_for_test,
            scores_for_predictions_fn=_scores_for_predictions,
            extract_linear_coefficients_fn=_extract_linear_coefficients,
            persist_models=True,
            persist_fold_models=True,
            persist_final_refit_model=True,
            experiment_id="E16",
            feature_space="whole_brain_masked",
        )
    )

    assert fit_output.model_bundle_artifact_id is not None
    assert fit_output.final_refit_artifact_id is not None

    artifact_types = {
        record.artifact_type
        for record in list_artifacts_for_run(registry_path=registry_path, run_id="run_model_bundle")
    }
    assert ARTIFACT_TYPE_MODEL_BUNDLE in artifact_types
    assert ARTIFACT_TYPE_MODEL_REFIT_BUNDLE in artifact_types
