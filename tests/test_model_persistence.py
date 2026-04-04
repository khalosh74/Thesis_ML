from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Thesis_ML.experiments.model_persistence import load_trained_estimator, save_trained_estimator
from Thesis_ML.experiments.run_experiment import (
    _build_pipeline,
    _extract_linear_coefficients,
    _scores_for_predictions,
)
from Thesis_ML.experiments.section_models import ModelFitInput
from Thesis_ML.experiments.sections_impl import execute_model_fit
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


def _within_subject_fixture() -> tuple[np.ndarray, pd.DataFrame]:
    x_matrix = np.asarray(
        [
            [2.0, 0.0, 0.1],
            [-2.0, 0.0, -0.1],
            [2.2, 0.2, 0.0],
            [-2.2, -0.2, 0.0],
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
    return x_matrix, metadata_df


def test_save_and_load_trained_estimator_roundtrip(tmp_path: Path) -> None:
    model = _build_pipeline_for_test(model_name="ridge", seed=7)
    x = np.asarray([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]], dtype=np.float64)
    y = np.asarray(["positive", "negative", "positive", "negative"], dtype=object)
    model.fit(x, y)

    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model.metadata.json"
    save_trained_estimator(
        estimator=model,
        model_path=model_path,
        metadata_path=metadata_path,
        metadata_payload={"artifact_role": "fold_model", "run_id": "run_1"},
    )

    loaded_model, metadata = load_trained_estimator(
        model_path=model_path,
        metadata_path=metadata_path,
    )
    assert metadata["artifact_role"] == "fold_model"
    np.testing.assert_array_equal(loaded_model.predict(x), model.predict(x))


def test_fold_model_reload_reproduces_saved_predictions(tmp_path: Path) -> None:
    x_matrix, metadata_df = _within_subject_fixture()
    report_dir = tmp_path / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    output = execute_model_fit(
        ModelFitInput(
            x_matrix=x_matrix,
            metadata_df=metadata_df,
            target_column="coarse_affect",
            cv_mode="within_subject_loso_session",
            model="ridge",
            subject="sub-001",
            seed=13,
            run_id="run_model_persist",
            config_filename="config.json",
            report_dir=report_dir,
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

    assert output["model_summary_path"] is not None
    assert output["model_artifacts_csv_path"] is not None
    assert output["final_refit_model_path"] is not None
    assert output["final_refit_metadata_path"] is not None

    first_fold_model = report_dir / "models" / "fold_000_model.joblib"
    first_fold_meta = report_dir / "models" / "fold_000_model.metadata.json"
    loaded_model, metadata = load_trained_estimator(
        model_path=first_fold_model,
        metadata_path=first_fold_meta,
    )
    assert metadata["artifact_role"] == "fold_model"

    fold_0_test_idx = output["splits"][0][1]
    fold_0_predictions = loaded_model.predict(x_matrix[fold_0_test_idx])
    saved_predictions = [
        row["y_pred"]
        for row in output["prediction_rows"]
        if int(row["fold"]) == 0
    ]
    assert list(fold_0_predictions.astype(str)) == [str(value) for value in saved_predictions]

    _, final_meta = load_trained_estimator(
        model_path=Path(output["final_refit_model_path"]),
        metadata_path=Path(output["final_refit_metadata_path"]),
    )
    assert final_meta["artifact_role"] == "final_refit_after_confirmatory_evaluation"
