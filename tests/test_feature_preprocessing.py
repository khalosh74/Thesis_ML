from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Thesis_ML.experiments.metrics import evaluate_permutations
from Thesis_ML.features.preprocessing import (
    BASELINE_STANDARD_SCALER_RECIPE_ID,
    PerSampleMeanCenter,
    SAMPLE_CENTER_STANDARD_SCALER_RECIPE_ID,
    SAMPLE_CENTER_VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID,
    VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID,
    build_feature_preprocessing_recipe,
    resolve_feature_recipe_id,
)


def test_per_sample_mean_center_zero_centers_each_row() -> None:
    transformer = PerSampleMeanCenter()
    matrix = np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float64)
    transformed = transformer.fit_transform(matrix)
    np.testing.assert_allclose(np.mean(transformed, axis=1), np.zeros(2), atol=1e-12)


def test_baseline_recipe_returns_standard_scaler() -> None:
    recipe = build_feature_preprocessing_recipe(BASELINE_STANDARD_SCALER_RECIPE_ID)
    assert isinstance(recipe, StandardScaler)
    assert bool(recipe.with_mean) is True
    assert bool(recipe.with_std) is True


def test_sample_center_recipe_chain() -> None:
    recipe = build_feature_preprocessing_recipe(SAMPLE_CENTER_STANDARD_SCALER_RECIPE_ID)
    assert isinstance(recipe, Pipeline)
    assert list(recipe.named_steps) == ["sample_center", "standard_scaler"]


def test_variance_filter_recipe_chain() -> None:
    recipe = build_feature_preprocessing_recipe(VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID)
    assert isinstance(recipe, Pipeline)
    assert list(recipe.named_steps) == ["variance_filter", "standard_scaler"]


def test_sample_center_variance_filter_recipe_chain() -> None:
    recipe = build_feature_preprocessing_recipe(
        SAMPLE_CENTER_VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID
    )
    assert isinstance(recipe, Pipeline)
    assert list(recipe.named_steps) == [
        "sample_center",
        "variance_filter",
        "standard_scaler",
    ]


def test_resolve_feature_recipe_id_normalizes_and_validates() -> None:
    assert (
        resolve_feature_recipe_id("  SAMPLE_CENTER_STANDARD_SCALER_V1  ")
        == SAMPLE_CENTER_STANDARD_SCALER_RECIPE_ID
    )


def test_permutation_path_supports_pipeline_scaler_step() -> None:
    x_matrix = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.0],
            [0.1, 0.0, 0.1, 0.0],
            [1.0, 1.1, 1.0, 0.0],
            [1.1, 1.0, 1.1, 0.0],
            [2.0, 2.1, 2.0, 0.0],
            [2.1, 2.0, 2.1, 0.0],
        ],
        dtype=np.float64,
    )
    y = np.asarray(["neg", "pos", "neg", "pos", "neg", "pos"])
    splits = [
        (np.asarray([2, 3, 4, 5]), np.asarray([0, 1])),
        (np.asarray([0, 1, 4, 5]), np.asarray([2, 3])),
        (np.asarray([0, 1, 2, 3]), np.asarray([4, 5])),
    ]
    pipeline_template = Pipeline(
        steps=[
            (
                "scaler",
                build_feature_preprocessing_recipe(
                    SAMPLE_CENTER_VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID
                ),
            ),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    random_state=13,
                    max_iter=1000,
                ),
            ),
        ]
    )
    payload = evaluate_permutations(
        pipeline_template=pipeline_template,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=7,
        n_permutations=3,
        metric_name="balanced_accuracy",
        observed_metric=0.5,
    )
    assert payload["n_permutations"] == 3
    assert payload["metric_name"] == "balanced_accuracy"
    assert "p_value" in payload

