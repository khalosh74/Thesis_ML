from __future__ import annotations

import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Thesis_ML.experiments import model_factory
from Thesis_ML.features.preprocessing import (
    BASELINE_STANDARD_SCALER_RECIPE_ID,
    FEATURE_RECIPE_IDS,
    SAMPLE_CENTER_STANDARD_SCALER_RECIPE_ID,
    SAMPLE_CENTER_VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID,
    VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID,
)


@pytest.mark.parametrize("model_name", ["dummy", "linearsvc", "logreg", "ridge", "xgboost"])
def test_model_preprocess_kind_is_standard_scaler(model_name: str) -> None:
    assert model_factory.model_preprocess_kind(model_name) == "standard_scaler"


def test_model_factory_flags_xgboost_as_exploratory_only() -> None:
    assert model_factory.model_is_officially_admitted("ridge") is True
    assert model_factory.model_is_officially_admitted("xgboost") is False
    assert model_factory.model_supports_linear_interpretability("xgboost") is False


def test_supported_feature_recipe_ids_match_registry() -> None:
    assert tuple(model_factory.SUPPORTED_FEATURE_RECIPE_IDS) == tuple(FEATURE_RECIPE_IDS)


def test_build_pipeline_for_xgboost_uses_standard_scaler_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SentinelEstimator:
        pass

    monkeypatch.setattr(
        model_factory,
        "make_model",
        lambda **_: _SentinelEstimator(),
    )

    pipeline = model_factory.build_pipeline(
        model_name="xgboost",
        seed=17,
        class_weight_policy="none",
        compute_policy=None,
    )
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "model"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["model"], _SentinelEstimator)


def test_xgboost_rejects_nonbaseline_feature_recipe() -> None:
    with pytest.raises(ValueError, match="xgboost only supports feature_recipe_id"):
        model_factory.resolve_preprocessing_recipe(
            recipe_id=SAMPLE_CENTER_STANDARD_SCALER_RECIPE_ID,
            model_name="xgboost",
        )


def test_build_feature_preprocessor_baseline_is_plain_standard_scaler() -> None:
    scaler = model_factory.build_feature_preprocessor(
        recipe_id=BASELINE_STANDARD_SCALER_RECIPE_ID,
        model_name="ridge",
    )
    assert isinstance(scaler, StandardScaler)
    assert bool(scaler.with_mean) is True
    assert bool(scaler.with_std) is True


def test_build_feature_preprocessor_sample_center_recipe_is_pipeline() -> None:
    preprocessor = model_factory.build_feature_preprocessor(
        recipe_id=SAMPLE_CENTER_STANDARD_SCALER_RECIPE_ID,
        model_name="ridge",
    )
    assert isinstance(preprocessor, Pipeline)
    assert list(preprocessor.named_steps) == ["sample_center", "standard_scaler"]


def test_build_feature_preprocessor_variance_filter_recipe_is_pipeline() -> None:
    preprocessor = model_factory.build_feature_preprocessor(
        recipe_id=VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID,
        model_name="ridge",
    )
    assert isinstance(preprocessor, Pipeline)
    assert list(preprocessor.named_steps) == ["variance_filter", "standard_scaler"]


def test_build_feature_preprocessor_sample_center_variance_filter_recipe_is_pipeline() -> None:
    preprocessor = model_factory.build_feature_preprocessor(
        recipe_id=SAMPLE_CENTER_VARIANCE_FILTER_STANDARD_SCALER_RECIPE_ID,
        model_name="ridge",
    )
    assert isinstance(preprocessor, Pipeline)
    assert list(preprocessor.named_steps) == [
        "sample_center",
        "variance_filter",
        "standard_scaler",
    ]
