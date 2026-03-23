from __future__ import annotations

import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from Thesis_ML.experiments import model_factory


@pytest.mark.parametrize("model_name", ["dummy", "linearsvc", "logreg", "ridge"])
def test_model_preprocess_kind_for_existing_models_is_standard_scaler(model_name: str) -> None:
    assert model_factory.model_preprocess_kind(model_name) == "standard_scaler"


def test_model_preprocess_kind_for_xgboost_is_passthrough() -> None:
    assert model_factory.model_preprocess_kind("xgboost") == "passthrough"


def test_model_factory_flags_xgboost_as_exploratory_only() -> None:
    assert model_factory.model_is_officially_admitted("ridge") is True
    assert model_factory.model_is_officially_admitted("xgboost") is False
    assert model_factory.model_supports_linear_interpretability("xgboost") is False


def test_build_pipeline_for_xgboost_uses_passthrough_scaler_step(
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
    assert isinstance(pipeline.named_steps["scaler"], FunctionTransformer)
    assert bool(pipeline.named_steps["scaler"].validate) is False
    assert isinstance(pipeline.named_steps["model"], _SentinelEstimator)


def test_build_pipeline_for_linear_model_uses_standard_scaler(
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
        model_name="ridge",
        seed=9,
        class_weight_policy="none",
        compute_policy=None,
    )
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["model"], _SentinelEstimator)
