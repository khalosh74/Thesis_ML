from __future__ import annotations

from typing import Any, Literal

from sklearn.pipeline import Pipeline

from Thesis_ML.experiments.backend_registry import resolve_backend_constructor
from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy
from Thesis_ML.features.preprocessing import (
    BASELINE_STANDARD_SCALER_RECIPE_ID,
    FEATURE_RECIPE_IDS,
    build_feature_preprocessing_recipe,
    resolve_feature_recipe_id,
)

# Official thesis-facing model taxonomy for this milestone.
OFFICIAL_LINEAR_MODEL_NAMES = ("ridge", "logreg", "linearsvc")
CONTROL_MODEL_NAMES = ("dummy",)
EXPLORATORY_EXTENSION_MODEL_NAMES = ("xgboost",)

OFFICIAL_MODEL_NAMES = OFFICIAL_LINEAR_MODEL_NAMES + CONTROL_MODEL_NAMES
ALL_MODEL_NAMES = OFFICIAL_MODEL_NAMES + EXPLORATORY_EXTENSION_MODEL_NAMES

# Backward-compatible aliases.
MODEL_NAMES = OFFICIAL_LINEAR_MODEL_NAMES + EXPLORATORY_EXTENSION_MODEL_NAMES
EXPLORATORY_ONLY_MODEL_NAMES = EXPLORATORY_EXTENSION_MODEL_NAMES
DEFAULT_BATCH_MODEL_NAMES = OFFICIAL_LINEAR_MODEL_NAMES

SUPPORTED_FEATURE_RECIPE_IDS: tuple[str, ...] = FEATURE_RECIPE_IDS

PreprocessKind = Literal["standard_scaler"]

_MODEL_PREPROCESS_KINDS: dict[str, PreprocessKind] = {
    "dummy": "standard_scaler",
    "linearsvc": "standard_scaler",
    "logreg": "standard_scaler",
    "ridge": "standard_scaler",
    "xgboost": "standard_scaler",
}


def make_model(
    name: str,
    seed: int,
    class_weight_policy: str = "none",
    compute_policy: ResolvedComputePolicy | None = None,
) -> Any:
    # Keep model hyperparameters fixed across runs to avoid hidden dependence on
    # full selected dataset geometry before fold-level train/test splitting.
    resolution = resolve_backend_constructor(
        model_name=name,
        compute_policy=compute_policy,
    )
    return resolution.build_estimator(
        seed=seed,
        class_weight_policy=class_weight_policy,
    )


def model_preprocess_kind(model_name: str) -> PreprocessKind:
    normalized_model_name = str(model_name).strip().lower()
    preprocess_kind = _MODEL_PREPROCESS_KINDS.get(normalized_model_name)
    if preprocess_kind is None:
        allowed = ", ".join(sorted(_MODEL_PREPROCESS_KINDS))
        raise ValueError(f"Unsupported model '{model_name}'. Allowed values: {allowed}.")
    return preprocess_kind


def model_supports_linear_interpretability(model_name: str) -> bool:
    return model_is_official_linear_family(model_name)


def model_is_official_linear_family(model_name: str) -> bool:
    return str(model_name).strip().lower() in set(OFFICIAL_LINEAR_MODEL_NAMES)


def model_is_control_model(model_name: str) -> bool:
    return str(model_name).strip().lower() in set(CONTROL_MODEL_NAMES)


def model_is_exploratory_extension(model_name: str) -> bool:
    return str(model_name).strip().lower() in set(EXPLORATORY_EXTENSION_MODEL_NAMES)


def model_is_officially_admitted(model_name: str) -> bool:
    return str(model_name).strip().lower() in set(OFFICIAL_MODEL_NAMES)


def resolve_preprocessing_recipe(
    *,
    recipe_id: str | None,
    model_name: str,
) -> str:
    resolved_recipe_id = resolve_feature_recipe_id(recipe_id)
    normalized_model_name = str(model_name).strip().lower()
    if (
        normalized_model_name == "xgboost"
        and resolved_recipe_id != BASELINE_STANDARD_SCALER_RECIPE_ID
    ):
        raise ValueError(
            "xgboost only supports feature_recipe_id="
            f"'{BASELINE_STANDARD_SCALER_RECIPE_ID}'. "
            f"Received '{resolved_recipe_id}'."
        )
    return resolved_recipe_id


def build_feature_preprocessor(*, recipe_id: str | None, model_name: str) -> Any:
    resolved_recipe_id = resolve_preprocessing_recipe(
        recipe_id=recipe_id,
        model_name=model_name,
    )
    return build_feature_preprocessing_recipe(resolved_recipe_id)


def build_pipeline(
    model_name: str,
    seed: int,
    class_weight_policy: str = "none",
    compute_policy: ResolvedComputePolicy | None = None,
    feature_recipe_id: str | None = BASELINE_STANDARD_SCALER_RECIPE_ID,
) -> Pipeline:
    model = make_model(
        name=model_name,
        seed=seed,
        class_weight_policy=class_weight_policy,
        compute_policy=compute_policy,
    )
    return Pipeline(
        steps=[
            # Keep historical `scaler` step name for compatibility with tuning/permutation helpers.
            (
                "scaler",
                build_feature_preprocessor(
                    recipe_id=feature_recipe_id,
                    model_name=model_name,
                ),
            ),
            ("model", model),
        ]
    )
