from __future__ import annotations

from typing import Any, Literal, cast

from sklearn.pipeline import Pipeline

from Thesis_ML.experiments.backend_registry import resolve_backend_constructor
from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy
from Thesis_ML.experiments.model_admission import (
    model_is_exploratory_only,
    model_is_official,
)
from Thesis_ML.experiments.model_registry import (
    control_model_names,
    default_batch_model_names,
    exploratory_only_model_names,
    get_model_spec,
    official_linear_model_names,
    official_model_names,
    registered_model_names,
)
from Thesis_ML.features.preprocessing import (
    FEATURE_RECIPE_IDS,
    build_feature_preprocessing_recipe,
    resolve_feature_recipe_id,
)

OFFICIAL_LINEAR_MODEL_NAMES = official_linear_model_names()
CONTROL_MODEL_NAMES = control_model_names()
EXPLORATORY_EXTENSION_MODEL_NAMES = exploratory_only_model_names()
OFFICIAL_MODEL_NAMES = official_model_names()
ALL_MODEL_NAMES = registered_model_names()

# Backward-compatible aliases.
MODEL_NAMES = OFFICIAL_LINEAR_MODEL_NAMES + EXPLORATORY_EXTENSION_MODEL_NAMES
EXPLORATORY_ONLY_MODEL_NAMES = EXPLORATORY_EXTENSION_MODEL_NAMES
DEFAULT_BATCH_MODEL_NAMES = default_batch_model_names()

SUPPORTED_FEATURE_RECIPE_IDS: tuple[str, ...] = FEATURE_RECIPE_IDS

PreprocessKind = Literal["standard_scaler"]


def make_model(
    name: str,
    seed: int,
    class_weight_policy: str = "none",
    compute_policy: ResolvedComputePolicy | None = None,
) -> Any:
    # Keep model hyperparameters fixed across runs to avoid hidden dependence on
    # full selected dataset geometry before fold-level train/test splitting.
    spec = get_model_spec(name)
    normalized_class_weight_policy = str(class_weight_policy).strip().lower()
    if normalized_class_weight_policy not in set(spec.supported_class_weight_policies):
        allowed = ", ".join(spec.supported_class_weight_policies)
        raise ValueError(
            f"Model '{spec.logical_name}' does not support class_weight_policy="
            f"'{class_weight_policy}'. Allowed values: {allowed}."
        )
    resolution = resolve_backend_constructor(
        model_name=spec.logical_name,
        compute_policy=compute_policy,
    )
    return resolution.build_estimator(
        seed=seed,
        class_weight_policy=normalized_class_weight_policy,
    )


def model_preprocess_kind(model_name: str) -> PreprocessKind:
    return cast(PreprocessKind, get_model_spec(model_name).preprocess_kind)


def model_supports_linear_interpretability(model_name: str) -> bool:
    return bool(get_model_spec(model_name).capabilities.linear_interpretability)


def model_supports_probability_outputs(model_name: str) -> bool:
    return bool(get_model_spec(model_name).capabilities.probability_outputs)


def model_is_official_linear_family(model_name: str) -> bool:
    normalized = str(model_name).strip().lower()
    return normalized in set(OFFICIAL_LINEAR_MODEL_NAMES)


def model_is_control_model(model_name: str) -> bool:
    normalized = str(model_name).strip().lower()
    return normalized in set(CONTROL_MODEL_NAMES)


def model_is_exploratory_extension(model_name: str) -> bool:
    return bool(model_is_exploratory_only(model_name))


def model_is_officially_admitted(model_name: str) -> bool:
    return bool(model_is_official(model_name))


def resolve_preprocessing_recipe(
    *,
    recipe_id: str | None,
    model_name: str,
) -> str:
    resolved_recipe_id = resolve_feature_recipe_id(recipe_id)
    spec = get_model_spec(model_name)
    if resolved_recipe_id not in set(spec.allowed_feature_recipe_ids):
        allowed = ", ".join(spec.allowed_feature_recipe_ids)
        raise ValueError(
            f"{spec.logical_name} only supports feature_recipe_id values: {allowed}. "
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
    feature_recipe_id: str | None = None,
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
