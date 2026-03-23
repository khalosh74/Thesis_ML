from __future__ import annotations

from typing import Any, Literal

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler

from Thesis_ML.experiments.backend_registry import resolve_backend_constructor
from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy

MODEL_NAMES = ("logreg", "linearsvc", "ridge", "xgboost")
CONTROL_MODEL_NAMES = ("dummy",)
ALL_MODEL_NAMES = MODEL_NAMES + CONTROL_MODEL_NAMES
EXPLORATORY_ONLY_MODEL_NAMES = ("xgboost",)
DEFAULT_BATCH_MODEL_NAMES = tuple(
    name for name in MODEL_NAMES if name not in EXPLORATORY_ONLY_MODEL_NAMES
)
OFFICIAL_MODEL_NAMES = tuple(
    name for name in ALL_MODEL_NAMES if name not in EXPLORATORY_ONLY_MODEL_NAMES
)

PreprocessKind = Literal["standard_scaler", "passthrough"]

_MODEL_PREPROCESS_KINDS: dict[str, PreprocessKind] = {
    "dummy": "standard_scaler",
    "linearsvc": "standard_scaler",
    "logreg": "standard_scaler",
    "ridge": "standard_scaler",
    "xgboost": "passthrough",
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
        raise ValueError(
            f"Unsupported model '{model_name}'. Allowed values: {allowed}."
        )
    return preprocess_kind


def model_supports_linear_interpretability(model_name: str) -> bool:
    return str(model_name).strip().lower() in {"logreg", "linearsvc", "ridge"}


def model_is_officially_admitted(model_name: str) -> bool:
    return str(model_name).strip().lower() in set(OFFICIAL_MODEL_NAMES)


def _build_pipeline_preprocessor(model_name: str) -> Any:
    preprocess_kind = model_preprocess_kind(model_name)
    if preprocess_kind == "standard_scaler":
        # fMRI voxel vectors are dense numeric arrays; centered scaling is appropriate.
        return StandardScaler(with_mean=True, with_std=True)
    if preprocess_kind == "passthrough":
        return FunctionTransformer(validate=False)
    raise ValueError(f"Unsupported preprocess kind '{preprocess_kind}'.")


def build_pipeline(
    model_name: str,
    seed: int,
    class_weight_policy: str = "none",
    compute_policy: ResolvedComputePolicy | None = None,
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
            ("scaler", _build_pipeline_preprocessor(model_name)),
            ("model", model),
        ]
    )
