from __future__ import annotations

from typing import Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Thesis_ML.experiments.backend_registry import resolve_backend_constructor
from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy

MODEL_NAMES = ("logreg", "linearsvc", "ridge")
CONTROL_MODEL_NAMES = ("dummy",)
ALL_MODEL_NAMES = MODEL_NAMES + CONTROL_MODEL_NAMES


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
    # fMRI voxel vectors are dense numeric arrays; centered scaling is appropriate.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", model),
        ]
    )
