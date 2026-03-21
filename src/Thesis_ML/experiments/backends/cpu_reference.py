from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from Thesis_ML.experiments.backends.common import EstimatorConstructor, normalize_model_name

CPU_REFERENCE_BACKEND_ID = "cpu_reference"


def resolve_cpu_reference_class_weight(class_weight_policy: str) -> str | None:
    normalized = str(class_weight_policy).strip().lower()
    if normalized == "none":
        return None
    if normalized == "balanced":
        return "balanced"
    raise ValueError(
        f"Unsupported class_weight_policy '{class_weight_policy}'. Allowed values: none, balanced."
    )


def make_logreg_cpu_reference(
    *,
    seed: int,
    class_weight_policy: str = "none",
) -> LogisticRegression:
    return LogisticRegression(
        solver="saga",
        max_iter=5000,
        random_state=seed,
        class_weight=resolve_cpu_reference_class_weight(class_weight_policy),
    )


def make_linearsvc_cpu_reference(
    *,
    seed: int,
    class_weight_policy: str = "none",
) -> LinearSVC:
    return LinearSVC(
        dual=True,
        random_state=seed,
        max_iter=5000,
        class_weight=resolve_cpu_reference_class_weight(class_weight_policy),
    )


def make_ridge_cpu_reference(
    *,
    seed: int,
    class_weight_policy: str = "none",
) -> RidgeClassifier:
    return RidgeClassifier(
        random_state=seed,
        class_weight=resolve_cpu_reference_class_weight(class_weight_policy),
    )


def make_dummy_cpu_reference(
    *,
    seed: int,
    class_weight_policy: str = "none",
) -> DummyClassifier:
    del seed
    del class_weight_policy
    # Deterministic fixed baseline used in protocol-level control suites.
    return DummyClassifier(strategy="most_frequent")


CPU_REFERENCE_MODEL_CONSTRUCTORS: dict[str, EstimatorConstructor] = {
    "logreg": make_logreg_cpu_reference,
    "linearsvc": make_linearsvc_cpu_reference,
    "ridge": make_ridge_cpu_reference,
    "dummy": make_dummy_cpu_reference,
}


def resolve_cpu_reference_constructor(model_name: str) -> EstimatorConstructor:
    normalized = normalize_model_name(model_name)
    constructor = CPU_REFERENCE_MODEL_CONSTRUCTORS.get(normalized)
    if constructor is None:
        allowed = ", ".join(sorted(CPU_REFERENCE_MODEL_CONSTRUCTORS))
        raise ValueError(
            "Unsupported CPU reference model "
            f"'{model_name}'. Allowed values: {allowed}."
        )
    return constructor


def make_cpu_reference_model(
    name: str,
    seed: int,
    class_weight_policy: str = "none",
) -> Any:
    constructor = resolve_cpu_reference_constructor(name)
    return constructor(
        seed=seed,
        class_weight_policy=class_weight_policy,
    )


def build_cpu_reference_pipeline(
    model_name: str,
    seed: int,
    class_weight_policy: str = "none",
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "model",
                make_cpu_reference_model(
                    name=model_name,
                    seed=seed,
                    class_weight_policy=class_weight_policy,
                ),
            ),
        ]
    )


__all__ = [
    "CPU_REFERENCE_BACKEND_ID",
    "CPU_REFERENCE_MODEL_CONSTRUCTORS",
    "build_cpu_reference_pipeline",
    "make_cpu_reference_model",
    "make_dummy_cpu_reference",
    "make_linearsvc_cpu_reference",
    "make_logreg_cpu_reference",
    "make_ridge_cpu_reference",
    "resolve_cpu_reference_class_weight",
    "resolve_cpu_reference_constructor",
]
