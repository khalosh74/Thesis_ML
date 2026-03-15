from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

MODEL_NAMES = ("logreg", "linearsvc", "ridge")
CONTROL_MODEL_NAMES = ("dummy",)
ALL_MODEL_NAMES = MODEL_NAMES + CONTROL_MODEL_NAMES


def _resolve_class_weight(class_weight_policy: str) -> str | None:
    normalized = str(class_weight_policy).strip().lower()
    if normalized == "none":
        return None
    if normalized == "balanced":
        return "balanced"
    raise ValueError(
        f"Unsupported class_weight_policy '{class_weight_policy}'. Allowed values: none, balanced."
    )


def make_model(name: str, seed: int, class_weight_policy: str = "none") -> Any:
    # Keep model hyperparameters fixed across runs to avoid hidden dependence on
    # full selected dataset geometry before fold-level train/test splitting.
    class_weight = _resolve_class_weight(class_weight_policy)
    if name == "logreg":
        return LogisticRegression(
            solver="saga",
            max_iter=5000,
            random_state=seed,
            class_weight=class_weight,
        )
    if name == "linearsvc":
        return LinearSVC(
            dual=True,
            random_state=seed,
            max_iter=5000,
            class_weight=class_weight,
        )
    if name == "ridge":
        return RidgeClassifier(random_state=seed, class_weight=class_weight)
    if name == "dummy":
        # Deterministic fixed baseline used in protocol-level control suites.
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(
    model_name: str,
    seed: int,
    class_weight_policy: str = "none",
) -> Pipeline:
    model = make_model(
        name=model_name,
        seed=seed,
        class_weight_policy=class_weight_policy,
    )
    # fMRI voxel vectors are dense numeric arrays; centered scaling is appropriate.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", model),
        ]
    )
