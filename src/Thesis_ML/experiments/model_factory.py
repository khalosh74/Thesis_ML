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


def make_model(name: str, seed: int) -> Any:
    # Keep model hyperparameters fixed across runs to avoid hidden dependence on
    # full selected dataset geometry before fold-level train/test splitting.
    if name == "logreg":
        return LogisticRegression(
            solver="saga",
            max_iter=5000,
            random_state=seed,
        )
    if name == "linearsvc":
        return LinearSVC(dual=True, random_state=seed, max_iter=5000)
    if name == "ridge":
        return RidgeClassifier(random_state=seed)
    if name == "dummy":
        # Deterministic fixed baseline used in protocol-level control suites.
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(model_name: str, seed: int) -> Pipeline:
    model = make_model(name=model_name, seed=seed)
    # fMRI voxel vectors are dense numeric arrays; centered scaling is appropriate.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", model),
        ]
    )
