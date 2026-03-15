from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, make_scorer

MetricName = Literal["balanced_accuracy", "macro_f1", "accuracy"]
SUPPORTED_CLASSIFICATION_METRICS = frozenset({"balanced_accuracy", "macro_f1", "accuracy"})


def _macro_f1(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


_METRIC_FNS: dict[str, Callable[[list[str] | np.ndarray, list[str] | np.ndarray], float]] = {
    "balanced_accuracy": lambda y_true, y_pred: float(
        balanced_accuracy_score(y_true, y_pred)
    ),
    "macro_f1": _macro_f1,
    "accuracy": lambda y_true, y_pred: float(accuracy_score(y_true, y_pred)),
}


def validate_metric_name(metric_name: str) -> str:
    normalized = str(metric_name).strip()
    if normalized not in SUPPORTED_CLASSIFICATION_METRICS:
        allowed = ", ".join(sorted(SUPPORTED_CLASSIFICATION_METRICS))
        raise ValueError(f"Unsupported metric '{metric_name}'. Allowed values: {allowed}.")
    return normalized


def classification_metric_score(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    metric_name: str,
) -> float:
    normalized = validate_metric_name(metric_name)
    return _METRIC_FNS[normalized](y_true, y_pred)


def metric_scorer(metric_name: str):
    normalized = validate_metric_name(metric_name)
    return make_scorer(_METRIC_FNS[normalized])


def metric_bundle(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    metric_names: Iterable[str] | None = None,
) -> dict[str, float]:
    names = list(metric_names) if metric_names is not None else sorted(SUPPORTED_CLASSIFICATION_METRICS)
    scores: dict[str, float] = {}
    for metric_name in names:
        normalized = validate_metric_name(metric_name)
        scores[normalized] = classification_metric_score(
            y_true=y_true,
            y_pred=y_pred,
            metric_name=normalized,
        )
    return scores

