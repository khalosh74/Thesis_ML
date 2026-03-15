from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, make_scorer

MetricName = Literal["balanced_accuracy", "macro_f1", "accuracy"]
SUPPORTED_CLASSIFICATION_METRICS = frozenset({"balanced_accuracy", "macro_f1", "accuracy"})
_METRIC_ALIASES: dict[str, str] = {
    "balanced accuracy": "balanced_accuracy",
    "balanced-accuracy": "balanced_accuracy",
    "macro f1": "macro_f1",
    "macro-f1": "macro_f1",
    "f1_macro": "macro_f1",
    "f1-macro": "macro_f1",
    "f1 macro": "macro_f1",
}


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
_METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "balanced_accuracy": True,
    "macro_f1": True,
    "accuracy": True,
}


@dataclass(frozen=True)
class EffectiveMetricPolicy:
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    decision_metric: str
    tuning_metric: str
    permutation_metric: str
    higher_is_better: bool


def validate_metric_name(metric_name: str) -> str:
    raw = str(metric_name).strip()
    lowered = raw.lower()
    normalized = _METRIC_ALIASES.get(lowered, lowered)
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


def metric_higher_is_better(metric_name: str) -> bool:
    normalized = validate_metric_name(metric_name)
    return bool(_METRIC_HIGHER_IS_BETTER[normalized])


def _coerce_metric_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def extract_metric_value(
    payload: Mapping[str, Any],
    metric_name: str,
    *,
    require: bool = False,
    payload_label: str = "metrics payload",
) -> float | None:
    normalized = validate_metric_name(metric_name)
    direct_value = _coerce_metric_float(payload.get(normalized))
    if direct_value is not None:
        return direct_value

    payload_primary_name = payload.get("primary_metric_name")
    if isinstance(payload_primary_name, str):
        payload_primary_name_normalized = validate_metric_name(payload_primary_name)
        if payload_primary_name_normalized == normalized:
            primary_value = _coerce_metric_float(payload.get("primary_metric_value"))
            if primary_value is not None:
                return primary_value

    if require:
        raise ValueError(
            f"{payload_label} is missing required metric '{normalized}' "
            "(expected direct metric field or matching primary_metric_name/primary_metric_value)."
        )
    return None


def resolve_effective_metric_policy(
    *,
    primary_metric: str,
    secondary_metrics: Iterable[str] | None = None,
    decision_metric: str | None = None,
    tuning_metric: str | None = None,
    permutation_metric: str | None = None,
) -> EffectiveMetricPolicy:
    resolved_primary = validate_metric_name(primary_metric)
    resolved_secondary: list[str] = []
    for metric_name in list(secondary_metrics or []):
        normalized = validate_metric_name(metric_name)
        if normalized == resolved_primary:
            continue
        if normalized in resolved_secondary:
            continue
        resolved_secondary.append(normalized)

    resolved_decision = validate_metric_name(decision_metric or resolved_primary)
    resolved_tuning = validate_metric_name(tuning_metric or resolved_primary)
    resolved_permutation = validate_metric_name(permutation_metric or resolved_primary)
    return EffectiveMetricPolicy(
        primary_metric=resolved_primary,
        secondary_metrics=tuple(resolved_secondary),
        decision_metric=resolved_decision,
        tuning_metric=resolved_tuning,
        permutation_metric=resolved_permutation,
        higher_is_better=metric_higher_is_better(resolved_primary),
    )


def enforce_primary_metric_alignment(
    effective_policy: EffectiveMetricPolicy,
    *,
    context: str,
) -> EffectiveMetricPolicy:
    mismatches: list[str] = []
    if effective_policy.decision_metric != effective_policy.primary_metric:
        mismatches.append(
            "decision_metric="
            + effective_policy.decision_metric
            + f" != primary_metric={effective_policy.primary_metric}"
        )
    if effective_policy.tuning_metric != effective_policy.primary_metric:
        mismatches.append(
            "tuning_metric="
            + effective_policy.tuning_metric
            + f" != primary_metric={effective_policy.primary_metric}"
        )
    if effective_policy.permutation_metric != effective_policy.primary_metric:
        mismatches.append(
            "permutation_metric="
            + effective_policy.permutation_metric
            + f" != primary_metric={effective_policy.primary_metric}"
        )
    if mismatches:
        raise ValueError(
            f"{context} metric policy drift detected: " + "; ".join(mismatches)
        )
    return effective_policy
