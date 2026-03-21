from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy

EstimatorConstructor = Callable[..., Any]


def normalize_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    if not normalized:
        raise ValueError("model_name must be a non-empty string.")
    return normalized


def effective_backend_family_for_resolution(
    compute_policy: ResolvedComputePolicy | None,
) -> str:
    if compute_policy is None:
        return "sklearn_cpu"
    return str(compute_policy.effective_backend_family)


@dataclass(frozen=True)
class BackendSupport:
    model_name: str
    effective_backend_family: str
    supported: bool
    backend_id: str | None
    reason: str | None


@dataclass(frozen=True)
class BackendResolution:
    model_name: str
    effective_backend_family: str
    backend_id: str
    constructor: EstimatorConstructor
    compute_policy: ResolvedComputePolicy | None = None

    def build_estimator(
        self,
        *,
        seed: int,
        class_weight_policy: str = "none",
    ) -> Any:
        return self.constructor(
            seed=seed,
            class_weight_policy=class_weight_policy,
        )


__all__ = [
    "BackendResolution",
    "BackendSupport",
    "EstimatorConstructor",
    "effective_backend_family_for_resolution",
    "normalize_model_name",
]
