from __future__ import annotations

from Thesis_ML.experiments.backends.common import (
    BackendResolution,
    BackendSupport,
    effective_backend_family_for_resolution,
    normalize_model_name,
)
from Thesis_ML.experiments.backends.cpu_reference import (
    CPU_REFERENCE_BACKEND_ID,
    resolve_cpu_reference_constructor,
)
from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy


def _unsupported_backend_message(
    *,
    model_name: str,
    effective_backend_family: str,
) -> str:
    return (
        "Unsupported backend resolution for PR 2: "
        f"model='{model_name}', effective_backend_family='{effective_backend_family}'. "
        "Only the CPU reference backend ('sklearn_cpu') is implemented."
    )


def resolve_backend_support(
    model_name: str,
    compute_policy: ResolvedComputePolicy | None = None,
) -> BackendSupport:
    normalized_model_name = normalize_model_name(model_name)
    effective_backend_family = effective_backend_family_for_resolution(compute_policy)

    if effective_backend_family != "sklearn_cpu":
        return BackendSupport(
            model_name=normalized_model_name,
            effective_backend_family=effective_backend_family,
            supported=False,
            backend_id=None,
            reason=_unsupported_backend_message(
                model_name=normalized_model_name,
                effective_backend_family=effective_backend_family,
            ),
        )

    try:
        resolve_cpu_reference_constructor(normalized_model_name)
    except ValueError as exc:
        return BackendSupport(
            model_name=normalized_model_name,
            effective_backend_family=effective_backend_family,
            supported=False,
            backend_id=None,
            reason=str(exc),
        )

    return BackendSupport(
        model_name=normalized_model_name,
        effective_backend_family=effective_backend_family,
        supported=True,
        backend_id=CPU_REFERENCE_BACKEND_ID,
        reason=None,
    )


def resolve_backend_constructor(
    model_name: str,
    compute_policy: ResolvedComputePolicy | None = None,
) -> BackendResolution:
    support = resolve_backend_support(model_name, compute_policy)
    if not support.supported or support.backend_id is None:
        raise ValueError(
            support.reason
            or _unsupported_backend_message(
                model_name=normalize_model_name(model_name),
                effective_backend_family=effective_backend_family_for_resolution(compute_policy),
            )
        )

    constructor = resolve_cpu_reference_constructor(support.model_name)
    return BackendResolution(
        model_name=support.model_name,
        effective_backend_family=support.effective_backend_family,
        backend_id=support.backend_id,
        constructor=constructor,
        compute_policy=compute_policy,
    )


__all__ = [
    "BackendResolution",
    "BackendSupport",
    "resolve_backend_constructor",
    "resolve_backend_support",
]
