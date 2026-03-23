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
from Thesis_ML.experiments.backends.torch_logreg import (
    TORCH_LOGREG_BACKEND_ID,
    make_torch_logreg_estimator,
)
from Thesis_ML.experiments.backends.torch_ridge import (
    TORCH_RIDGE_BACKEND_ID,
    make_torch_ridge_estimator,
)
from Thesis_ML.experiments.backends.xgboost_cpu import (
    XGBOOST_CPU_BACKEND_ID,
    make_xgboost_cpu_estimator,
    xgboost_cpu_support_status,
)
from Thesis_ML.experiments.backends.xgboost_gpu import (
    XGBOOST_GPU_BACKEND_ID,
    make_xgboost_gpu_estimator,
    xgboost_gpu_support_status,
)
from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy


def _unsupported_backend_message(
    *,
    model_name: str,
    effective_backend_family: str,
) -> str:
    return (
        "Unsupported backend resolution: "
        f"model='{model_name}', effective_backend_family='{effective_backend_family}'. "
        "Supported backend families are: "
        "'sklearn_cpu' (cpu_reference + xgboost_cpu) and "
        "'torch_gpu' (torch ridge/logreg + xgboost_gpu)."
    )


def _unsupported_torch_gpu_model_message(model_name: str) -> str:
    return (
        "Unsupported torch_gpu backend request: "
        f"model='{model_name}'. Supported models: ridge, logreg, xgboost."
    )


def resolve_backend_support(
    model_name: str,
    compute_policy: ResolvedComputePolicy | None = None,
) -> BackendSupport:
    normalized_model_name = normalize_model_name(model_name)
    effective_backend_family = effective_backend_family_for_resolution(compute_policy)

    if effective_backend_family == "sklearn_cpu":
        if normalized_model_name == "xgboost":
            supported, reason = xgboost_cpu_support_status()
            return BackendSupport(
                model_name=normalized_model_name,
                effective_backend_family=effective_backend_family,
                supported=bool(supported),
                backend_id=(XGBOOST_CPU_BACKEND_ID if supported else None),
                reason=(None if supported else str(reason or "xgboost_cpu_backend_unavailable")),
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

    if effective_backend_family == "torch_gpu":
        if normalized_model_name == "xgboost":
            gpu_device_id = (
                int(compute_policy.gpu_device_id)
                if compute_policy is not None and compute_policy.gpu_device_id is not None
                else None
            )
            supported, reason = xgboost_gpu_support_status(gpu_device_id=gpu_device_id)
            return BackendSupport(
                model_name=normalized_model_name,
                effective_backend_family=effective_backend_family,
                supported=bool(supported),
                backend_id=(XGBOOST_GPU_BACKEND_ID if supported else None),
                reason=(None if supported else str(reason or "xgboost_gpu_backend_unavailable")),
            )
        if normalized_model_name not in {"ridge", "logreg"}:
            return BackendSupport(
                model_name=normalized_model_name,
                effective_backend_family=effective_backend_family,
                supported=False,
                backend_id=None,
                reason=_unsupported_torch_gpu_model_message(normalized_model_name),
            )
        backend_id = (
            TORCH_RIDGE_BACKEND_ID
            if normalized_model_name == "ridge"
            else TORCH_LOGREG_BACKEND_ID
        )
        return BackendSupport(
            model_name=normalized_model_name,
            effective_backend_family=effective_backend_family,
            supported=True,
            backend_id=backend_id,
            reason=None,
        )
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

    if support.backend_id == CPU_REFERENCE_BACKEND_ID:
        constructor = resolve_cpu_reference_constructor(support.model_name)
    elif support.backend_id == XGBOOST_CPU_BACKEND_ID:

        def constructor(*, seed: int, class_weight_policy: str = "none"):
            return make_xgboost_cpu_estimator(
                seed=seed,
                class_weight_policy=class_weight_policy,
            )

    elif support.backend_id == TORCH_RIDGE_BACKEND_ID:
        if compute_policy is None or compute_policy.gpu_device_id is None:
            raise ValueError(
                "torch_gpu backend resolution requires compute_policy with a resolved gpu_device_id."
            )

        def constructor(*, seed: int, class_weight_policy: str = "none"):
            return make_torch_ridge_estimator(
                seed=seed,
                class_weight_policy=class_weight_policy,
                gpu_device_id=int(compute_policy.gpu_device_id),
                deterministic_compute=bool(compute_policy.deterministic_compute),
            )

    elif support.backend_id == TORCH_LOGREG_BACKEND_ID:
        if compute_policy is None or compute_policy.gpu_device_id is None:
            raise ValueError(
                "torch_gpu backend resolution requires compute_policy with a resolved gpu_device_id."
            )

        def constructor(*, seed: int, class_weight_policy: str = "none"):
            return make_torch_logreg_estimator(
                seed=seed,
                class_weight_policy=class_weight_policy,
                gpu_device_id=int(compute_policy.gpu_device_id),
                deterministic_compute=bool(compute_policy.deterministic_compute),
            )

    elif support.backend_id == XGBOOST_GPU_BACKEND_ID:
        if compute_policy is None or compute_policy.gpu_device_id is None:
            raise ValueError(
                "xgboost_gpu backend resolution requires compute_policy with a resolved gpu_device_id."
            )

        def constructor(*, seed: int, class_weight_policy: str = "none"):
            return make_xgboost_gpu_estimator(
                seed=seed,
                class_weight_policy=class_weight_policy,
                gpu_device_id=int(compute_policy.gpu_device_id),
                deterministic_compute=bool(compute_policy.deterministic_compute),
            )

    else:
        raise ValueError(
            _unsupported_backend_message(
                model_name=support.model_name,
                effective_backend_family=support.effective_backend_family,
            )
        )

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
