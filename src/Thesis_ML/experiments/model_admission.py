from __future__ import annotations

from typing import Literal

from Thesis_ML.config.framework_mode import FrameworkMode, coerce_framework_mode
from Thesis_ML.experiments.model_registry import get_model_spec, iter_model_specs

HardwareMode = Literal["cpu_only", "gpu_only", "max_both"]


def model_is_exploratory_only(model_name: str) -> bool:
    spec = get_model_spec(model_name)
    return spec.official_admission.status == "exploratory_only"


def model_is_official(model_name: str) -> bool:
    spec = get_model_spec(model_name)
    return spec.official_admission.status == "official"


def model_allowed_in_locked_comparison(model_name: str) -> bool:
    return bool(get_model_spec(model_name).official_admission.locked_comparison_allowed)


def model_allowed_in_confirmatory(model_name: str) -> bool:
    return bool(get_model_spec(model_name).official_admission.confirmatory_allowed)


def admitted_models_for_framework(
    framework_mode: FrameworkMode | str,
) -> tuple[str, ...]:
    resolved_mode = coerce_framework_mode(framework_mode)
    admitted: list[str] = []
    for spec in iter_model_specs():
        if resolved_mode == FrameworkMode.CONFIRMATORY and spec.official_admission.confirmatory_allowed:
            admitted.append(spec.logical_name)
        if (
            resolved_mode == FrameworkMode.LOCKED_COMPARISON
            and spec.official_admission.locked_comparison_allowed
        ):
            admitted.append(spec.logical_name)
        if resolved_mode == FrameworkMode.EXPLORATORY:
            admitted.append(spec.logical_name)
    return tuple(admitted)


def official_gpu_only_backend_pairs(
    framework_mode: FrameworkMode | str,
) -> tuple[tuple[str, str], ...]:
    resolved_mode = coerce_framework_mode(framework_mode)
    pairs: list[tuple[str, str]] = []
    for spec in iter_model_specs():
        if resolved_mode == FrameworkMode.CONFIRMATORY:
            families = spec.official_admission.confirmatory_gpu_only_backend_families
        elif resolved_mode == FrameworkMode.LOCKED_COMPARISON:
            families = spec.official_admission.locked_comparison_gpu_only_backend_families
        else:
            families = ()
        for family in families:
            pairs.append((spec.logical_name, str(family)))
    return tuple(sorted(pairs))


def official_max_both_gpu_lane_backend_pairs(
    framework_mode: FrameworkMode | str,
) -> tuple[tuple[str, str], ...]:
    resolved_mode = coerce_framework_mode(framework_mode)
    if resolved_mode != FrameworkMode.LOCKED_COMPARISON:
        return ()
    pairs: list[tuple[str, str]] = []
    for spec in iter_model_specs():
        for family in spec.official_admission.locked_comparison_max_both_gpu_lane_backend_families:
            pairs.append((spec.logical_name, str(family)))
    return tuple(sorted(pairs))


def official_gpu_only_model_backend_allowed(
    *,
    framework_mode: FrameworkMode | str,
    model_name: str,
    backend_family: str,
) -> bool:
    normalized_backend = str(backend_family).strip().lower()
    spec = get_model_spec(model_name)
    resolved_mode = coerce_framework_mode(framework_mode)
    if resolved_mode == FrameworkMode.CONFIRMATORY:
        allowed = spec.official_admission.confirmatory_gpu_only_backend_families
    elif resolved_mode == FrameworkMode.LOCKED_COMPARISON:
        allowed = spec.official_admission.locked_comparison_gpu_only_backend_families
    else:
        return False
    return normalized_backend in {str(value).strip().lower() for value in allowed}


def official_max_both_gpu_lane_eligible(
    *,
    framework_mode: FrameworkMode | str,
    model_name: str,
    backend_family: str,
) -> bool:
    resolved_mode = coerce_framework_mode(framework_mode)
    if resolved_mode != FrameworkMode.LOCKED_COMPARISON:
        return False
    normalized_backend = str(backend_family).strip().lower()
    allowed = get_model_spec(model_name).official_admission.locked_comparison_max_both_gpu_lane_backend_families
    return normalized_backend in {str(value).strip().lower() for value in allowed}


def official_hardware_mode_allowed(
    *,
    framework_mode: FrameworkMode | str,
    hardware_mode: HardwareMode | str,
) -> bool:
    resolved_mode = coerce_framework_mode(framework_mode)
    normalized_mode = str(hardware_mode).strip().lower()
    if resolved_mode == FrameworkMode.EXPLORATORY:
        return normalized_mode in {"cpu_only", "gpu_only", "max_both"}
    if resolved_mode == FrameworkMode.CONFIRMATORY:
        return normalized_mode in {"cpu_only", "gpu_only"}
    if resolved_mode == FrameworkMode.LOCKED_COMPARISON:
        return normalized_mode in {"cpu_only", "gpu_only", "max_both"}
    return False


def official_deterministic_compute_required(
    *,
    framework_mode: FrameworkMode | str,
    hardware_mode: HardwareMode | str,
) -> bool:
    resolved_mode = coerce_framework_mode(framework_mode)
    normalized_mode = str(hardware_mode).strip().lower()
    if resolved_mode not in {FrameworkMode.CONFIRMATORY, FrameworkMode.LOCKED_COMPARISON}:
        return False
    if normalized_mode == "gpu_only":
        return True
    if normalized_mode == "max_both" and resolved_mode == FrameworkMode.LOCKED_COMPARISON:
        return True
    return False


def official_backend_fallback_allowed(
    *,
    framework_mode: FrameworkMode | str,
    hardware_mode: HardwareMode | str,
) -> bool:
    resolved_mode = coerce_framework_mode(framework_mode)
    if resolved_mode in {FrameworkMode.CONFIRMATORY, FrameworkMode.LOCKED_COMPARISON}:
        return False
    return True


def official_admission_summary(
    *,
    framework_mode: FrameworkMode | str,
    model_name: str,
    backend_family: str | None,
    hardware_mode: HardwareMode | str,
) -> dict[str, object]:
    resolved_mode = coerce_framework_mode(framework_mode)
    normalized_backend = str(backend_family).strip().lower() if backend_family is not None else None
    normalized_hardware = str(hardware_mode).strip().lower()

    if resolved_mode == FrameworkMode.EXPLORATORY:
        admitted = True
    elif resolved_mode == FrameworkMode.CONFIRMATORY:
        admitted = model_allowed_in_confirmatory(model_name)
    else:
        admitted = model_allowed_in_locked_comparison(model_name)

    gpu_only_admitted = False
    max_both_gpu_lane_eligible = False
    if normalized_backend is not None:
        gpu_only_admitted = official_gpu_only_model_backend_allowed(
            framework_mode=resolved_mode,
            model_name=model_name,
            backend_family=normalized_backend,
        )
        max_both_gpu_lane_eligible = official_max_both_gpu_lane_eligible(
            framework_mode=resolved_mode,
            model_name=model_name,
            backend_family=normalized_backend,
        )

    return {
        "framework_mode": resolved_mode.value,
        "model": str(model_name).strip().lower(),
        "backend_family": normalized_backend,
        "hardware_mode": normalized_hardware,
        "model_admitted_for_framework": bool(admitted),
        "gpu_only_model_backend_admitted": bool(gpu_only_admitted),
        "max_both_gpu_lane_eligible": bool(max_both_gpu_lane_eligible),
        "deterministic_compute_required": bool(
            official_deterministic_compute_required(
                framework_mode=resolved_mode,
                hardware_mode=normalized_hardware,
            )
        ),
        "allow_backend_fallback": bool(
            official_backend_fallback_allowed(
                framework_mode=resolved_mode,
                hardware_mode=normalized_hardware,
            )
        ),
    }


__all__ = [
    "HardwareMode",
    "admitted_models_for_framework",
    "model_allowed_in_confirmatory",
    "model_allowed_in_locked_comparison",
    "model_is_exploratory_only",
    "model_is_official",
    "official_admission_summary",
    "official_backend_fallback_allowed",
    "official_deterministic_compute_required",
    "official_gpu_only_backend_pairs",
    "official_gpu_only_model_backend_allowed",
    "official_hardware_mode_allowed",
    "official_max_both_gpu_lane_backend_pairs",
    "official_max_both_gpu_lane_eligible",
]
