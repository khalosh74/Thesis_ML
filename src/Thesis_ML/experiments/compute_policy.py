from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from Thesis_ML.config.framework_mode import FrameworkMode, coerce_framework_mode
from Thesis_ML.experiments.compute_capabilities import (
    ComputeCapabilitySnapshot,
    detect_compute_capabilities,
)

HardwareMode = Literal["cpu_only", "gpu_only", "max_both"]
BackendFamily = Literal["sklearn_cpu", "torch_gpu", "auto_mixed"]

HARDWARE_MODE_CHOICES: tuple[HardwareMode, ...] = ("cpu_only", "gpu_only", "max_both")
CPU_ONLY: HardwareMode = "cpu_only"
GPU_ONLY: HardwareMode = "gpu_only"
MAX_BOTH: HardwareMode = "max_both"
CPU_REFERENCE_BACKEND_STACK_ID = "sklearn_cpu_reference_v1"
GPU_BACKEND_NOT_IMPLEMENTED_REASON = "gpu_backend_not_implemented_pr1"
TORCH_GPU_BACKEND_STACK_ID_FALLBACK = "torch_gpu_reference_v1"
COMPUTE_POLICY_FIELD_NAMES: tuple[str, ...] = (
    "hardware_mode_requested",
    "hardware_mode_effective",
    "requested_backend_family",
    "effective_backend_family",
    "gpu_device_id",
    "gpu_device_name",
    "gpu_device_total_memory_mb",
    "deterministic_compute",
    "allow_backend_fallback",
    "backend_stack_id",
    "backend_fallback_used",
    "backend_fallback_reason",
    "gpu_memory_peak_mb",
    "device_transfer_seconds",
    "torch_deterministic_enforced",
    "torch_deterministic_limitations",
)


@dataclass(frozen=True)
class ResolvedComputePolicy:
    hardware_mode_requested: HardwareMode
    hardware_mode_effective: HardwareMode
    requested_backend_family: BackendFamily
    effective_backend_family: BackendFamily
    gpu_device_id: int | None
    gpu_device_name: str | None
    gpu_device_total_memory_mb: int | None
    deterministic_compute: bool
    allow_backend_fallback: bool
    backend_stack_id: str
    backend_fallback_used: bool
    backend_fallback_reason: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "hardware_mode_requested": str(self.hardware_mode_requested),
            "hardware_mode_effective": str(self.hardware_mode_effective),
            "requested_backend_family": str(self.requested_backend_family),
            "effective_backend_family": str(self.effective_backend_family),
            "gpu_device_id": self.gpu_device_id,
            "gpu_device_name": self.gpu_device_name,
            "gpu_device_total_memory_mb": self.gpu_device_total_memory_mb,
            "deterministic_compute": bool(self.deterministic_compute),
            "allow_backend_fallback": bool(self.allow_backend_fallback),
            "backend_stack_id": str(self.backend_stack_id),
            "backend_fallback_used": bool(self.backend_fallback_used),
            "backend_fallback_reason": self.backend_fallback_reason,
        }


def normalize_hardware_mode(value: HardwareMode | str) -> HardwareMode:
    normalized = str(value).strip().lower()
    if normalized not in HARDWARE_MODE_CHOICES:
        allowed = ", ".join(HARDWARE_MODE_CHOICES)
        raise ValueError(
            f"Unsupported hardware_mode '{value}'. Allowed values: {allowed}."
        )
    return cast(HardwareMode, normalized)


def requested_backend_family_for_mode(hardware_mode: HardwareMode | str) -> BackendFamily:
    normalized = normalize_hardware_mode(hardware_mode)
    if normalized == CPU_ONLY:
        return "sklearn_cpu"
    if normalized == GPU_ONLY:
        return "torch_gpu"
    return "auto_mixed"


def _cpu_reference_policy(
    *,
    requested_mode: HardwareMode,
    deterministic_compute: bool,
    allow_backend_fallback: bool,
) -> ResolvedComputePolicy:
    return ResolvedComputePolicy(
        hardware_mode_requested=requested_mode,
        hardware_mode_effective=CPU_ONLY,
        requested_backend_family=requested_backend_family_for_mode(requested_mode),
        effective_backend_family="sklearn_cpu",
        gpu_device_id=None,
        gpu_device_name=None,
        gpu_device_total_memory_mb=None,
        deterministic_compute=bool(deterministic_compute),
        allow_backend_fallback=bool(allow_backend_fallback),
        backend_stack_id=CPU_REFERENCE_BACKEND_STACK_ID,
        backend_fallback_used=False,
        backend_fallback_reason=None,
    )


def _gpu_compatible(snapshot: ComputeCapabilitySnapshot) -> bool:
    return bool(snapshot.gpu_available and snapshot.device_id is not None)


def _compatibility_error_message(
    *,
    hardware_mode: HardwareMode,
    snapshot: ComputeCapabilitySnapshot,
) -> str:
    reasons = ", ".join(snapshot.incompatibility_reasons) or snapshot.compatibility_status
    return (
        f"hardware_mode='{hardware_mode}' requires compatible GPU capability. "
        f"compatibility_status='{snapshot.compatibility_status}'. reasons={reasons}"
    )


def resolve_compute_policy(
    *,
    framework_mode: FrameworkMode | str,
    hardware_mode: HardwareMode | str = CPU_ONLY,
    gpu_device_id: int | None = None,
    deterministic_compute: bool = False,
    allow_backend_fallback: bool = False,
    capability_snapshot: ComputeCapabilitySnapshot | None = None,
) -> ResolvedComputePolicy:
    resolved_framework_mode = coerce_framework_mode(framework_mode)
    requested_mode = normalize_hardware_mode(hardware_mode)

    if requested_mode == CPU_ONLY and gpu_device_id is not None:
        raise ValueError(
            "hardware_mode='cpu_only' cannot be combined with gpu_device_id. "
            "GPU device selection is valid only for gpu_only or max_both."
        )

    if resolved_framework_mode in {
        FrameworkMode.CONFIRMATORY,
        FrameworkMode.LOCKED_COMPARISON,
    }:
        if requested_mode != CPU_ONLY:
            raise ValueError(
                "Official compute controls are conservative in the current rollout: "
                "hardware_mode must remain 'cpu_only' for confirmatory and locked comparison runs."
            )
        if allow_backend_fallback:
            raise ValueError(
                "allow_backend_fallback is exploratory-only in the current rollout and is not "
                "allowed for official runs."
            )
        return _cpu_reference_policy(
            requested_mode=requested_mode,
            deterministic_compute=deterministic_compute,
            allow_backend_fallback=False,
        )

    if requested_mode == CPU_ONLY:
        return _cpu_reference_policy(
            requested_mode=requested_mode,
            deterministic_compute=deterministic_compute,
            allow_backend_fallback=allow_backend_fallback,
        )

    snapshot = capability_snapshot or detect_compute_capabilities(
        requested_device_id=gpu_device_id
    )

    if requested_mode == GPU_ONLY:
        if _gpu_compatible(snapshot):
            return ResolvedComputePolicy(
                hardware_mode_requested=requested_mode,
                hardware_mode_effective=GPU_ONLY,
                requested_backend_family="torch_gpu",
                effective_backend_family="torch_gpu",
                gpu_device_id=snapshot.device_id,
                gpu_device_name=snapshot.device_name,
                gpu_device_total_memory_mb=snapshot.device_total_memory_mb,
                deterministic_compute=bool(deterministic_compute),
                allow_backend_fallback=bool(allow_backend_fallback),
                backend_stack_id=(
                    str(snapshot.tested_stack_id).strip() or TORCH_GPU_BACKEND_STACK_ID_FALLBACK
                ),
                backend_fallback_used=False,
                backend_fallback_reason=None,
            )
        if not allow_backend_fallback:
            raise ValueError(
                _compatibility_error_message(
                    hardware_mode=requested_mode,
                    snapshot=snapshot,
                )
            )
        return ResolvedComputePolicy(
            hardware_mode_requested=requested_mode,
            hardware_mode_effective=CPU_ONLY,
            requested_backend_family="torch_gpu",
            effective_backend_family="sklearn_cpu",
            gpu_device_id=None,
            gpu_device_name=None,
            gpu_device_total_memory_mb=None,
            deterministic_compute=bool(deterministic_compute),
            allow_backend_fallback=True,
            backend_stack_id=CPU_REFERENCE_BACKEND_STACK_ID,
            backend_fallback_used=True,
            backend_fallback_reason=(
                "gpu_capability_unavailable:"
                f"{str(snapshot.compatibility_status).strip() or 'unknown'}"
            ),
        )

    if snapshot.requested_device_visible is False:
        raise ValueError(
            _compatibility_error_message(
                hardware_mode=requested_mode,
                snapshot=snapshot,
            )
        )

    if _gpu_compatible(snapshot):
        return ResolvedComputePolicy(
            hardware_mode_requested=requested_mode,
            hardware_mode_effective=MAX_BOTH,
            requested_backend_family="auto_mixed",
            effective_backend_family="sklearn_cpu",
            gpu_device_id=snapshot.device_id,
            gpu_device_name=snapshot.device_name,
            gpu_device_total_memory_mb=snapshot.device_total_memory_mb,
            deterministic_compute=bool(deterministic_compute),
            allow_backend_fallback=bool(allow_backend_fallback),
            backend_stack_id=CPU_REFERENCE_BACKEND_STACK_ID,
            backend_fallback_used=False,
            backend_fallback_reason=None,
        )

    return ResolvedComputePolicy(
        hardware_mode_requested=requested_mode,
        hardware_mode_effective=CPU_ONLY,
        requested_backend_family="auto_mixed",
        effective_backend_family="sklearn_cpu",
        gpu_device_id=None,
        gpu_device_name=None,
        gpu_device_total_memory_mb=None,
        deterministic_compute=bool(deterministic_compute),
        allow_backend_fallback=bool(allow_backend_fallback),
        backend_stack_id=CPU_REFERENCE_BACKEND_STACK_ID,
        backend_fallback_used=False,
        backend_fallback_reason=None,
    )


def extract_compute_policy_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    nested = payload.get("compute_policy")
    source = nested if isinstance(nested, dict) else payload
    resolved = {
        key: source.get(key)
        for key in COMPUTE_POLICY_FIELD_NAMES
        if key in source
    }
    if not resolved:
        return None
    required = {
        "hardware_mode_requested",
        "hardware_mode_effective",
        "requested_backend_family",
        "effective_backend_family",
        "backend_stack_id",
    }
    if not required.issubset(resolved):
        return None
    return resolved


def stamp_compute_policy_metadata(
    *,
    payload: dict[str, Any],
    compute_policy: ResolvedComputePolicy | dict[str, Any] | None,
    compute_runtime_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if compute_policy is None:
        return payload
    if isinstance(compute_policy, ResolvedComputePolicy):
        compute_policy_payload = compute_policy.to_payload()
    elif isinstance(compute_policy, dict):
        extracted = extract_compute_policy_payload(compute_policy)
        if extracted is None:
            return payload
        compute_policy_payload = extracted
    else:
        return payload
    runtime_metadata: dict[str, Any] = {}
    if isinstance(compute_runtime_metadata, dict):
        for key in (
            "gpu_memory_peak_mb",
            "device_transfer_seconds",
            "torch_deterministic_enforced",
            "torch_deterministic_limitations",
        ):
            if key in compute_runtime_metadata:
                runtime_metadata[key] = compute_runtime_metadata.get(key)
    payload.update(compute_policy_payload)
    if runtime_metadata:
        payload.update(runtime_metadata)
    nested = dict(compute_policy_payload)
    nested.update(runtime_metadata)
    payload["compute_policy"] = nested
    return payload


__all__ = [
    "BackendFamily",
    "COMPUTE_POLICY_FIELD_NAMES",
    "CPU_ONLY",
    "CPU_REFERENCE_BACKEND_STACK_ID",
    "GPU_BACKEND_NOT_IMPLEMENTED_REASON",
    "GPU_ONLY",
    "HARDWARE_MODE_CHOICES",
    "HardwareMode",
    "MAX_BOTH",
    "TORCH_GPU_BACKEND_STACK_ID_FALLBACK",
    "ResolvedComputePolicy",
    "extract_compute_policy_payload",
    "normalize_hardware_mode",
    "requested_backend_family_for_mode",
    "resolve_compute_policy",
    "stamp_compute_policy_metadata",
]
