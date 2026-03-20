from __future__ import annotations

import pytest

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.compute_capabilities import ComputeCapabilitySnapshot
from Thesis_ML.experiments.compute_policy import (
    CPU_REFERENCE_BACKEND_STACK_ID,
    GPU_BACKEND_NOT_IMPLEMENTED_REASON,
    resolve_compute_policy,
)


def _gpu_capability_snapshot(*, device_id: int = 0) -> ComputeCapabilitySnapshot:
    return ComputeCapabilitySnapshot(
        torch_installed=True,
        torch_version="2.4.1",
        cuda_available=True,
        cuda_runtime_version="12.1",
        gpu_available=True,
        gpu_count=2,
        requested_device_visible=True,
        device_id=device_id,
        device_name=f"GPU {device_id}",
        device_total_memory_mb=8192,
        compatibility_status="gpu_compatible",
        incompatibility_reasons=(),
        tested_stack_id="torch_2.4.1__cuda_12.1",
    )


def _missing_gpu_snapshot() -> ComputeCapabilitySnapshot:
    return ComputeCapabilitySnapshot(
        torch_installed=False,
        torch_version=None,
        cuda_available=False,
        cuda_runtime_version=None,
        gpu_available=False,
        gpu_count=0,
        requested_device_visible=None,
        device_id=None,
        device_name=None,
        device_total_memory_mb=None,
        compatibility_status="torch_unavailable",
        incompatibility_reasons=("torch_not_installed",),
        tested_stack_id="torch_unavailable",
    )


def test_cpu_only_resolves_without_gpu_stack() -> None:
    resolved = resolve_compute_policy(
        framework_mode=FrameworkMode.EXPLORATORY,
        hardware_mode="cpu_only",
    )

    assert resolved.hardware_mode_requested == "cpu_only"
    assert resolved.hardware_mode_effective == "cpu_only"
    assert resolved.requested_backend_family == "sklearn_cpu"
    assert resolved.effective_backend_family == "sklearn_cpu"
    assert resolved.backend_stack_id == CPU_REFERENCE_BACKEND_STACK_ID
    assert resolved.backend_fallback_used is False


def test_gpu_only_fails_clearly_when_capability_is_unavailable() -> None:
    with pytest.raises(ValueError, match="requires compatible GPU capability"):
        resolve_compute_policy(
            framework_mode=FrameworkMode.EXPLORATORY,
            hardware_mode="gpu_only",
            capability_snapshot=_missing_gpu_snapshot(),
        )


def test_gpu_only_uses_cpu_reference_fallback_when_allowed() -> None:
    resolved = resolve_compute_policy(
        framework_mode=FrameworkMode.EXPLORATORY,
        hardware_mode="gpu_only",
        allow_backend_fallback=True,
        deterministic_compute=True,
        capability_snapshot=_gpu_capability_snapshot(device_id=1),
    )

    assert resolved.hardware_mode_requested == "gpu_only"
    assert resolved.hardware_mode_effective == "gpu_only"
    assert resolved.requested_backend_family == "torch_gpu"
    assert resolved.effective_backend_family == "sklearn_cpu"
    assert resolved.gpu_device_id == 1
    assert resolved.backend_stack_id == CPU_REFERENCE_BACKEND_STACK_ID
    assert resolved.backend_fallback_used is True
    assert resolved.backend_fallback_reason == GPU_BACKEND_NOT_IMPLEMENTED_REASON
    assert resolved.deterministic_compute is True


def test_max_both_without_gpu_degrades_to_cpu_only_without_fallback() -> None:
    resolved = resolve_compute_policy(
        framework_mode=FrameworkMode.EXPLORATORY,
        hardware_mode="max_both",
        capability_snapshot=_missing_gpu_snapshot(),
    )

    assert resolved.hardware_mode_requested == "max_both"
    assert resolved.hardware_mode_effective == "cpu_only"
    assert resolved.requested_backend_family == "auto_mixed"
    assert resolved.effective_backend_family == "sklearn_cpu"
    assert resolved.backend_fallback_used is False
    assert resolved.backend_fallback_reason is None
    assert resolved.gpu_device_id is None


def test_max_both_with_gpu_capability_records_mode_but_keeps_cpu_backend() -> None:
    resolved = resolve_compute_policy(
        framework_mode=FrameworkMode.EXPLORATORY,
        hardware_mode="max_both",
        capability_snapshot=_gpu_capability_snapshot(device_id=0),
    )

    assert resolved.hardware_mode_effective == "max_both"
    assert resolved.requested_backend_family == "auto_mixed"
    assert resolved.effective_backend_family == "sklearn_cpu"
    assert resolved.backend_fallback_used is False
    assert resolved.gpu_device_id == 0


def test_official_paths_reject_non_cpu_modes_and_backend_fallback() -> None:
    with pytest.raises(ValueError, match="hardware_mode must remain 'cpu_only'"):
        resolve_compute_policy(
            framework_mode=FrameworkMode.CONFIRMATORY,
            hardware_mode="gpu_only",
        )

    with pytest.raises(ValueError, match="exploratory-only"):
        resolve_compute_policy(
            framework_mode=FrameworkMode.LOCKED_COMPARISON,
            hardware_mode="cpu_only",
            allow_backend_fallback=True,
        )
