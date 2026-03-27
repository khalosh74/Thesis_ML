from __future__ import annotations

import pytest

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.compute_capabilities import ComputeCapabilitySnapshot
from Thesis_ML.experiments.compute_policy import (
    CPU_REFERENCE_BACKEND_STACK_ID,
    TORCH_GPU_BACKEND_STACK_ID_FALLBACK,
    resolve_compute_policy,
    stamp_compute_policy_metadata,
)
from Thesis_ML.experiments.runtime_policies import resolve_methodology_runtime


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


def test_gpu_only_resolves_to_torch_backend_when_capable() -> None:
    resolved = resolve_compute_policy(
        framework_mode=FrameworkMode.EXPLORATORY,
        hardware_mode="gpu_only",
        deterministic_compute=True,
        capability_snapshot=_gpu_capability_snapshot(device_id=1),
    )

    assert resolved.hardware_mode_requested == "gpu_only"
    assert resolved.hardware_mode_effective == "gpu_only"
    assert resolved.requested_backend_family == "torch_gpu"
    assert resolved.effective_backend_family == "torch_gpu"
    assert resolved.gpu_device_id == 1
    assert resolved.backend_stack_id in {
        "torch_2.4.1__cuda_12.1",
        TORCH_GPU_BACKEND_STACK_ID_FALLBACK,
    }
    assert resolved.backend_fallback_used is False
    assert resolved.backend_fallback_reason is None
    assert resolved.deterministic_compute is True


def test_gpu_only_can_fallback_to_cpu_when_capability_is_unavailable_and_allowed() -> None:
    resolved = resolve_compute_policy(
        framework_mode=FrameworkMode.EXPLORATORY,
        hardware_mode="gpu_only",
        allow_backend_fallback=True,
        capability_snapshot=_missing_gpu_snapshot(),
    )

    assert resolved.hardware_mode_requested == "gpu_only"
    assert resolved.hardware_mode_effective == "cpu_only"
    assert resolved.requested_backend_family == "torch_gpu"
    assert resolved.effective_backend_family == "sklearn_cpu"
    assert resolved.backend_stack_id == CPU_REFERENCE_BACKEND_STACK_ID
    assert resolved.backend_fallback_used is True
    assert resolved.backend_fallback_reason is not None
    assert "gpu_capability_unavailable" in resolved.backend_fallback_reason


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


def test_confirmatory_paths_enforce_pr7_gates() -> None:
    with pytest.raises(ValueError, match="do not admit hardware_mode='max_both'"):
        resolve_compute_policy(
            framework_mode=FrameworkMode.CONFIRMATORY,
            hardware_mode="max_both",
        )

    with pytest.raises(ValueError, match="deterministic_compute=true"):
        resolve_compute_policy(
            framework_mode=FrameworkMode.CONFIRMATORY,
            hardware_mode="gpu_only",
            deterministic_compute=False,
            capability_snapshot=_gpu_capability_snapshot(),
        )

    with pytest.raises(ValueError, match="requires compatible GPU capability"):
        resolve_compute_policy(
            framework_mode=FrameworkMode.CONFIRMATORY,
            hardware_mode="gpu_only",
            deterministic_compute=True,
            capability_snapshot=_missing_gpu_snapshot(),
        )

    with pytest.raises(ValueError, match="exploratory-only"):
        resolve_compute_policy(
            framework_mode=FrameworkMode.CONFIRMATORY,
            hardware_mode="cpu_only",
            allow_backend_fallback=True,
        )


def test_confirmatory_gpu_only_resolves_when_deterministic_and_capable() -> None:
    resolved = resolve_compute_policy(
        framework_mode=FrameworkMode.CONFIRMATORY,
        hardware_mode="gpu_only",
        deterministic_compute=True,
        capability_snapshot=_gpu_capability_snapshot(device_id=0),
    )

    assert resolved.hardware_mode_requested == "gpu_only"
    assert resolved.hardware_mode_effective == "gpu_only"
    assert resolved.requested_backend_family == "torch_gpu"
    assert resolved.effective_backend_family == "torch_gpu"
    assert resolved.gpu_device_id == 0
    assert resolved.deterministic_compute is True
    assert resolved.allow_backend_fallback is False
    assert resolved.backend_fallback_used is False


def test_locked_comparison_paths_enforce_pr6_pr8_gates() -> None:
    with pytest.raises(ValueError, match="max_both execution requires deterministic_compute=true"):
        resolve_compute_policy(
            framework_mode=FrameworkMode.LOCKED_COMPARISON,
            hardware_mode="max_both",
        )

    with pytest.raises(ValueError, match="requires compatible GPU capability"):
        resolve_compute_policy(
            framework_mode=FrameworkMode.LOCKED_COMPARISON,
            hardware_mode="max_both",
            deterministic_compute=True,
            capability_snapshot=_missing_gpu_snapshot(),
        )

    resolved_max_both = resolve_compute_policy(
        framework_mode=FrameworkMode.LOCKED_COMPARISON,
        hardware_mode="max_both",
        deterministic_compute=True,
        capability_snapshot=_gpu_capability_snapshot(device_id=0),
    )
    assert resolved_max_both.hardware_mode_requested == "max_both"
    assert resolved_max_both.hardware_mode_effective == "max_both"
    assert resolved_max_both.requested_backend_family == "auto_mixed"
    assert resolved_max_both.effective_backend_family == "sklearn_cpu"
    assert resolved_max_both.gpu_device_id == 0
    assert resolved_max_both.deterministic_compute is True
    assert resolved_max_both.allow_backend_fallback is False
    assert resolved_max_both.backend_fallback_used is False

    with pytest.raises(ValueError, match="deterministic_compute=true"):
        resolve_compute_policy(
            framework_mode=FrameworkMode.LOCKED_COMPARISON,
            hardware_mode="gpu_only",
            deterministic_compute=False,
            capability_snapshot=_gpu_capability_snapshot(),
        )

    with pytest.raises(ValueError, match="requires compatible GPU capability"):
        resolve_compute_policy(
            framework_mode=FrameworkMode.LOCKED_COMPARISON,
            hardware_mode="gpu_only",
            deterministic_compute=True,
            capability_snapshot=_missing_gpu_snapshot(),
        )

    with pytest.raises(ValueError, match="exploratory-only"):
        resolve_compute_policy(
            framework_mode=FrameworkMode.LOCKED_COMPARISON,
            hardware_mode="cpu_only",
            allow_backend_fallback=True,
        )


def test_locked_comparison_gpu_only_resolves_when_deterministic_and_capable() -> None:
    resolved = resolve_compute_policy(
        framework_mode=FrameworkMode.LOCKED_COMPARISON,
        hardware_mode="gpu_only",
        deterministic_compute=True,
        capability_snapshot=_gpu_capability_snapshot(device_id=0),
    )

    assert resolved.hardware_mode_requested == "gpu_only"
    assert resolved.hardware_mode_effective == "gpu_only"
    assert resolved.requested_backend_family == "torch_gpu"
    assert resolved.effective_backend_family == "torch_gpu"
    assert resolved.gpu_device_id == 0
    assert resolved.deterministic_compute is True
    assert resolved.allow_backend_fallback is False
    assert resolved.backend_fallback_used is False


def test_fixed_baselines_runtime_resolution_drops_tuning_fields() -> None:
    methodology_policy, _ = resolve_methodology_runtime(
        framework_mode=FrameworkMode.EXPLORATORY,
        methodology_policy_name="fixed_baselines_only",
        class_weight_policy="none",
        tuning_enabled=False,
        tuning_search_space_id="linear-grouped-nested-v1",
        tuning_search_space_version="1.0.0",
        tuning_inner_cv_scheme="grouped_leave_one_group_out",
        tuning_inner_group_field="session",
        subgroup_reporting_enabled=True,
        subgroup_dimensions=["label"],
        subgroup_min_samples_per_group=1,
        evidence_run_role=None,
        protocol_context={},
        comparison_context={},
    )

    assert methodology_policy.policy_name.value == "fixed_baselines_only"
    assert methodology_policy.tuning_enabled is False
    assert methodology_policy.inner_cv_scheme is None
    assert methodology_policy.inner_group_field is None
    assert methodology_policy.tuning_search_space_id is None
    assert methodology_policy.tuning_search_space_version is None


def test_compute_policy_stamping_includes_runtime_gpu_diagnostics_additively() -> None:
    resolved = resolve_compute_policy(
        framework_mode=FrameworkMode.EXPLORATORY,
        hardware_mode="cpu_only",
    )
    payload: dict[str, object] = {}
    stamp_compute_policy_metadata(
        payload=payload,
        compute_policy=resolved,
        compute_runtime_metadata={
            "backend_id": "cpu_reference",
            "actual_estimator_backend_id": "cpu_reference",
            "actual_estimator_backend_family": "sklearn_cpu",
            "gpu_memory_peak_mb": 512.25,
            "device_transfer_seconds": 0.123,
            "torch_deterministic_enforced": True,
            "torch_deterministic_limitations": None,
        },
    )

    assert payload["actual_estimator_backend_id"] == "cpu_reference"
    assert payload["actual_estimator_backend_family"] == "sklearn_cpu"
    assert payload["gpu_memory_peak_mb"] == 512.25
    assert payload["device_transfer_seconds"] == 0.123
    assert payload["torch_deterministic_enforced"] is True
    assert payload["compute_policy"]["actual_estimator_backend_id"] == "cpu_reference"
    assert payload["compute_policy"]["gpu_memory_peak_mb"] == 512.25
    assert payload["compute_policy"]["torch_deterministic_enforced"] is True
