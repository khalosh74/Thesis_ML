from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from Thesis_ML.experiments.backend_registry import resolve_backend_support
from Thesis_ML.experiments.compute_policy import (
    CPU_ONLY,
    CPU_REFERENCE_BACKEND_STACK_ID,
    GPU_ONLY,
    MAX_BOTH,
    TORCH_GPU_BACKEND_STACK_ID_FALLBACK,
    BackendFamily,
    ComputeLane,
    HardwareMode,
    ResolvedComputePolicy,
)

AssignedBackendFamily = Literal["sklearn_cpu", "torch_gpu"]


@dataclass(frozen=True)
class ComputeRunRequest:
    order_index: int
    run_id: str
    model_name: str


@dataclass(frozen=True)
class ComputeRunAssignment:
    order_index: int
    run_id: str
    model_name: str
    assigned_compute_lane: ComputeLane
    assigned_backend_family: AssignedBackendFamily
    lane_assignment_reason: str
    scheduler_mode_effective: HardwareMode
    gpu_device_id: int | None
    backend_fallback_used: bool
    backend_fallback_reason: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "order_index": int(self.order_index),
            "run_id": str(self.run_id),
            "model_name": str(self.model_name),
            "assigned_compute_lane": str(self.assigned_compute_lane),
            "assigned_backend_family": str(self.assigned_backend_family),
            "lane_assignment_reason": str(self.lane_assignment_reason),
            "scheduler_mode_effective": str(self.scheduler_mode_effective),
            "gpu_device_id": self.gpu_device_id,
            "backend_fallback_used": bool(self.backend_fallback_used),
            "backend_fallback_reason": self.backend_fallback_reason,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        default_order_index: int = 0,
        default_run_id: str = "",
        default_model_name: str = "",
    ) -> ComputeRunAssignment:
        if not isinstance(payload, dict):
            raise ValueError("scheduled_compute_assignment must be a JSON object when provided.")

        lane_raw = str(payload.get("assigned_compute_lane", "")).strip().lower()
        if lane_raw not in {"cpu", "gpu"}:
            raise ValueError(
                "scheduled_compute_assignment.assigned_compute_lane must be one of: cpu, gpu."
            )
        lane = cast(ComputeLane, lane_raw)

        backend_raw = str(payload.get("assigned_backend_family", "")).strip().lower()
        if backend_raw not in {"sklearn_cpu", "torch_gpu"}:
            raise ValueError(
                "scheduled_compute_assignment.assigned_backend_family must be one of: "
                "sklearn_cpu, torch_gpu."
            )
        assigned_backend_family = cast(AssignedBackendFamily, backend_raw)

        if lane == "cpu" and assigned_backend_family != "sklearn_cpu":
            raise ValueError(
                "CPU lane assignments must use assigned_backend_family='sklearn_cpu'."
            )
        if lane == "gpu" and assigned_backend_family != "torch_gpu":
            raise ValueError("GPU lane assignments must use assigned_backend_family='torch_gpu'.")

        scheduler_mode_raw = str(
            payload.get("scheduler_mode_effective", payload.get("hardware_mode_effective", ""))
        ).strip().lower()
        if scheduler_mode_raw not in {CPU_ONLY, GPU_ONLY, MAX_BOTH}:
            raise ValueError(
                "scheduled_compute_assignment.scheduler_mode_effective must be one of: "
                "cpu_only, gpu_only, max_both."
            )
        scheduler_mode_effective = cast(HardwareMode, scheduler_mode_raw)

        gpu_device_id_raw = payload.get("gpu_device_id")
        gpu_device_id = int(gpu_device_id_raw) if gpu_device_id_raw is not None else None
        if lane == "cpu":
            gpu_device_id = None

        order_index_raw = payload.get("order_index", default_order_index)
        run_id_raw = payload.get("run_id", default_run_id)
        model_name_raw = payload.get("model_name", default_model_name)
        lane_assignment_reason = str(payload.get("lane_assignment_reason", "")).strip()
        if not lane_assignment_reason:
            lane_assignment_reason = "externally_assigned"

        backend_fallback_used = bool(payload.get("backend_fallback_used", False))
        backend_fallback_reason_raw = payload.get("backend_fallback_reason")
        backend_fallback_reason = (
            str(backend_fallback_reason_raw).strip()
            if isinstance(backend_fallback_reason_raw, str)
            and str(backend_fallback_reason_raw).strip()
            else None
        )
        if not backend_fallback_used:
            backend_fallback_reason = None

        return cls(
            order_index=int(order_index_raw),
            run_id=str(run_id_raw),
            model_name=str(model_name_raw),
            assigned_compute_lane=lane,
            assigned_backend_family=assigned_backend_family,
            lane_assignment_reason=lane_assignment_reason,
            scheduler_mode_effective=scheduler_mode_effective,
            gpu_device_id=gpu_device_id,
            backend_fallback_used=backend_fallback_used,
            backend_fallback_reason=backend_fallback_reason,
        )


def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be >= 1.")
    return resolved


def _coerce_non_negative_int(value: Any, *, field_name: str) -> int:
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{field_name} must be >= 0.")
    return resolved


def _normalize_requests(
    run_requests: Sequence[ComputeRunRequest],
) -> list[ComputeRunRequest]:
    normalized: list[ComputeRunRequest] = []
    for item in run_requests:
        normalized.append(
            ComputeRunRequest(
                order_index=int(item.order_index),
                run_id=str(item.run_id),
                model_name=str(item.model_name),
            )
        )
    return sorted(
        normalized,
        key=lambda request: (
            int(request.order_index),
            str(request.run_id),
            str(request.model_name),
        ),
    )


def _torch_gpu_support_for_model(
    *,
    model_name: str,
    base_compute_policy: ResolvedComputePolicy,
    gpu_model_allowlist: set[str] | None = None,
) -> tuple[bool, str | None]:
    normalized_model_name = str(model_name).strip().lower()
    if (
        gpu_model_allowlist is not None
        and normalized_model_name not in gpu_model_allowlist
    ):
        return (
            False,
            f"gpu_lane_not_allowed_by_policy:{normalized_model_name}",
        )
    if base_compute_policy.gpu_device_id is None:
        return False, "gpu_capability_unavailable"
    probe_policy = replace(
        base_compute_policy,
        effective_backend_family=cast(BackendFamily, "torch_gpu"),
    )
    support = resolve_backend_support(normalized_model_name, probe_policy)
    return bool(support.supported), support.reason


def _cpu_assignment(
    *,
    request: ComputeRunRequest,
    scheduler_mode_effective: HardwareMode,
    reason: str,
    fallback_used: bool,
    fallback_reason: str | None,
) -> ComputeRunAssignment:
    return ComputeRunAssignment(
        order_index=int(request.order_index),
        run_id=str(request.run_id),
        model_name=str(request.model_name),
        assigned_compute_lane=cast(ComputeLane, "cpu"),
        assigned_backend_family=cast(AssignedBackendFamily, "sklearn_cpu"),
        lane_assignment_reason=str(reason),
        scheduler_mode_effective=scheduler_mode_effective,
        gpu_device_id=None,
        backend_fallback_used=bool(fallback_used),
        backend_fallback_reason=(
            str(fallback_reason).strip() if isinstance(fallback_reason, str) else None
        ),
    )


def _gpu_assignment(
    *,
    request: ComputeRunRequest,
    scheduler_mode_effective: HardwareMode,
    reason: str,
    gpu_device_id: int | None,
) -> ComputeRunAssignment:
    return ComputeRunAssignment(
        order_index=int(request.order_index),
        run_id=str(request.run_id),
        model_name=str(request.model_name),
        assigned_compute_lane=cast(ComputeLane, "gpu"),
        assigned_backend_family=cast(AssignedBackendFamily, "torch_gpu"),
        lane_assignment_reason=str(reason),
        scheduler_mode_effective=scheduler_mode_effective,
        gpu_device_id=(int(gpu_device_id) if gpu_device_id is not None else None),
        backend_fallback_used=False,
        backend_fallback_reason=None,
    )


def plan_compute_schedule(
    *,
    run_requests: Sequence[ComputeRunRequest],
    base_compute_policy: ResolvedComputePolicy,
    max_parallel_runs: int = 1,
    max_parallel_gpu_runs: int = 1,
    gpu_model_allowlist: set[str] | None = None,
) -> list[ComputeRunAssignment]:
    requests = _normalize_requests(run_requests)
    if not requests:
        return []

    resolved_max_parallel_runs = _coerce_positive_int(
        max_parallel_runs,
        field_name="max_parallel_runs",
    )
    resolved_max_parallel_gpu_runs = _coerce_non_negative_int(
        max_parallel_gpu_runs,
        field_name="max_parallel_gpu_runs",
    )
    if resolved_max_parallel_gpu_runs > resolved_max_parallel_runs:
        raise ValueError("max_parallel_gpu_runs cannot exceed max_parallel_runs.")

    normalized_gpu_model_allowlist = (
        {str(value).strip().lower() for value in gpu_model_allowlist}
        if gpu_model_allowlist is not None
        else None
    )

    requested_mode = cast(HardwareMode, base_compute_policy.hardware_mode_requested)
    scheduler_mode_effective = cast(HardwareMode, base_compute_policy.hardware_mode_effective)
    gpu_stack_available = bool(
        base_compute_policy.gpu_device_id is not None
        and scheduler_mode_effective in {GPU_ONLY, MAX_BOTH}
    )

    if requested_mode == CPU_ONLY:
        return [
            _cpu_assignment(
                request=request,
                scheduler_mode_effective=scheduler_mode_effective,
                reason="hardware_mode_cpu_only",
                fallback_used=False,
                fallback_reason=None,
            )
            for request in requests
        ]

    if requested_mode == GPU_ONLY:
        if not gpu_stack_available or str(base_compute_policy.effective_backend_family) != "torch_gpu":
            if not bool(base_compute_policy.allow_backend_fallback):
                raise ValueError(
                    "gpu_only scheduling requires a resolved GPU backend capability. "
                    "Set allow_backend_fallback=true for exploratory CPU fallback."
                )
            fallback_reason = (
                str(base_compute_policy.backend_fallback_reason).strip()
                if isinstance(base_compute_policy.backend_fallback_reason, str)
                and str(base_compute_policy.backend_fallback_reason).strip()
                else "gpu_capability_unavailable"
            )
            return [
                _cpu_assignment(
                    request=request,
                    scheduler_mode_effective=scheduler_mode_effective,
                    reason="gpu_only_fallback_cpu_lane",
                    fallback_used=True,
                    fallback_reason=fallback_reason,
                )
                for request in requests
            ]

        if resolved_max_parallel_gpu_runs <= 0:
            if not bool(base_compute_policy.allow_backend_fallback):
                raise ValueError(
                    "gpu_only scheduling requires max_parallel_gpu_runs >= 1 "
                    "when fallback is disallowed."
                )
            return [
                _cpu_assignment(
                    request=request,
                    scheduler_mode_effective=scheduler_mode_effective,
                    reason="gpu_only_gpu_parallelism_disabled",
                    fallback_used=True,
                    fallback_reason="gpu_parallelism_disabled",
                )
                for request in requests
            ]

        assignments: list[ComputeRunAssignment] = []
        for request in requests:
            supported, unsupported_reason = _torch_gpu_support_for_model(
                model_name=request.model_name,
                base_compute_policy=base_compute_policy,
                gpu_model_allowlist=normalized_gpu_model_allowlist,
            )
            if supported:
                assignments.append(
                    _gpu_assignment(
                        request=request,
                        scheduler_mode_effective=scheduler_mode_effective,
                        reason="gpu_only_gpu_backend_assigned",
                        gpu_device_id=base_compute_policy.gpu_device_id,
                    )
                )
                continue
            if bool(base_compute_policy.allow_backend_fallback):
                assignments.append(
                    _cpu_assignment(
                        request=request,
                        scheduler_mode_effective=scheduler_mode_effective,
                        reason="gpu_only_model_fallback_cpu_lane",
                        fallback_used=True,
                        fallback_reason=(
                            "gpu_backend_unsupported_for_model:"
                            f"{request.model_name}"
                        ),
                    )
                )
                continue
            support_text = (
                str(unsupported_reason).strip()
                if isinstance(unsupported_reason, str) and str(unsupported_reason).strip()
                else "unknown_reason"
            )
            raise ValueError(
                "gpu_only scheduling requires GPU backend support for every run model. "
                f"model='{request.model_name}' support_reason='{support_text}'."
            )
        return assignments

    if not gpu_stack_available:
        if not bool(base_compute_policy.allow_backend_fallback):
            raise ValueError(
                "max_both scheduling requires visible GPU capability when "
                "allow_backend_fallback=false."
            )
        fallback_reason = (
            str(base_compute_policy.backend_fallback_reason).strip()
            if isinstance(base_compute_policy.backend_fallback_reason, str)
            and str(base_compute_policy.backend_fallback_reason).strip()
            else "gpu_capability_unavailable_for_max_both"
        )
        return [
            _cpu_assignment(
                request=request,
                scheduler_mode_effective=scheduler_mode_effective,
                reason="max_both_fallback_cpu_lane",
                fallback_used=True,
                fallback_reason=fallback_reason,
            )
            for request in requests
        ]

    if resolved_max_parallel_gpu_runs <= 0:
        return [
            _cpu_assignment(
                request=request,
                scheduler_mode_effective=scheduler_mode_effective,
                reason="max_both_gpu_parallelism_disabled",
                fallback_used=False,
                fallback_reason=None,
            )
            for request in requests
        ]

    assignments: list[ComputeRunAssignment] = []
    for batch_start in range(0, len(requests), resolved_max_parallel_runs):
        batch = requests[batch_start : batch_start + resolved_max_parallel_runs]
        remaining_gpu_slots = int(resolved_max_parallel_gpu_runs)

        for request in batch:
            supported, unsupported_reason = _torch_gpu_support_for_model(
                model_name=request.model_name,
                base_compute_policy=base_compute_policy,
                gpu_model_allowlist=normalized_gpu_model_allowlist,
            )
            if supported and remaining_gpu_slots > 0:
                assignments.append(
                    _gpu_assignment(
                        request=request,
                        scheduler_mode_effective=scheduler_mode_effective,
                        reason="max_both_gpu_eligible_assigned_gpu",
                        gpu_device_id=base_compute_policy.gpu_device_id,
                    )
                )
                remaining_gpu_slots -= 1
                continue

            if supported:
                assignments.append(
                    _cpu_assignment(
                        request=request,
                        scheduler_mode_effective=scheduler_mode_effective,
                        reason="max_both_gpu_lane_budget_exhausted",
                        fallback_used=False,
                        fallback_reason=None,
                    )
                )
                continue

            cpu_reason = "max_both_model_cpu_only"
            if isinstance(unsupported_reason, str) and unsupported_reason.startswith(
                "gpu_lane_not_allowed_by_policy:"
            ):
                cpu_reason = "max_both_gpu_disallowed_by_policy"
            assignments.append(
                _cpu_assignment(
                    request=request,
                    scheduler_mode_effective=scheduler_mode_effective,
                    reason=cpu_reason,
                    fallback_used=False,
                    fallback_reason=None,
                )
            )

    return sorted(
        assignments,
        key=lambda item: (
            int(item.order_index),
            str(item.run_id),
            str(item.model_name),
        ),
    )


def _backend_stack_id_for_assignment(
    *,
    base_compute_policy: ResolvedComputePolicy,
    assigned_backend_family: AssignedBackendFamily,
) -> str:
    if assigned_backend_family == "sklearn_cpu":
        return CPU_REFERENCE_BACKEND_STACK_ID
    existing_stack_id = str(base_compute_policy.backend_stack_id).strip()
    if not existing_stack_id or existing_stack_id == CPU_REFERENCE_BACKEND_STACK_ID:
        return TORCH_GPU_BACKEND_STACK_ID_FALLBACK
    return existing_stack_id


def materialize_scheduled_compute_policy(
    *,
    base_compute_policy: ResolvedComputePolicy,
    assignment: ComputeRunAssignment,
) -> ResolvedComputePolicy:
    backend_family = cast(BackendFamily, assignment.assigned_backend_family)
    return ResolvedComputePolicy(
        hardware_mode_requested=base_compute_policy.hardware_mode_requested,
        hardware_mode_effective=base_compute_policy.hardware_mode_effective,
        requested_backend_family=base_compute_policy.requested_backend_family,
        effective_backend_family=backend_family,
        gpu_device_id=(
            int(assignment.gpu_device_id)
            if assignment.assigned_compute_lane == "gpu"
            and assignment.gpu_device_id is not None
            else None
        ),
        gpu_device_name=(
            base_compute_policy.gpu_device_name
            if assignment.assigned_compute_lane == "gpu"
            else None
        ),
        gpu_device_total_memory_mb=(
            base_compute_policy.gpu_device_total_memory_mb
            if assignment.assigned_compute_lane == "gpu"
            else None
        ),
        deterministic_compute=bool(base_compute_policy.deterministic_compute),
        allow_backend_fallback=bool(base_compute_policy.allow_backend_fallback),
        backend_stack_id=_backend_stack_id_for_assignment(
            base_compute_policy=base_compute_policy,
            assigned_backend_family=assignment.assigned_backend_family,
        ),
        backend_fallback_used=bool(assignment.backend_fallback_used),
        backend_fallback_reason=assignment.backend_fallback_reason,
        assigned_compute_lane=assignment.assigned_compute_lane,
        assigned_backend_family=backend_family,
        lane_assignment_reason=str(assignment.lane_assignment_reason),
        scheduler_mode_effective=assignment.scheduler_mode_effective,
    )


__all__ = [
    "ComputeRunAssignment",
    "ComputeRunRequest",
    "plan_compute_schedule",
    "materialize_scheduled_compute_policy",
]
