from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stack_id(torch_version: str | None, cuda_runtime_version: str | None) -> str:
    if not torch_version:
        return "torch_unavailable"
    cuda_token = str(cuda_runtime_version).strip() or "none"
    return f"torch_{torch_version}__cuda_{cuda_token}"


@dataclass(frozen=True)
class ComputeCapabilitySnapshot:
    torch_installed: bool
    torch_version: str | None
    cuda_available: bool
    cuda_runtime_version: str | None
    gpu_available: bool
    gpu_count: int
    requested_device_visible: bool | None
    device_id: int | None
    device_name: str | None
    device_total_memory_mb: int | None
    compatibility_status: str
    incompatibility_reasons: tuple[str, ...]
    tested_stack_id: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "torch_installed": bool(self.torch_installed),
            "torch_version": self.torch_version,
            "cuda_available": bool(self.cuda_available),
            "cuda_runtime_version": self.cuda_runtime_version,
            "gpu_available": bool(self.gpu_available),
            "gpu_count": int(self.gpu_count),
            "requested_device_visible": self.requested_device_visible,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_total_memory_mb": self.device_total_memory_mb,
            "compatibility_status": str(self.compatibility_status),
            "incompatibility_reasons": list(self.incompatibility_reasons),
            "tested_stack_id": str(self.tested_stack_id),
        }


def detect_compute_capabilities(
    *,
    requested_device_id: int | None = None,
) -> ComputeCapabilitySnapshot:
    requested_visible: bool | None = None
    normalized_device_id: int | None = None
    incompatibility_reasons: list[str] = []

    if requested_device_id is not None:
        normalized_device_id = _safe_int(requested_device_id)
        if normalized_device_id is None or normalized_device_id < 0:
            requested_visible = False
            incompatibility_reasons.append(f"invalid_gpu_device_id:{requested_device_id}")
            normalized_device_id = None

    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        if requested_visible is None and requested_device_id is not None:
            requested_visible = False
        incompatibility_reasons.append("torch_not_installed")
        return ComputeCapabilitySnapshot(
            torch_installed=False,
            torch_version=None,
            cuda_available=False,
            cuda_runtime_version=None,
            gpu_available=False,
            gpu_count=0,
            requested_device_visible=requested_visible,
            device_id=None,
            device_name=None,
            device_total_memory_mb=None,
            compatibility_status="torch_unavailable",
            incompatibility_reasons=tuple(incompatibility_reasons),
            tested_stack_id="torch_unavailable",
        )
    except Exception as exc:  # pragma: no cover - defensive path
        if requested_visible is None and requested_device_id is not None:
            requested_visible = False
        incompatibility_reasons.append(f"torch_import_failed:{exc.__class__.__name__}")
        return ComputeCapabilitySnapshot(
            torch_installed=False,
            torch_version=None,
            cuda_available=False,
            cuda_runtime_version=None,
            gpu_available=False,
            gpu_count=0,
            requested_device_visible=requested_visible,
            device_id=None,
            device_name=None,
            device_total_memory_mb=None,
            compatibility_status="torch_import_failed",
            incompatibility_reasons=tuple(incompatibility_reasons),
            tested_stack_id="torch_import_failed",
        )

    torch_version = str(getattr(torch, "__version__", "")).strip() or None
    cuda_runtime_version = str(getattr(getattr(torch, "version", None), "cuda", "")).strip() or None
    tested_stack_id = _stack_id(torch_version, cuda_runtime_version)
    cuda_module = getattr(torch, "cuda", None)

    if cuda_module is None:
        if requested_visible is None and requested_device_id is not None:
            requested_visible = False
        incompatibility_reasons.append("torch_cuda_module_unavailable")
        return ComputeCapabilitySnapshot(
            torch_installed=True,
            torch_version=torch_version,
            cuda_available=False,
            cuda_runtime_version=cuda_runtime_version,
            gpu_available=False,
            gpu_count=0,
            requested_device_visible=requested_visible,
            device_id=None,
            device_name=None,
            device_total_memory_mb=None,
            compatibility_status="cuda_module_unavailable",
            incompatibility_reasons=tuple(incompatibility_reasons),
            tested_stack_id=tested_stack_id,
        )

    try:
        cuda_available = bool(cuda_module.is_available())
    except Exception as exc:  # pragma: no cover - defensive path
        cuda_available = False
        incompatibility_reasons.append(f"cuda_availability_probe_failed:{exc.__class__.__name__}")

    try:
        gpu_count = int(cuda_module.device_count()) if cuda_available else 0
    except Exception as exc:  # pragma: no cover - defensive path
        gpu_count = 0
        incompatibility_reasons.append(f"gpu_count_probe_failed:{exc.__class__.__name__}")

    gpu_available = bool(cuda_available and gpu_count > 0)
    if not cuda_available:
        incompatibility_reasons.append("cuda_not_available")
    elif gpu_count <= 0:
        incompatibility_reasons.append("no_visible_gpus")

    selected_device_id: int | None = None
    if gpu_available:
        if normalized_device_id is None and requested_device_id is None:
            selected_device_id = 0
        elif normalized_device_id is not None and 0 <= normalized_device_id < gpu_count:
            selected_device_id = normalized_device_id
            requested_visible = True
        elif requested_device_id is not None:
            requested_visible = False
            incompatibility_reasons.append(
                f"requested_gpu_device_not_visible:{requested_device_id}"
            )
    elif requested_device_id is not None and requested_visible is None:
        requested_visible = False
        incompatibility_reasons.append("requested_gpu_device_without_gpu_support")

    device_name: str | None = None
    device_total_memory_mb: int | None = None
    if selected_device_id is not None:
        try:
            device_name = str(cuda_module.get_device_name(selected_device_id))
        except Exception as exc:  # pragma: no cover - defensive path
            incompatibility_reasons.append(f"gpu_name_probe_failed:{exc.__class__.__name__}")
        try:
            properties = cuda_module.get_device_properties(selected_device_id)
            total_memory = _safe_int(getattr(properties, "total_memory", None))
            if total_memory is not None:
                device_total_memory_mb = int(total_memory // (1024 * 1024))
        except Exception as exc:  # pragma: no cover - defensive path
            incompatibility_reasons.append(f"gpu_properties_probe_failed:{exc.__class__.__name__}")

    if requested_device_id is not None and requested_visible is False:
        compatibility_status = "requested_device_unavailable"
    elif gpu_available and selected_device_id is not None:
        compatibility_status = "gpu_compatible"
    elif not cuda_available:
        compatibility_status = "cuda_unavailable"
    elif gpu_count <= 0:
        compatibility_status = "no_visible_gpus"
    else:
        compatibility_status = "cpu_reference_only"

    return ComputeCapabilitySnapshot(
        torch_installed=True,
        torch_version=torch_version,
        cuda_available=bool(cuda_available),
        cuda_runtime_version=cuda_runtime_version,
        gpu_available=bool(gpu_available),
        gpu_count=int(gpu_count),
        requested_device_visible=requested_visible,
        device_id=selected_device_id,
        device_name=device_name,
        device_total_memory_mb=device_total_memory_mb,
        compatibility_status=compatibility_status,
        incompatibility_reasons=tuple(incompatibility_reasons),
        tested_stack_id=tested_stack_id,
    )


__all__ = ["ComputeCapabilitySnapshot", "detect_compute_capabilities"]
