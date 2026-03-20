from __future__ import annotations

from types import SimpleNamespace

from Thesis_ML.experiments import compute_capabilities


class _FakeCuda:
    def __init__(
        self,
        *,
        available: bool,
        device_count: int,
        names: dict[int, str] | None = None,
        total_memory_bytes: int = 8 * 1024 * 1024 * 1024,
    ) -> None:
        self._available = available
        self._device_count = device_count
        self._names = names or {}
        self._total_memory_bytes = total_memory_bytes

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._device_count

    def get_device_name(self, device_id: int) -> str:
        return self._names[device_id]

    def get_device_properties(self, device_id: int) -> SimpleNamespace:
        if device_id not in self._names:
            raise KeyError(device_id)
        return SimpleNamespace(total_memory=self._total_memory_bytes)


class _FakeTorch:
    def __init__(self, *, cuda: _FakeCuda, torch_version: str = "2.4.1", cuda_version: str = "12.1") -> None:
        self.__version__ = torch_version
        self.version = SimpleNamespace(cuda=cuda_version)
        self.cuda = cuda


def test_detect_compute_capabilities_handles_missing_torch(
    monkeypatch,
) -> None:
    original_import_module = compute_capabilities.importlib.import_module

    def _fake_import_module(name: str):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import_module(name)

    monkeypatch.setattr(
        compute_capabilities.importlib,
        "import_module",
        _fake_import_module,
    )

    snapshot = compute_capabilities.detect_compute_capabilities()

    assert snapshot.torch_installed is False
    assert snapshot.cuda_available is False
    assert snapshot.gpu_available is False
    assert snapshot.gpu_count == 0
    assert snapshot.compatibility_status == "torch_unavailable"
    assert "torch_not_installed" in snapshot.incompatibility_reasons
    assert snapshot.tested_stack_id == "torch_unavailable"


def test_detect_compute_capabilities_handles_torch_without_cuda(
    monkeypatch,
) -> None:
    fake_torch = _FakeTorch(cuda=_FakeCuda(available=False, device_count=0), cuda_version="")
    original_import_module = compute_capabilities.importlib.import_module

    def _fake_import_module(name: str):
        if name == "torch":
            return fake_torch
        return original_import_module(name)

    monkeypatch.setattr(
        compute_capabilities.importlib,
        "import_module",
        _fake_import_module,
    )

    snapshot = compute_capabilities.detect_compute_capabilities()

    assert snapshot.torch_installed is True
    assert snapshot.torch_version == "2.4.1"
    assert snapshot.cuda_available is False
    assert snapshot.cuda_runtime_version is None
    assert snapshot.gpu_available is False
    assert snapshot.compatibility_status == "cuda_unavailable"
    assert "cuda_not_available" in snapshot.incompatibility_reasons


def test_detect_compute_capabilities_selects_requested_gpu_device(
    monkeypatch,
) -> None:
    fake_torch = _FakeTorch(
        cuda=_FakeCuda(
            available=True,
            device_count=2,
            names={0: "GPU Zero", 1: "GPU One"},
            total_memory_bytes=16 * 1024 * 1024 * 1024,
        )
    )
    original_import_module = compute_capabilities.importlib.import_module

    def _fake_import_module(name: str):
        if name == "torch":
            return fake_torch
        return original_import_module(name)

    monkeypatch.setattr(
        compute_capabilities.importlib,
        "import_module",
        _fake_import_module,
    )

    snapshot = compute_capabilities.detect_compute_capabilities(requested_device_id=1)

    assert snapshot.gpu_available is True
    assert snapshot.gpu_count == 2
    assert snapshot.requested_device_visible is True
    assert snapshot.device_id == 1
    assert snapshot.device_name == "GPU One"
    assert snapshot.device_total_memory_mb == 16 * 1024
    assert snapshot.compatibility_status == "gpu_compatible"


def test_detect_compute_capabilities_rejects_invalid_gpu_device_selection(
    monkeypatch,
) -> None:
    fake_torch = _FakeTorch(
        cuda=_FakeCuda(
            available=True,
            device_count=2,
            names={0: "GPU Zero", 1: "GPU One"},
        )
    )
    original_import_module = compute_capabilities.importlib.import_module

    def _fake_import_module(name: str):
        if name == "torch":
            return fake_torch
        return original_import_module(name)

    monkeypatch.setattr(
        compute_capabilities.importlib,
        "import_module",
        _fake_import_module,
    )

    snapshot = compute_capabilities.detect_compute_capabilities(requested_device_id=4)

    assert snapshot.requested_device_visible is False
    assert snapshot.device_id is None
    assert snapshot.compatibility_status == "requested_device_unavailable"
    assert "requested_gpu_device_not_visible:4" in snapshot.incompatibility_reasons
