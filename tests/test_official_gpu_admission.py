from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from Thesis_ML.comparisons.loader import load_comparison_spec
from Thesis_ML.comparisons.runner import (
    _validate_official_comparison_gpu_admission,
    compile_and_run_comparison,
)
from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.compute_capabilities import ComputeCapabilitySnapshot
from Thesis_ML.experiments.compute_policy import resolve_compute_policy
from tests._config_refs import model_family_v1_comparison_variant_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


_comparison_spec_path = model_family_v1_comparison_variant_path


def _demo_dataset_paths() -> dict[str, Path]:
    root = _repo_root() / "demo_data" / "synthetic_v1"
    return {
        "index_csv": root / "dataset_index.csv",
        "data_root": root / "data_root",
        "cache_dir": root / "cache",
    }


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


@pytest.mark.parametrize("model_name", ["linearsvc", "dummy", "xgboost"])
def test_official_comparison_gpu_only_allowlist_rejects_unsupported_models(
    model_name: str,
) -> None:
    compute_policy = resolve_compute_policy(
        framework_mode=FrameworkMode.LOCKED_COMPARISON,
        hardware_mode="gpu_only",
        deterministic_compute=True,
        capability_snapshot=_gpu_capability_snapshot(),
    )
    manifest = SimpleNamespace(runs=[SimpleNamespace(run_id=f"run_{model_name}", model=model_name)])

    with pytest.raises(ValueError, match="Official locked comparison gpu_only admission rejected"):
        _validate_official_comparison_gpu_admission(
            compiled_manifest=manifest,  # type: ignore[arg-type]
            compute_policy=compute_policy,
        )


def test_official_comparison_gpu_only_ridge_is_admitted_and_stamped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_paths = _demo_dataset_paths()
    spec = load_comparison_spec(_comparison_spec_path())

    monkeypatch.setattr(
        "Thesis_ML.experiments.compute_policy.detect_compute_capabilities",
        lambda requested_device_id=None: _gpu_capability_snapshot(
            device_id=0 if requested_device_id is None else int(requested_device_id)
        ),
    )

    def _successful_watchdog(*, run_kwargs: dict[str, Any], **_: object) -> dict[str, object]:
        run_id = str(run_kwargs["run_id"])
        report_dir = tmp_path / "fake_reports" / run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        compute_policy = resolve_compute_policy(
            framework_mode=FrameworkMode.LOCKED_COMPARISON,
            hardware_mode=str(run_kwargs.get("hardware_mode", "cpu_only")),
            gpu_device_id=(
                int(run_kwargs["gpu_device_id"])
                if run_kwargs.get("gpu_device_id") is not None
                else None
            ),
            deterministic_compute=bool(run_kwargs.get("deterministic_compute", False)),
            allow_backend_fallback=bool(run_kwargs.get("allow_backend_fallback", False)),
        )
        metrics_payload = {
            "balanced_accuracy": 0.5,
            "macro_f1": 0.5,
            "accuracy": 0.5,
            "n_folds": 1,
            "primary_metric_name": "balanced_accuracy",
            "primary_metric_value": 0.5,
        }
        return {
            "status": "success",
            "run_payload": {
                "report_dir": str(report_dir),
                "config_path": str(report_dir / "config.json"),
                "metrics_path": str(report_dir / "metrics.json"),
                "metrics": metrics_payload,
                "compute_policy": compute_policy.to_payload(),
            },
        }

    monkeypatch.setattr(
        "Thesis_ML.comparisons.runner.execute_run_with_timeout_watchdog",
        _successful_watchdog,
    )
    monkeypatch.setattr(
        "Thesis_ML.comparisons.runner.verify_official_artifacts",
        lambda **_: {"passed": True, "issues": []},
    )

    result = compile_and_run_comparison(
        comparison=spec,
        index_csv=dataset_paths["index_csv"],
        data_root=dataset_paths["data_root"],
        cache_dir=dataset_paths["cache_dir"],
        reports_root=tmp_path / "comparison_reports",
        variant_ids=["ridge"],
        dry_run=False,
        hardware_mode="gpu_only",
        deterministic_compute=True,
    )

    assert int(result["n_failed"]) == 0
    successful = [row for row in result["run_results"] if row["status"] == "success"]
    assert successful
    compute_policy_payload = successful[0]["compute_policy"]
    assert isinstance(compute_policy_payload, dict)
    assert compute_policy_payload["hardware_mode_requested"] == "gpu_only"
    assert compute_policy_payload["hardware_mode_effective"] == "gpu_only"
    assert compute_policy_payload["effective_backend_family"] == "torch_gpu"
    assert compute_policy_payload["deterministic_compute"] is True
    assert compute_policy_payload["backend_fallback_used"] is False
