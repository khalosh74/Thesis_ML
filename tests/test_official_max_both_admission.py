from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from Thesis_ML.comparisons.loader import load_comparison_spec
from Thesis_ML.comparisons.runner import compile_and_run_comparison
from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.compute_capabilities import ComputeCapabilitySnapshot
from Thesis_ML.experiments.compute_policy import resolve_compute_policy
from Thesis_ML.experiments.compute_scheduler import (
    ComputeRunAssignment,
    materialize_scheduled_compute_policy,
)
from Thesis_ML.protocols.loader import load_protocol
from Thesis_ML.protocols.runner import compile_and_run_protocol


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _comparison_spec_path() -> Path:
    return _repo_root() / "configs" / "comparisons" / "model_family_comparison_v1.json"


def _protocol_path() -> Path:
    return _repo_root() / "configs" / "protocols" / "thesis_canonical_v1.json"


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


def test_official_locked_comparison_max_both_assigns_gpu_lane_only_to_ridge(
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
        model_name = str(run_kwargs["model"])
        report_dir = tmp_path / "fake_reports" / run_id
        report_dir.mkdir(parents=True, exist_ok=True)

        base_policy = resolve_compute_policy(
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

        scheduled_assignment_raw = run_kwargs.get("scheduled_compute_assignment")
        if isinstance(scheduled_assignment_raw, dict):
            assignment = ComputeRunAssignment.from_payload(
                scheduled_assignment_raw,
                default_order_index=0,
                default_run_id=run_id,
                default_model_name=model_name,
            )
            resolved_policy = materialize_scheduled_compute_policy(
                base_compute_policy=base_policy,
                assignment=assignment,
            )
        else:
            resolved_policy = base_policy

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
                "compute_policy": resolved_policy.to_payload(),
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
        variant_ids=["ridge", "logreg", "linearsvc"],
        dry_run=False,
        hardware_mode="max_both",
        deterministic_compute=True,
        max_parallel_runs=1,
    )

    assert int(result["n_failed"]) == 0
    successful = [row for row in result["run_results"] if row["status"] == "success"]
    assert successful
    assert all(row["compute_policy"]["hardware_mode_requested"] == "max_both" for row in successful)
    assert all(row["compute_policy"]["deterministic_compute"] is True for row in successful)
    assert all(row["compute_policy"]["backend_fallback_used"] is False for row in successful)

    ridge_rows = [row for row in successful if row["variant_id"] == "ridge"]
    logreg_rows = [row for row in successful if row["variant_id"] == "logreg"]
    linearsvc_rows = [row for row in successful if row["variant_id"] == "linearsvc"]
    assert ridge_rows and logreg_rows and linearsvc_rows

    assert any(row["compute_policy"]["assigned_compute_lane"] == "gpu" for row in ridge_rows)
    assert all(
        row["compute_policy"]["assigned_backend_family"] == "torch_gpu"
        for row in ridge_rows
        if row["compute_policy"]["assigned_compute_lane"] == "gpu"
    )
    assert all(row["compute_policy"]["assigned_compute_lane"] == "cpu" for row in logreg_rows)
    assert all(
        row["compute_policy"]["assigned_backend_family"] == "sklearn_cpu" for row in logreg_rows
    )
    assert all(row["compute_policy"]["assigned_compute_lane"] == "cpu" for row in linearsvc_rows)
    assert all(
        row["compute_policy"]["assigned_backend_family"] == "sklearn_cpu" for row in linearsvc_rows
    )


def test_official_locked_comparison_max_both_requires_deterministic_compute(
    tmp_path: Path,
) -> None:
    dataset_paths = _demo_dataset_paths()
    spec = load_comparison_spec(_comparison_spec_path())
    with pytest.raises(ValueError, match="max_both execution requires deterministic_compute=true"):
        compile_and_run_comparison(
            comparison=spec,
            index_csv=dataset_paths["index_csv"],
            data_root=dataset_paths["data_root"],
            cache_dir=dataset_paths["cache_dir"],
            reports_root=tmp_path / "comparison_reports",
            variant_ids=["ridge"],
            dry_run=True,
            hardware_mode="max_both",
        )


def test_official_confirmatory_max_both_still_rejected(
    tmp_path: Path,
) -> None:
    dataset_paths = _demo_dataset_paths()
    protocol = load_protocol(_protocol_path())
    with pytest.raises(ValueError, match="do not admit hardware_mode='max_both'"):
        compile_and_run_protocol(
            protocol=protocol,
            index_csv=dataset_paths["index_csv"],
            data_root=dataset_paths["data_root"],
            cache_dir=dataset_paths["cache_dir"],
            reports_root=tmp_path / "protocol_reports",
            suite_ids=["primary_within_subject"],
            dry_run=True,
            hardware_mode="max_both",
            deterministic_compute=True,
        )
