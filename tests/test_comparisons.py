from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from Thesis_ML.comparisons.compiler import compile_comparison
from Thesis_ML.comparisons.loader import load_comparison_spec
from Thesis_ML.comparisons.models import ComparisonStatus
from Thesis_ML.comparisons.runner import compile_and_run_comparison
from Thesis_ML.data.index_dataset import build_dataset_index
from Thesis_ML.experiments.errors import OfficialContractValidationError


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _comparison_spec_path() -> Path:
    return _repo_root() / "configs" / "comparisons" / "model_family_comparison_v1.json"


def _nested_comparison_spec_path() -> Path:
    return (
        _repo_root() / "configs" / "comparisons" / "model_family_grouped_nested_comparison_v1.json"
    )


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4, dtype=np.float64))
    nib.save(image, str(path))


def _create_glm_session(
    glm_dir: Path,
    labels: list[str],
    *,
    class_signal: bool,
    shape: tuple[int, int, int] = (3, 3, 3),
) -> None:
    glm_dir.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape, dtype=np.float32)
    mask[1:, 1:, 1:] = 1.0
    _write_nifti(glm_dir / "mask.nii", mask)
    pd.Series(labels).to_csv(glm_dir / "regressor_labels.csv", index=False, header=False)

    for idx, label in enumerate(labels, start=1):
        beta = np.full(shape, fill_value=float(idx), dtype=np.float32)
        if class_signal:
            if "_anger_" in label:
                beta[1:, 1:, 1:] += 5.0
            if "_happiness_" in label:
                beta[1:, 1:, 1:] -= 5.0
        _write_nifti(glm_dir / f"beta_{idx:04d}.nii", beta)


@pytest.fixture
def comparison_dataset(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
    ]
    for subject in ("sub-001", "sub-002"):
        for session in ("ses-01", "ses-02"):
            _create_glm_session(
                glm_dir=data_root / subject / session / "BAS2",
                labels=labels,
                class_signal=True,
            )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)
    return {
        "index_csv": index_csv,
        "data_root": data_root,
        "cache_dir": tmp_path / "cache",
        "reports_root": tmp_path / "reports" / "comparisons",
    }


def test_load_comparison_spec_validates() -> None:
    spec = load_comparison_spec(_comparison_spec_path())
    assert spec.comparison_schema_version == "comparison-spec-v1"
    assert spec.framework_mode == "locked_comparison"
    assert spec.scientific_contract.target == "coarse_affect"
    assert spec.scientific_contract.split_mode == "within_subject_loso_session"
    assert spec.methodology_policy.policy_name.value == "fixed_baselines_only"
    assert spec.metric_policy.primary_metric == "balanced_accuracy"
    assert {variant.variant_id for variant in spec.allowed_variants} == {
        "ridge",
        "logreg",
        "linearsvc",
    }
    assert int(spec.evidence_policy.repeat_evaluation.repeat_count) == 3
    assert bool(spec.evidence_policy.paired_comparisons.require_significant_win) is True
    assert "logreg" in set(spec.cost_policy.explicit_benchmark_expensive_models)
    assert int(spec.cost_policy.max_projected_runtime_seconds_per_run) > 0


def test_compile_comparison_expands_variants_and_subjects(
    comparison_dataset: dict[str, Path],
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())
    manifest = compile_comparison(spec, index_csv=comparison_dataset["index_csv"])
    assert manifest.framework_mode == "locked_comparison"
    assert set(manifest.variant_ids) == {"ridge", "logreg", "linearsvc"}
    expected_runs = (
        len(spec.allowed_variants)
        * len(pd.read_csv(comparison_dataset["index_csv"])["subject"].astype(str).unique())
        * int(spec.evidence_policy.repeat_evaluation.repeat_count)
    )
    assert len(manifest.runs) == expected_runs
    assert all(run.framework_mode == "locked_comparison" for run in manifest.runs)
    assert all(run.canonical_run is False for run in manifest.runs)
    assert all(run.subject in {"sub-001", "sub-002"} for run in manifest.runs)
    assert all(run.methodology_policy_name.value == "fixed_baselines_only" for run in manifest.runs)


def test_compile_comparison_rejects_unknown_variant(
    comparison_dataset: dict[str, Path],
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())
    with pytest.raises(ValueError, match="Unknown comparison variant"):
        compile_comparison(
            spec,
            index_csv=comparison_dataset["index_csv"],
            variant_ids=["not_registered_variant"],
        )


def test_comparison_requires_explicit_methodology_policy(tmp_path: Path) -> None:
    payload = json.loads(_comparison_spec_path().read_text(encoding="utf-8"))
    payload.pop("methodology_policy", None)
    spec_path = tmp_path / "missing_methodology_comparison.json"
    spec_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="methodology_policy"):
        load_comparison_spec(spec_path)


def test_comparison_validation_rejects_metric_drift(tmp_path: Path) -> None:
    payload = json.loads(_comparison_spec_path().read_text(encoding="utf-8"))
    payload["decision_policy"]["primary_metric"] = "macro_f1"
    decision_drift_path = tmp_path / "decision_metric_drift_comparison.json"
    decision_drift_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="decision_policy.primary_metric must match metric_policy.primary_metric",
    ):
        load_comparison_spec(decision_drift_path)

    payload = json.loads(_comparison_spec_path().read_text(encoding="utf-8"))
    payload["control_policy"]["permutation_metric"] = "macro_f1"
    permutation_drift_path = tmp_path / "permutation_metric_drift_comparison.json"
    permutation_drift_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="control_policy.permutation_metric must match metric_policy.primary_metric",
    ):
        load_comparison_spec(permutation_drift_path)


def test_comparison_runner_dry_run_emits_artifacts(
    comparison_dataset: dict[str, Path],
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())
    result = compile_and_run_comparison(
        comparison=spec,
        index_csv=comparison_dataset["index_csv"],
        data_root=comparison_dataset["data_root"],
        cache_dir=comparison_dataset["cache_dir"],
        reports_root=comparison_dataset["reports_root"],
        variant_ids=["ridge"],
        dry_run=True,
    )
    assert result["n_failed"] == 0
    assert result["n_completed"] == 0
    assert result["n_planned"] == 2 * int(spec.evidence_policy.repeat_evaluation.repeat_count)
    assert int(result["max_parallel_runs_effective"]) == 1
    for artifact_path in result["artifact_paths"].values():
        assert Path(artifact_path).exists()
    assert "repeated_run_metrics" in result["artifact_paths"]
    assert "repeated_run_summary" in result["artifact_paths"]
    assert "confidence_intervals" in result["artifact_paths"]
    assert "metric_intervals" in result["artifact_paths"]
    assert "paired_model_comparisons" in result["artifact_paths"]
    assert "paired_model_comparisons_csv" in result["artifact_paths"]
    assert Path(result["artifact_paths"]["comparison_decision"]).exists()

    status_payload = json.loads(
        Path(result["artifact_paths"]["execution_status"]).read_text(encoding="utf-8")
    )
    assert status_payload["framework_mode"] == "locked_comparison"
    assert status_payload["dry_run"] is True
    assert all(run["status"] == "planned" for run in status_payload["runs"])

    summary_payload = json.loads(
        Path(result["artifact_paths"]["comparison_summary"]).read_text(encoding="utf-8")
    )
    decision_payload = json.loads(
        Path(result["artifact_paths"]["comparison_decision"]).read_text(encoding="utf-8")
    )
    manifest_payload = json.loads(
        Path(result["artifact_paths"]["compiled_comparison_manifest"]).read_text(encoding="utf-8")
    )
    assert summary_payload["metric_policy_effective"]["primary_metric"] == "balanced_accuracy"
    assert summary_payload["metric_policy_effective"]["decision_metric"] == "balanced_accuracy"
    assert isinstance(summary_payload["required_evidence_status"], dict)
    assert isinstance(summary_payload["paired_comparisons"], dict)
    assert decision_payload["metric_policy_effective"]["primary_metric"] == "balanced_accuracy"
    assert decision_payload["metric_policy_effective"]["decision_metric"] == "balanced_accuracy"
    assert manifest_payload["metric_policy_effective"]["primary_metric"] == "balanced_accuracy"
    assert manifest_payload["metric_policy_effective"]["decision_metric"] == "balanced_accuracy"


def test_comparison_runner_rejects_non_cpu_compute_controls(
    comparison_dataset: dict[str, Path],
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())

    with pytest.raises(ValueError, match="hardware_mode must remain 'cpu_only'"):
        compile_and_run_comparison(
            comparison=spec,
            index_csv=comparison_dataset["index_csv"],
            data_root=comparison_dataset["data_root"],
            cache_dir=comparison_dataset["cache_dir"],
            reports_root=comparison_dataset["reports_root"],
            variant_ids=["ridge"],
            dry_run=True,
            hardware_mode="gpu_only",
        )

    with pytest.raises(ValueError, match="allow_backend_fallback is exploratory-only"):
        compile_and_run_comparison(
            comparison=spec,
            index_csv=comparison_dataset["index_csv"],
            data_root=comparison_dataset["data_root"],
            cache_dir=comparison_dataset["cache_dir"],
            reports_root=comparison_dataset["reports_root"],
            variant_ids=["ridge"],
            dry_run=True,
            allow_backend_fallback=True,
        )


def test_comparison_runner_surfaces_structured_failure_metadata(
    comparison_dataset: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())

    def _failing_watchdog(**_: object) -> dict[str, object]:
        failure = OfficialContractValidationError(
            "synthetic comparison preflight failure",
            details={"reason": "unit_test"},
        )
        return {
            "status": "failed",
            "run_payload": None,
            "report_dir": None,
            "error": str(failure),
            "error_code": "official_contract_validation_error",
            "error_type": "OfficialContractValidationError",
            "failure_stage": "preflight_validation",
            "error_details": {"reason": "unit_test"},
            "timeout_seconds": None,
            "elapsed_seconds": None,
            "timeout_diagnostics_path": None,
            "child_pid": None,
            "termination_method": "normal_exit",
            "command": [],
        }

    monkeypatch.setattr(
        "Thesis_ML.comparisons.runner.execute_run_with_timeout_watchdog",
        _failing_watchdog,
    )
    result = compile_and_run_comparison(
        comparison=spec,
        index_csv=comparison_dataset["index_csv"],
        data_root=comparison_dataset["data_root"],
        cache_dir=comparison_dataset["cache_dir"],
        reports_root=comparison_dataset["reports_root"],
        variant_ids=["ridge"],
        dry_run=False,
    )
    assert result["n_failed"] > 0
    failed = [row for row in result["run_results"] if row["status"] == "failed"]
    assert failed
    first = failed[0]
    assert first["error"] == "synthetic comparison preflight failure"
    assert first["error_code"] == "official_contract_validation_error"
    assert first["error_type"] == "OfficialContractValidationError"
    assert first["failure_stage"] == "preflight_validation"
    assert first["error_details"] == {"reason": "unit_test"}


def test_comparison_runner_real_run_stamps_metadata(
    comparison_dataset: dict[str, Path],
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())
    result = compile_and_run_comparison(
        comparison=spec,
        index_csv=comparison_dataset["index_csv"],
        data_root=comparison_dataset["data_root"],
        cache_dir=comparison_dataset["cache_dir"],
        reports_root=comparison_dataset["reports_root"],
        variant_ids=["ridge"],
        dry_run=False,
    )
    assert result["n_failed"] == 0
    successful = [row for row in result["run_results"] if row["status"] == "success"]
    assert successful

    config_path = Path(str(successful[0]["config_path"]))
    metrics_path = Path(str(successful[0]["metrics_path"]))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert config["framework_mode"] == "locked_comparison"
    assert config["canonical_run"] is False
    assert config["comparison_id"] == spec.comparison_id
    assert config["comparison_version"] == spec.comparison_version
    assert config["comparison_variant_id"] == "ridge"
    assert config["methodology_policy_name"] == "fixed_baselines_only"
    assert config["subgroup_reporting_enabled"] is True
    assert isinstance(config["data_policy_effective"], dict)
    assert isinstance(config["data_artifacts"], dict)
    assert config["metric_policy_effective"]["primary_metric"] == "balanced_accuracy"
    assert config["metric_policy_effective"]["decision_metric"] == "balanced_accuracy"
    assert config["metric_policy_effective"]["tuning_metric"] == "balanced_accuracy"
    assert config["metric_policy_effective"]["permutation_metric"] == "balanced_accuracy"
    assert config["metric_policy_effective"]["higher_is_better"] is True
    assert config["hardware_mode_requested"] == "cpu_only"
    assert config["hardware_mode_effective"] == "cpu_only"
    assert config["requested_backend_family"] == "sklearn_cpu"
    assert config["effective_backend_family"] == "sklearn_cpu"
    assert config["gpu_device_id"] is None
    assert config["gpu_device_name"] is None
    assert config["gpu_device_total_memory_mb"] is None
    assert config["deterministic_compute"] is False
    assert config["allow_backend_fallback"] is False
    assert config["backend_stack_id"] == "sklearn_cpu_reference_v1"
    assert config["backend_fallback_used"] is False
    assert config["backend_fallback_reason"] is None
    assert isinstance(config["compute_policy"], dict)
    assert config["compute_policy"]["hardware_mode_effective"] == "cpu_only"

    assert metrics["framework_mode"] == "locked_comparison"
    assert metrics["canonical_run"] is False
    assert metrics["comparison_id"] == spec.comparison_id
    assert metrics["comparison_version"] == spec.comparison_version
    assert metrics["comparison_variant_id"] == "ridge"
    assert metrics["methodology_policy_name"] == "fixed_baselines_only"
    assert isinstance(metrics["data_policy_effective"], dict)
    assert isinstance(metrics["data_artifacts"], dict)
    assert "subgroup_reporting" in metrics
    assert metrics["decision_metric_name"] == "balanced_accuracy"
    assert metrics["tuning_metric_name"] == "balanced_accuracy"
    assert metrics["permutation_metric_name"] == "balanced_accuracy"
    assert metrics["metric_policy_effective"]["primary_metric"] == "balanced_accuracy"
    assert metrics["metric_policy_effective"]["decision_metric"] == "balanced_accuracy"
    assert metrics["metric_policy_effective"]["tuning_metric"] == "balanced_accuracy"
    assert metrics["metric_policy_effective"]["permutation_metric"] == "balanced_accuracy"
    assert metrics["metric_policy_effective"]["higher_is_better"] is True
    assert metrics["hardware_mode_requested"] == "cpu_only"
    assert metrics["hardware_mode_effective"] == "cpu_only"
    assert metrics["requested_backend_family"] == "sklearn_cpu"
    assert metrics["effective_backend_family"] == "sklearn_cpu"
    assert metrics["gpu_device_id"] is None
    assert metrics["gpu_device_name"] is None
    assert metrics["gpu_device_total_memory_mb"] is None
    assert metrics["deterministic_compute"] is False
    assert metrics["allow_backend_fallback"] is False
    assert metrics["backend_stack_id"] == "sklearn_cpu_reference_v1"
    assert metrics["backend_fallback_used"] is False
    assert metrics["backend_fallback_reason"] is None
    assert isinstance(metrics["compute_policy"], dict)
    assert metrics["compute_policy"]["hardware_mode_effective"] == "cpu_only"
    assert metrics["calibration"]["policy_status"] in {
        "required_if_probabilities_available",
        "probabilities_required_for_validity",
    }
    assert isinstance(metrics["calibration"]["probability_support_detected"], bool)
    assert Path(str(config["dataset_card_json_path"])).exists()
    assert Path(str(config["dataset_summary_json_path"])).exists()
    assert Path(str(config["data_quality_report_path"])).exists()
    assert Path(str(config["class_balance_report_path"])).exists()
    assert Path(str(config["missingness_report_path"])).exists()
    assert Path(str(config["leakage_audit_path"])).exists()

    execution_status = json.loads(
        Path(result["artifact_paths"]["execution_status"]).read_text(encoding="utf-8")
    )
    successful_rows = [row for row in execution_status["runs"] if row["status"] == "success"]
    assert successful_rows
    assert successful_rows[0]["compute_policy"]["hardware_mode_requested"] == "cpu_only"
    assert successful_rows[0]["compute_policy"]["hardware_mode_effective"] == "cpu_only"
    assert successful_rows[0]["compute_policy"]["backend_stack_id"] == "sklearn_cpu_reference_v1"
    assert successful_rows[0]["compute_policy"]["backend_fallback_used"] is False


def test_comparison_runner_timed_out_runs_are_explicit_and_invalidate_decision(
    comparison_dataset: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())

    def _timed_out_watchdog(*, run_kwargs: dict[str, object], **_: object) -> dict[str, object]:
        report_dir = Path(str(run_kwargs["reports_root"])) / str(run_kwargs["run_id"])
        report_dir.mkdir(parents=True, exist_ok=True)
        timeout_path = report_dir / "timeout_diagnostics.json"
        timeout_path.write_text(
            json.dumps(
                {
                    "run_id": str(run_kwargs["run_id"]),
                    "termination_method": "terminate",
                    "timeout_budget_seconds": 1,
                    "elapsed_seconds": 1.1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "timed_out",
            "run_payload": None,
            "report_dir": str(report_dir),
            "error": "run_exceeded_timeout_budget",
            "error_code": "run_timeout",
            "error_type": "RunTimeoutError",
            "failure_stage": "watchdog_timeout",
            "error_details": {"timeout_budget_seconds": 1},
            "timeout_seconds": 1.0,
            "elapsed_seconds": 1.1,
            "timeout_diagnostics_path": str(timeout_path),
            "child_pid": 12345,
            "termination_method": "terminate",
            "command": ["python", "-m", "Thesis_ML.experiments.supervised_worker"],
        }

    monkeypatch.setattr(
        "Thesis_ML.comparisons.runner.execute_run_with_timeout_watchdog",
        _timed_out_watchdog,
    )
    result = compile_and_run_comparison(
        comparison=spec,
        index_csv=comparison_dataset["index_csv"],
        data_root=comparison_dataset["data_root"],
        cache_dir=comparison_dataset["cache_dir"],
        reports_root=comparison_dataset["reports_root"],
        variant_ids=["ridge"],
        dry_run=False,
    )

    assert result["n_success"] == 0
    assert result["n_failed"] == 0
    assert result["n_timed_out"] > 0

    decision = json.loads(
        Path(result["artifact_paths"]["comparison_decision"]).read_text(encoding="utf-8")
    )
    assert decision["decision_status"] == "invalid_comparison"
    assert "timed_out" in str(decision["reason"])

    report_index = pd.read_csv(Path(result["artifact_paths"]["report_index"]))
    assert set(report_index["status"].astype(str)) == {"timed_out"}


def test_comparison_runner_rejects_draft_or_retired_status(
    comparison_dataset: dict[str, Path],
) -> None:
    spec = load_comparison_spec(_comparison_spec_path()).model_copy(deep=True)

    spec.status = ComparisonStatus.DRAFT
    with pytest.raises(ValueError, match="status is 'draft'"):
        compile_and_run_comparison(
            comparison=spec,
            index_csv=comparison_dataset["index_csv"],
            data_root=comparison_dataset["data_root"],
            cache_dir=comparison_dataset["cache_dir"],
            reports_root=comparison_dataset["reports_root"],
            dry_run=True,
        )

    spec.status = ComparisonStatus.RETIRED
    with pytest.raises(ValueError, match="status is 'retired'"):
        compile_and_run_comparison(
            comparison=spec,
            index_csv=comparison_dataset["index_csv"],
            data_root=comparison_dataset["data_root"],
            cache_dir=comparison_dataset["cache_dir"],
            reports_root=comparison_dataset["reports_root"],
            dry_run=True,
        )


def test_grouped_nested_comparison_spec_supports_dry_run(
    comparison_dataset: dict[str, Path],
) -> None:
    spec = load_comparison_spec(_nested_comparison_spec_path())
    assert spec.methodology_policy.policy_name.value == "grouped_nested_tuning"
    assert int(spec.evidence_policy.repeat_evaluation.repeat_count) == 3
    assert bool(spec.evidence_policy.paired_comparisons.require_significant_win) is True
    result = compile_and_run_comparison(
        comparison=spec,
        index_csv=comparison_dataset["index_csv"],
        data_root=comparison_dataset["data_root"],
        cache_dir=comparison_dataset["cache_dir"],
        reports_root=comparison_dataset["reports_root"],
        variant_ids=["ridge"],
        dry_run=True,
    )
    assert result["n_failed"] == 0
    decision_payload = json.loads(
        Path(result["artifact_paths"]["comparison_decision"]).read_text(encoding="utf-8")
    )
    assert decision_payload["decision_status"] in {
        "winner_selected",
        "inconclusive",
        "invalid_comparison",
    }


def test_grouped_nested_comparison_compiler_expands_repeats_and_untuned_ablation(
    comparison_dataset: dict[str, Path],
) -> None:
    spec = load_comparison_spec(_nested_comparison_spec_path()).model_copy(deep=True)
    spec.evidence_policy.repeat_evaluation.repeat_count = 2
    spec.evidence_policy.repeat_evaluation.seed_stride = 17

    manifest = compile_comparison(
        spec,
        index_csv=comparison_dataset["index_csv"],
        variant_ids=["ridge"],
    )
    primary_runs = [run for run in manifest.runs if run.evidence_run_role.value == "primary"]
    untuned_runs = [
        run for run in manifest.runs if run.evidence_run_role.value == "untuned_baseline"
    ]
    assert len(primary_runs) == 4
    assert len(untuned_runs) == 4

    assert {run.repeat_id for run in primary_runs} == {1, 2}
    assert {run.repeat_count for run in primary_runs} == {2}
    assert all(run.tuning_enabled is True for run in primary_runs)
    assert all(run.tuning_enabled is False for run in untuned_runs)
    assert all(run.run_id.endswith("__untuned") for run in untuned_runs)

    for primary in primary_runs:
        expected_untuned = f"{primary.run_id}__untuned"
        assert any(run.run_id == expected_untuned for run in untuned_runs)


def test_comparison_runner_resume_reuses_completed_runs_without_rerun(
    comparison_dataset: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())
    call_counter = {"count": 0}

    def _successful_watchdog(*, run_kwargs: dict[str, object], **_: object) -> dict[str, object]:
        call_counter["count"] += 1
        run_id = str(run_kwargs["run_id"])
        report_dir = Path(str(run_kwargs["reports_root"])) / run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        config_path = report_dir / "config.json"
        metrics_path = report_dir / "metrics.json"
        metrics_payload = {
            "balanced_accuracy": 0.5,
            "macro_f1": 0.5,
            "accuracy": 0.5,
            "n_folds": 1,
            "primary_metric_name": "balanced_accuracy",
            "primary_metric_value": 0.5,
        }
        config_path.write_text(f"{json.dumps({'run_id': run_id}, indent=2)}\n", encoding="utf-8")
        metrics_path.write_text(f"{json.dumps(metrics_payload, indent=2)}\n", encoding="utf-8")
        (report_dir / "run_status.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "success",
                    "updated_at_utc": "2026-03-19T00:00:00+00:00",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "success",
            "run_payload": {
                "report_dir": str(report_dir),
                "config_path": str(config_path),
                "metrics_path": str(metrics_path),
                "metrics": metrics_payload,
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

    initial = compile_and_run_comparison(
        comparison=spec,
        index_csv=comparison_dataset["index_csv"],
        data_root=comparison_dataset["data_root"],
        cache_dir=comparison_dataset["cache_dir"],
        reports_root=comparison_dataset["reports_root"],
        variant_ids=["ridge"],
        dry_run=False,
    )
    assert initial["n_success"] > 0

    call_counter["count"] = 0
    resumed = compile_and_run_comparison(
        comparison=spec,
        index_csv=comparison_dataset["index_csv"],
        data_root=comparison_dataset["data_root"],
        cache_dir=comparison_dataset["cache_dir"],
        reports_root=comparison_dataset["reports_root"],
        variant_ids=["ridge"],
        resume=True,
        dry_run=False,
    )
    assert resumed["n_failed"] == 0
    assert call_counter["count"] == 0
    reconciliation = resumed["resume_reconciliation"]
    assert int(reconciliation["n_reused"]) == int(resumed["n_success"])
    assert int(reconciliation["n_rerun"]) == 0


def test_comparison_runner_resume_reruns_only_timed_out_runs(
    comparison_dataset: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())
    call_counter = {"count": 0}

    def _successful_watchdog(*, run_kwargs: dict[str, object], **_: object) -> dict[str, object]:
        call_counter["count"] += 1
        run_id = str(run_kwargs["run_id"])
        report_dir = Path(str(run_kwargs["reports_root"])) / run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        config_path = report_dir / "config.json"
        metrics_path = report_dir / "metrics.json"
        metrics_payload = {
            "balanced_accuracy": 0.5,
            "macro_f1": 0.5,
            "accuracy": 0.5,
            "n_folds": 1,
            "primary_metric_name": "balanced_accuracy",
            "primary_metric_value": 0.5,
        }
        config_path.write_text(f"{json.dumps({'run_id': run_id}, indent=2)}\n", encoding="utf-8")
        metrics_path.write_text(f"{json.dumps(metrics_payload, indent=2)}\n", encoding="utf-8")
        (report_dir / "run_status.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "success",
                    "updated_at_utc": "2026-03-19T00:00:00+00:00",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "success",
            "run_payload": {
                "report_dir": str(report_dir),
                "config_path": str(config_path),
                "metrics_path": str(metrics_path),
                "metrics": metrics_payload,
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

    initial = compile_and_run_comparison(
        comparison=spec,
        index_csv=comparison_dataset["index_csv"],
        data_root=comparison_dataset["data_root"],
        cache_dir=comparison_dataset["cache_dir"],
        reports_root=comparison_dataset["reports_root"],
        variant_ids=["ridge"],
        dry_run=False,
    )
    assert initial["n_success"] > 0

    first_run_id = str(initial["run_results"][0]["run_id"])
    first_report_dir = Path(str(initial["run_results"][0]["report_dir"]))
    (first_report_dir / "run_status.json").write_text(
        json.dumps(
            {
                "run_id": first_run_id,
                "status": "timed_out",
                "updated_at_utc": "2026-03-19T00:10:00+00:00",
                "error": "run_exceeded_timeout_budget",
                "error_code": "run_timeout",
                "error_type": "RunTimeoutError",
                "failure_stage": "watchdog_timeout",
                "timeout_seconds": 1.0,
                "elapsed_seconds": 1.2,
                "timeout_diagnostics_path": str(first_report_dir / "timeout_diagnostics.json"),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    call_counter["count"] = 0
    resumed = compile_and_run_comparison(
        comparison=spec,
        index_csv=comparison_dataset["index_csv"],
        data_root=comparison_dataset["data_root"],
        cache_dir=comparison_dataset["cache_dir"],
        reports_root=comparison_dataset["reports_root"],
        variant_ids=["ridge"],
        resume=True,
        dry_run=False,
    )
    assert call_counter["count"] == 1
    reconciliation = resumed["resume_reconciliation"]
    assert int(reconciliation["n_existing_timed_out"]) == 1
    assert int(reconciliation["n_rerun"]) == 1


def test_comparison_runner_force_reruns_even_when_completed(
    comparison_dataset: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())
    call_counter = {"count": 0}

    def _successful_watchdog(*, run_kwargs: dict[str, object], **_: object) -> dict[str, object]:
        call_counter["count"] += 1
        run_id = str(run_kwargs["run_id"])
        report_dir = Path(str(run_kwargs["reports_root"])) / run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        config_path = report_dir / "config.json"
        metrics_path = report_dir / "metrics.json"
        metrics_payload = {
            "balanced_accuracy": 0.5,
            "macro_f1": 0.5,
            "accuracy": 0.5,
            "n_folds": 1,
            "primary_metric_name": "balanced_accuracy",
            "primary_metric_value": 0.5,
        }
        config_path.write_text(f"{json.dumps({'run_id': run_id}, indent=2)}\n", encoding="utf-8")
        metrics_path.write_text(f"{json.dumps(metrics_payload, indent=2)}\n", encoding="utf-8")
        (report_dir / "run_status.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "success",
                    "updated_at_utc": "2026-03-19T00:00:00+00:00",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "success",
            "run_payload": {
                "report_dir": str(report_dir),
                "config_path": str(config_path),
                "metrics_path": str(metrics_path),
                "metrics": metrics_payload,
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

    initial = compile_and_run_comparison(
        comparison=spec,
        index_csv=comparison_dataset["index_csv"],
        data_root=comparison_dataset["data_root"],
        cache_dir=comparison_dataset["cache_dir"],
        reports_root=comparison_dataset["reports_root"],
        variant_ids=["ridge"],
        dry_run=False,
    )
    assert initial["n_success"] > 0

    call_counter["count"] = 0
    forced = compile_and_run_comparison(
        comparison=spec,
        index_csv=comparison_dataset["index_csv"],
        data_root=comparison_dataset["data_root"],
        cache_dir=comparison_dataset["cache_dir"],
        reports_root=comparison_dataset["reports_root"],
        variant_ids=["ridge"],
        force=True,
        dry_run=False,
    )
    assert forced["n_failed"] == 0
    assert call_counter["count"] == int(forced["n_success"])
    reconciliation = forced["resume_reconciliation"]
    assert int(reconciliation["n_rerun"]) == int(forced["n_success"])


def test_comparison_runner_fresh_refuses_existing_outputs(
    comparison_dataset: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())

    def _successful_watchdog(*, run_kwargs: dict[str, object], **_: object) -> dict[str, object]:
        run_id = str(run_kwargs["run_id"])
        report_dir = Path(str(run_kwargs["reports_root"])) / run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        config_path = report_dir / "config.json"
        metrics_path = report_dir / "metrics.json"
        metrics_payload = {
            "balanced_accuracy": 0.5,
            "macro_f1": 0.5,
            "accuracy": 0.5,
            "n_folds": 1,
            "primary_metric_name": "balanced_accuracy",
            "primary_metric_value": 0.5,
        }
        config_path.write_text(f"{json.dumps({'run_id': run_id}, indent=2)}\n", encoding="utf-8")
        metrics_path.write_text(f"{json.dumps(metrics_payload, indent=2)}\n", encoding="utf-8")
        (report_dir / "run_status.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "success",
                    "updated_at_utc": "2026-03-19T00:00:00+00:00",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "success",
            "run_payload": {
                "report_dir": str(report_dir),
                "config_path": str(config_path),
                "metrics_path": str(metrics_path),
                "metrics": metrics_payload,
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

    first = compile_and_run_comparison(
        comparison=spec,
        index_csv=comparison_dataset["index_csv"],
        data_root=comparison_dataset["data_root"],
        cache_dir=comparison_dataset["cache_dir"],
        reports_root=comparison_dataset["reports_root"],
        variant_ids=["ridge"],
        dry_run=False,
    )
    assert first["n_success"] > 0

    with pytest.raises(RuntimeError, match="Fresh comparison execution refused"):
        compile_and_run_comparison(
            comparison=spec,
            index_csv=comparison_dataset["index_csv"],
            data_root=comparison_dataset["data_root"],
            cache_dir=comparison_dataset["cache_dir"],
            reports_root=comparison_dataset["reports_root"],
            variant_ids=["ridge"],
            dry_run=False,
        )


def test_comparison_runner_rejects_nonpositive_max_parallel_runs(
    comparison_dataset: dict[str, Path],
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())
    with pytest.raises(ValueError, match="max_parallel_runs must be >= 1"):
        compile_and_run_comparison(
            comparison=spec,
            index_csv=comparison_dataset["index_csv"],
            data_root=comparison_dataset["data_root"],
            cache_dir=comparison_dataset["cache_dir"],
            reports_root=comparison_dataset["reports_root"],
            variant_ids=["ridge"],
            dry_run=True,
            max_parallel_runs=0,
        )
