from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.verification.performance_baseline import (
    BenchmarkCaseResult,
    BenchmarkCaseSpec,
    BenchmarkFingerprint,
    compare_baseline_bundles,
    run_baseline_suite,
    write_baseline_bundle,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bundle_payload(case_payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "performance-baseline-bundle-v1",
        "generated_at_utc": "2026-03-30T00:00:00+00:00",
        "baseline_mode": "ci_synthetic",
        "suite_id": "phase0_runtime_baseline",
        "output_root": "C:/tmp/baseline",
        "case_specs": [],
        "case_results": [case_payload],
        "aggregate_summary": {"n_cases": 1, "n_passed": 1, "n_failed": 0, "failed_case_ids": []},
    }


def test_run_baseline_suite_and_write_bundle_on_shipped_assets(tmp_path: Path) -> None:
    shipped_index = _repo_root() / "demo_data" / "synthetic_v1" / "dataset_index.csv"
    assert shipped_index.exists()

    case_spec = BenchmarkCaseSpec(
        case_id="shipped_asset_probe",
        mode="ci_synthetic",
        description="Minimal shipped-asset probe.",
        inputs={"dataset_index": str(shipped_index.resolve())},
    )

    def _case_runner(_: BenchmarkCaseSpec, case_dir: Path) -> BenchmarkCaseResult:
        marker = case_dir / "marker.json"
        marker.write_text(
            json.dumps({"dataset_index_exists": bool(shipped_index.exists())}, indent=2) + "\n",
            encoding="utf-8",
        )
        return BenchmarkCaseResult(
            case_id="shipped_asset_probe",
            status="passed",
            elapsed_seconds=0.001,
            artifact_refs={"marker_path": str(marker.resolve())},
            fingerprints=BenchmarkFingerprint(config_fingerprint="cfg"),
            scientific_metrics={},
            observability={
                "expected_artifacts": [str(marker.resolve())],
                "artifact_presence": {str(marker.resolve()): True},
                "schema_checks": {"dataset_index_exists": True},
            },
            summary={},
        )

    bundle = run_baseline_suite(
        suite_id="phase0_runtime_baseline",
        baseline_mode="ci_synthetic",
        output_root=tmp_path / "bundle_root",
        case_specs=[case_spec],
        case_runner=_case_runner,
    )
    assert bundle.aggregate_summary["n_failed"] == 0
    bundle_path = write_baseline_bundle(bundle, tmp_path / "bundle_root" / "baseline_bundle.json")
    assert bundle_path.exists()
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "performance-baseline-bundle-v1"
    assert payload["case_results"][0]["case_id"] == "shipped_asset_probe"


def test_compare_baseline_bundles_detects_scientific_drift() -> None:
    left = _bundle_payload(
        {
            "case_id": "single_run_ridge_direct",
            "fingerprints": {
                "selected_sample_fingerprint": "same",
                "fold_split_fingerprint": "same",
                "prediction_fingerprint": "same",
                "tuning_fingerprint": None,
                "stage_execution_fingerprint": "same",
            },
            "scientific_metrics": {"primary_metric_value": 0.81},
            "observability": {"expected_artifacts": [], "artifact_presence": {}, "schema_checks": {}},
        }
    )
    right = _bundle_payload(
        {
            "case_id": "single_run_ridge_direct",
            "fingerprints": {
                "selected_sample_fingerprint": "same",
                "fold_split_fingerprint": "same",
                "prediction_fingerprint": "same",
                "tuning_fingerprint": None,
                "stage_execution_fingerprint": "same",
            },
            "scientific_metrics": {"primary_metric_value": 0.72},
            "observability": {"expected_artifacts": [], "artifact_presence": {}, "schema_checks": {}},
        }
    )

    comparison = compare_baseline_bundles(
        left_bundle=left,
        right_bundle=right,
        metrics_tolerance=1e-6,
    )
    assert comparison["scientific_parity"] is False
    assert any(
        mismatch.get("code") == "scientific_metric_drift"
        for mismatch in comparison["mismatches"]["scientific"]
    )


def test_compare_baseline_bundles_tolerates_pure_timing_drift() -> None:
    left = _bundle_payload(
        {
            "case_id": "single_run_ridge_direct",
            "elapsed_seconds": 10.0,
            "fingerprints": {
                "selected_sample_fingerprint": "same",
                "fold_split_fingerprint": "same",
                "prediction_fingerprint": "same",
                "tuning_fingerprint": "same",
                "stage_execution_fingerprint": "same",
            },
            "scientific_metrics": {"primary_metric_value": 0.81},
            "observability": {"expected_artifacts": [], "artifact_presence": {}, "schema_checks": {}},
            "summary": {"wall_clock_elapsed_seconds": 10.0},
        }
    )
    right = _bundle_payload(
        {
            "case_id": "single_run_ridge_direct",
            "elapsed_seconds": 14.0,
            "fingerprints": {
                "selected_sample_fingerprint": "same",
                "fold_split_fingerprint": "same",
                "prediction_fingerprint": "same",
                "tuning_fingerprint": "same",
                "stage_execution_fingerprint": "same",
            },
            "scientific_metrics": {"primary_metric_value": 0.8100000001},
            "observability": {"expected_artifacts": [], "artifact_presence": {}, "schema_checks": {}},
            "summary": {"wall_clock_elapsed_seconds": 14.0},
        }
    )

    comparison = compare_baseline_bundles(
        left_bundle=left,
        right_bundle=right,
        metrics_tolerance=1e-6,
    )
    assert comparison["scientific_parity"] is True
    assert comparison["observability_parity"] is True


def test_compare_baseline_bundles_detects_missing_observability_artifacts() -> None:
    expected_artifact = str((_repo_root() / "does_not_exist.json").resolve())
    left = _bundle_payload(
        {
            "case_id": "single_run_ridge_direct",
            "fingerprints": {
                "selected_sample_fingerprint": "same",
                "fold_split_fingerprint": "same",
                "prediction_fingerprint": "same",
                "tuning_fingerprint": None,
                "stage_execution_fingerprint": "same",
            },
            "scientific_metrics": {"primary_metric_value": 0.81},
            "observability": {
                "expected_artifacts": [expected_artifact],
                "artifact_presence": {expected_artifact: False},
                "schema_checks": {"status_schema_valid": True},
            },
        }
    )
    right = _bundle_payload(
        {
            "case_id": "single_run_ridge_direct",
            "fingerprints": {
                "selected_sample_fingerprint": "same",
                "fold_split_fingerprint": "same",
                "prediction_fingerprint": "same",
                "tuning_fingerprint": None,
                "stage_execution_fingerprint": "same",
            },
            "scientific_metrics": {"primary_metric_value": 0.81},
            "observability": {
                "expected_artifacts": [expected_artifact],
                "artifact_presence": {expected_artifact: True},
                "schema_checks": {"status_schema_valid": True},
            },
        }
    )

    comparison = compare_baseline_bundles(
        left_bundle=left,
        right_bundle=right,
    )
    assert comparison["observability_parity"] is False
    assert any(
        mismatch.get("code") == "missing_observability_artifact"
        for mismatch in comparison["mismatches"]["observability"]
    )
