from __future__ import annotations

import csv
import json
from pathlib import Path

from Thesis_ML.verification.official_artifacts import verify_official_artifacts
from Thesis_ML.verification.reproducibility import compare_official_outputs


def _write_report_index(path: Path, report_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "status", "report_dir"])
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "run_001",
                "status": "completed",
                "report_dir": str(report_dir.resolve()),
            }
        )


def _write_confirmatory_output(root: Path) -> Path:
    run_dir = root / "runs" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "framework_mode": "confirmatory",
        "canonical_run": True,
        "methodology_policy_name": "fixed_baselines_only",
        "protocol_id": "thesis-canonical",
        "protocol_version": "1.0.0",
        "suite_id": "primary_within_subject",
        "claim_ids": ["C01"],
        "metric_policy_effective": {
            "primary_metric": "balanced_accuracy",
            "secondary_metrics": ["macro_f1", "accuracy"],
            "decision_metric": "balanced_accuracy",
            "tuning_metric": "balanced_accuracy",
            "permutation_metric": "balanced_accuracy",
            "higher_is_better": True,
        },
        "primary_metric_name": "balanced_accuracy",
    }
    metrics_payload = {
        **config_payload,
        "balanced_accuracy": 0.75,
        "macro_f1": 0.74,
        "accuracy": 0.76,
        "primary_metric_value": 0.75,
        "permutation_test": {
            "metric_name": "balanced_accuracy",
            "p_value": 0.03,
            "observed_score": 0.75,
            "n_permutations": 10,
        },
    }

    (run_dir / "config.json").write_text(f"{json.dumps(config_payload, indent=2)}\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        f"{json.dumps(metrics_payload, indent=2)}\n", encoding="utf-8"
    )
    (run_dir / "fold_splits.csv").write_text("fold,train,test\n0,a,b\n", encoding="utf-8")
    (run_dir / "predictions.csv").write_text("y_true,y_pred\nanger,anger\n", encoding="utf-8")

    (root / "protocol.json").write_text(
        json.dumps(
            {
                "artifact_contract": {
                    "required_run_metadata_fields": [
                        "framework_mode",
                        "canonical_run",
                        "methodology_policy_name",
                        "protocol_id",
                        "protocol_version",
                        "suite_id",
                        "claim_ids",
                    ]
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "compiled_protocol_manifest.json").write_text(
        json.dumps(
            {
                "required_run_artifacts": [
                    "config.json",
                    "metrics.json",
                    "fold_splits.csv",
                    "predictions.csv",
                ],
                "required_run_metadata_fields": [
                    "framework_mode",
                    "canonical_run",
                    "methodology_policy_name",
                    "protocol_id",
                    "protocol_version",
                    "suite_id",
                    "claim_ids",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "claim_to_run_map.json").write_text("{}\n", encoding="utf-8")
    (root / "suite_summary.json").write_text("{}\n", encoding="utf-8")
    (root / "execution_status.json").write_text(
        json.dumps(
            {
                "framework_mode": "confirmatory",
                "runs": [{"run_id": "run_001", "status": "completed"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report_index(root / "report_index.csv", run_dir)
    return run_dir


def test_verify_official_artifacts_passes_for_valid_confirmatory_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "protocol_runs" / "thesis-canonical__1.0.0"
    _write_confirmatory_output(output_dir)

    summary = verify_official_artifacts(output_dir=output_dir)
    assert summary["passed"] is True
    assert summary["framework_mode"] == "confirmatory"
    assert summary["n_completed_runs_checked"] == 1


def test_verify_official_artifacts_fails_when_required_run_artifact_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "protocol_runs" / "thesis-canonical__1.0.0"
    run_dir = _write_confirmatory_output(output_dir)
    (run_dir / "metrics.json").unlink()

    summary = verify_official_artifacts(output_dir=output_dir)
    assert summary["passed"] is False
    assert any(issue["code"] == "run_artifact_missing" for issue in summary["issues"])


def test_compare_official_outputs_detects_deterministic_mismatch(tmp_path: Path) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"

    _write_confirmatory_output(left_dir)
    _write_confirmatory_output(right_dir)

    equal_summary = compare_official_outputs(left_dir=left_dir, right_dir=right_dir)
    assert equal_summary["passed"] is True

    right_metrics_path = right_dir / "runs" / "run_001" / "metrics.json"
    metrics_payload = json.loads(right_metrics_path.read_text(encoding="utf-8"))
    metrics_payload["primary_metric_value"] = 0.11
    right_metrics_path.write_text(f"{json.dumps(metrics_payload, indent=2)}\n", encoding="utf-8")

    mismatch_summary = compare_official_outputs(left_dir=left_dir, right_dir=right_dir)
    assert mismatch_summary["passed"] is False
    assert any(
        mismatch["code"] == "run_artifacts_mismatch"
        for mismatch in mismatch_summary["mismatches"]
    )
