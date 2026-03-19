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

    dataset_fingerprint = {
        "index_csv_sha256": "abc123",
        "selected_sample_id_sha256": "def456",
        "target_column": "coarse_affect",
        "cv_mode": "within_subject_loso_session",
    }
    config_payload = {
        "framework_mode": "confirmatory",
        "canonical_run": True,
        "methodology_policy_name": "fixed_baselines_only",
        "class_weight_policy": "none",
        "tuning_enabled": False,
        "model_cost_tier": "official_fast",
        "projected_runtime_seconds": 1200,
        "evidence_run_role": "primary",
        "repeat_id": 1,
        "repeat_count": 1,
        "base_run_id": "run_001",
        "protocol_id": "thesis-canonical",
        "protocol_version": "1.0.0",
        "protocol_schema_version": "thesis-protocol-v1",
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
        "data_policy_effective": {
            "class_balance": {
                "enabled": True,
                "axes": ["overall"],
                "min_class_fraction_warning": 0.05,
                "min_class_fraction_blocking": None,
            },
            "missingness": {
                "enabled": True,
                "max_missing_fraction_warning": 0.1,
                "max_missing_fraction_blocking": None,
            },
            "leakage": {
                "enabled": True,
                "fail_on_duplicate_sample_id": True,
                "warn_on_duplicate_beta_path": True,
                "fail_on_duplicate_beta_path": False,
                "fail_on_subject_overlap_for_transfer": True,
                "fail_on_cv_group_overlap": True,
            },
            "external_validation": {
                "enabled": False,
                "mode": "compatibility_only",
                "require_compatible": False,
                "require_for_official_runs": False,
                "datasets": [],
            },
            "required_index_columns": [],
            "intended_use": "unit test",
            "not_intended_use": [],
            "known_limitations": [],
        },
        "data_artifacts": {
            "dataset_card_json": str((run_dir / "dataset_card.json").resolve()),
            "dataset_summary_json": str((run_dir / "dataset_summary.json").resolve()),
            "data_quality_report_json": str((run_dir / "data_quality_report.json").resolve()),
            "leakage_audit_json": str((run_dir / "leakage_audit.json").resolve()),
        },
        "evidence_policy_effective": {
            "repeat_evaluation": {"repeat_count": 1, "seed_stride": 1000},
            "confidence_intervals": {
                "method": "grouped_bootstrap_percentile",
                "confidence_level": 0.95,
                "n_bootstrap": 10,
                "seed": 2026,
            },
            "paired_comparisons": {
                "method": "paired_sign_flip_permutation",
                "n_permutations": 100,
                "alpha": 0.05,
                "require_significant_win": False,
            },
            "permutation": {
                "alpha": 0.05,
                "minimum_permutations": 10,
                "require_pass_for_validity": False,
            },
            "calibration": {
                "enabled": True,
                "n_bins": 10,
                "require_probabilities_for_validity": False,
            },
            "required_package": {
                "require_dummy_baseline": True,
                "require_permutation_control": True,
                "require_untuned_baseline_if_tuning": False,
            },
        },
        "primary_metric_name": "balanced_accuracy",
        "dataset_fingerprint": dataset_fingerprint,
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

    (run_dir / "config.json").write_text(
        f"{json.dumps(config_payload, indent=2)}\n", encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        f"{json.dumps(metrics_payload, indent=2)}\n", encoding="utf-8"
    )
    (run_dir / "fold_splits.csv").write_text("fold,train,test\n0,a,b\n", encoding="utf-8")
    (run_dir / "predictions.csv").write_text("y_true,y_pred\nanger,anger\n", encoding="utf-8")
    (run_dir / "dataset_card.json").write_text(
        json.dumps(
            {
                "framework_mode": "confirmatory",
                "dataset_identity": {"dataset_fingerprint": dataset_fingerprint},
                "target_definition": {"target_column": "coarse_affect"},
                "coverage": {"selected_subset": {"n_rows": 2}},
                "external_validation": {
                    "enabled": False,
                    "mode": "compatibility_only",
                    "status": "not_configured",
                    "datasets": [],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "dataset_card.md").write_text("# Dataset Card\n", encoding="utf-8")
    (run_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "framework_mode": "confirmatory",
                "target_column": "coarse_affect",
                "full_index": {"n_rows": 2},
                "selected_subset": {"n_rows": 2},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "dataset_summary.csv").write_text(
        "scope,n_rows,n_subjects,n_sessions\nselected_subset,2,1,2\n",
        encoding="utf-8",
    )
    (run_dir / "data_quality_report.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "n_blocking_issues": 0,
                "n_warnings": 0,
                "blocking_issues": [],
                "warnings": [],
                "leakage_audit_verdict": "pass",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "class_balance_report.csv").write_text(
        "scope,axis,group_value,n_samples,n_classes,min_class_fraction,status\nselected_subset,overall,__overall__,2,2,0.5,ok\n",
        encoding="utf-8",
    )
    (run_dir / "missingness_report.csv").write_text(
        "scope,column,missing_count,missing_fraction,status\nselected_subset,coarse_affect,0,0.0,ok\n",
        encoding="utf-8",
    )
    (run_dir / "leakage_audit.json").write_text(
        json.dumps({"verdict": "pass", "checks": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "external_dataset_card.json").write_text(
        json.dumps({"enabled": False, "datasets": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "external_dataset_summary.json").write_text(
        json.dumps({"enabled": False, "n_datasets": 0, "datasets": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "external_validation_compatibility.json").write_text(
        json.dumps(
            {
                "enabled": False,
                "mode": "compatibility_only",
                "status": "not_configured",
                "datasets": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (root / "protocol.json").write_text(
        json.dumps(
            {
                "artifact_contract": {
                    "required_run_metadata_fields": [
                        "framework_mode",
                        "canonical_run",
                        "methodology_policy_name",
                        "model_cost_tier",
                        "projected_runtime_seconds",
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
                    "dataset_card.json",
                    "dataset_card.md",
                    "dataset_summary.json",
                    "dataset_summary.csv",
                    "data_quality_report.json",
                    "class_balance_report.csv",
                    "missingness_report.csv",
                    "leakage_audit.json",
                    "external_dataset_card.json",
                    "external_dataset_summary.json",
                    "external_validation_compatibility.json",
                    "fold_splits.csv",
                    "predictions.csv",
                ],
                "required_run_metadata_fields": [
                    "framework_mode",
                    "canonical_run",
                    "methodology_policy_name",
                    "model_cost_tier",
                    "projected_runtime_seconds",
                    "data_policy_effective",
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
    (root / "suite_summary.json").write_text(
        json.dumps(
            {
                "required_evidence_status": {
                    "valid": True,
                    "required_checks": [
                        "dummy_baseline",
                        "permutation_control",
                    ],
                    "missing_checks": [],
                },
                "confirmatory_reporting_contract": {
                    "protocol_id": "thesis-canonical",
                    "protocol_version": "1.0.0",
                    "dataset_fingerprint": {
                        "n_completed_runs": 1,
                        "n_with_fingerprint": 1,
                        "n_missing_fingerprint": 0,
                        "missing_run_ids": [],
                        "unique_fingerprint_count": 1,
                        "consistent_across_runs": True,
                        "sources": ["config"],
                    },
                    "target_mapping_version": "affect_mapping_v1",
                    "target_mapping_hash": "hash001",
                    "primary_split": "within_subject_loso_session",
                    "primary_metric": "balanced_accuracy",
                    "model_family": "ridge",
                    "controls_status": {
                        "dummy_baseline_required": True,
                        "dummy_baseline_present": True,
                        "permutation_required": True,
                        "minimum_permutations": 10,
                        "dummy_requirement_satisfied": True,
                        "permutation_requirement_satisfied": True,
                        "controls_valid_for_confirmatory": True,
                    },
                    "multiplicity_policy": {
                        "primary_hypotheses": 1,
                        "primary_alpha": 0.05,
                        "secondary_policy": "descriptive_only",
                        "exploratory_claims_allowed": False,
                    },
                    "interpretation_limits": {"no_causal_claims": True},
                    "subgroup_evidence_policy": {
                        "evidence_role": "descriptive_only",
                        "primary_evidence_substitution_allowed": False,
                    },
                    "deviations_from_protocol": {
                        "n_total_deviations": 0,
                        "n_science_critical_deviations": 0,
                        "science_critical_deviation_detected": False,
                        "controls_valid_for_confirmatory": True,
                        "confirmatory_status": "confirmatory",
                        "explicit_no_deviation_record": True,
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "execution_status.json").write_text(
        json.dumps(
            {
                "framework_mode": "confirmatory",
                "confirmatory_status": "confirmatory",
                "runs": [{"run_id": "run_001", "status": "completed"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "repeated_run_metrics.csv").write_text(
        "group_key,run_id,base_run_id,evidence_run_role,repeat_id,repeat_count,primary_metric_name,primary_metric_value\n"
        "primary_within_subject,run_001,run_001,primary,1,1,balanced_accuracy,0.75\n",
        encoding="utf-8",
    )
    (root / "repeated_run_summary.json").write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "group_key": "primary_within_subject",
                        "n_runs": 1,
                        "primary_metric_name": "balanced_accuracy",
                        "mean_primary_metric_value": 0.75,
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "confidence_intervals.json").write_text(
        json.dumps(
            {
                "intervals": [
                    {
                        "group_key": "primary_within_subject",
                        "metric_name": "balanced_accuracy",
                        "confidence_level": 0.95,
                        "lower_bound": 0.75,
                        "upper_bound": 0.75,
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "metric_intervals.csv").write_text(
        "group_key,metric_name,confidence_level,lower_bound,upper_bound\n"
        "primary_within_subject,balanced_accuracy,0.95,0.75,0.75\n",
        encoding="utf-8",
    )
    (root / "deviation_log.json").write_text(
        json.dumps(
            {
                "framework_mode": "confirmatory",
                "protocol_id": "thesis-canonical",
                "protocol_version": "1.0.0",
                "science_critical_deviation_detected": False,
                "confirmatory_status": "confirmatory",
                "n_total_deviations": 0,
                "n_science_critical_deviations": 0,
                "deviations": [
                    {
                        "run_id": None,
                        "suite_id": None,
                        "status": "no_deviation",
                        "science_critical": False,
                        "reason": "No protocol deviations detected.",
                    }
                ],
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


def test_verify_official_artifacts_fails_when_strict_confirmatory_multiplicity_missing(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "protocol_runs" / "thesis-confirmatory__1.0.0"
    _write_confirmatory_output(output_dir)

    protocol_payload = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))
    protocol_payload["protocol_id"] = "thesis_confirmatory_v1"
    (output_dir / "protocol.json").write_text(
        f"{json.dumps(protocol_payload, indent=2)}\n", encoding="utf-8"
    )

    summary_payload = json.loads((output_dir / "suite_summary.json").read_text(encoding="utf-8"))
    contract = summary_payload["confirmatory_reporting_contract"]
    contract.pop("multiplicity_policy", None)
    (output_dir / "suite_summary.json").write_text(
        f"{json.dumps(summary_payload, indent=2)}\n", encoding="utf-8"
    )

    summary = verify_official_artifacts(output_dir=output_dir, mode="confirmatory")
    assert summary["passed"] is False
    assert any(
        issue["code"]
        in {
            "confirmatory_reporting_field_missing",
            "confirmatory_freeze_multiplicity_policy_missing",
        }
        for issue in summary["issues"]
    )


def test_verify_official_artifacts_resolves_relative_report_dirs(tmp_path: Path) -> None:
    output_dir = tmp_path / "protocol_runs" / "thesis-canonical__1.0.0"
    run_dir = _write_confirmatory_output(output_dir)

    report_index_path = output_dir / "report_index.csv"
    report_index_path.write_text(
        "run_id,status,report_dir\n"
        f"run_001,completed,{run_dir.relative_to(output_dir).as_posix()}\n",
        encoding="utf-8",
    )

    summary = verify_official_artifacts(output_dir=output_dir)
    assert summary["passed"] is True


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
        mismatch["code"] == "run_artifacts_mismatch" for mismatch in mismatch_summary["mismatches"]
    )
