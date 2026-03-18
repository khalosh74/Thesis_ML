from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

FrameworkModeLiteral = Literal["confirmatory", "locked_comparison"]
_REQUIRED_PROTOCOL_ARTIFACTS = (
    "protocol.json",
    "compiled_protocol_manifest.json",
    "claim_to_run_map.json",
    "suite_summary.json",
    "execution_status.json",
    "deviation_log.json",
    "repeated_run_metrics.csv",
    "repeated_run_summary.json",
    "confidence_intervals.json",
    "metric_intervals.csv",
    "report_index.csv",
)
_REQUIRED_COMPARISON_ARTIFACTS = (
    "comparison.json",
    "compiled_comparison_manifest.json",
    "comparison_summary.json",
    "comparison_decision.json",
    "execution_status.json",
    "repeated_run_metrics.csv",
    "repeated_run_summary.json",
    "confidence_intervals.json",
    "metric_intervals.csv",
    "paired_model_comparisons.json",
    "paired_model_comparisons.csv",
    "report_index.csv",
)
_REQUIRED_DATA_RUN_ARTIFACTS = (
    "dataset_card.json",
    "dataset_summary.json",
    "data_quality_report.json",
    "class_balance_report.csv",
    "missingness_report.csv",
    "leakage_audit.json",
    "external_dataset_card.json",
    "external_dataset_summary.json",
    "external_validation_compatibility.json",
)


def _add_issue(
    issues: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    path: Path | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "code": str(code),
        "message": str(message),
    }
    if path is not None:
        payload["path"] = str(path)
    if details:
        payload["details"] = details
    issues.append(payload)


def _load_json(path: Path, issues: list[dict[str, Any]], *, code_prefix: str) -> dict[str, Any] | None:
    if not path.exists():
        _add_issue(
            issues,
            code=f"{code_prefix}_missing",
            message=f"Missing required JSON file: {path.name}",
            path=path,
        )
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _add_issue(
            issues,
            code=f"{code_prefix}_invalid_json",
            message=f"Invalid JSON in {path.name}: {exc}",
            path=path,
        )
        return None
    if not isinstance(payload, dict):
        _add_issue(
            issues,
            code=f"{code_prefix}_invalid_shape",
            message=f"Expected JSON object in {path.name}.",
            path=path,
        )
        return None
    return payload


def _expected_mode(mode_hint: str | None, execution_status: dict[str, Any]) -> FrameworkModeLiteral | None:
    detected = execution_status.get("framework_mode")
    if not isinstance(detected, str):
        return None
    if mode_hint is None:
        if detected == "confirmatory":
            return "confirmatory"
        if detected == "locked_comparison":
            return "locked_comparison"
        return None
    normalized_hint = str(mode_hint).strip().lower()
    if normalized_hint in {"confirmatory", "protocol"}:
        return "confirmatory"
    if normalized_hint in {"locked_comparison", "comparison"}:
        return "locked_comparison"
    return None


def _required_top_level_files(mode: FrameworkModeLiteral) -> list[str]:
    if mode == "confirmatory":
        return list(_REQUIRED_PROTOCOL_ARTIFACTS)
    return list(_REQUIRED_COMPARISON_ARTIFACTS)


def _load_report_index(path: Path, issues: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not path.exists():
        _add_issue(
            issues,
            code="report_index_missing",
            message="Missing report_index.csv.",
            path=path,
        )
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{str(k): str(v) for k, v in row.items()} for row in reader]


def _resolve_report_dir(root: Path, report_dir_raw: str) -> Path:
    report_dir = Path(report_dir_raw)
    if report_dir.is_absolute():
        return report_dir
    return root / report_dir


def _verify_metric_policy(
    *,
    config_payload: dict[str, Any],
    metrics_payload: dict[str, Any],
    issues: list[dict[str, Any]],
    report_dir: Path,
) -> None:
    config_policy = config_payload.get("metric_policy_effective")
    metrics_policy = metrics_payload.get("metric_policy_effective")
    if not isinstance(config_policy, dict) or not isinstance(metrics_policy, dict):
        _add_issue(
            issues,
            code="metric_policy_missing",
            message="Run config/metrics must include metric_policy_effective.",
            path=report_dir,
        )
        return
    evidence_policy_config = config_payload.get("evidence_policy_effective")
    evidence_policy_metrics = metrics_payload.get("evidence_policy_effective")
    if not isinstance(evidence_policy_config, dict) or not isinstance(evidence_policy_metrics, dict):
        _add_issue(
            issues,
            code="evidence_policy_missing",
            message="Run config/metrics must include evidence_policy_effective.",
            path=report_dir,
        )
    for key in (
        "primary_metric",
        "decision_metric",
        "tuning_metric",
        "permutation_metric",
        "higher_is_better",
    ):
        if key not in config_policy or key not in metrics_policy:
            _add_issue(
                issues,
                code="metric_policy_field_missing",
                message=f"metric_policy_effective missing field '{key}'.",
                path=report_dir,
            )
    permutation_payload = metrics_payload.get("permutation_test")
    if isinstance(permutation_payload, dict):
        metric_name = permutation_payload.get("metric_name")
        primary_metric = config_policy.get("primary_metric") if isinstance(config_policy, dict) else None
        if isinstance(metric_name, str) and isinstance(primary_metric, str):
            if metric_name.strip() != primary_metric.strip():
                _add_issue(
                    issues,
                    code="permutation_metric_drift",
                    message="Permutation metric does not match effective primary metric.",
                    path=report_dir,
                    details={"metric_name": metric_name, "primary_metric": primary_metric},
                ) 


def _verify_data_layer_artifacts(
    *,
    config_payload: dict[str, Any],
    metrics_payload: dict[str, Any],
    issues: list[dict[str, Any]],
    report_dir: Path,
) -> None:
    config_data_policy = config_payload.get("data_policy_effective")
    metrics_data_policy = metrics_payload.get("data_policy_effective")
    if not isinstance(config_data_policy, dict) or not isinstance(metrics_data_policy, dict):
        _add_issue(
            issues,
            code="data_policy_missing",
            message="Run config/metrics must include data_policy_effective.",
            path=report_dir,
        )
    config_data_artifacts = config_payload.get("data_artifacts")
    metrics_data_artifacts = metrics_payload.get("data_artifacts")
    if not isinstance(config_data_artifacts, dict) or not isinstance(metrics_data_artifacts, dict):
        _add_issue(
            issues,
            code="data_artifacts_metadata_missing",
            message="Run config/metrics must include data_artifacts metadata object.",
            path=report_dir,
        )

    for filename in _REQUIRED_DATA_RUN_ARTIFACTS:
        artifact_path = report_dir / filename
        if not artifact_path.exists():
            _add_issue(
                issues,
                code="data_run_artifact_missing",
                message=f"Missing required data-layer run artifact '{filename}'.",
                path=artifact_path,
            )

    dataset_card = _load_json(report_dir / "dataset_card.json", issues, code_prefix="dataset_card")
    if isinstance(dataset_card, dict):
        for key in (
            "framework_mode",
            "dataset_identity",
            "target_definition",
            "coverage",
            "external_validation",
        ):
            if key not in dataset_card:
                _add_issue(
                    issues,
                    code="dataset_card_field_missing",
                    message=f"dataset_card.json missing required field '{key}'.",
                    path=report_dir / "dataset_card.json",
                )

    quality_report = _load_json(
        report_dir / "data_quality_report.json",
        issues,
        code_prefix="data_quality_report",
    )
    if isinstance(quality_report, dict):
        for key in (
            "status",
            "n_blocking_issues",
            "n_warnings",
            "blocking_issues",
            "warnings",
            "leakage_audit_verdict",
        ):
            if key not in quality_report:
                _add_issue(
                    issues,
                    code="data_quality_field_missing",
                    message=f"data_quality_report.json missing required field '{key}'.",
                    path=report_dir / "data_quality_report.json",
                )

    leakage_report = _load_json(
        report_dir / "leakage_audit.json",
        issues,
        code_prefix="leakage_audit",
    )
    if isinstance(leakage_report, dict):
        if "verdict" not in leakage_report:
            _add_issue(
                issues,
                code="leakage_audit_verdict_missing",
                message="leakage_audit.json must include verdict.",
                path=report_dir / "leakage_audit.json",
            )
        if "checks" not in leakage_report:
            _add_issue(
                issues,
                code="leakage_audit_checks_missing",
                message="leakage_audit.json must include checks.",
                path=report_dir / "leakage_audit.json",
            )

    external_payload = _load_json(
        report_dir / "external_validation_compatibility.json",
        issues,
        code_prefix="external_validation_compatibility",
    )
    if isinstance(external_payload, dict):
        for key in ("enabled", "mode", "status", "datasets"):
            if key not in external_payload:
                _add_issue(
                    issues,
                    code="external_validation_field_missing",
                    message=f"external_validation_compatibility.json missing required field '{key}'.",
                    path=report_dir / "external_validation_compatibility.json",
                )


def _verify_confirmatory_reporting_contract(
    *,
    suite_summary: dict[str, Any] | None,
    execution_status: dict[str, Any],
    deviation_log: dict[str, Any] | None,
    source_contract: dict[str, Any] | None,
    issues: list[dict[str, Any]],
    root: Path,
) -> None:
    if not isinstance(suite_summary, dict):
        _add_issue(
            issues,
            code="suite_summary_missing",
            message="Missing or invalid suite_summary.json for confirmatory reporting contract.",
            path=root / "suite_summary.json",
        )
        return
    if not isinstance(deviation_log, dict):
        _add_issue(
            issues,
            code="deviation_log_missing",
            message="Missing or invalid deviation_log.json for confirmatory reporting contract.",
            path=root / "deviation_log.json",
        )
        return

    contract = suite_summary.get("confirmatory_reporting_contract")
    if not isinstance(contract, dict):
        _add_issue(
            issues,
            code="confirmatory_reporting_contract_missing",
            message="suite_summary.json must include confirmatory_reporting_contract.",
            path=root / "suite_summary.json",
        )
        return

    source_protocol_id = (
        str(source_contract.get("protocol_id"))
        if isinstance(source_contract, dict) and source_contract.get("protocol_id") is not None
        else ""
    )
    strict_confirmatory_freeze = source_protocol_id == "thesis_confirmatory_v1"

    for key in (
        "protocol_id",
        "protocol_version",
        "dataset_fingerprint",
        "target_mapping_version",
        "target_mapping_hash",
        "primary_split",
        "primary_metric",
        "model_family",
        "controls_status",
        "multiplicity_policy",
        "interpretation_limits",
        "subgroup_evidence_policy",
        "deviations_from_protocol",
    ):
        if key not in contract:
            _add_issue(
                issues,
                code="confirmatory_reporting_field_missing",
                message=f"confirmatory_reporting_contract missing required field '{key}'.",
                path=root / "suite_summary.json",
            )

    deviations_payload = contract.get("deviations_from_protocol")
    if not isinstance(deviations_payload, dict):
        _add_issue(
            issues,
            code="deviation_summary_missing",
            message="confirmatory_reporting_contract.deviations_from_protocol must be an object.",
            path=root / "suite_summary.json",
        )
    else:
        for key in (
            "n_total_deviations",
            "n_science_critical_deviations",
            "science_critical_deviation_detected",
            "controls_valid_for_confirmatory",
            "confirmatory_status",
            "explicit_no_deviation_record",
        ):
            if key not in deviations_payload:
                _add_issue(
                    issues,
                    code="deviation_summary_field_missing",
                    message=f"deviations_from_protocol missing required field '{key}'.",
                    path=root / "suite_summary.json",
                )

    subgroup_policy = contract.get("subgroup_evidence_policy")
    if not isinstance(subgroup_policy, dict):
        _add_issue(
            issues,
            code="subgroup_evidence_policy_missing",
            message="confirmatory_reporting_contract.subgroup_evidence_policy must be an object.",
            path=root / "suite_summary.json",
        )
    elif strict_confirmatory_freeze:
        if bool(subgroup_policy.get("primary_evidence_substitution_allowed", True)):
            _add_issue(
                issues,
                code="subgroup_primary_evidence_guardrail_missing",
                message=(
                    "Confirmatory subgroup evidence must not be eligible as primary-evidence "
                    "substitute."
                ),
                path=root / "suite_summary.json",
            )

    controls_status = contract.get("controls_status")
    if strict_confirmatory_freeze and isinstance(controls_status, dict):
        if not bool(controls_status.get("controls_valid_for_confirmatory", False)):
            _add_issue(
                issues,
                code="confirmatory_controls_invalid",
                message=(
                    "Confirmatory controls are required for run validity and are not fully "
                    "satisfied."
                ),
                path=root / "suite_summary.json",
                details={"controls_status": controls_status},
            )

    deviation_entries = deviation_log.get("deviations")
    if not isinstance(deviation_entries, list) or not deviation_entries:
        _add_issue(
            issues,
            code="deviation_log_entries_missing",
            message="deviation_log.json must contain a non-empty deviations list.",
            path=root / "deviation_log.json",
        )
    else:
        if all(str(entry.get("status")) != "no_deviation" for entry in deviation_entries):
            total_deviations = int(deviation_log.get("n_total_deviations", 0))
            if total_deviations == 0:
                _add_issue(
                    issues,
                    code="deviation_log_no_explicit_no_deviation",
                    message=(
                        "deviation_log.json must include an explicit no_deviation record "
                        "when no deviations occurred."
                    ),
                    path=root / "deviation_log.json",
                )

    status_in_log = str(deviation_log.get("confirmatory_status", ""))
    status_in_execution = str(execution_status.get("confirmatory_status", ""))
    if status_in_log and status_in_execution and status_in_log != status_in_execution:
        _add_issue(
            issues,
            code="confirmatory_status_drift",
            message="confirmatory_status differs between execution_status and deviation_log.",
            path=root / "execution_status.json",
            details={
                "execution_status_confirmatory_status": status_in_execution,
                "deviation_log_confirmatory_status": status_in_log,
            },
        )

    if strict_confirmatory_freeze:
        for key in (
            "target_mapping_version",
            "target_mapping_hash",
            "primary_split",
            "primary_metric",
            "model_family",
        ):
            value = contract.get(key)
            if not isinstance(value, str) or not value.strip():
                _add_issue(
                    issues,
                    code="confirmatory_freeze_reporting_field_invalid",
                    message=f"confirmatory_reporting_contract field '{key}' must be non-empty.",
                    path=root / "suite_summary.json",
                )
        multiplicity_policy = contract.get("multiplicity_policy")
        if not isinstance(multiplicity_policy, dict):
            _add_issue(
                issues,
                code="confirmatory_freeze_multiplicity_policy_missing",
                message="confirmatory_reporting_contract.multiplicity_policy must be an object.",
                path=root / "suite_summary.json",
            )
        else:
            for key in (
                "primary_hypotheses",
                "primary_alpha",
                "secondary_policy",
                "exploratory_claims_allowed",
            ):
                if key not in multiplicity_policy:
                    _add_issue(
                        issues,
                        code="confirmatory_freeze_multiplicity_field_missing",
                        message=f"multiplicity_policy missing required field '{key}'.",
                        path=root / "suite_summary.json",
                    )
        interpretation_limits = contract.get("interpretation_limits")
        if isinstance(interpretation_limits, dict):
            for key in (
                "no_causal_claims",
                "no_clinical_claims",
                "no_localization_claims_from_coefficients",
                "no_external_generalization_claim",
                "secondary_results_not_primary_evidence",
            ):
                if key not in interpretation_limits:
                    _add_issue(
                        issues,
                        code="confirmatory_freeze_interpretation_limit_missing",
                        message=f"interpretation_limits missing required field '{key}'.",
                        path=root / "suite_summary.json",
                    )
                elif interpretation_limits.get(key) is not True:
                    _add_issue(
                        issues,
                        code="confirmatory_freeze_interpretation_limit_invalid",
                        message=f"interpretation_limits field '{key}' must be true for strict confirmatory freeze.",
                        path=root / "suite_summary.json",
                    )
        if not isinstance(contract.get("controls_status"), dict):
            _add_issue(
                issues,
                code="confirmatory_freeze_controls_status_missing",
                message="confirmatory_reporting_contract.controls_status must be an object.",
                path=root / "suite_summary.json",
            )
        if not isinstance(contract.get("interpretation_limits"), dict):
            _add_issue(
                issues,
                code="confirmatory_freeze_interpretation_limits_missing",
                message="confirmatory_reporting_contract.interpretation_limits must be an object.",
                path=root / "suite_summary.json",
            )

def verify_official_artifacts(
    *,
    output_dir: Path | str,
    mode: str | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    issues: list[dict[str, Any]] = []

    if not root.exists() or not root.is_dir():
        _add_issue(
            issues,
            code="output_dir_missing",
            message="Official artifact directory does not exist or is not a directory.",
            path=root,
        )
        return {
            "passed": False,
            "output_dir": str(root),
            "framework_mode": None,
            "issues": issues,
            "n_completed_runs_checked": 0,
        }

    execution_status_path = root / "execution_status.json"
    execution_status = _load_json(execution_status_path, issues, code_prefix="execution_status")
    if execution_status is None:
        return {
            "passed": False,
            "output_dir": str(root),
            "framework_mode": None,
            "issues": issues,
            "n_completed_runs_checked": 0,
        }

    framework_mode = _expected_mode(mode, execution_status)
    if framework_mode is None:
        _add_issue(
            issues,
            code="framework_mode_invalid",
            message=(
                "Could not resolve framework mode from execution_status.json; "
                "expected confirmatory or locked_comparison."
            ),
            path=execution_status_path,
            details={"detected": execution_status.get("framework_mode"), "mode_hint": mode},
        )
        return {
            "passed": False,
            "output_dir": str(root),
            "framework_mode": None,
            "issues": issues,
            "n_completed_runs_checked": 0,
        }

    expected_top_level = _required_top_level_files(framework_mode)
    for filename in expected_top_level:
        artifact_path = root / filename
        if not artifact_path.exists():
            _add_issue(
                issues,
                code="top_level_artifact_missing",
                message=f"Missing required top-level artifact '{filename}'.",
                path=artifact_path,
            )

    if framework_mode == "confirmatory":
        manifest_path = root / "compiled_protocol_manifest.json"
        source_path = root / "protocol.json"
    else:
        manifest_path = root / "compiled_comparison_manifest.json"
        source_path = root / "comparison.json"

    compiled_manifest = _load_json(manifest_path, issues, code_prefix="compiled_manifest")
    source_contract = _load_json(source_path, issues, code_prefix="source_contract")
    suite_summary_payload: dict[str, Any] | None = None
    deviation_log_payload: dict[str, Any] | None = None
    if framework_mode == "confirmatory":
        suite_summary_payload = _load_json(
            root / "suite_summary.json",
            issues,
            code_prefix="suite_summary",
        )
        deviation_log_payload = _load_json(
            root / "deviation_log.json",
            issues,
            code_prefix="deviation_log",
        )

    required_run_artifacts: list[str] = ["config.json", "metrics.json"]
    required_run_metadata_fields: list[str] = ["framework_mode", "canonical_run"]

    if isinstance(compiled_manifest, dict):
        raw_artifacts = compiled_manifest.get("required_run_artifacts")
        if isinstance(raw_artifacts, list) and raw_artifacts:
            required_run_artifacts = [str(value) for value in raw_artifacts]
        raw_metadata = compiled_manifest.get("required_run_metadata_fields")
        if isinstance(raw_metadata, list) and raw_metadata:
            required_run_metadata_fields = [str(value) for value in raw_metadata]

    if isinstance(source_contract, dict):
        artifact_contract = source_contract.get("artifact_contract")
        if isinstance(artifact_contract, dict):
            raw_metadata = artifact_contract.get("required_run_metadata_fields")
            if isinstance(raw_metadata, list) and raw_metadata:
                required_run_metadata_fields = [str(value) for value in raw_metadata]

    if framework_mode == "confirmatory":
        _verify_confirmatory_reporting_contract(
            suite_summary=suite_summary_payload,
            execution_status=execution_status,
            deviation_log=deviation_log_payload,
            source_contract=source_contract,
            issues=issues,
            root=root,
        )
    repeated_summary_payload = _load_json(
        root / "repeated_run_summary.json",
        issues,
        code_prefix="repeated_run_summary",
    )
    confidence_payload = _load_json(
        root / "confidence_intervals.json",
        issues,
        code_prefix="confidence_intervals",
    )
    if isinstance(repeated_summary_payload, dict) and "groups" not in repeated_summary_payload:
        _add_issue(
            issues,
            code="repeated_run_summary_invalid",
            message="repeated_run_summary.json must include 'groups'.",
            path=root / "repeated_run_summary.json",
        )
    if isinstance(confidence_payload, dict) and "intervals" not in confidence_payload:
        _add_issue(
            issues,
            code="confidence_intervals_invalid",
            message="confidence_intervals.json must include 'intervals'.",
            path=root / "confidence_intervals.json",
        )
    if framework_mode == "locked_comparison":
        paired_payload = _load_json(
            root / "paired_model_comparisons.json",
            issues,
            code_prefix="paired_model_comparisons",
        )
        if isinstance(paired_payload, dict) and not isinstance(paired_payload.get("pairs"), list):
            _add_issue(
                issues,
                code="paired_model_comparisons_invalid",
                message="paired_model_comparisons.json must include list field 'pairs'.",
                path=root / "paired_model_comparisons.json",
            )

    report_rows = _load_report_index(root / "report_index.csv", issues)
    n_completed_runs_checked = 0
    expected_canonical = framework_mode == "confirmatory"

    for row in report_rows:
        status = str(row.get("status", "")).strip().lower()
        if status != "completed":
            continue
        n_completed_runs_checked += 1
        report_dir_raw = str(row.get("report_dir", "")).strip()
        if not report_dir_raw:
            _add_issue(
                issues,
                code="report_dir_missing",
                message="Completed run row is missing report_dir.",
                details={"run_id": row.get("run_id")},
            )
            continue
        report_dir = _resolve_report_dir(root, report_dir_raw)
        if not report_dir.exists() or not report_dir.is_dir():
            _add_issue(
                issues,
                code="report_dir_invalid",
                message="Completed run report_dir does not exist.",
                path=report_dir,
                details={"run_id": row.get("run_id")},
            )
            continue

        for filename in required_run_artifacts:
            run_artifact_path = report_dir / filename
            if not run_artifact_path.exists():
                _add_issue(
                    issues,
                    code="run_artifact_missing",
                    message=f"Missing required run artifact '{filename}'.",
                    path=run_artifact_path,
                    details={"run_id": row.get("run_id")},
                )

        config_payload = _load_json(report_dir / "config.json", issues, code_prefix="run_config")
        metrics_payload = _load_json(report_dir / "metrics.json", issues, code_prefix="run_metrics")
        if config_payload is None or metrics_payload is None:
            continue

        for key in required_run_metadata_fields:
            if key not in config_payload or key not in metrics_payload:
                _add_issue(
                    issues,
                    code="run_metadata_missing",
                    message=f"Missing required run metadata key '{key}' in config/metrics.",
                    path=report_dir,
                    details={"run_id": row.get("run_id")},
                )

        config_mode = str(config_payload.get("framework_mode", "")).strip()
        metrics_mode = str(metrics_payload.get("framework_mode", "")).strip()
        if config_mode != framework_mode or metrics_mode != framework_mode:
            _add_issue(
                issues,
                code="framework_mode_drift",
                message="Run artifact framework_mode does not match mode-level framework_mode.",
                path=report_dir,
                details={
                    "expected": framework_mode,
                    "config_mode": config_mode,
                    "metrics_mode": metrics_mode,
                },
            )

        config_canonical = bool(config_payload.get("canonical_run"))
        metrics_canonical = bool(metrics_payload.get("canonical_run"))
        if config_canonical != expected_canonical or metrics_canonical != expected_canonical:
            _add_issue(
                issues,
                code="canonical_flag_drift",
                message="Run artifact canonical_run flag does not match framework mode expectation.",
                path=report_dir,
                details={
                    "expected": expected_canonical,
                    "config_canonical": config_canonical,
                    "metrics_canonical": metrics_canonical,
                },
            )

        if framework_mode == "confirmatory":
            if not isinstance(config_payload.get("dataset_fingerprint"), dict):
                _add_issue(
                    issues,
                    code="confirmatory_dataset_fingerprint_missing_config",
                    message="Confirmatory run config.json must include dataset_fingerprint.",
                    path=report_dir / "config.json",
                    details={"run_id": row.get("run_id")},
                )
            if not isinstance(metrics_payload.get("dataset_fingerprint"), dict):
                _add_issue(
                    issues,
                    code="confirmatory_dataset_fingerprint_missing_metrics",
                    message="Confirmatory run metrics.json must include dataset_fingerprint.",
                    path=report_dir / "metrics.json",
                    details={"run_id": row.get("run_id")},
                )

        _verify_metric_policy(
            config_payload=config_payload,
            metrics_payload=metrics_payload,
            issues=issues,
            report_dir=report_dir,
        )
        _verify_data_layer_artifacts(
            config_payload=config_payload,
            metrics_payload=metrics_payload,
            issues=issues,
            report_dir=report_dir,
        )

    return {
        "passed": not issues,
        "output_dir": str(root.resolve()),
        "framework_mode": framework_mode,
        "issues": issues,
        "n_completed_runs_checked": int(n_completed_runs_checked),
    }


__all__ = ["verify_official_artifacts"]
