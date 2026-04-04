from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Thesis_ML.config.paths import (
    DEFAULT_DECISION_SUPPORT_THESIS_RUNTIME_REGISTRY,
    PROJECT_ROOT,
)
from Thesis_ML.experiments.run_states import is_run_success_status
from Thesis_ML.verification.confirmatory_scope_runtime_alignment import (
    verify_confirmatory_scope_runtime_alignment,
)
from Thesis_ML.verification.official_artifacts import verify_official_artifacts


def _load_json(
    path: Path, *, issues: list[dict[str, Any]], code_prefix: str
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(
            {
                "code": f"{code_prefix}_invalid_json",
                "message": f"Invalid JSON in '{path.name}'.",
                "details": {"error": str(exc), "path": str(path.resolve())},
            }
        )
        return None
    if not isinstance(payload, dict):
        issues.append(
            {
                "code": f"{code_prefix}_invalid_shape",
                "message": f"Expected JSON object in '{path.name}'.",
                "details": {"path": str(path.resolve())},
            }
        )
        return None
    return payload


def _criterion(name: str, *, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": str(name),
        "passed": bool(passed),
    }
    if details:
        payload["details"] = details
    return payload


def verify_confirmatory_ready(
    *,
    output_dir: Path | str,
    reproducibility_summary: Path | str | None = None,
    scope_config_path: Path | str | None = None,
    runtime_registry_path: Path | str | None = None,
    scope_exceptions_path: Path | str | None = None,
    require_control_coverage: bool = False,
) -> dict[str, Any]:
    root = Path(output_dir)
    criteria: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    resolved_scope_path = (
        Path(scope_config_path)
        if scope_config_path is not None
        else (PROJECT_ROOT / "configs" / "confirmatory" / "confirmatory_scope_v1.json")
    )
    resolved_runtime_registry_path = (
        Path(runtime_registry_path)
        if runtime_registry_path is not None
        else Path(DEFAULT_DECISION_SUPPORT_THESIS_RUNTIME_REGISTRY)
    )
    resolved_scope_exceptions_path = (
        Path(scope_exceptions_path) if scope_exceptions_path is not None else None
    )

    try:
        scope_runtime_summary = verify_confirmatory_scope_runtime_alignment(
            scope_config_path=resolved_scope_path,
            runtime_registry_path=resolved_runtime_registry_path,
            exceptions_config_path=resolved_scope_exceptions_path,
        )
    except Exception as exc:
        scope_runtime_summary = {
            "passed": False,
            "issues": [
                {
                    "code": "scope_runtime_alignment_exception",
                    "message": str(exc),
                }
            ],
        }

    scope_runtime_passed = bool(scope_runtime_summary.get("passed", False))
    criteria.append(
        _criterion(
            "confirmatory_scope_runtime_alignment",
            passed=scope_runtime_passed,
            details={
                "scope_config_path": str(resolved_scope_path.resolve()),
                "runtime_registry_path": str(resolved_runtime_registry_path.resolve()),
            },
        )
    )
    if not scope_runtime_passed:
        issues.append(
            {
                "code": "confirmatory_scope_runtime_mismatch",
                "message": "Scientific confirmatory scope and thesis runtime registry are not aligned.",
                "details": {"issues": scope_runtime_summary.get("issues", [])},
            }
        )

    official_summary = verify_official_artifacts(output_dir=root, mode="confirmatory")
    official_passed = bool(official_summary.get("passed", False))
    criteria.append(
        _criterion(
            "official_artifact_verification",
            passed=official_passed,
            details={"framework_mode": official_summary.get("framework_mode")},
        )
    )
    if not official_passed:
        issues.append(
            {
                "code": "official_artifact_verification_failed",
                "message": "Official artifact verification failed for confirmatory output.",
                "details": {"issues": official_summary.get("issues", [])},
            }
        )

    execution_status_path = root / "execution_status.json"
    suite_summary_path = root / "suite_summary.json"
    deviation_log_path = root / "deviation_log.json"

    missing_files = [
        str(path.name)
        for path in (execution_status_path, suite_summary_path, deviation_log_path)
        if not path.exists()
    ]
    if missing_files:
        issue = {
            "code": "confirmatory_ready_files_missing",
            "message": "Missing required confirmatory status files.",
            "details": {"missing_files": missing_files},
        }
        issues.append(issue)
        criteria.append(
            _criterion(
                "confirmatory_reporting_files_present",
                passed=False,
                details={"missing_files": missing_files},
            )
        )
        return {
            "passed": False,
            "output_dir": str(root.resolve()),
            "criteria": criteria,
            "issues": issues,
            "official_artifact_summary": official_summary,
            "reproducibility_summary_path": (
                str(Path(reproducibility_summary).resolve()) if reproducibility_summary else None
            ),
        }

    execution_status = _load_json(
        execution_status_path,
        issues=issues,
        code_prefix="execution_status",
    )
    suite_summary = _load_json(
        suite_summary_path,
        issues=issues,
        code_prefix="suite_summary",
    )
    deviation_log = _load_json(
        deviation_log_path,
        issues=issues,
        code_prefix="deviation_log",
    )
    if (
        not isinstance(execution_status, dict)
        or not isinstance(suite_summary, dict)
        or not isinstance(deviation_log, dict)
    ):
        criteria.append(
            _criterion(
                "confirmatory_reporting_files_parseable",
                passed=False,
            )
        )
        return {
            "passed": False,
            "output_dir": str(root.resolve()),
            "criteria": criteria,
            "issues": issues,
            "official_artifact_summary": official_summary,
            "reproducibility_summary_path": (
                str(Path(reproducibility_summary).resolve()) if reproducibility_summary else None
            ),
        }

    confirmatory_status = str(execution_status.get("confirmatory_status", "")).strip()
    status_ok = confirmatory_status == "confirmatory"
    criteria.append(
        _criterion(
            "confirmatory_status_not_downgraded",
            passed=status_ok,
            details={"confirmatory_status": confirmatory_status or None},
        )
    )
    if not status_ok:
        issues.append(
            {
                "code": "confirmatory_status_invalid",
                "message": "Confirmatory output is not in confirmatory status.",
                "details": {"confirmatory_status": confirmatory_status},
            }
        )

    science_critical = bool(execution_status.get("science_critical_deviation_detected", False))
    criteria.append(
        _criterion(
            "no_science_critical_deviation",
            passed=not science_critical,
            details={"science_critical_deviation_detected": science_critical},
        )
    )
    if science_critical:
        issues.append(
            {
                "code": "science_critical_deviation_detected",
                "message": "Science-critical deviation detected in execution_status.",
            }
        )

    deviation_science_critical = bool(
        deviation_log.get("science_critical_deviation_detected", False)
    )
    criteria.append(
        _criterion(
            "deviation_log_science_critical_clear",
            passed=not deviation_science_critical,
            details={
                "science_critical_deviation_detected": deviation_science_critical,
            },
        )
    )
    if deviation_science_critical:
        issues.append(
            {
                "code": "deviation_log_science_critical_deviation_detected",
                "message": "Science-critical deviation detected in deviation_log.",
            }
        )

    reporting_contract = suite_summary.get("confirmatory_reporting_contract")
    controls_valid = False
    dataset_fingerprint_present = False
    if isinstance(reporting_contract, dict):
        controls_status = reporting_contract.get("controls_status")
        if isinstance(controls_status, dict):
            controls_valid = bool(controls_status.get("controls_valid_for_confirmatory", False))
        dataset_fingerprint_payload = reporting_contract.get("dataset_fingerprint")
        if isinstance(dataset_fingerprint_payload, dict):
            n_with_fingerprint = int(dataset_fingerprint_payload.get("n_with_fingerprint", 0))
            n_missing_fingerprint = int(dataset_fingerprint_payload.get("n_missing_fingerprint", 0))
            dataset_fingerprint_present = n_with_fingerprint > 0 and n_missing_fingerprint == 0
    criteria.append(
        _criterion(
            "controls_valid_for_confirmatory",
            passed=controls_valid,
            details={"controls_valid_for_confirmatory": controls_valid},
        )
    )
    if not controls_valid:
        issues.append(
            {
                "code": "controls_not_valid_for_confirmatory",
                "message": "Confirmatory controls are not valid per reporting contract.",
            }
        )

    required_evidence_status = suite_summary.get("required_evidence_status")
    evidence_valid = bool(
        isinstance(required_evidence_status, dict) and required_evidence_status.get("valid", False)
    )
    criteria.append(
        _criterion(
            "required_evidence_status_valid",
            passed=evidence_valid,
            details={"required_evidence_status": required_evidence_status},
        )
    )
    if not evidence_valid:
        issues.append(
            {
                "code": "required_evidence_status_invalid",
                "message": "required_evidence_status.valid is false.",
            }
        )

    criteria.append(
        _criterion(
            "dataset_fingerprint_present",
            passed=dataset_fingerprint_present,
            details={"dataset_fingerprint_present": dataset_fingerprint_present},
        )
    )
    if not dataset_fingerprint_present:
        issues.append(
            {
                "code": "dataset_fingerprint_missing_or_incomplete",
                "message": (
                    "Confirmatory reporting contract dataset_fingerprint is missing or "
                    "contains missing fingerprints."
                ),
            }
        )

    runs = execution_status.get("runs", [])
    all_runs_completed = isinstance(runs, list) and all(
        isinstance(entry, dict) and is_run_success_status(str(entry.get("status", "")).strip())
        for entry in runs
    )
    criteria.append(
        _criterion(
            "all_runs_completed",
            passed=all_runs_completed,
            details={"n_runs": len(runs) if isinstance(runs, list) else 0},
        )
    )
    if not all_runs_completed:
        issues.append(
            {
                "code": "confirmatory_runs_incomplete",
                "message": "Not all confirmatory runs are completed.",
            }
        )

    if reproducibility_summary is not None:
        reproducibility_path = Path(reproducibility_summary)
        if not reproducibility_path.exists():
            criteria.append(
                _criterion(
                    "reproducibility_summary_present",
                    passed=False,
                    details={"path": str(reproducibility_path.resolve())},
                )
            )
            issues.append(
                {
                    "code": "reproducibility_summary_missing",
                    "message": "Provided reproducibility summary path does not exist.",
                    "details": {"path": str(reproducibility_path.resolve())},
                }
            )
        else:
            reproducibility_payload = _load_json(
                reproducibility_path,
                issues=issues,
                code_prefix="reproducibility_summary",
            )
            if not isinstance(reproducibility_payload, dict):
                criteria.append(
                    _criterion(
                        "reproducibility_summary_parseable",
                        passed=False,
                        details={"path": str(reproducibility_path.resolve())},
                    )
                )
                passed = bool(all(entry.get("passed", False) for entry in criteria))
                return {
                    "passed": passed,
                    "output_dir": str(root.resolve()),
                    "criteria": criteria,
                    "issues": issues,
                    "official_artifact_summary": official_summary,
                    "reproducibility_summary_path": str(reproducibility_path.resolve()),
                }
            reproducibility_passed = bool(reproducibility_payload.get("passed", False))
            criteria.append(
                _criterion(
                    "reproducibility_summary_passed",
                    passed=reproducibility_passed,
                    details={"path": str(reproducibility_path.resolve())},
                )
            )
            if not reproducibility_passed:
                issues.append(
                    {
                        "code": "reproducibility_summary_failed",
                        "message": "Reproducibility summary reports passed=false.",
                        "details": {"path": str(reproducibility_path.resolve())},
                    }
                )

    control_coverage_path = root / "special_aggregations" / "confirmatory"
    control_coverage_json = control_coverage_path / "confirmatory_anchor_control_coverage.json"
    if require_control_coverage:
        control_coverage_present = (
            control_coverage_json.exists() and control_coverage_json.is_file()
        )
        criteria.append(
            _criterion(
                "confirmatory_control_coverage_artifact_present",
                passed=control_coverage_present,
                details={"path": str(control_coverage_json.resolve())},
            )
        )
        if not control_coverage_present:
            issues.append(
                {
                    "code": "confirmatory_control_coverage_artifact_missing",
                    "message": "Required confirmatory control coverage artifact is missing.",
                    "details": {"path": str(control_coverage_json.resolve())},
                }
            )

    passed = bool(all(entry.get("passed", False) for entry in criteria))
    return {
        "passed": passed,
        "output_dir": str(root.resolve()),
        "criteria": criteria,
        "issues": issues,
        "official_artifact_summary": official_summary,
        "scope_runtime_alignment": scope_runtime_summary,
        "reproducibility_summary_path": (
            str(Path(reproducibility_summary).resolve()) if reproducibility_summary else None
        ),
    }


__all__ = ["verify_confirmatory_ready"]
