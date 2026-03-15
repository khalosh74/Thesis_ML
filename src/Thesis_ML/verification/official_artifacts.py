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
    "report_index.csv",
)
_REQUIRED_COMPARISON_ARTIFACTS = (
    "comparison.json",
    "compiled_comparison_manifest.json",
    "comparison_summary.json",
    "comparison_decision.json",
    "execution_status.json",
    "report_index.csv",
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
        if detected in {"confirmatory", "locked_comparison"}:
            return detected
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
        report_dir = Path(report_dir_raw)
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

        _verify_metric_policy(
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
