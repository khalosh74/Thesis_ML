from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from Thesis_ML.experiments.stage_execution import StageExecutionResult


_ARTIFACT_EXPECTED_STAGES = {
    "feature_cache_build",
    "feature_matrix_load",
    "evaluation",
    "reporting",
}


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _issue(
    *,
    code: str,
    stage_key: str,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": str(code),
        "stage_key": str(stage_key),
        "severity": str(severity),
        "message": str(message),
    }
    if isinstance(details, dict) and details:
        payload["details"] = dict(details)
    return payload


def _normalize_stage_execution(
    value: StageExecutionResult | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, StageExecutionResult):
        return value.model_dump(mode="json")
    try:
        return StageExecutionResult.model_validate(dict(value)).model_dump(mode="json")
    except Exception:
        return None


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _looks_gpu_execution(
    *,
    observed_backend_family: Any,
    observed_compute_lane: Any,
    observed_executor_id: Any,
) -> bool:
    backend_value = str(observed_backend_family or "").strip().lower()
    lane_value = str(observed_compute_lane or "").strip().lower()
    executor_value = str(observed_executor_id or "").strip().lower()
    if lane_value == "gpu":
        return True
    if backend_value in {"torch_gpu", "xgboost_gpu"}:
        return True
    if executor_value and ("gpu" in executor_value or executor_value.startswith("torch_")):
        return True
    return False


def verify_stage_execution_evidence(
    *,
    stage_execution: StageExecutionResult | Mapping[str, Any] | None,
    observed_stage_evidence: Mapping[str, Any] | None = None,
    run_status: Mapping[str, Any] | None = None,
    process_profile_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stage_execution_payload = _normalize_stage_execution(stage_execution)
    findings: list[dict[str, Any]] = []

    if stage_execution_payload is None:
        findings.append(
            _issue(
                code="missing_stage_execution",
                stage_key="run",
                severity="error",
                message="stage_execution payload is missing or invalid.",
            )
        )
        return {
            "schema_version": "stage-execution-verifier-v1",
            "passed": False,
            "findings": findings,
            "finding_counts": {"error": 1},
        }

    telemetry_rows = stage_execution_payload.get("telemetry")
    if not isinstance(telemetry_rows, list):
        findings.append(
            _issue(
                code="missing_stage_telemetry",
                stage_key="run",
                severity="error",
                message="stage_execution telemetry is missing.",
            )
        )
        return {
            "schema_version": "stage-execution-verifier-v1",
            "passed": False,
            "findings": findings,
            "finding_counts": {"error": 1},
        }

    observed_rows: dict[str, dict[str, Any]] = {}
    if isinstance(observed_stage_evidence, Mapping):
        rows_payload = observed_stage_evidence.get("stages")
        if isinstance(rows_payload, list):
            for row in rows_payload:
                if not isinstance(row, Mapping):
                    continue
                key = row.get("stage_key")
                if isinstance(key, str) and key.strip():
                    observed_rows[str(key)] = {str(k): v for k, v in dict(row).items()}

    for row in telemetry_rows:
        if not isinstance(row, Mapping):
            continue
        stage_key = str(row.get("stage") or "unknown")
        status = str(row.get("status") or "unknown")
        backend_match = row.get("backend_match")
        lane_match = row.get("lane_match")
        executor_match = row.get("executor_match")
        planning_match = row.get("planning_match")
        fallback_used = bool(row.get("fallback_used", False))
        fallback_expected = bool(row.get("fallback_expected", False))
        observed_present = bool(row.get("observed_evidence_present", False))
        missing_observed = bool(row.get("missing_observed_evidence", False))
        lease_required = bool(row.get("lease_required", False))
        lease_acquired_raw = row.get("lease_acquired")
        lease_acquired = (
            bool(lease_acquired_raw)
            if isinstance(lease_acquired_raw, (bool, int))
            else False
        )
        lease_released_raw = row.get("lease_released")
        lease_released = (
            bool(lease_released_raw)
            if isinstance(lease_released_raw, (bool, int))
            else False
        )
        lease_acquired_at_utc = row.get("lease_acquired_at_utc")
        lease_released_at_utc = row.get("lease_released_at_utc")
        primary_artifacts = row.get("primary_artifacts")
        derived_from_stage = row.get("derived_from_stage")
        evidence_quality = str(row.get("evidence_quality") or "")
        resource_coverage = str(row.get("resource_coverage") or "")
        observed_backend_family = row.get("observed_backend_family")
        observed_compute_lane = row.get("observed_compute_lane")
        observed_executor_id = row.get("observed_executor_id")

        if lease_required and not lease_acquired:
            findings.append(
                _issue(
                    code="required_gpu_lease_missing",
                    stage_key=stage_key,
                    severity="error",
                    message="Stage requires GPU lease evidence but no lease acquisition was recorded.",
                )
            )

        observed_gpu_execution = _looks_gpu_execution(
            observed_backend_family=observed_backend_family,
            observed_compute_lane=observed_compute_lane,
            observed_executor_id=observed_executor_id,
        )
        if observed_gpu_execution and not lease_acquired:
            findings.append(
                _issue(
                    code="observed_gpu_without_lease",
                    stage_key=stage_key,
                    severity="error",
                    message="Observed GPU execution has no lease acquisition evidence.",
                )
            )

        if lease_acquired:
            if not isinstance(lease_acquired_at_utc, str) or not lease_acquired_at_utc.strip():
                findings.append(
                    _issue(
                        code="lease_missing_acquired_timestamp",
                        stage_key=stage_key,
                        severity="warning",
                        message="Lease acquisition is recorded but lease_acquired_at_utc is missing.",
                    )
                )
            if not isinstance(lease_released_at_utc, str) or not lease_released_at_utc.strip():
                findings.append(
                    _issue(
                        code="lease_missing_released_timestamp",
                        stage_key=stage_key,
                        severity="warning",
                        message="Lease acquisition is recorded but lease_released_at_utc is missing.",
                    )
                )

        stage_started = _parse_utc(row.get("started_at_utc"))
        stage_ended = _parse_utc(row.get("ended_at_utc"))
        lease_started = _parse_utc(lease_acquired_at_utc)
        lease_ended = _parse_utc(lease_released_at_utc)
        if (
            lease_acquired
            and stage_started is not None
            and stage_ended is not None
            and lease_started is not None
            and lease_ended is not None
        ):
            interval_tolerance = timedelta(seconds=1)
            if (
                lease_started < (stage_started - interval_tolerance)
                or lease_ended > (stage_ended + interval_tolerance)
            ):
                findings.append(
                    _issue(
                        code="lease_outside_stage_interval",
                        stage_key=stage_key,
                        severity="warning",
                        message="Lease interval falls outside the observed stage interval.",
                    )
                )

        if str(status) in {"failed", "missing"} and lease_acquired and not lease_released:
            findings.append(
                _issue(
                    code="lease_leak_on_failure_path",
                    stage_key=stage_key,
                    severity="error",
                    message="Stage failed/missing with lease acquisition but no release evidence.",
                )
            )

        if backend_match is False:
            findings.append(
                _issue(
                    code="backend_mismatch",
                    stage_key=stage_key,
                    severity="warning",
                    message="Observed backend family does not match planned backend family.",
                )
            )
        if lane_match is False:
            findings.append(
                _issue(
                    code="lane_mismatch",
                    stage_key=stage_key,
                    severity="warning",
                    message="Observed compute lane does not match planned compute lane.",
                )
            )
        if executor_match is False:
            findings.append(
                _issue(
                    code="executor_mismatch",
                    stage_key=stage_key,
                    severity="warning",
                    message="Observed executor does not match planned executor.",
                )
            )
        if planning_match is False and backend_match is None and lane_match is None and executor_match is None:
            findings.append(
                _issue(
                    code="planning_mismatch",
                    stage_key=stage_key,
                    severity="warning",
                    message="Planning match is false but detailed match fields are unavailable.",
                )
            )
        if fallback_used and not fallback_expected:
            findings.append(
                _issue(
                    code="unexpected_fallback",
                    stage_key=stage_key,
                    severity="warning",
                    message="Observed fallback was used although it was not expected in planning.",
                    details={"fallback_reason": row.get("fallback_reason")},
                )
            )
        if status in {"executed", "reused", "missing"} and (missing_observed or not observed_present):
            findings.append(
                _issue(
                    code="missing_evidence",
                    stage_key=stage_key,
                    severity="warning",
                    message="Stage is marked executed/reused but observed evidence is incomplete.",
                )
            )

        has_primary_artifacts = isinstance(primary_artifacts, list) and len(primary_artifacts) > 0
        if status == "executed" and stage_key in _ARTIFACT_EXPECTED_STAGES and not has_primary_artifacts:
            findings.append(
                _issue(
                    code="suspicious_missing_artifacts",
                    stage_key=stage_key,
                    severity="warning",
                    message="Stage executed but no primary artifacts were recorded.",
                )
            )
        if status == "reused" and derived_from_stage in (None, "") and not has_primary_artifacts:
            findings.append(
                _issue(
                    code="suspicious_reused_without_lineage",
                    stage_key=stage_key,
                    severity="warning",
                    message="Stage reused without lineage or artifact references.",
                )
            )
        if evidence_quality.lower() == "low" or (
            resource_coverage.lower() == "none" and status in {"executed", "reused", "missing"}
        ):
            findings.append(
                _issue(
                    code="low_confidence_evidence",
                    stage_key=stage_key,
                    severity="warning",
                    message="Stage evidence confidence is low due to sparse or missing coverage.",
                    details={
                        "evidence_quality": evidence_quality,
                        "resource_coverage": resource_coverage,
                    },
                )
            )

        if stage_key in observed_rows and status == "not_planned":
            findings.append(
                _issue(
                    code="unexpected_observed_for_not_planned",
                    stage_key=stage_key,
                    severity="warning",
                    message="Observed stage evidence exists for a stage marked not_planned.",
                )
            )

    if isinstance(run_status, Mapping):
        run_status_state = str(run_status.get("status") or "")
        if run_status_state.lower() in {"completed", "success"}:
            missing_rows = [
                row
                for row in telemetry_rows
                if isinstance(row, Mapping)
                and str(row.get("status")) in {"missing", "executed", "reused"}
                and not bool(row.get("observed_evidence_present", False))
            ]
            if missing_rows:
                findings.append(
                    _issue(
                        code="run_completed_with_missing_stage_evidence",
                        stage_key="run",
                        severity="warning",
                        message="Run completed but one or more stages have missing observed evidence.",
                        details={"missing_stage_count": int(len(missing_rows))},
                    )
                )

    if isinstance(process_profile_summary, Mapping):
        sample_count = process_profile_summary.get("sample_count")
        if isinstance(sample_count, int) and sample_count <= 0:
            findings.append(
                _issue(
                    code="no_process_samples",
                    stage_key="run",
                    severity="warning",
                    message="Process profile has no samples; stage resource evidence is low confidence.",
                )
            )

    severity_counts = Counter(str(item.get("severity") or "warning") for item in findings)
    return {
        "schema_version": "stage-execution-verifier-v1",
        "passed": len(findings) == 0,
        "findings": findings,
        "finding_counts": {str(key): int(value) for key, value in sorted(severity_counts.items())},
        "stage_count": int(len(telemetry_rows)),
    }


def verify_stage_execution_from_report_dir(report_dir: Path) -> dict[str, Any]:
    report_root = Path(report_dir)
    config_payload = _safe_load_json(report_root / "config.json") or {}
    run_status_payload = _safe_load_json(report_root / "run_status.json") or {}
    observed_payload = _safe_load_json(report_root / "stage_observed_evidence.json") or {}
    process_summary_payload = _safe_load_json(report_root / "process_profile_summary.json") or {}

    return verify_stage_execution_evidence(
        stage_execution=config_payload.get("stage_execution"),
        observed_stage_evidence=observed_payload,
        run_status=run_status_payload,
        process_profile_summary=process_summary_payload,
    )


__all__ = [
    "verify_stage_execution_evidence",
    "verify_stage_execution_from_report_dir",
]
