from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Thesis_ML.config.methodology import EvidenceRunRole
from Thesis_ML.experiments.run_states import is_run_success_status, normalize_run_status
from Thesis_ML.release.runtime_protocol.models import (
    ClaimCategory,
    ClaimRole,
    ClaimSpec,
    CompiledProtocolManifest,
    ProtocolRunResult,
    ThesisProtocol,
)

_PRIMARY_CONTROL_METRIC_TOLERANCE = 1e-9


def _load_json(path_text: str | None) -> dict[str, Any] | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _result_by_run_id(run_results: list[ProtocolRunResult]) -> dict[str, ProtocolRunResult]:
    return {result.run_id: result for result in run_results}


def _manifest_run_specs_by_id(compiled_manifest: CompiledProtocolManifest) -> dict[str, Any]:
    return {
        str(spec.run_id): spec
        for spec in getattr(compiled_manifest, "runs", [])
        if hasattr(spec, "run_id")
    }


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalized_evidence_run_role(value: Any) -> str:
    if isinstance(value, EvidenceRunRole):
        return str(value.value)
    normalized = getattr(value, "value", value)
    return str(normalized)


def _normalized_methodology_policy_name(value: Any) -> str:
    normalized = getattr(value, "value", value)
    return str(normalized) if normalized is not None else ""


def _is_dummy_run_spec(spec: Any) -> bool:
    controls = getattr(spec, "controls", None)
    controls_dummy = bool(getattr(controls, "dummy_baseline_run", False))
    return bool(controls_dummy or str(getattr(spec, "model", "")) == "dummy")


def _is_primary_evidence_run_spec(spec: Any) -> bool:
    evidence_role = _normalized_evidence_run_role(getattr(spec, "evidence_run_role", ""))
    if evidence_role != EvidenceRunRole.PRIMARY.value:
        return False
    if _is_dummy_run_spec(spec):
        return False
    return True


def _run_match_key(spec: Any, *, include_model: bool) -> tuple[str, str, str, str, int, int, str]:
    model_name = str(getattr(spec, "model", "")) if include_model else ""
    return (
        str(getattr(spec, "cv_mode", "")),
        str(getattr(spec, "subject", "") or ""),
        str(getattr(spec, "train_subject", "") or ""),
        str(getattr(spec, "test_subject", "") or ""),
        int(getattr(spec, "repeat_id", 1)),
        int(getattr(spec, "repeat_count", 1)),
        model_name,
    )


def _resolve_supporting_control_claims(
    *,
    protocol: ThesisProtocol,
    primary_claim: ClaimSpec,
) -> list[ClaimSpec]:
    return [
        claim
        for claim in protocol.claims
        if (
            claim.role == ClaimRole.SUPPORTING
            and claim.category == ClaimCategory.CONTROL_EVIDENCE
            and claim.estimand_scope == primary_claim.estimand_scope
        )
    ]


def _completed_claim_metric_rows(
    *,
    claim_run_ids: list[str],
    run_results: list[ProtocolRunResult],
) -> list[dict[str, Any]]:
    by_run = _result_by_run_id(run_results)
    rows: list[dict[str, Any]] = []

    for run_id in claim_run_ids:
        result = by_run.get(run_id)
        if result is None:
            continue
        if not is_run_success_status(result.status):
            continue

        metrics_payload = _load_json(result.metrics_path)
        if not isinstance(metrics_payload, dict):
            continue

        rows.append(
            {
                "run_id": run_id,
                "suite_id": result.suite_id,
                "status": normalize_run_status(result.status),
                "metrics": metrics_payload,
            }
        )

    return rows


def _primary_controls_valid(reporting_contract: dict[str, Any]) -> bool:
    controls_status = reporting_contract.get("controls_status", {})
    if not isinstance(controls_status, dict):
        return False
    return bool(controls_status.get("controls_valid_for_confirmatory", False))


def _science_critical_deviation_detected(reporting_contract: dict[str, Any]) -> bool:
    deviations = reporting_contract.get("deviations_from_protocol", {})
    if not isinstance(deviations, dict):
        return True
    return bool(deviations.get("science_critical_deviation_detected", False))


def _primary_metric_name_from_payload(payload: dict[str, Any]) -> str:
    primary_metric_name = payload.get("primary_metric_name")
    if isinstance(primary_metric_name, str) and primary_metric_name.strip():
        return str(primary_metric_name)
    metric_policy = payload.get("metric_policy_effective")
    if isinstance(metric_policy, dict):
        policy_metric = metric_policy.get("primary_metric")
        if isinstance(policy_metric, str) and policy_metric.strip():
            return str(policy_metric)
    direct_primary_metric = payload.get("primary_metric")
    if isinstance(direct_primary_metric, str) and direct_primary_metric.strip():
        return str(direct_primary_metric)
    return ""


def _permutation_requirement_status(
    permutation_payload: Any,
    *,
    alpha: float,
) -> str:
    if not isinstance(permutation_payload, dict):
        return "missing"
    p_value = _safe_float(permutation_payload.get("p_value"))
    passes_threshold_raw = permutation_payload.get("passes_threshold")
    has_passes_threshold = isinstance(passes_threshold_raw, bool)
    if p_value is None and not has_passes_threshold:
        return "missing"
    if p_value is not None:
        passes_by_p = bool(p_value <= float(alpha))
        if has_passes_threshold and bool(passes_threshold_raw) != passes_by_p:
            return "invalid"
        return "pass" if passes_by_p else "fail"
    return "pass" if bool(passes_threshold_raw) else "fail"


def _expected_cv_mode_for_estimand_scope(estimand_scope: Any) -> str:
    estimand_scope_value = str(getattr(estimand_scope, "value", estimand_scope))
    if estimand_scope_value == "within_subject_loso_session":
        return "within_subject_loso_session"
    if estimand_scope_value == "frozen_cross_person_transfer":
        return "frozen_cross_person_transfer"
    return ""


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted(set(values))


def _build_strict_gate_summary(
    *,
    required_conditions: list[str],
    condition_summary: dict[str, bool],
    supporting_control_claim_ids: list[str],
    supporting_control_run_ids: list[str],
    compared_primary_run_ids: list[str],
    matched_control_run_ids: list[str],
    matched_dummy_run_ids: list[str],
    missing_primary_run_ids: list[str],
    metric_mismatch_run_ids: list[str],
    cv_mismatch_run_ids: list[str],
    missing_control_run_ids: list[str],
    missing_dummy_run_ids: list[str],
    baseline_fail_run_ids: list[str],
    permutation_fail_run_ids: list[str],
    protocol_invalid_run_ids: list[str],
) -> dict[str, Any]:
    failed_condition_names = [
        condition
        for condition in required_conditions
        if not bool(condition_summary.get(condition, False))
    ]
    return {
        "required_conditions": list(required_conditions),
        "all_required_conditions_passed": bool(not failed_condition_names),
        "failed_condition_names": failed_condition_names,
        "condition_summary": dict(condition_summary),
        "supporting_control_claim_ids": _sorted_unique(supporting_control_claim_ids),
        "supporting_control_run_ids": _sorted_unique(supporting_control_run_ids),
        "compared_primary_run_ids": _sorted_unique(compared_primary_run_ids),
        "matched_control_run_ids": _sorted_unique(matched_control_run_ids),
        "matched_dummy_run_ids": _sorted_unique(matched_dummy_run_ids),
        "missing_primary_run_ids": _sorted_unique(missing_primary_run_ids),
        "metric_mismatch_run_ids": _sorted_unique(metric_mismatch_run_ids),
        "cv_mismatch_run_ids": _sorted_unique(cv_mismatch_run_ids),
        "missing_control_run_ids": _sorted_unique(missing_control_run_ids),
        "missing_dummy_run_ids": _sorted_unique(missing_dummy_run_ids),
        "baseline_fail_run_ids": _sorted_unique(baseline_fail_run_ids),
        "permutation_fail_run_ids": _sorted_unique(permutation_fail_run_ids),
        "protocol_invalid_run_ids": _sorted_unique(protocol_invalid_run_ids),
    }


def _evaluate_primary_claim(
    *,
    protocol: ThesisProtocol,
    primary_claim: ClaimSpec,
    compiled_manifest: CompiledProtocolManifest,
    claim_run_ids: list[str],
    run_results: list[ProtocolRunResult],
    reporting_contract: dict[str, Any],
) -> dict[str, Any]:
    run_specs_by_id = _manifest_run_specs_by_id(compiled_manifest)
    run_results_by_id = _result_by_run_id(run_results)

    expected_primary_run_ids = [
        run_id
        for run_id in claim_run_ids
        if (
            run_id in run_specs_by_id
            and _is_primary_evidence_run_spec(run_specs_by_id[run_id])
            and _normalized_evidence_run_role(
                getattr(run_specs_by_id[run_id], "evidence_run_role", "")
            )
            != EvidenceRunRole.UNTUNED_BASELINE.value
        )
    ]
    expected_primary_run_id_set = set(expected_primary_run_ids)

    require_complete = bool(protocol.success_criteria.require_complete_primary_suite_evidence)
    require_dummy = bool(protocol.success_criteria.require_dummy_baseline_outperformance)
    require_permutation = bool(protocol.success_criteria.require_permutation_pass)
    permutation_alpha = float(protocol.success_criteria.permutation_alpha)

    protocol_invalid_conditions: list[str] = []
    confirmatory_valid = bool(reporting_contract.get("confirmatory_valid", True))
    if not confirmatory_valid:
        protocol_invalid_conditions.append("protocol_confirmatory_valid")

    science_critical_deviation = bool(_science_critical_deviation_detected(reporting_contract))
    if science_critical_deviation:
        protocol_invalid_conditions.append("no_science_critical_deviation")

    controls_valid_for_confirmatory = bool(_primary_controls_valid(reporting_contract))
    if not controls_valid_for_confirmatory:
        protocol_invalid_conditions.append("controls_valid_for_confirmatory")

    completed_primary_rows: list[dict[str, Any]] = []
    missing_primary_run_ids: list[str] = []
    for run_id in expected_primary_run_ids:
        result = run_results_by_id.get(run_id)
        if result is None or not is_run_success_status(result.status):
            missing_primary_run_ids.append(run_id)
            continue

        config_payload = _load_json(result.config_path)
        metrics_payload = _load_json(result.metrics_path)
        if not isinstance(config_payload, dict) or not isinstance(metrics_payload, dict):
            missing_primary_run_ids.append(run_id)
            continue

        primary_metric_value = _safe_float(metrics_payload.get("primary_metric_value"))
        if primary_metric_value is None:
            missing_primary_run_ids.append(run_id)
            continue

        completed_primary_rows.append(
            {
                "run_id": run_id,
                "run_spec": run_specs_by_id[run_id],
                "result": result,
                "config": config_payload,
                "metrics": metrics_payload,
                "primary_metric_value": float(primary_metric_value),
            }
        )

    compared_primary_run_ids = [str(row["run_id"]) for row in completed_primary_rows]

    metric_mismatch_run_ids: list[str] = []
    cv_mismatch_run_ids: list[str] = []
    missing_control_run_ids: list[str] = []
    missing_dummy_run_ids: list[str] = []
    baseline_fail_run_ids: list[str] = []
    permutation_fail_run_ids: list[str] = []
    matched_control_run_ids: list[str] = []
    matched_dummy_run_ids: list[str] = []
    protocol_invalid_run_ids = sorted(expected_primary_run_id_set)

    expected_metric_from_claim = str(primary_claim.decision_metric)
    expected_metric_from_protocol = str(protocol.metric_policy.primary_metric)
    expected_cv_for_estimand = _expected_cv_mode_for_estimand_scope(primary_claim.estimand_scope)
    expected_methodology_from_protocol = _normalized_methodology_policy_name(
        protocol.methodology_policy.policy_name
    )
    expected_tuning_from_protocol = bool(protocol.methodology_policy.tuning_enabled)

    for row in completed_primary_rows:
        run_id = str(row["run_id"])
        run_spec = row["run_spec"]
        config_payload = row["config"]
        metrics_payload = row["metrics"]

        run_spec_metric = str(getattr(run_spec, "primary_metric", "") or "")
        config_metric = _primary_metric_name_from_payload(config_payload)
        metrics_metric = _primary_metric_name_from_payload(metrics_payload)

        metric_values = {
            expected_metric_from_claim,
            expected_metric_from_protocol,
            run_spec_metric,
            config_metric,
            metrics_metric,
        }
        if "" in metric_values or len(metric_values) != 1:
            metric_mismatch_run_ids.append(run_id)

        run_cv_mode = str(getattr(run_spec, "cv_mode", "") or "")
        config_cv_mode = str(config_payload.get("cv_mode") or config_payload.get("cv") or "")
        run_methodology = _normalized_methodology_policy_name(
            getattr(run_spec, "methodology_policy_name", "")
        )
        config_methodology = _normalized_methodology_policy_name(
            config_payload.get("methodology_policy_name")
        )
        metrics_methodology = _normalized_methodology_policy_name(
            metrics_payload.get("methodology_policy_name")
        )
        run_tuning_enabled = bool(getattr(run_spec, "tuning_enabled", False))
        config_tuning_enabled = config_payload.get("tuning_enabled")
        metrics_tuning_enabled = metrics_payload.get("tuning_enabled")

        methodology_mismatch = False
        if not run_cv_mode or not config_cv_mode or run_cv_mode != config_cv_mode:
            methodology_mismatch = True
        if expected_cv_for_estimand and run_cv_mode != expected_cv_for_estimand:
            methodology_mismatch = True
        if not run_methodology or run_methodology != expected_methodology_from_protocol:
            methodology_mismatch = True
        if run_methodology != config_methodology or run_methodology != metrics_methodology:
            methodology_mismatch = True
        if run_tuning_enabled != expected_tuning_from_protocol:
            methodology_mismatch = True
        if (
            not isinstance(config_tuning_enabled, bool)
            or bool(config_tuning_enabled) != run_tuning_enabled
        ):
            methodology_mismatch = True
        if (
            not isinstance(metrics_tuning_enabled, bool)
            or bool(metrics_tuning_enabled) != run_tuning_enabled
        ):
            methodology_mismatch = True

        if run_methodology == "grouped_nested_tuning":
            for field_name in (
                "tuning_inner_cv_scheme",
                "tuning_inner_group_field",
                "tuning_search_space_id",
                "tuning_search_space_version",
            ):
                run_value = getattr(run_spec, field_name, None)
                config_value = config_payload.get(field_name)
                metrics_value = metrics_payload.get(field_name)
                if run_value is None or config_value is None or metrics_value is None:
                    methodology_mismatch = True
                    continue
                if str(run_value) != str(config_value) or str(run_value) != str(metrics_value):
                    methodology_mismatch = True

            if str(getattr(protocol.methodology_policy, "inner_cv_scheme", "") or "") != str(
                getattr(run_spec, "tuning_inner_cv_scheme", "") or ""
            ):
                methodology_mismatch = True
            if str(getattr(protocol.methodology_policy, "inner_group_field", "") or "") != str(
                getattr(run_spec, "tuning_inner_group_field", "") or ""
            ):
                methodology_mismatch = True
            if str(getattr(protocol.methodology_policy, "tuning_search_space_id", "") or "") != str(
                getattr(run_spec, "tuning_search_space_id", "") or ""
            ):
                methodology_mismatch = True
            if str(
                getattr(protocol.methodology_policy, "tuning_search_space_version", "") or ""
            ) != str(getattr(run_spec, "tuning_search_space_version", "") or ""):
                methodology_mismatch = True

        if methodology_mismatch:
            cv_mismatch_run_ids.append(run_id)

    supporting_control_claims = _resolve_supporting_control_claims(
        protocol=protocol,
        primary_claim=primary_claim,
    )
    supporting_control_claim_ids = [claim.claim_id for claim in supporting_control_claims]
    supporting_control_run_ids = sorted(
        {
            run_id
            for claim_id in supporting_control_claim_ids
            for run_id in compiled_manifest.claim_to_run_map.get(claim_id, [])
        }
    )

    need_supporting_control_evidence = bool(require_dummy or require_permutation)
    non_dummy_control_by_key: dict[tuple[str, str, str, str, int, int, str], Any] = {}
    dummy_control_by_key: dict[tuple[str, str, str, str, int, int, str], Any] = {}
    duplicate_control_match_keys: list[str] = []
    duplicate_dummy_match_keys: list[str] = []

    if need_supporting_control_evidence and (
        not supporting_control_claim_ids or not supporting_control_run_ids
    ):
        missing_control_run_ids.extend(compared_primary_run_ids)
    else:
        for run_id in supporting_control_run_ids:
            run_spec = run_specs_by_id.get(run_id)
            if run_spec is None:
                continue
            if (
                _normalized_evidence_run_role(getattr(run_spec, "evidence_run_role", ""))
                != EvidenceRunRole.PRIMARY.value
            ):
                continue
            if _is_dummy_run_spec(run_spec):
                dummy_key = _run_match_key(run_spec, include_model=False)
                if dummy_key in dummy_control_by_key:
                    duplicate_dummy_match_keys.append("|".join(str(part) for part in dummy_key))
                    continue
                dummy_control_by_key[dummy_key] = run_spec
                continue

            model_key = _run_match_key(run_spec, include_model=True)
            if model_key in non_dummy_control_by_key:
                duplicate_control_match_keys.append("|".join(str(part) for part in model_key))
                continue
            non_dummy_control_by_key[model_key] = run_spec

        for row in completed_primary_rows:
            primary_run_id = str(row["run_id"])
            primary_run_spec = row["run_spec"]
            primary_metric_value = float(row["primary_metric_value"])
            model_match_key = _run_match_key(primary_run_spec, include_model=True)
            dummy_match_key = _run_match_key(primary_run_spec, include_model=False)

            matched_control_spec = non_dummy_control_by_key.get(model_match_key)
            if matched_control_spec is None:
                missing_control_run_ids.append(primary_run_id)
                continue

            matched_control_run_id = str(matched_control_spec.run_id)
            matched_control_run_ids.append(matched_control_run_id)
            matched_control_result = run_results_by_id.get(matched_control_run_id)
            if matched_control_result is None or not is_run_success_status(
                matched_control_result.status
            ):
                missing_control_run_ids.append(primary_run_id)
                continue

            matched_control_metrics = _load_json(matched_control_result.metrics_path)
            if not isinstance(matched_control_metrics, dict):
                missing_control_run_ids.append(primary_run_id)
                continue

            matched_control_metric_value = _safe_float(
                matched_control_metrics.get("primary_metric_value")
            )
            if matched_control_metric_value is None:
                missing_control_run_ids.append(primary_run_id)
                continue

            if (
                abs(float(primary_metric_value) - float(matched_control_metric_value))
                > _PRIMARY_CONTROL_METRIC_TOLERANCE
            ):
                metric_mismatch_run_ids.append(primary_run_id)

            if require_permutation:
                permutation_status = _permutation_requirement_status(
                    matched_control_metrics.get("permutation_test"),
                    alpha=permutation_alpha,
                )
                if permutation_status in {"missing", "invalid"}:
                    missing_control_run_ids.append(primary_run_id)
                elif permutation_status == "fail":
                    permutation_fail_run_ids.append(primary_run_id)

            if require_dummy:
                matched_dummy_spec = dummy_control_by_key.get(dummy_match_key)
                if matched_dummy_spec is None:
                    missing_dummy_run_ids.append(primary_run_id)
                    continue

                matched_dummy_run_id = str(matched_dummy_spec.run_id)
                matched_dummy_run_ids.append(matched_dummy_run_id)
                matched_dummy_result = run_results_by_id.get(matched_dummy_run_id)
                if matched_dummy_result is None or not is_run_success_status(
                    matched_dummy_result.status
                ):
                    missing_dummy_run_ids.append(primary_run_id)
                    continue

                matched_dummy_metrics = _load_json(matched_dummy_result.metrics_path)
                if not isinstance(matched_dummy_metrics, dict):
                    missing_dummy_run_ids.append(primary_run_id)
                    continue

                matched_dummy_metric_value = _safe_float(
                    matched_dummy_metrics.get("primary_metric_value")
                )
                if matched_dummy_metric_value is None:
                    missing_dummy_run_ids.append(primary_run_id)
                    continue

                if float(matched_control_metric_value) <= float(matched_dummy_metric_value):
                    baseline_fail_run_ids.append(primary_run_id)

    if duplicate_control_match_keys:
        missing_control_run_ids.extend(compared_primary_run_ids)
    if duplicate_dummy_match_keys:
        missing_dummy_run_ids.extend(compared_primary_run_ids)

    required_conditions = [
        "protocol_confirmatory_valid",
        "no_science_critical_deviation",
        "controls_valid_for_confirmatory",
        "usable_primary_evidence_present",
        "primary_metric_lock",
        "methodology_lock",
    ]
    if require_complete:
        required_conditions.append("complete_primary_evidence")
    if need_supporting_control_evidence:
        required_conditions.append("matched_control_evidence")
    if require_dummy:
        required_conditions.append("matched_dummy_evidence")
        required_conditions.append("dummy_baseline_outperformance")
    if require_permutation:
        required_conditions.append("permutation_pass")

    condition_summary = {
        "protocol_confirmatory_valid": bool(confirmatory_valid),
        "no_science_critical_deviation": bool(not science_critical_deviation),
        "controls_valid_for_confirmatory": bool(controls_valid_for_confirmatory),
        "usable_primary_evidence_present": bool(len(completed_primary_rows) > 0),
        "complete_primary_evidence": bool((not require_complete) or (not missing_primary_run_ids)),
        "primary_metric_lock": bool(not metric_mismatch_run_ids),
        "methodology_lock": bool(not cv_mismatch_run_ids),
        "matched_control_evidence": bool(
            not missing_control_run_ids and not duplicate_control_match_keys
        ),
        "matched_dummy_evidence": bool(
            not missing_dummy_run_ids and not duplicate_dummy_match_keys
        ),
        "dummy_baseline_outperformance": bool((not require_dummy) or (not baseline_fail_run_ids)),
        "permutation_pass": bool((not require_permutation) or (not permutation_fail_run_ids)),
    }

    strict_gate_summary = _build_strict_gate_summary(
        required_conditions=required_conditions,
        condition_summary=condition_summary,
        supporting_control_claim_ids=supporting_control_claim_ids,
        supporting_control_run_ids=supporting_control_run_ids,
        compared_primary_run_ids=compared_primary_run_ids,
        matched_control_run_ids=matched_control_run_ids,
        matched_dummy_run_ids=matched_dummy_run_ids,
        missing_primary_run_ids=missing_primary_run_ids,
        metric_mismatch_run_ids=metric_mismatch_run_ids,
        cv_mismatch_run_ids=cv_mismatch_run_ids,
        missing_control_run_ids=missing_control_run_ids,
        missing_dummy_run_ids=missing_dummy_run_ids,
        baseline_fail_run_ids=baseline_fail_run_ids,
        permutation_fail_run_ids=permutation_fail_run_ids,
        protocol_invalid_run_ids=(protocol_invalid_run_ids if protocol_invalid_conditions else []),
    )

    all_required_conditions_passed = bool(strict_gate_summary["all_required_conditions_passed"])
    observed_metric_values = [
        float(primary_row["primary_metric_value"]) for primary_row in completed_primary_rows
    ]

    invalid_failure_map: list[tuple[str, str]] = [
        ("protocol_confirmatory_valid", "confirmatory_valid_false"),
        ("no_science_critical_deviation", "science_critical_deviation_detected"),
        ("controls_valid_for_confirmatory", "required_controls_invalid"),
        ("complete_primary_evidence", "incomplete_primary_evidence"),
        ("primary_metric_lock", "primary_metric_mismatch"),
        ("methodology_lock", "methodology_or_cv_mismatch"),
        ("matched_control_evidence", "missing_or_inconsistent_supporting_control_evidence"),
        ("matched_dummy_evidence", "missing_or_incomplete_dummy_evidence"),
    ]

    for condition_name, reason in invalid_failure_map:
        if condition_name in required_conditions and not bool(
            condition_summary.get(condition_name, False)
        ):
            return {
                "verdict": "invalid",
                "reason": reason,
                "n_expected_primary_runs": len(expected_primary_run_ids),
                "n_completed_runs": len(completed_primary_rows),
                "observed_metric_values": observed_metric_values,
                "all_required_conditions_passed": all_required_conditions_passed,
                "strict_gate_summary": strict_gate_summary,
                "control_evidence_summary": strict_gate_summary,
            }

    if not completed_primary_rows:
        return {
            "verdict": "inconclusive",
            "reason": "no_usable_completed_primary_evidence",
            "n_expected_primary_runs": len(expected_primary_run_ids),
            "n_completed_runs": 0,
            "all_required_conditions_passed": all_required_conditions_passed,
            "strict_gate_summary": strict_gate_summary,
            "control_evidence_summary": strict_gate_summary,
        }

    if require_permutation and permutation_fail_run_ids:
        return {
            "verdict": "not_supported",
            "reason": "permutation_requirement_not_met",
            "n_expected_primary_runs": len(expected_primary_run_ids),
            "n_completed_runs": len(completed_primary_rows),
            "observed_metric_values": observed_metric_values,
            "all_required_conditions_passed": all_required_conditions_passed,
            "strict_gate_summary": strict_gate_summary,
            "control_evidence_summary": strict_gate_summary,
        }

    if require_dummy and baseline_fail_run_ids:
        return {
            "verdict": "not_supported",
            "reason": "dummy_baseline_not_outperformed",
            "n_expected_primary_runs": len(expected_primary_run_ids),
            "n_completed_runs": len(completed_primary_rows),
            "observed_metric_values": observed_metric_values,
            "all_required_conditions_passed": all_required_conditions_passed,
            "strict_gate_summary": strict_gate_summary,
            "control_evidence_summary": strict_gate_summary,
        }

    return {
        "verdict": "supported",
        "reason": "all_required_primary_conditions_passed",
        "n_expected_primary_runs": len(expected_primary_run_ids),
        "n_completed_runs": len(completed_primary_rows),
        "observed_metric_values": observed_metric_values,
        "all_required_conditions_passed": all_required_conditions_passed,
        "strict_gate_summary": strict_gate_summary,
        "control_evidence_summary": strict_gate_summary,
    }


def _evaluate_secondary_transfer_claim(
    *,
    claim_run_ids: list[str],
    run_results: list[ProtocolRunResult],
) -> dict[str, Any]:
    rows = _completed_claim_metric_rows(
        claim_run_ids=claim_run_ids,
        run_results=run_results,
    )

    if not rows:
        return {
            "verdict": "inconclusive",
            "reason": "no_completed_claim_runs",
        }

    return {
        "verdict": "inconclusive",
        "reason": "secondary_descriptive_only",
        "n_completed_runs": len(rows),
    }


def _evaluate_supporting_claim(
    *,
    claim_run_ids: list[str],
    run_results: list[ProtocolRunResult],
) -> dict[str, Any]:
    rows = _completed_claim_metric_rows(
        claim_run_ids=claim_run_ids,
        run_results=run_results,
    )

    if not rows:
        return {
            "verdict": "inconclusive",
            "reason": "no_completed_claim_runs",
        }

    return {
        "verdict": "inconclusive",
        "reason": "supporting_evidence_only",
        "n_completed_runs": len(rows),
    }


def evaluate_claim_outcomes(
    *,
    protocol: ThesisProtocol,
    compiled_manifest: CompiledProtocolManifest,
    run_results: list[ProtocolRunResult],
    reporting_contract: dict[str, Any],
) -> dict[str, Any]:
    claim_to_run_map = compiled_manifest.claim_to_run_map
    outcomes: list[dict[str, Any]] = []

    for claim in protocol.claims:
        claim_run_ids = list(claim_to_run_map.get(claim.claim_id, []))

        if claim.role == ClaimRole.PRIMARY:
            outcome = _evaluate_primary_claim(
                protocol=protocol,
                primary_claim=claim,
                compiled_manifest=compiled_manifest,
                claim_run_ids=claim_run_ids,
                run_results=run_results,
                reporting_contract=reporting_contract,
            )
        elif claim.category == ClaimCategory.CROSS_PERSON_TRANSFER:
            outcome = _evaluate_secondary_transfer_claim(
                claim_run_ids=claim_run_ids,
                run_results=run_results,
            )
        else:
            outcome = _evaluate_supporting_claim(
                claim_run_ids=claim_run_ids,
                run_results=run_results,
            )

        outcomes.append(
            {
                "claim_id": claim.claim_id,
                "title": claim.title,
                "role": claim.role.value,
                "category": claim.category.value,
                "decision_metric": claim.decision_metric,
                "suite_ids": list(claim.suite_ids),
                "run_ids": claim_run_ids,
                **outcome,
            }
        )

    primary_claim_id = protocol.success_criteria.primary_claim_id
    primary_claim_outcome = next(
        (entry for entry in outcomes if entry["claim_id"] == primary_claim_id),
        None,
    )

    return {
        "framework_mode": "confirmatory",
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "primary_claim_id": primary_claim_id,
        "primary_claim_verdict": (
            primary_claim_outcome["verdict"]
            if isinstance(primary_claim_outcome, dict)
            else "invalid"
        ),
        "primary_claim_reason": (
            primary_claim_outcome.get("reason")
            if isinstance(primary_claim_outcome, dict)
            else "primary_claim_outcome_missing"
        ),
        "primary_claim_all_required_conditions_passed": bool(
            primary_claim_outcome.get("all_required_conditions_passed", False)
            if isinstance(primary_claim_outcome, dict)
            else False
        ),
        "primary_claim_failed_condition_names": (
            list(
                primary_claim_outcome.get("strict_gate_summary", {}).get(
                    "failed_condition_names",
                    [],
                )
            )
            if isinstance(primary_claim_outcome, dict)
            else []
        ),
        "claims": outcomes,
    }

