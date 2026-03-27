from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Thesis_ML.config.methodology import EvidenceRunRole
from Thesis_ML.experiments.run_states import is_run_success_status, normalize_run_status
from Thesis_ML.protocols.models import (
    ClaimCategory,
    ClaimRole,
    CompiledProtocolManifest,
    ClaimSpec,
    ProtocolRunResult,
    ThesisProtocol,
)

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
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalized_evidence_run_role(value: Any) -> str:
    if isinstance(value, EvidenceRunRole):
        return str(value.value)
    return str(value)


def _is_dummy_run_spec(spec: Any) -> bool:
    controls = getattr(spec, "controls", None)
    controls_dummy = bool(getattr(controls, "dummy_baseline_run", False))
    return bool(controls_dummy or str(getattr(spec, "model", "")) == "dummy")


def _is_primary_evidence_run_spec(spec: Any) -> bool:
    return (
        _normalized_evidence_run_role(getattr(spec, "evidence_run_role", ""))
        == EvidenceRunRole.PRIMARY.value
    ) and (not _is_dummy_run_spec(spec))


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


def _baseline_evidence_summary(
    *,
    supporting_control_claim_ids: list[str],
    supporting_control_run_ids: list[str],
    compared_primary_run_ids: list[str],
    matched_dummy_run_ids: list[str],
    baseline_fail_run_ids: list[str],
    missing_primary_runs: list[str],
    missing_dummy_runs: list[str],
    duplicate_dummy_match_keys: list[str],
) -> dict[str, Any]:
    return {
        "supporting_control_claim_ids": supporting_control_claim_ids,
        "supporting_control_run_ids": supporting_control_run_ids,
        "compared_primary_run_ids": sorted(set(compared_primary_run_ids)),
        "matched_dummy_run_ids": sorted(set(matched_dummy_run_ids)),
        "baseline_fail_run_ids": sorted(set(baseline_fail_run_ids)),
        "missing_primary_runs": sorted(set(missing_primary_runs)),
        "missing_dummy_runs": sorted(set(missing_dummy_runs)),
        "duplicate_dummy_match_keys": sorted(set(duplicate_dummy_match_keys)),
    }


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


def _evaluate_primary_claim(
    *,
    protocol: ThesisProtocol,
    primary_claim: ClaimSpec,
    compiled_manifest: CompiledProtocolManifest,
    claim_run_ids: list[str],
    run_results: list[ProtocolRunResult],
    reporting_contract: dict[str, Any],
) -> dict[str, Any]:
    if _science_critical_deviation_detected(reporting_contract):
        return {
            "verdict": "invalid",
            "reason": "science_critical_deviation_detected",
        }

    if not _primary_controls_valid(reporting_contract):
        return {
            "verdict": "invalid",
            "reason": "required_controls_invalid",
        }

    run_specs_by_id = _manifest_run_specs_by_id(compiled_manifest)
    run_results_by_id = _result_by_run_id(run_results)

    required_primary_run_ids = [
        run_id
        for run_id in claim_run_ids
        if (
            run_id in run_specs_by_id
            and _is_primary_evidence_run_spec(run_specs_by_id[run_id])
        )
    ]

    completed_primary_rows: list[dict[str, Any]] = []
    missing_primary_runs: list[str] = []
    for run_id in required_primary_run_ids:
        result = run_results_by_id.get(run_id)
        if result is None or not is_run_success_status(result.status):
            missing_primary_runs.append(run_id)
            continue
        metrics_payload = _load_json(result.metrics_path)
        if not isinstance(metrics_payload, dict):
            missing_primary_runs.append(run_id)
            continue
        primary_metric_value = _safe_float(metrics_payload.get("primary_metric_value"))
        if primary_metric_value is None:
            missing_primary_runs.append(run_id)
            continue
        completed_primary_rows.append(
            {
                "run_id": run_id,
                "run_spec": run_specs_by_id[run_id],
                "result": result,
                "metrics": metrics_payload,
                "primary_metric_value": primary_metric_value,
            }
        )

    if not completed_primary_rows:
        return {
            "verdict": "inconclusive",
            "reason": "no_completed_primary_evidence_runs",
            "control_evidence_summary": _baseline_evidence_summary(
                supporting_control_claim_ids=[],
                supporting_control_run_ids=[],
                compared_primary_run_ids=[],
                matched_dummy_run_ids=[],
                baseline_fail_run_ids=[],
                missing_primary_runs=missing_primary_runs,
                missing_dummy_runs=[],
                duplicate_dummy_match_keys=[],
            ),
        }

    require_dummy = bool(protocol.success_criteria.require_dummy_baseline_outperformance)
    require_complete = bool(protocol.success_criteria.require_complete_primary_suite_evidence)

    if not require_dummy:
        permutation_passes = 0
        for primary_row in completed_primary_rows:
            permutation_payload = primary_row["metrics"].get("permutation_test")
            if isinstance(permutation_payload, dict) and bool(
                permutation_payload.get("passes_threshold", False)
            ):
                permutation_passes += 1
        if permutation_passes > 0:
            return {
                "verdict": "supported",
                "reason": "completed_runs_with_permutation_pass",
                "n_completed_runs": len(completed_primary_rows),
                "n_permutation_pass_runs": permutation_passes,
                "observed_metric_values": [
                    float(primary_row["primary_metric_value"])
                    for primary_row in completed_primary_rows
                ],
                "control_evidence_summary": _baseline_evidence_summary(
                    supporting_control_claim_ids=[],
                    supporting_control_run_ids=[],
                    compared_primary_run_ids=[row["run_id"] for row in completed_primary_rows],
                    matched_dummy_run_ids=[],
                    baseline_fail_run_ids=[],
                    missing_primary_runs=missing_primary_runs,
                    missing_dummy_runs=[],
                    duplicate_dummy_match_keys=[],
                ),
            }
        return {
            "verdict": "not_supported",
            "reason": "completed_runs_without_permutation_pass",
            "n_completed_runs": len(completed_primary_rows),
            "n_permutation_pass_runs": 0,
            "observed_metric_values": [
                float(primary_row["primary_metric_value"])
                for primary_row in completed_primary_rows
            ],
            "control_evidence_summary": _baseline_evidence_summary(
                supporting_control_claim_ids=[],
                supporting_control_run_ids=[],
                compared_primary_run_ids=[row["run_id"] for row in completed_primary_rows],
                matched_dummy_run_ids=[],
                baseline_fail_run_ids=[],
                missing_primary_runs=missing_primary_runs,
                missing_dummy_runs=[],
                duplicate_dummy_match_keys=[],
            ),
        }

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

    if not supporting_control_claim_ids or not supporting_control_run_ids:
        return {
            "verdict": "invalid",
            "reason": "missing_supporting_dummy_control_evidence",
            "control_evidence_summary": _baseline_evidence_summary(
                supporting_control_claim_ids=supporting_control_claim_ids,
                supporting_control_run_ids=supporting_control_run_ids,
                compared_primary_run_ids=[row["run_id"] for row in completed_primary_rows],
                matched_dummy_run_ids=[],
                baseline_fail_run_ids=[],
                missing_primary_runs=missing_primary_runs,
                missing_dummy_runs=[row["run_id"] for row in completed_primary_rows],
                duplicate_dummy_match_keys=[],
            ),
        }

    dummy_control_by_key: dict[tuple[str, str, str, str, int, int, str], Any] = {}
    duplicate_dummy_match_keys: list[str] = []
    for run_id in supporting_control_run_ids:
        run_spec = run_specs_by_id.get(run_id)
        if run_spec is None:
            continue
        if (
            _normalized_evidence_run_role(getattr(run_spec, "evidence_run_role", ""))
            != EvidenceRunRole.PRIMARY.value
        ):
            continue
        if not _is_dummy_run_spec(run_spec):
            continue
        match_key = _run_match_key(run_spec, include_model=False)
        if match_key in dummy_control_by_key:
            duplicate_dummy_match_keys.append("|".join(str(part) for part in match_key))
            continue
        dummy_control_by_key[match_key] = run_spec

    compared_primary_run_ids: list[str] = []
    matched_dummy_run_ids: list[str] = []
    missing_dummy_runs: list[str] = []
    baseline_fail_run_ids: list[str] = []

    for primary_row in completed_primary_rows:
        primary_run_id = str(primary_row["run_id"])
        primary_run_spec = primary_row["run_spec"]
        primary_metric_value = float(primary_row["primary_metric_value"])
        compared_primary_run_ids.append(primary_run_id)
        matched_dummy_spec = dummy_control_by_key.get(
            _run_match_key(primary_run_spec, include_model=False)
        )
        if matched_dummy_spec is None:
            missing_dummy_runs.append(primary_run_id)
            continue

        dummy_run_id = str(matched_dummy_spec.run_id)
        matched_dummy_run_ids.append(dummy_run_id)
        dummy_result = run_results_by_id.get(dummy_run_id)
        if dummy_result is None or not is_run_success_status(dummy_result.status):
            missing_dummy_runs.append(primary_run_id)
            continue

        dummy_metrics_payload = _load_json(dummy_result.metrics_path)
        if not isinstance(dummy_metrics_payload, dict):
            missing_dummy_runs.append(primary_run_id)
            continue

        dummy_metric_value = _safe_float(dummy_metrics_payload.get("primary_metric_value"))
        if dummy_metric_value is None:
            missing_dummy_runs.append(primary_run_id)
            continue

        if float(primary_metric_value) <= float(dummy_metric_value):
            baseline_fail_run_ids.append(primary_run_id)

    summary = _baseline_evidence_summary(
        supporting_control_claim_ids=supporting_control_claim_ids,
        supporting_control_run_ids=supporting_control_run_ids,
        compared_primary_run_ids=compared_primary_run_ids,
        matched_dummy_run_ids=matched_dummy_run_ids,
        baseline_fail_run_ids=baseline_fail_run_ids,
        missing_primary_runs=missing_primary_runs,
        missing_dummy_runs=missing_dummy_runs,
        duplicate_dummy_match_keys=duplicate_dummy_match_keys,
    )

    observed_metric_values = [
        float(primary_row["primary_metric_value"]) for primary_row in completed_primary_rows
    ]

    if require_complete and summary["missing_primary_runs"]:
        return {
            "verdict": "invalid",
            "reason": "incomplete_primary_evidence",
            "n_completed_runs": len(completed_primary_rows),
            "observed_metric_values": observed_metric_values,
            "control_evidence_summary": summary,
        }

    if summary["duplicate_dummy_match_keys"]:
        return {
            "verdict": "invalid",
            "reason": "duplicate_dummy_baseline_matches",
            "n_completed_runs": len(completed_primary_rows),
            "observed_metric_values": observed_metric_values,
            "control_evidence_summary": summary,
        }

    if summary["missing_dummy_runs"]:
        return {
            "verdict": "invalid",
            "reason": "missing_or_incomplete_dummy_evidence",
            "n_completed_runs": len(completed_primary_rows),
            "observed_metric_values": observed_metric_values,
            "control_evidence_summary": summary,
        }

    if summary["baseline_fail_run_ids"]:
        return {
            "verdict": "not_supported",
            "reason": "dummy_baseline_not_outperformed",
            "n_completed_runs": len(completed_primary_rows),
            "observed_metric_values": observed_metric_values,
            "control_evidence_summary": summary,
        }

    return {
        "verdict": "supported",
        "reason": "strictly_outperforms_dummy_baseline",
        "n_completed_runs": len(completed_primary_rows),
        "observed_metric_values": observed_metric_values,
        "control_evidence_summary": summary,
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
        "claims": outcomes,
    }
