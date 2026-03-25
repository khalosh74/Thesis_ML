from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Thesis_ML.experiments.run_states import is_run_success_status, normalize_run_status
from Thesis_ML.protocols.models import (
    ClaimCategory,
    ClaimRole,
    CompiledProtocolManifest,
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

    rows = _completed_claim_metric_rows(
        claim_run_ids=claim_run_ids,
        run_results=run_results,
    )

    if not rows:
        return {
            "verdict": "inconclusive",
            "reason": "no_completed_claim_runs",
        }

    permutation_passes = 0
    observed_metric_values: list[float] = []

    for row in rows:
        metrics_payload = row["metrics"]

        observed = metrics_payload.get("primary_metric_value")
        if isinstance(observed, (int, float)):
            observed_metric_values.append(float(observed))

        permutation_payload = metrics_payload.get("permutation_test")
        if isinstance(permutation_payload, dict):
            if bool(permutation_payload.get("passes_threshold", False)):
                permutation_passes += 1

    if permutation_passes > 0:
        return {
            "verdict": "supported",
            "reason": "completed_runs_with_permutation_pass",
            "n_completed_runs": len(rows),
            "n_permutation_pass_runs": permutation_passes,
            "observed_metric_values": observed_metric_values,
        }

    return {
        "verdict": "not_supported",
        "reason": "completed_runs_without_permutation_pass",
        "n_completed_runs": len(rows),
        "n_permutation_pass_runs": 0,
        "observed_metric_values": observed_metric_values,
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
