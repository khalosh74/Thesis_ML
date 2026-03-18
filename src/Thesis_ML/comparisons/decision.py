from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Thesis_ML.comparisons.models import (
    ComparisonDecisionStatus,
    ComparisonRunResult,
    ComparisonSpec,
    CompiledComparisonManifest,
)
from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.metric_policy import (
    enforce_primary_metric_alignment,
    extract_metric_value,
    resolve_effective_metric_policy,
    validate_metric_name,
)
from Thesis_ML.experiments.run_states import (
    RUN_STATUS_FAILED,
    RUN_STATUS_PLANNED,
    RUN_STATUS_SKIPPED_DUE_TO_POLICY,
    RUN_STATUS_SUCCESS,
    RUN_STATUS_TIMED_OUT,
    is_run_success_status,
    normalize_run_status,
)


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_metrics_payload(result: ComparisonRunResult) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if isinstance(result.metrics, dict):
        payload.update(result.metrics)
    if result.metrics_path:
        metrics_path = Path(result.metrics_path)
        if metrics_path.exists():
            try:
                loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                payload.update(loaded)
    return payload


def _metric_policy_payload(metric_policy_effective: Any) -> dict[str, Any]:
    return {
        "primary_metric": metric_policy_effective.primary_metric,
        "secondary_metrics": list(metric_policy_effective.secondary_metrics),
        "decision_metric": metric_policy_effective.decision_metric,
        "tuning_metric": metric_policy_effective.tuning_metric,
        "permutation_metric": metric_policy_effective.permutation_metric,
        "higher_is_better": bool(metric_policy_effective.higher_is_better),
    }


def _subgroup_reports_ok(variant_results: list[ComparisonRunResult]) -> tuple[bool, str | None]:
    for result in variant_results:
        if not is_run_success_status(result.status) or not result.report_dir:
            continue
        subgroup_json_path = Path(result.report_dir) / "subgroup_metrics.json"
        if not subgroup_json_path.exists():
            return False, f"Missing subgroup report: {subgroup_json_path}"
        try:
            payload = json.loads(subgroup_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False, f"Invalid subgroup JSON: {subgroup_json_path}"
        if not isinstance(payload, dict):
            return False, f"Invalid subgroup payload shape: {subgroup_json_path}"
        if bool(payload.get("enabled")) and int(payload.get("n_rows", 0)) <= 0:
            return False, f"Subgroup report enabled but empty: {subgroup_json_path}"
    return True, None


def build_comparison_decision(
    *,
    comparison: ComparisonSpec,
    compiled_manifest: CompiledComparisonManifest,
    run_results: list[ComparisonRunResult],
    paired_comparison_payload: dict[str, Any] | None = None,
    required_evidence_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric_policy_effective = enforce_primary_metric_alignment(
        resolve_effective_metric_policy(
            primary_metric=comparison.metric_policy.primary_metric,
            secondary_metrics=comparison.metric_policy.secondary_metrics,
            decision_metric=comparison.decision_policy.primary_metric,
            tuning_metric=comparison.metric_policy.primary_metric,
            permutation_metric=(
                comparison.control_policy.permutation_metric
                or comparison.metric_policy.primary_metric
            ),
        ),
        context="locked comparison decision",
    )
    primary_metric = metric_policy_effective.primary_metric
    decision_policy = comparison.decision_policy
    result_by_run_id = {result.run_id: result for result in run_results}
    variant_scores: dict[str, dict[str, Any]] = {
        variant_id: {
            "n_runs": 0,
            "n_success": 0,
            "n_completed": 0,
            "n_failed": 0,
            "n_timed_out": 0,
            "n_skipped_due_to_policy": 0,
            "n_planned": 0,
            "primary_metric_values": [],
            "mean_primary_metric": None,
            "permutation_p_values": [],
        }
        for variant_id in compiled_manifest.variant_ids
    }
    missing_runs: list[str] = []
    missing_metric_runs: list[str] = []

    for spec in compiled_manifest.runs:
        if str(spec.evidence_run_role.value) != "primary":
            continue
        bucket = variant_scores.setdefault(
            spec.variant_id,
            {
                "n_runs": 0,
                "n_success": 0,
                "n_completed": 0,
                "n_failed": 0,
                "n_timed_out": 0,
                "n_skipped_due_to_policy": 0,
                "n_planned": 0,
                "primary_metric_values": [],
                "mean_primary_metric": None,
                "permutation_p_values": [],
            },
        )
        bucket["n_runs"] = int(bucket["n_runs"]) + 1
        result = result_by_run_id.get(spec.run_id)
        if result is None:
            missing_runs.append(spec.run_id)
            bucket["n_planned"] = int(bucket["n_planned"]) + 1
            continue
        normalized_status = normalize_run_status(result.status)
        if normalized_status == RUN_STATUS_SUCCESS:
            bucket["n_success"] = int(bucket["n_success"]) + 1
            bucket["n_completed"] = int(bucket["n_completed"]) + 1
            metrics_payload = _load_metrics_payload(result)
            payload_primary_metric_name = metrics_payload.get("primary_metric_name")
            if (
                payload_primary_metric_name is not None
                and validate_metric_name(str(payload_primary_metric_name)) != primary_metric
            ):
                missing_metric_runs.append(spec.run_id)
                continue
            metric_value = extract_metric_value(
                metrics_payload,
                primary_metric,
                require=False,
                payload_label=f"run '{spec.run_id}' metrics",
            )
            if metric_value is None:
                missing_metric_runs.append(spec.run_id)
            else:
                bucket["primary_metric_values"].append(metric_value)
            permutation_payload = metrics_payload.get("permutation_test")
            if isinstance(permutation_payload, dict):
                p_value = _safe_float(permutation_payload.get("p_value"))
                if p_value is not None:
                    bucket["permutation_p_values"].append(p_value)
        elif normalized_status == RUN_STATUS_FAILED:
            bucket["n_failed"] = int(bucket["n_failed"]) + 1
        elif normalized_status == RUN_STATUS_TIMED_OUT:
            bucket["n_timed_out"] = int(bucket["n_timed_out"]) + 1
        elif normalized_status == RUN_STATUS_SKIPPED_DUE_TO_POLICY:
            bucket["n_skipped_due_to_policy"] = int(bucket["n_skipped_due_to_policy"]) + 1
        elif normalized_status == RUN_STATUS_PLANNED:
            bucket["n_planned"] = int(bucket["n_planned"]) + 1
        else:
            bucket["n_planned"] = int(bucket["n_planned"]) + 1

    for bucket in variant_scores.values():
        values = [float(value) for value in bucket["primary_metric_values"]]
        bucket["mean_primary_metric"] = float(sum(values) / len(values)) if values else None

    timed_out_runs = sorted(
        result.run_id
        for result in run_results
        if normalize_run_status(result.status) == RUN_STATUS_TIMED_OUT
    )
    skipped_runs = sorted(
        result.run_id
        for result in run_results
        if normalize_run_status(result.status) == RUN_STATUS_SKIPPED_DUE_TO_POLICY
    )

    if isinstance(required_evidence_status, dict) and not bool(
        required_evidence_status.get("valid", False)
    ):
        return {
            "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
            "comparison_id": comparison.comparison_id,
            "comparison_version": comparison.comparison_version,
            "decision_status": ComparisonDecisionStatus.INVALID_COMPARISON.value,
            "primary_metric": primary_metric,
            "metric_policy_effective": _metric_policy_payload(metric_policy_effective),
            "variant_scores": variant_scores,
            "selected_variant": None,
            "reason": "required_evidence_package_not_satisfied",
            "required_evidence_status": required_evidence_status,
            "paired_model_comparisons": paired_comparison_payload or {"pairs": []},
            "control_results_summary": {},
        }

    if decision_policy.require_all_runs_completed:
        non_completed = [
            result.run_id
            for result in run_results
            if normalize_run_status(result.status) != RUN_STATUS_SUCCESS
        ]
        if missing_runs or non_completed:
            reason = "not_all_runs_completed"
            if timed_out_runs:
                reason = "required_runs_timed_out"
            elif skipped_runs:
                reason = "required_runs_skipped_due_to_policy"
            return {
                "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
                "comparison_id": comparison.comparison_id,
                "comparison_version": comparison.comparison_version,
                "decision_status": ComparisonDecisionStatus.INVALID_COMPARISON.value,
                "primary_metric": primary_metric,
                "metric_policy_effective": _metric_policy_payload(metric_policy_effective),
                "variant_scores": variant_scores,
                "selected_variant": None,
                "reason": reason,
                "missing_runs": sorted(missing_runs),
                "non_completed_runs": sorted(non_completed),
                "timed_out_runs": timed_out_runs,
                "skipped_due_to_policy_runs": skipped_runs,
                "control_results_summary": {},
            }

    required_variants_timed_out = sorted(
        variant_id
        for variant_id, payload in variant_scores.items()
        if int(payload.get("n_timed_out", 0)) > 0 and int(payload.get("n_success", 0)) == 0
    )
    if required_variants_timed_out:
        return {
            "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
            "comparison_id": comparison.comparison_id,
            "comparison_version": comparison.comparison_version,
            "decision_status": ComparisonDecisionStatus.INVALID_COMPARISON.value,
            "primary_metric": primary_metric,
            "metric_policy_effective": _metric_policy_payload(metric_policy_effective),
            "variant_scores": variant_scores,
            "selected_variant": None,
            "reason": "required_variants_timed_out",
            "required_variants_timed_out": required_variants_timed_out,
            "timed_out_runs": timed_out_runs,
            "control_results_summary": {},
        }

    if decision_policy.invalid_on_missing_metrics and missing_metric_runs:
        return {
            "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
            "comparison_id": comparison.comparison_id,
            "comparison_version": comparison.comparison_version,
            "decision_status": ComparisonDecisionStatus.INVALID_COMPARISON.value,
            "primary_metric": primary_metric,
            "metric_policy_effective": _metric_policy_payload(metric_policy_effective),
            "variant_scores": variant_scores,
            "selected_variant": None,
            "reason": "missing_primary_metric_values",
            "missing_metric_runs": sorted(missing_metric_runs),
            "control_results_summary": {},
        }

    candidate_scores = {
        variant_id: float(payload["mean_primary_metric"])
        for variant_id, payload in variant_scores.items()
        if payload.get("mean_primary_metric") is not None
    }
    if not candidate_scores:
        return {
            "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
            "comparison_id": comparison.comparison_id,
            "comparison_version": comparison.comparison_version,
            "decision_status": ComparisonDecisionStatus.INVALID_COMPARISON.value,
            "primary_metric": primary_metric,
            "metric_policy_effective": _metric_policy_payload(metric_policy_effective),
            "variant_scores": variant_scores,
            "selected_variant": None,
            "reason": "no_variant_scores",
            "control_results_summary": {},
        }

    control_results_summary: dict[str, Any] = {
        "require_permutation_control_pass": bool(decision_policy.require_permutation_control_pass),
        "permutation_p_value_threshold": float(decision_policy.permutation_p_value_threshold),
        "require_significant_win": bool(
            comparison.evidence_policy.paired_comparisons.require_significant_win
        ),
        "variants": {},
    }
    if decision_policy.require_permutation_control_pass:
        filtered_scores: dict[str, float] = {}
        for variant_id, score in candidate_scores.items():
            p_values = [
                float(value) for value in variant_scores[variant_id].get("permutation_p_values", [])
            ]
            if not p_values:
                control_results_summary["variants"][variant_id] = {
                    "status": "missing_permutation_control",
                    "max_p_value": None,
                }
                continue
            max_p_value = max(p_values)
            passed = max_p_value <= float(decision_policy.permutation_p_value_threshold)
            control_results_summary["variants"][variant_id] = {
                "status": "passed" if passed else "failed",
                "max_p_value": float(max_p_value),
            }
            if passed:
                filtered_scores[variant_id] = score
        if not filtered_scores:
            return {
                "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
                "comparison_id": comparison.comparison_id,
                "comparison_version": comparison.comparison_version,
                "decision_status": ComparisonDecisionStatus.INCONCLUSIVE.value,
                "primary_metric": primary_metric,
                "metric_policy_effective": _metric_policy_payload(metric_policy_effective),
                "variant_scores": variant_scores,
                "selected_variant": None,
                "reason": "controls_not_passed",
                "control_results_summary": control_results_summary,
            }
        candidate_scores = filtered_scores

    best_score = max(candidate_scores.values())
    tied_variants = [
        variant_id
        for variant_id, score in candidate_scores.items()
        if abs(score - best_score) <= float(decision_policy.tie_tolerance)
    ]
    if len(tied_variants) > 1:
        return {
            "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
            "comparison_id": comparison.comparison_id,
            "comparison_version": comparison.comparison_version,
            "decision_status": ComparisonDecisionStatus.INCONCLUSIVE.value,
            "primary_metric": primary_metric,
            "metric_policy_effective": _metric_policy_payload(metric_policy_effective),
            "variant_scores": variant_scores,
            "selected_variant": None,
            "reason": "tie_within_tolerance",
            "tie_variants": sorted(tied_variants),
            "control_results_summary": control_results_summary,
        }

    selected_variant = tied_variants[0]

    if bool(comparison.evidence_policy.paired_comparisons.require_significant_win):
        pair_rows = (
            paired_comparison_payload.get("pairs", [])
            if isinstance(paired_comparison_payload, dict)
            else []
        )
        if not isinstance(pair_rows, list):
            pair_rows = []
        comparison_outcomes: dict[str, bool] = {}
        for row in pair_rows:
            if not isinstance(row, dict):
                continue
            left_variant = str(row.get("left_variant_id", ""))
            right_variant = str(row.get("right_variant_id", ""))
            if selected_variant not in {left_variant, right_variant}:
                continue
            significant = bool(row.get("significant", False))
            observed_diff = _safe_float(row.get("observed_mean_difference"))
            if observed_diff is None:
                comparison_outcomes[
                    right_variant if selected_variant == left_variant else left_variant
                ] = False
                continue
            if selected_variant == left_variant:
                comparison_outcomes[right_variant] = bool(significant and observed_diff > 0.0)
            else:
                comparison_outcomes[left_variant] = bool(significant and observed_diff < 0.0)
        other_variants = sorted(set(candidate_scores.keys()) - {selected_variant})
        missing_or_non_significant = [
            variant_id
            for variant_id in other_variants
            if not bool(comparison_outcomes.get(variant_id, False))
        ]
        if missing_or_non_significant:
            return {
                "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
                "comparison_id": comparison.comparison_id,
                "comparison_version": comparison.comparison_version,
                "decision_status": ComparisonDecisionStatus.INCONCLUSIVE.value,
                "primary_metric": primary_metric,
                "metric_policy_effective": _metric_policy_payload(metric_policy_effective),
                "variant_scores": variant_scores,
                "selected_variant": None,
                "reason": "selected_variant_lacks_significant_paired_wins",
                "non_significant_or_missing_pairs": missing_or_non_significant,
                "required_evidence_status": required_evidence_status,
                "paired_model_comparisons": paired_comparison_payload or {"pairs": []},
                "control_results_summary": control_results_summary,
            }

    if decision_policy.block_on_subgroup_failures:
        variant_results = [row for row in run_results if row.variant_id == selected_variant]
        subgroup_ok, subgroup_reason = _subgroup_reports_ok(variant_results)
        if not subgroup_ok:
            status = (
                ComparisonDecisionStatus.INVALID_COMPARISON.value
                if decision_policy.invalid_on_missing_metrics
                else ComparisonDecisionStatus.INCONCLUSIVE.value
            )
            return {
                "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
                "comparison_id": comparison.comparison_id,
                "comparison_version": comparison.comparison_version,
                "decision_status": status,
                "primary_metric": primary_metric,
                "metric_policy_effective": _metric_policy_payload(metric_policy_effective),
                "variant_scores": variant_scores,
                "selected_variant": None,
                "reason": "subgroup_reporting_failure",
                "subgroup_reason": subgroup_reason,
                "control_results_summary": control_results_summary,
            }

    return {
        "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
        "comparison_id": comparison.comparison_id,
        "comparison_version": comparison.comparison_version,
        "decision_status": ComparisonDecisionStatus.WINNER_SELECTED.value,
        "primary_metric": primary_metric,
        "metric_policy_effective": _metric_policy_payload(metric_policy_effective),
        "variant_scores": variant_scores,
        "selected_variant": selected_variant,
        "reason": "best_primary_metric",
        "required_evidence_status": required_evidence_status,
        "paired_model_comparisons": paired_comparison_payload or {"pairs": []},
        "control_results_summary": control_results_summary,
    }
