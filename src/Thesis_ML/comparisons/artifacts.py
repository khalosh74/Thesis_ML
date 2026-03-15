from __future__ import annotations

import csv
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


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


def _comparison_summary(
    comparison: ComparisonSpec,
    compiled_manifest: CompiledComparisonManifest,
    run_results: list[ComparisonRunResult],
) -> dict[str, Any]:
    by_variant: dict[str, dict[str, int]] = {}
    for variant_id in compiled_manifest.variant_ids:
        by_variant[variant_id] = {"planned": 0, "completed": 0, "failed": 0}
    for result in run_results:
        variant_counts = by_variant.setdefault(
            result.variant_id,
            {"planned": 0, "completed": 0, "failed": 0},
        )
        variant_counts[result.status] = int(variant_counts.get(result.status, 0)) + 1
    return {
        "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
        "comparison_id": comparison.comparison_id,
        "comparison_version": comparison.comparison_version,
        "comparison_dimension": comparison.comparison_dimension,
        "methodology_policy_name": comparison.methodology_policy.policy_name.value,
        "primary_metric": comparison.metric_policy.primary_metric,
        "variant_status_counts": by_variant,
        "n_runs": int(len(run_results)),
    }


def _subgroup_reports_ok(variant_results: list[ComparisonRunResult]) -> tuple[bool, str | None]:
    for result in variant_results:
        if result.status != "completed" or not result.report_dir:
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
) -> dict[str, Any]:
    primary_metric = comparison.decision_policy.primary_metric
    decision_policy = comparison.decision_policy
    result_by_run_id = {result.run_id: result for result in run_results}
    variant_scores: dict[str, dict[str, Any]] = {
        variant_id: {
            "n_runs": 0,
            "n_completed": 0,
            "n_failed": 0,
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
        bucket = variant_scores.setdefault(
            spec.variant_id,
            {
                "n_runs": 0,
                "n_completed": 0,
                "n_failed": 0,
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
        if result.status == "completed":
            bucket["n_completed"] = int(bucket["n_completed"]) + 1
            metrics_payload = _load_metrics_payload(result)
            metric_value = _safe_float(metrics_payload.get(primary_metric))
            if metric_value is None:
                metric_value = _safe_float(metrics_payload.get("primary_metric_value"))
            if metric_value is None:
                missing_metric_runs.append(spec.run_id)
            else:
                bucket["primary_metric_values"].append(metric_value)
            permutation_payload = metrics_payload.get("permutation_test")
            if isinstance(permutation_payload, dict):
                p_value = _safe_float(permutation_payload.get("p_value"))
                if p_value is not None:
                    bucket["permutation_p_values"].append(p_value)
        elif result.status == "failed":
            bucket["n_failed"] = int(bucket["n_failed"]) + 1
        else:
            bucket["n_planned"] = int(bucket["n_planned"]) + 1

    for bucket in variant_scores.values():
        values = [float(value) for value in bucket["primary_metric_values"]]
        bucket["mean_primary_metric"] = float(sum(values) / len(values)) if values else None

    if decision_policy.require_all_runs_completed:
        non_completed = [
            result.run_id for result in run_results if result.status in {"failed", "planned"}
        ]
        if missing_runs or non_completed:
            return {
                "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
                "comparison_id": comparison.comparison_id,
                "comparison_version": comparison.comparison_version,
                "decision_status": ComparisonDecisionStatus.INVALID_COMPARISON.value,
                "primary_metric": primary_metric,
                "variant_scores": variant_scores,
                "selected_variant": None,
                "reason": "not_all_runs_completed",
                "missing_runs": sorted(missing_runs),
                "non_completed_runs": sorted(non_completed),
                "control_results_summary": {},
            }

    if decision_policy.invalid_on_missing_metrics and missing_metric_runs:
        return {
            "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
            "comparison_id": comparison.comparison_id,
            "comparison_version": comparison.comparison_version,
            "decision_status": ComparisonDecisionStatus.INVALID_COMPARISON.value,
            "primary_metric": primary_metric,
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
            "variant_scores": variant_scores,
            "selected_variant": None,
            "reason": "no_variant_scores",
            "control_results_summary": {},
        }

    control_results_summary: dict[str, Any] = {
        "require_permutation_control_pass": bool(decision_policy.require_permutation_control_pass),
        "permutation_p_value_threshold": float(decision_policy.permutation_p_value_threshold),
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
            "variant_scores": variant_scores,
            "selected_variant": None,
            "reason": "tie_within_tolerance",
            "tie_variants": sorted(tied_variants),
            "control_results_summary": control_results_summary,
        }

    selected_variant = tied_variants[0]
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
        "variant_scores": variant_scores,
        "selected_variant": selected_variant,
        "reason": "best_primary_metric",
        "control_results_summary": control_results_summary,
    }


def _execution_status_payload(
    comparison: ComparisonSpec,
    compiled_manifest: CompiledComparisonManifest,
    run_results: list[ComparisonRunResult],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
        "comparison_id": comparison.comparison_id,
        "comparison_version": comparison.comparison_version,
        "comparison_schema_version": comparison.comparison_schema_version,
        "compiled_schema_version": compiled_manifest.compiled_schema_version,
        "dry_run": bool(dry_run),
        "runs": [result.model_dump(mode="json") for result in run_results],
    }


def _report_index_rows(
    compiled_manifest: CompiledComparisonManifest,
    run_results: list[ComparisonRunResult],
) -> list[dict[str, Any]]:
    result_by_run_id = {result.run_id: result for result in run_results}
    rows: list[dict[str, Any]] = []
    for spec in compiled_manifest.runs:
        result = result_by_run_id.get(spec.run_id)
        metrics = result.metrics if result and result.metrics else {}
        rows.append(
            {
                "run_id": spec.run_id,
                "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
                "comparison_id": spec.comparison_id,
                "comparison_version": spec.comparison_version,
                "variant_id": spec.variant_id,
                "claim_ids": "|".join(spec.claim_ids),
                "status": result.status if result is not None else "planned",
                "target": spec.target,
                "model": spec.model,
                "cv_mode": spec.cv_mode,
                "subject": spec.subject,
                "train_subject": spec.train_subject,
                "test_subject": spec.test_subject,
                "seed": int(spec.seed),
                "primary_metric": spec.primary_metric,
                "methodology_policy_name": spec.methodology_policy_name.value,
                "class_weight_policy": spec.class_weight_policy.value,
                "tuning_enabled": bool(spec.tuning_enabled),
                "permutation_enabled": bool(spec.controls.permutation_enabled),
                "n_permutations": int(spec.controls.n_permutations),
                "permutation_metric": spec.controls.permutation_metric,
                "dummy_baseline_enabled": bool(spec.controls.dummy_baseline_enabled),
                "interpretability_enabled": bool(spec.interpretability_enabled),
                "subgroup_reporting_enabled": bool(spec.subgroup_reporting_enabled),
                "canonical_run": bool(spec.canonical_run),
                "report_dir": result.report_dir if result is not None else None,
                "config_path": result.config_path if result is not None else None,
                "metrics_path": result.metrics_path if result is not None else None,
                "error": result.error if result is not None else None,
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "accuracy": metrics.get("accuracy"),
                "primary_metric_value": metrics.get("primary_metric_value"),
            }
        )
    return rows


def write_comparison_artifacts(
    *,
    comparison: ComparisonSpec,
    compiled_manifest: CompiledComparisonManifest,
    run_results: list[ComparisonRunResult],
    output_dir: Path | str,
    dry_run: bool,
) -> dict[str, str]:
    comparison_dir = Path(output_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_json_path = comparison_dir / "comparison.json"
    compiled_manifest_path = comparison_dir / "compiled_comparison_manifest.json"
    comparison_summary_path = comparison_dir / "comparison_summary.json"
    comparison_decision_path = comparison_dir / "comparison_decision.json"
    execution_status_path = comparison_dir / "execution_status.json"
    report_index_path = comparison_dir / "report_index.csv"

    _write_json(comparison_json_path, comparison.model_dump(mode="json"))
    _write_json(compiled_manifest_path, compiled_manifest.model_dump(mode="json"))
    _write_json(
        comparison_summary_path,
        _comparison_summary(comparison, compiled_manifest, run_results),
    )
    _write_json(
        comparison_decision_path,
        build_comparison_decision(
            comparison=comparison,
            compiled_manifest=compiled_manifest,
            run_results=run_results,
        ),
    )
    _write_json(
        execution_status_path,
        _execution_status_payload(
            comparison=comparison,
            compiled_manifest=compiled_manifest,
            run_results=run_results,
            dry_run=dry_run,
        ),
    )

    report_rows = _report_index_rows(compiled_manifest, run_results)
    with report_index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(report_rows[0].keys()) if report_rows else ["run_id"],
        )
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)

    return {
        "comparison_json": str(comparison_json_path.resolve()),
        "compiled_comparison_manifest": str(compiled_manifest_path.resolve()),
        "comparison_summary": str(comparison_summary_path.resolve()),
        "comparison_decision": str(comparison_decision_path.resolve()),
        "execution_status": str(execution_status_path.resolve()),
        "report_index": str(report_index_path.resolve()),
    }
