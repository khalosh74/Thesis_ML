from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from Thesis_ML.comparisons.decision import build_comparison_decision
from Thesis_ML.comparisons.models import (
    ComparisonRunResult,
    ComparisonSpec,
    CompiledComparisonManifest,
)
from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.metric_policy import (
    enforce_primary_metric_alignment,
    resolve_effective_metric_policy,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _relative_path(path_text: str | None) -> str | None:
    if not path_text:
        return None
    path_obj = Path(path_text)
    try:
        return str(path_obj.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path_obj)


def _comparison_summary(
    comparison: ComparisonSpec,
    compiled_manifest: CompiledComparisonManifest,
    run_results: list[ComparisonRunResult],
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
        context="locked comparison",
    )
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
        "metric_policy_effective": {
            "primary_metric": metric_policy_effective.primary_metric,
            "secondary_metrics": list(metric_policy_effective.secondary_metrics),
            "decision_metric": metric_policy_effective.decision_metric,
            "tuning_metric": metric_policy_effective.tuning_metric,
            "permutation_metric": metric_policy_effective.permutation_metric,
            "higher_is_better": bool(metric_policy_effective.higher_is_better),
        },
        "variant_status_counts": by_variant,
        "n_runs": int(len(run_results)),
    }


def _execution_status_payload(
    comparison: ComparisonSpec,
    compiled_manifest: CompiledComparisonManifest,
    run_results: list[ComparisonRunResult],
    *,
    dry_run: bool,
    stage_timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
        "comparison_id": comparison.comparison_id,
        "comparison_version": comparison.comparison_version,
        "comparison_schema_version": comparison.comparison_schema_version,
        "compiled_schema_version": compiled_manifest.compiled_schema_version,
        "dry_run": bool(dry_run),
        "runs": [result.model_dump(mode="json") for result in run_results],
    }
    if stage_timings:
        payload["stage_timings_seconds"] = {
            key: round(float(value), 6) for key, value in stage_timings.items()
        }
    return payload


def _report_index_rows(
    compiled_manifest: CompiledComparisonManifest,
    run_results: list[ComparisonRunResult],
) -> list[dict[str, Any]]:
    result_by_run_id = {result.run_id: result for result in run_results}
    metric_policy_cache: dict[
        tuple[str, tuple[str, ...], str, str],
        dict[str, Any],
    ] = {}
    rows: list[dict[str, Any]] = []
    for spec in compiled_manifest.runs:
        cache_key = (
            spec.primary_metric,
            tuple(compiled_manifest.metric_policy.secondary_metrics),
            compiled_manifest.decision_policy.primary_metric,
            spec.controls.permutation_metric,
        )
        metric_policy_effective = metric_policy_cache.get(cache_key)
        if metric_policy_effective is None:
            resolved_metric_policy = resolve_effective_metric_policy(
                primary_metric=spec.primary_metric,
                secondary_metrics=compiled_manifest.metric_policy.secondary_metrics,
                decision_metric=compiled_manifest.decision_policy.primary_metric,
                tuning_metric=spec.primary_metric,
                permutation_metric=spec.controls.permutation_metric,
            )
            metric_policy_effective = {
                "primary_metric": resolved_metric_policy.primary_metric,
                "secondary_metrics": list(resolved_metric_policy.secondary_metrics),
                "decision_metric": resolved_metric_policy.decision_metric,
                "tuning_metric": resolved_metric_policy.tuning_metric,
                "permutation_metric": resolved_metric_policy.permutation_metric,
                "higher_is_better": bool(resolved_metric_policy.higher_is_better),
            }
            metric_policy_cache[cache_key] = metric_policy_effective
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
                "decision_metric": metric_policy_effective["decision_metric"],
                "tuning_metric": metric_policy_effective["tuning_metric"],
                "methodology_policy_name": spec.methodology_policy_name.value,
                "class_weight_policy": spec.class_weight_policy.value,
                "tuning_enabled": bool(spec.tuning_enabled),
                "permutation_enabled": bool(spec.controls.permutation_enabled),
                "n_permutations": int(spec.controls.n_permutations),
                "permutation_metric": spec.controls.permutation_metric,
                "higher_is_better": bool(metric_policy_effective["higher_is_better"]),
                "dummy_baseline_enabled": bool(spec.controls.dummy_baseline_enabled),
                "interpretability_enabled": bool(spec.interpretability_enabled),
                "subgroup_reporting_enabled": bool(spec.subgroup_reporting_enabled),
                "canonical_run": bool(spec.canonical_run),
                "report_dir": result.report_dir if result is not None else None,
                "report_dir_relative": (
                    _relative_path(result.report_dir) if result is not None else None
                ),
                "config_path": result.config_path if result is not None else None,
                "config_path_relative": (
                    _relative_path(result.config_path) if result is not None else None
                ),
                "metrics_path": result.metrics_path if result is not None else None,
                "metrics_path_relative": (
                    _relative_path(result.metrics_path) if result is not None else None
                ),
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
    stage_timings: dict[str, float] | None = None,
) -> dict[str, str]:
    comparison_dir = Path(output_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_json_path = comparison_dir / "comparison.json"
    compiled_manifest_path = comparison_dir / "compiled_comparison_manifest.json"
    comparison_summary_path = comparison_dir / "comparison_summary.json"
    comparison_decision_path = comparison_dir / "comparison_decision.json"
    execution_status_path = comparison_dir / "execution_status.json"
    report_index_path = comparison_dir / "report_index.csv"

    metric_policy_effective = enforce_primary_metric_alignment(
        resolve_effective_metric_policy(
            primary_metric=compiled_manifest.metric_policy.primary_metric,
            secondary_metrics=compiled_manifest.metric_policy.secondary_metrics,
            decision_metric=compiled_manifest.decision_policy.primary_metric,
            tuning_metric=compiled_manifest.metric_policy.primary_metric,
            permutation_metric=(
                comparison.control_policy.permutation_metric
                or compiled_manifest.metric_policy.primary_metric
            ),
        ),
        context="locked comparison artifacts",
    )
    compiled_manifest_payload = compiled_manifest.model_dump(mode="json")
    compiled_manifest_payload["metric_policy_effective"] = {
        "primary_metric": metric_policy_effective.primary_metric,
        "secondary_metrics": list(metric_policy_effective.secondary_metrics),
        "decision_metric": metric_policy_effective.decision_metric,
        "tuning_metric": metric_policy_effective.tuning_metric,
        "permutation_metric": metric_policy_effective.permutation_metric,
        "higher_is_better": bool(metric_policy_effective.higher_is_better),
    }

    _write_json(comparison_json_path, comparison.model_dump(mode="json"))
    _write_json(compiled_manifest_path, compiled_manifest_payload)
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
            stage_timings=stage_timings,
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
