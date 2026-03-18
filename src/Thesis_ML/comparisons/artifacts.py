from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.comparisons.decision import build_comparison_decision
from Thesis_ML.comparisons.models import (
    ComparisonRunResult,
    ComparisonSpec,
    CompiledComparisonManifest,
)
from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.methodology import MethodologyPolicyName
from Thesis_ML.config.metric_policy import (
    enforce_primary_metric_alignment,
    resolve_effective_metric_policy,
)
from Thesis_ML.experiments.evidence_statistics import (
    aggregate_repeated_runs,
    grouped_bootstrap_percentile_interval,
    paired_sign_flip_permutation,
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
        "data_policy_effective": compiled_manifest.data_policy.model_dump(mode="json"),
        "external_validation_status": {
            "enabled": bool(compiled_manifest.data_policy.external_validation.enabled),
            "mode": str(compiled_manifest.data_policy.external_validation.mode),
            "require_compatible": bool(
                compiled_manifest.data_policy.external_validation.require_compatible
            ),
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
                "evidence_run_role": spec.evidence_run_role.value,
                "repeat_id": int(spec.repeat_id),
                "repeat_count": int(spec.repeat_count),
                "base_run_id": str(spec.base_run_id),
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


def _completed_primary_rows(report_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in report_rows
        if str(row.get("status", "")).strip().lower() == "completed"
        and str(row.get("evidence_run_role", "")).strip() == "primary"
    ]


def _build_repeated_run_outputs(
    *,
    report_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    repeated_rows, repeated_summary_frame = aggregate_repeated_runs(
        report_rows,
        metric_key="primary_metric_value",
        group_keys=["comparison_id", "comparison_version", "variant_id", "model"],
    )
    summary_payload = {
        "n_rows": int(repeated_rows.shape[0]),
        "n_groups": int(repeated_summary_frame.shape[0]),
        "groups": repeated_summary_frame.to_dict(orient="records"),
    }
    return repeated_rows, summary_payload


def _build_confidence_interval_outputs(
    *,
    comparison: ComparisonSpec,
    report_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ci_policy = comparison.evidence_policy.confidence_intervals
    interval_rows: list[dict[str, Any]] = []
    for variant_id in sorted({str(row.get("variant_id", "")) for row in report_rows}):
        variant_rows = [row for row in report_rows if str(row.get("variant_id", "")) == variant_id]
        if not variant_rows:
            continue
        interval = grouped_bootstrap_percentile_interval(
            variant_rows,
            value_key="primary_metric_value",
            group_key="base_run_id",
            confidence_level=float(ci_policy.confidence_level),
            n_bootstrap=int(ci_policy.n_bootstrap),
            seed=int(ci_policy.seed),
        )
        interval_rows.append(
            {
                "comparison_id": comparison.comparison_id,
                "comparison_version": comparison.comparison_version,
                "variant_id": variant_id,
                "primary_metric": comparison.metric_policy.primary_metric,
                **interval,
            }
        )
    payload = {
        "method": str(ci_policy.method.value),
        "confidence_level": float(ci_policy.confidence_level),
        "n_bootstrap": int(ci_policy.n_bootstrap),
        "seed": int(ci_policy.seed),
        "intervals": interval_rows,
    }
    return interval_rows, payload


def _build_paired_comparison_outputs(
    *,
    comparison: ComparisonSpec,
    report_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = comparison.evidence_policy.paired_comparisons
    by_variant_key: dict[str, dict[tuple[Any, ...], float]] = {}
    for row in report_rows:
        variant_id = str(row.get("variant_id", ""))
        base_key = (
            int(row.get("repeat_id", 1)),
            str(row.get("subject", "")),
            str(row.get("train_subject", "")),
            str(row.get("test_subject", "")),
            str(row.get("cv_mode", "")),
        )
        raw_metric_value = row.get("primary_metric_value")
        if raw_metric_value is None:
            continue
        try:
            metric_value = float(raw_metric_value)
        except (TypeError, ValueError):
            continue
        by_variant_key.setdefault(variant_id, {})[base_key] = metric_value

    variant_ids = sorted(by_variant_key.keys())
    pair_rows: list[dict[str, Any]] = []
    for idx, left_variant in enumerate(variant_ids):
        for right_variant in variant_ids[idx + 1 :]:
            left_map = by_variant_key[left_variant]
            right_map = by_variant_key[right_variant]
            shared_keys = sorted(set(left_map.keys()) & set(right_map.keys()))
            paired_rows = [
                {"left_metric": left_map[key], "right_metric": right_map[key]}
                for key in shared_keys
            ]
            test_result = paired_sign_flip_permutation(
                paired_rows,
                left_key="left_metric",
                right_key="right_metric",
                n_permutations=int(policy.n_permutations),
                alpha=float(policy.alpha),
                seed=int(comparison.scientific_contract.seed_policy.global_seed),
            )
            pair_rows.append(
                {
                    "comparison_id": comparison.comparison_id,
                    "comparison_version": comparison.comparison_version,
                    "left_variant_id": left_variant,
                    "right_variant_id": right_variant,
                    "primary_metric": comparison.metric_policy.primary_metric,
                    "n_shared_pairs": int(len(shared_keys)),
                    **test_result,
                }
            )
    payload = {
        "method": str(policy.method.value),
        "n_permutations": int(policy.n_permutations),
        "alpha": float(policy.alpha),
        "require_significant_win": bool(policy.require_significant_win),
        "pairs": pair_rows,
    }
    return pair_rows, payload


def _required_evidence_status(
    *,
    comparison: ComparisonSpec,
    completed_rows: list[dict[str, Any]],
    paired_payload: dict[str, Any],
) -> dict[str, Any]:
    required = comparison.evidence_policy.required_package
    requires_dummy = bool(required.require_dummy_baseline)
    requires_permutation = bool(required.require_permutation_control)
    requires_untuned = bool(required.require_untuned_baseline_if_tuning)

    dummy_present = any(str(row.get("model", "")) == "dummy" for row in completed_rows)
    permutation_present = any(bool(row.get("permutation_enabled")) for row in completed_rows)
    untuned_present = any(
        str(row.get("evidence_run_role", "")) == "untuned_baseline" for row in completed_rows
    )
    valid = bool(
        (not requires_dummy or dummy_present)
        and (not requires_permutation or permutation_present)
        and (
            not requires_untuned
            or comparison.methodology_policy.policy_name != MethodologyPolicyName.GROUPED_NESTED_TUNING
            or untuned_present
        )
    )
    return {
        "require_dummy_baseline": requires_dummy,
        "require_permutation_control": requires_permutation,
        "require_untuned_baseline_if_tuning": requires_untuned,
        "dummy_baseline_present": bool(dummy_present),
        "permutation_control_present": bool(permutation_present),
        "untuned_baseline_present": bool(untuned_present),
        "paired_comparisons_available": bool(paired_payload.get("pairs")),
        "valid": bool(valid),
    }


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
    repeated_run_metrics_path = comparison_dir / "repeated_run_metrics.csv"
    repeated_run_summary_path = comparison_dir / "repeated_run_summary.json"
    confidence_intervals_path = comparison_dir / "confidence_intervals.json"
    metric_intervals_path = comparison_dir / "metric_intervals.csv"
    paired_model_comparisons_path = comparison_dir / "paired_model_comparisons.json"
    paired_model_comparisons_csv_path = comparison_dir / "paired_model_comparisons.csv"
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

    report_rows = _report_index_rows(compiled_manifest, run_results)
    with report_index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(report_rows[0].keys()) if report_rows else ["run_id"],
        )
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)

    primary_rows = _completed_primary_rows(report_rows)
    completed_rows = [
        row for row in report_rows if str(row.get("status", "")).strip().lower() == "completed"
    ]
    repeated_run_rows, repeated_summary_payload = _build_repeated_run_outputs(
        report_rows=primary_rows
    )
    repeated_run_rows.to_csv(repeated_run_metrics_path, index=False)
    _write_json(repeated_run_summary_path, repeated_summary_payload)

    interval_rows, confidence_payload = _build_confidence_interval_outputs(
        comparison=comparison,
        report_rows=primary_rows,
    )
    _write_json(confidence_intervals_path, confidence_payload)
    pd.DataFrame(interval_rows).to_csv(metric_intervals_path, index=False)

    paired_rows, paired_payload = _build_paired_comparison_outputs(
        comparison=comparison,
        report_rows=primary_rows,
    )
    _write_json(paired_model_comparisons_path, paired_payload)
    pd.DataFrame(paired_rows).to_csv(paired_model_comparisons_csv_path, index=False)

    required_evidence = _required_evidence_status(
        comparison=comparison,
        completed_rows=completed_rows,
        paired_payload=paired_payload,
    )

    _write_json(comparison_json_path, comparison.model_dump(mode="json"))
    _write_json(compiled_manifest_path, compiled_manifest_payload)
    summary_payload = _comparison_summary(comparison, compiled_manifest, run_results)
    summary_payload["required_evidence_status"] = required_evidence
    summary_payload["paired_comparisons"] = {
        "path": str(paired_model_comparisons_path.resolve()),
        "n_pairs": int(len(paired_rows)),
    }
    summary_payload["confidence_intervals"] = {
        "path": str(confidence_intervals_path.resolve()),
        "n_intervals": int(len(interval_rows)),
    }
    _write_json(comparison_summary_path, summary_payload)
    _write_json(
        comparison_decision_path,
        build_comparison_decision(
            comparison=comparison,
            compiled_manifest=compiled_manifest,
            run_results=run_results,
            paired_comparison_payload=paired_payload,
            required_evidence_status=required_evidence,
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

    return {
        "comparison_json": str(comparison_json_path.resolve()),
        "compiled_comparison_manifest": str(compiled_manifest_path.resolve()),
        "comparison_summary": str(comparison_summary_path.resolve()),
        "comparison_decision": str(comparison_decision_path.resolve()),
        "execution_status": str(execution_status_path.resolve()),
        "repeated_run_metrics": str(repeated_run_metrics_path.resolve()),
        "repeated_run_summary": str(repeated_run_summary_path.resolve()),
        "confidence_intervals": str(confidence_intervals_path.resolve()),
        "metric_intervals": str(metric_intervals_path.resolve()),
        "paired_model_comparisons": str(paired_model_comparisons_path.resolve()),
        "paired_model_comparisons_csv": str(paired_model_comparisons_csv_path.resolve()),
        "report_index": str(report_index_path.resolve()),
    }
