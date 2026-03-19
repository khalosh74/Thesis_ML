from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from Thesis_ML.comparisons.artifacts import write_comparison_artifacts
from Thesis_ML.comparisons.compiler import compile_comparison
from Thesis_ML.comparisons.models import (
    ComparisonRunResult,
    ComparisonSpec,
    ComparisonStatus,
    CompiledComparisonManifest,
    CompiledComparisonRunSpec,
)
from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.metric_policy import resolve_effective_metric_policy
from Thesis_ML.experiments.errors import exception_failure_payload
from Thesis_ML.experiments.execution_policy import read_run_status
from Thesis_ML.experiments.run_states import (
    RUN_STATUS_FAILED,
    RUN_STATUS_PLANNED,
    RUN_STATUS_SKIPPED_DUE_TO_POLICY,
    RUN_STATUS_SUCCESS,
    RUN_STATUS_TIMED_OUT,
    normalize_run_status,
)
from Thesis_ML.experiments.runtime_policies import (
    resolve_run_timeout_policy,
    validate_official_context_payload,
)
from Thesis_ML.experiments.timeout_watchdog import execute_run_with_timeout_watchdog
from Thesis_ML.verification.official_artifacts import verify_official_artifacts


def _comparison_output_dir(comparison: ComparisonSpec, reports_root: Path | str) -> Path:
    root = Path(reports_root)
    return root / "comparison_runs" / f"{comparison.comparison_id}__{comparison.comparison_version}"


def _comparison_context_payload(
    spec: CompiledComparisonRunSpec,
    *,
    secondary_metrics: list[str],
    decision_metric: str,
    required_run_metadata_fields: list[str],
    evidence_policy_payload: dict[str, Any],
    data_policy_payload: dict[str, Any],
) -> dict[str, Any]:
    metric_policy = resolve_effective_metric_policy(
        primary_metric=spec.primary_metric,
        secondary_metrics=secondary_metrics,
        decision_metric=decision_metric,
        tuning_metric=spec.primary_metric,
        permutation_metric=spec.controls.permutation_metric,
    )
    payload = {
        "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
        "comparison_id": spec.comparison_id,
        "comparison_version": spec.comparison_version,
        "variant_id": spec.variant_id,
        "claim_ids": list(spec.claim_ids),
        "methodology_policy_name": spec.methodology_policy_name.value,
        "class_weight_policy": spec.class_weight_policy.value,
        "tuning_enabled": bool(spec.tuning_enabled),
        "model_cost_tier": spec.model_cost_tier.value,
        "projected_runtime_seconds": int(spec.projected_runtime_seconds),
        "tuning_search_space_id": spec.tuning_search_space_id,
        "tuning_search_space_version": spec.tuning_search_space_version,
        "tuning_inner_cv_scheme": spec.tuning_inner_cv_scheme,
        "tuning_inner_group_field": spec.tuning_inner_group_field,
        "subgroup_reporting_enabled": bool(spec.subgroup_reporting_enabled),
        "subgroup_dimensions": list(spec.subgroup_dimensions),
        "subgroup_min_samples_per_group": int(spec.subgroup_min_samples_per_group),
        "artifact_requirements": list(spec.artifact_requirements),
        "required_run_metadata_fields": list(required_run_metadata_fields),
        "data_policy": dict(data_policy_payload),
        "repeat_id": int(spec.repeat_id),
        "repeat_count": int(spec.repeat_count),
        "base_run_id": str(spec.base_run_id),
        "evidence_run_role": spec.evidence_run_role.value,
        "evidence_policy": dict(evidence_policy_payload),
        "primary_metric": spec.primary_metric,
        "metric_policy": {
            "primary_metric": metric_policy.primary_metric,
            "secondary_metrics": list(metric_policy.secondary_metrics),
            "decision_metric": metric_policy.decision_metric,
            "tuning_metric": metric_policy.tuning_metric,
            "permutation_metric": metric_policy.permutation_metric,
            "higher_is_better": bool(metric_policy.higher_is_better),
        },
        "controls": spec.controls.model_dump(mode="json"),
        "interpretability_enabled": bool(spec.interpretability_enabled),
    }
    return validate_official_context_payload(
        framework_mode=FrameworkMode.LOCKED_COMPARISON,
        context_name="comparison_context",
        context=payload,
    )


def _to_run_result_success(
    spec: CompiledComparisonRunSpec,
    run_payload: dict[str, Any],
) -> ComparisonRunResult:
    metrics_payload = run_payload.get("metrics", {}) if isinstance(run_payload, dict) else {}
    metrics: dict[str, float | int | str | bool | None | dict[str, Any]] = {}
    if isinstance(metrics_payload, dict):
        for key in (
            "balanced_accuracy",
            "macro_f1",
            "accuracy",
            "n_folds",
            "primary_metric_name",
            "primary_metric_value",
        ):
            value = metrics_payload.get(key)
            if isinstance(value, (float, int, str, bool)) or value is None:
                metrics[key] = value
        permutation_payload = metrics_payload.get("permutation_test")
        if isinstance(permutation_payload, dict):
            p_value = permutation_payload.get("p_value")
            if isinstance(p_value, (float, int)):
                metrics["permutation_p_value"] = float(p_value)
            metric_name = permutation_payload.get("metric_name")
            if isinstance(metric_name, str):
                metrics["permutation_metric_name"] = metric_name

    return ComparisonRunResult(
        run_id=spec.run_id,
        framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
        comparison_id=spec.comparison_id,
        comparison_version=spec.comparison_version,
        variant_id=spec.variant_id,
        status=RUN_STATUS_SUCCESS,
        report_dir=(str(run_payload.get("report_dir")) if run_payload.get("report_dir") else None),
        config_path=(
            str(run_payload.get("config_path")) if run_payload.get("config_path") else None
        ),
        metrics_path=(
            str(run_payload.get("metrics_path")) if run_payload.get("metrics_path") else None
        ),
        metrics=metrics,
    )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_last_updated_utc(
    *,
    run_dir: Path,
    status_payload: dict[str, Any] | None,
) -> str:
    if isinstance(status_payload, dict):
        value = str(status_payload.get("updated_at_utc", "")).strip()
        if value:
            return value
    refreshed = read_run_status(run_dir)
    if isinstance(refreshed, dict):
        value = str(refreshed.get("updated_at_utc", "")).strip()
        if value:
            return value
    if run_dir.exists():
        return datetime.fromtimestamp(run_dir.stat().st_mtime, tz=UTC).isoformat()
    return _utc_now_iso()


def _metrics_snapshot_from_file(
    metrics_path: Path | None,
) -> dict[str, float | int | str | bool | None | dict[str, Any]]:
    if metrics_path is None:
        return {}
    payload = _read_json_object(metrics_path)
    if not isinstance(payload, dict):
        return {}
    snapshot: dict[str, float | int | str | bool | None | dict[str, Any]] = {}
    for key in (
        "balanced_accuracy",
        "macro_f1",
        "accuracy",
        "n_folds",
        "primary_metric_name",
        "primary_metric_value",
    ):
        value = payload.get(key)
        if isinstance(value, (float, int, str, bool)) or value is None:
            snapshot[key] = value
    permutation_payload = payload.get("permutation_test")
    if isinstance(permutation_payload, dict):
        p_value = permutation_payload.get("p_value")
        if isinstance(p_value, (float, int)):
            snapshot["permutation_p_value"] = float(p_value)
        metric_name = permutation_payload.get("metric_name")
        if isinstance(metric_name, str):
            snapshot["permutation_metric_name"] = metric_name
    return snapshot


def _classify_existing_run(
    *,
    run_dir: Path,
) -> tuple[str, dict[str, Any] | None]:
    if not run_dir.exists():
        return ("missing", None)
    if not run_dir.is_dir():
        return ("partial_incomplete", None)
    status_payload = read_run_status(run_dir)
    if not isinstance(status_payload, dict):
        return ("partial_incomplete", None)
    normalized = normalize_run_status(str(status_payload.get("status", "")))
    if normalized in {
        RUN_STATUS_SUCCESS,
        RUN_STATUS_FAILED,
        RUN_STATUS_TIMED_OUT,
        RUN_STATUS_SKIPPED_DUE_TO_POLICY,
    }:
        return (normalized, status_payload)
    return ("partial_incomplete", status_payload)


def _new_reconciliation_counts(
    *,
    execution_mode: str,
    n_planned: int,
) -> dict[str, int | str]:
    return {
        "execution_mode": execution_mode,
        "n_planned": int(n_planned),
        "n_existing_success": 0,
        "n_existing_failed": 0,
        "n_existing_timed_out": 0,
        "n_existing_skipped_due_to_policy": 0,
        "n_existing_partial_incomplete": 0,
        "n_missing": 0,
        "n_rerun": 0,
        "n_reused": 0,
        "n_skipped_as_already_complete": 0,
    }


def _increment_existing_count(
    *,
    reconciliation_counts: dict[str, int | str],
    classification: str,
) -> None:
    key_map = {
        RUN_STATUS_SUCCESS: "n_existing_success",
        RUN_STATUS_FAILED: "n_existing_failed",
        RUN_STATUS_TIMED_OUT: "n_existing_timed_out",
        RUN_STATUS_SKIPPED_DUE_TO_POLICY: "n_existing_skipped_due_to_policy",
        "partial_incomplete": "n_existing_partial_incomplete",
    }
    key = key_map.get(classification)
    if key is None:
        return
    reconciliation_counts[key] = int(reconciliation_counts.get(key, 0)) + 1


def _reused_run_result(
    *,
    spec: CompiledComparisonRunSpec,
    status: str,
    status_payload: dict[str, Any] | None,
    run_dir: Path,
) -> ComparisonRunResult:
    normalized = normalize_run_status(status)
    report_dir_value = str(run_dir.resolve()) if run_dir.exists() else str(run_dir)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    config_path_value = str(config_path.resolve()) if config_path.exists() else None
    metrics_path_value = str(metrics_path.resolve()) if metrics_path.exists() else None
    if normalized == RUN_STATUS_SUCCESS:
        return ComparisonRunResult(
            run_id=spec.run_id,
            framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
            comparison_id=spec.comparison_id,
            comparison_version=spec.comparison_version,
            variant_id=spec.variant_id,
            status=RUN_STATUS_SUCCESS,
            report_dir=report_dir_value,
            config_path=config_path_value,
            metrics_path=metrics_path_value,
            metrics=_metrics_snapshot_from_file(metrics_path if metrics_path.exists() else None),
        )
    if normalized == RUN_STATUS_SKIPPED_DUE_TO_POLICY:
        reason = str(
            (status_payload or {}).get("policy_reason")
            or (status_payload or {}).get("message")
            or "skipped_by_runtime_policy"
        ).strip()
        return ComparisonRunResult(
            run_id=spec.run_id,
            framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
            comparison_id=spec.comparison_id,
            comparison_version=spec.comparison_version,
            variant_id=spec.variant_id,
            status=RUN_STATUS_SKIPPED_DUE_TO_POLICY,
            policy_reason=reason or "skipped_by_runtime_policy",
        )
    raise ValueError(f"Unsupported reused run status for comparison run '{spec.run_id}': {status}")


def _run_index_row(
    *,
    spec: CompiledComparisonRunSpec,
    status: str,
    output_dir: Path,
    execution_mode: str,
    action: str,
    existing_classification: str,
    started_at: str | None,
    ended_at: str | None,
    last_updated: str | None,
) -> dict[str, Any]:
    normalized = normalize_run_status(status)
    return {
        "run_id": spec.run_id,
        "phase": FrameworkMode.LOCKED_COMPARISON.value,
        "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
        "execution_mode": execution_mode,
        "status": normalized or str(status),
        "action": action,
        "existing_classification": existing_classification,
        "output_dir": str(output_dir.resolve()) if output_dir.exists() else str(output_dir),
        "model": spec.model,
        "variant_id": spec.variant_id,
        "suite": None,
        "subject": spec.subject,
        "train_subject": spec.train_subject,
        "test_subject": spec.test_subject,
        "repeat": int(spec.repeat_id),
        "repeat_count": int(spec.repeat_count),
        "started_at": started_at,
        "ended_at": ended_at,
        "last_updated": last_updated,
        "eligible_for_resume": bool(
            normalize_run_status(status) in {RUN_STATUS_FAILED, RUN_STATUS_TIMED_OUT}
        ),
        "eligible_for_rerun": bool(
            normalize_run_status(status) in {RUN_STATUS_FAILED, RUN_STATUS_TIMED_OUT}
        ),
    }


def execute_compiled_comparison(
    *,
    comparison: ComparisonSpec,
    compiled_manifest: CompiledComparisonManifest,
    index_csv: Path | str,
    data_root: Path | str,
    cache_dir: Path | str,
    reports_root: Path | str,
    force: bool,
    resume: bool,
    dry_run: bool,
    compile_duration_seconds: float | None = None,
    timeout_policy_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if force and resume:
        raise ValueError(
            "force=True and resume=True are mutually exclusive for comparison execution."
        )

    run_results: list[ComparisonRunResult] = []
    run_index_rows: list[dict[str, Any]] = []
    reports_root_path = Path(reports_root)
    reports_root_path.mkdir(parents=True, exist_ok=True)
    run_loop_start = perf_counter()
    execution_mode = "dry_run" if dry_run else "force" if force else "resume" if resume else "fresh"
    reconciliation_counts = _new_reconciliation_counts(
        execution_mode=execution_mode,
        n_planned=len(compiled_manifest.runs),
    )
    run_plan: list[dict[str, Any]] = []
    fresh_conflicts: list[dict[str, str]] = []

    for spec in compiled_manifest.runs:
        run_dir = reports_root_path / spec.run_id
        existing_classification, status_payload = _classify_existing_run(run_dir=run_dir)
        if existing_classification == "missing":
            reconciliation_counts["n_missing"] = int(reconciliation_counts.get("n_missing", 0)) + 1
        else:
            _increment_existing_count(
                reconciliation_counts=reconciliation_counts,
                classification=existing_classification,
            )

        action = "rerun_fresh"
        if dry_run:
            action = "planned_dry_run"
        elif force:
            action = "rerun_force"
            reconciliation_counts["n_rerun"] = int(reconciliation_counts.get("n_rerun", 0)) + 1
        elif resume:
            if existing_classification == RUN_STATUS_SUCCESS:
                action = "reuse_success"
                reconciliation_counts["n_reused"] = (
                    int(reconciliation_counts.get("n_reused", 0)) + 1
                )
                reconciliation_counts["n_skipped_as_already_complete"] = (
                    int(reconciliation_counts.get("n_skipped_as_already_complete", 0)) + 1
                )
            elif existing_classification == RUN_STATUS_SKIPPED_DUE_TO_POLICY:
                action = "reuse_skipped_due_to_policy"
                reconciliation_counts["n_reused"] = (
                    int(reconciliation_counts.get("n_reused", 0)) + 1
                )
            else:
                action = "rerun_resume"
                reconciliation_counts["n_rerun"] = int(reconciliation_counts.get("n_rerun", 0)) + 1
        else:
            if existing_classification != "missing":
                action = "conflict_existing"
                fresh_conflicts.append(
                    {
                        "run_id": spec.run_id,
                        "status": existing_classification,
                        "output_dir": str(run_dir),
                    }
                )
            else:
                reconciliation_counts["n_rerun"] = int(reconciliation_counts.get("n_rerun", 0)) + 1

        run_plan.append(
            {
                "spec": spec,
                "run_dir": run_dir,
                "existing_classification": existing_classification,
                "status_payload": status_payload,
                "action": action,
            }
        )

    if fresh_conflicts:
        preview = "; ".join(
            f"{item['run_id']}[{item['status']}]@{item['output_dir']}"
            for item in fresh_conflicts[:5]
        )
        raise RuntimeError(
            "Fresh comparison execution refused because existing run outputs were found: "
            f"{preview}. Use resume mode to reuse/rerun selectively or force mode to rerun all."
        )

    for planned in run_plan:
        spec = planned["spec"]
        if not isinstance(spec, CompiledComparisonRunSpec):
            raise TypeError("Comparison run planner produced invalid spec entry.")
        run_dir = planned["run_dir"]
        if not isinstance(run_dir, Path):
            raise TypeError("Comparison run planner produced invalid run directory entry.")
        existing_classification = str(planned["existing_classification"])
        status_payload = planned["status_payload"]
        if status_payload is not None and not isinstance(status_payload, dict):
            status_payload = None
        action = str(planned["action"])

        if dry_run:
            run_results.append(
                ComparisonRunResult(
                    run_id=spec.run_id,
                    framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
                    comparison_id=spec.comparison_id,
                    comparison_version=spec.comparison_version,
                    variant_id=spec.variant_id,
                    status=RUN_STATUS_PLANNED,
                )
            )
            run_index_rows.append(
                _run_index_row(
                    spec=spec,
                    status=RUN_STATUS_PLANNED,
                    output_dir=run_dir,
                    execution_mode=execution_mode,
                    action=action,
                    existing_classification=existing_classification,
                    started_at=None,
                    ended_at=None,
                    last_updated=_resolve_last_updated_utc(
                        run_dir=run_dir,
                        status_payload=status_payload,
                    )
                    if run_dir.exists()
                    else None,
                )
            )
            continue

        if action in {"reuse_success", "reuse_skipped_due_to_policy"}:
            reused_result = _reused_run_result(
                spec=spec,
                status=str(existing_classification),
                status_payload=status_payload,
                run_dir=run_dir,
            )
            run_results.append(reused_result)
            run_index_rows.append(
                _run_index_row(
                    spec=spec,
                    status=reused_result.status,
                    output_dir=run_dir,
                    execution_mode=execution_mode,
                    action=action,
                    existing_classification=existing_classification,
                    started_at=None,
                    ended_at=None,
                    last_updated=_resolve_last_updated_utc(
                        run_dir=run_dir,
                        status_payload=status_payload,
                    ),
                )
            )
            continue

        run_force = bool(force)
        run_resume = bool(resume)
        if resume and not force:
            if existing_classification in {RUN_STATUS_FAILED, RUN_STATUS_TIMED_OUT}:
                run_force = True
                run_resume = False
            elif existing_classification == "partial_incomplete":
                run_force = False
                run_resume = True
            elif existing_classification == "missing":
                run_force = False
                run_resume = False

        started_at = _utc_now_iso()
        run_result: ComparisonRunResult
        try:
            timeout_policy_effective = resolve_run_timeout_policy(
                framework_mode=FrameworkMode.LOCKED_COMPARISON,
                model_name=spec.model,
                policy_overrides=timeout_policy_overrides,
            )
            comparison_context_payload = _comparison_context_payload(
                spec,
                secondary_metrics=list(compiled_manifest.metric_policy.secondary_metrics),
                decision_metric=compiled_manifest.decision_policy.primary_metric,
                required_run_metadata_fields=list(compiled_manifest.required_run_metadata_fields),
                evidence_policy_payload=compiled_manifest.evidence_policy.model_dump(mode="json"),
                data_policy_payload=compiled_manifest.data_policy.model_dump(mode="json"),
            )
            comparison_context_payload["timeout_policy"] = dict(timeout_policy_effective)
            run_kwargs: dict[str, Any] = {
                "index_csv": str(Path(index_csv)),
                "data_root": str(Path(data_root)),
                "cache_dir": str(Path(cache_dir)),
                "target": spec.target,
                "model": spec.model,
                "cv": spec.cv_mode,
                "subject": spec.subject,
                "train_subject": spec.train_subject,
                "test_subject": spec.test_subject,
                "seed": int(spec.seed),
                "filter_task": spec.filter_task,
                "filter_modality": spec.filter_modality,
                "n_permutations": int(spec.controls.n_permutations),
                "run_id": spec.run_id,
                "reports_root": str(reports_root_path),
                "force": bool(run_force),
                "resume": bool(run_resume),
                "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
                "primary_metric_name": spec.primary_metric,
                "permutation_metric_name": spec.controls.permutation_metric,
                "methodology_policy_name": spec.methodology_policy_name.value,
                "class_weight_policy": spec.class_weight_policy.value,
                "tuning_enabled": bool(spec.tuning_enabled),
                "model_cost_tier": spec.model_cost_tier.value,
                "projected_runtime_seconds": int(spec.projected_runtime_seconds),
                "tuning_search_space_id": spec.tuning_search_space_id,
                "tuning_search_space_version": spec.tuning_search_space_version,
                "tuning_inner_cv_scheme": spec.tuning_inner_cv_scheme,
                "tuning_inner_group_field": spec.tuning_inner_group_field,
                "subgroup_reporting_enabled": bool(spec.subgroup_reporting_enabled),
                "subgroup_dimensions": list(spec.subgroup_dimensions),
                "subgroup_min_samples_per_group": int(spec.subgroup_min_samples_per_group),
                "interpretability_enabled_override": bool(spec.interpretability_enabled),
                "comparison_context": comparison_context_payload,
                "repeat_id": int(spec.repeat_id),
                "repeat_count": int(spec.repeat_count),
                "base_run_id": str(spec.base_run_id),
                "evidence_run_role": spec.evidence_run_role.value,
                "evidence_policy": compiled_manifest.evidence_policy.model_dump(mode="json"),
                "timeout_policy_effective": dict(timeout_policy_effective),
            }
            supervised_result = execute_run_with_timeout_watchdog(
                run_kwargs=run_kwargs,
                timeout_policy=timeout_policy_effective,
                phase_name=FrameworkMode.LOCKED_COMPARISON.value,
                run_identity={
                    "run_id": spec.run_id,
                    "suite_id": None,
                    "variant_id": spec.variant_id,
                },
            )
            run_status = str(supervised_result.get("status", "failed"))
            supervised_error_details_raw = supervised_result.get("error_details")
            supervised_error_details = (
                dict(supervised_error_details_raw)
                if isinstance(supervised_error_details_raw, dict)
                else {}
            )
            if run_status == RUN_STATUS_SUCCESS:
                run_payload = supervised_result.get("run_payload")
                if not isinstance(run_payload, dict):
                    raise RuntimeError("Watchdog run returned success without run payload.")
                run_result = _to_run_result_success(spec, run_payload)
            elif run_status == RUN_STATUS_TIMED_OUT:
                run_result = ComparisonRunResult(
                    run_id=spec.run_id,
                    framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
                    comparison_id=spec.comparison_id,
                    comparison_version=spec.comparison_version,
                    variant_id=spec.variant_id,
                    status=RUN_STATUS_TIMED_OUT,
                    report_dir=(
                        str(supervised_result.get("report_dir"))
                        if supervised_result.get("report_dir")
                        else None
                    ),
                    error=str(supervised_result.get("error") or "run_exceeded_timeout_budget"),
                    error_code=str(supervised_result.get("error_code") or "run_timeout"),
                    error_type=str(supervised_result.get("error_type") or "RunTimeoutError"),
                    failure_stage=str(supervised_result.get("failure_stage") or "watchdog_timeout"),
                    error_details=supervised_error_details,
                    timeout_seconds=float(supervised_result.get("timeout_seconds") or 0.0),
                    elapsed_seconds=float(supervised_result.get("elapsed_seconds") or 0.0),
                    timeout_diagnostics_path=str(supervised_result.get("timeout_diagnostics_path"))
                    if supervised_result.get("timeout_diagnostics_path")
                    else None,
                )
            elif run_status == RUN_STATUS_SKIPPED_DUE_TO_POLICY:
                run_result = ComparisonRunResult(
                    run_id=spec.run_id,
                    framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
                    comparison_id=spec.comparison_id,
                    comparison_version=spec.comparison_version,
                    variant_id=spec.variant_id,
                    status=RUN_STATUS_SKIPPED_DUE_TO_POLICY,
                    policy_reason=str(
                        supervised_result.get("policy_reason") or "skipped_by_runtime_policy"
                    ),
                )
            else:
                run_result = ComparisonRunResult(
                    run_id=spec.run_id,
                    framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
                    comparison_id=spec.comparison_id,
                    comparison_version=spec.comparison_version,
                    variant_id=spec.variant_id,
                    status=RUN_STATUS_FAILED,
                    report_dir=(
                        str(supervised_result.get("report_dir"))
                        if supervised_result.get("report_dir")
                        else None
                    ),
                    error=str(supervised_result.get("error") or "run_execution_failed"),
                    error_code=str(supervised_result.get("error_code") or "run_execution_failed"),
                    error_type=str(supervised_result.get("error_type") or "RuntimeError"),
                    failure_stage=str(supervised_result.get("failure_stage") or "runtime"),
                    error_details=supervised_error_details,
                )
        except Exception as exc:
            failure_payload = exception_failure_payload(exc, default_stage="runtime")
            run_result = ComparisonRunResult(
                run_id=spec.run_id,
                framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
                comparison_id=spec.comparison_id,
                comparison_version=spec.comparison_version,
                variant_id=spec.variant_id,
                status=RUN_STATUS_FAILED,
                error=str(exc),
                error_code=str(failure_payload["error_code"]),
                error_type=str(failure_payload["error_type"]),
                failure_stage=str(failure_payload["failure_stage"]),
                error_details=dict(failure_payload["error_details"]),
            )
        ended_at = _utc_now_iso()
        run_results.append(run_result)
        run_index_rows.append(
            _run_index_row(
                spec=spec,
                status=run_result.status,
                output_dir=run_dir,
                execution_mode=execution_mode,
                action=action,
                existing_classification=existing_classification,
                started_at=started_at,
                ended_at=ended_at,
                last_updated=_resolve_last_updated_utc(
                    run_dir=run_dir,
                    status_payload=None,
                ),
            )
        )

    run_loop_duration_seconds = perf_counter() - run_loop_start
    artifact_write_start = perf_counter()
    comparison_output_dir = _comparison_output_dir(comparison, reports_root=reports_root_path)
    stage_timings: dict[str, float] = {
        "run_execution": float(run_loop_duration_seconds),
    }
    if compile_duration_seconds is not None:
        stage_timings["compile"] = float(compile_duration_seconds)
    artifact_paths = write_comparison_artifacts(
        comparison=comparison,
        compiled_manifest=compiled_manifest,
        run_results=run_results,
        output_dir=comparison_output_dir,
        dry_run=dry_run,
        stage_timings=stage_timings,
    )
    run_index_path = comparison_output_dir / "run_index.json"
    run_index_payload = {
        "schema_version": "official-run-index-v1",
        "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
        "phase": FrameworkMode.LOCKED_COMPARISON.value,
        "execution_mode": execution_mode,
        "generated_at_utc": _utc_now_iso(),
        "runs": run_index_rows,
    }
    run_index_path.write_text(f"{json.dumps(run_index_payload, indent=2)}\n", encoding="utf-8")

    resume_reconciliation_path = comparison_output_dir / "resume_reconciliation.json"
    resume_reconciliation_payload = {
        **reconciliation_counts,
        "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
        "phase": FrameworkMode.LOCKED_COMPARISON.value,
        "generated_at_utc": _utc_now_iso(),
    }
    resume_reconciliation_path.write_text(
        f"{json.dumps(resume_reconciliation_payload, indent=2)}\n",
        encoding="utf-8",
    )
    artifact_paths["run_index"] = str(run_index_path.resolve())
    artifact_paths["resume_reconciliation"] = str(resume_reconciliation_path.resolve())

    stage_timings["artifact_writing"] = float(perf_counter() - artifact_write_start)
    artifact_verification = verify_official_artifacts(
        output_dir=comparison_output_dir,
        mode="comparison",
    )
    if not bool(artifact_verification.get("passed", False)):
        raise ValueError(
            "Comparison artifact verification failed: "
            + "; ".join(
                str(issue.get("message"))
                for issue in list(artifact_verification.get("issues", []))[:5]
                if isinstance(issue, dict)
            )
        )

    n_success = sum(result.status == RUN_STATUS_SUCCESS for result in run_results)
    n_failed = sum(result.status == RUN_STATUS_FAILED for result in run_results)
    n_timed_out = sum(result.status == RUN_STATUS_TIMED_OUT for result in run_results)
    n_skipped_due_to_policy = sum(
        result.status == RUN_STATUS_SKIPPED_DUE_TO_POLICY for result in run_results
    )
    n_planned = sum(result.status == RUN_STATUS_PLANNED for result in run_results)
    return {
        "comparison_id": comparison.comparison_id,
        "comparison_version": comparison.comparison_version,
        "comparison_output_dir": str(comparison_output_dir.resolve()),
        "compiled_manifest": compiled_manifest.model_dump(mode="json"),
        "run_results": [result.model_dump(mode="json") for result in run_results],
        "n_success": int(n_success),
        "n_completed": int(n_success),
        "n_failed": int(n_failed),
        "n_timed_out": int(n_timed_out),
        "n_skipped_due_to_policy": int(n_skipped_due_to_policy),
        "n_planned": int(n_planned),
        "stage_timings_seconds": {key: round(value, 6) for key, value in stage_timings.items()},
        "artifact_paths": artifact_paths,
        "artifact_verification": artifact_verification,
        "resume_reconciliation": resume_reconciliation_payload,
    }


def compile_and_run_comparison(
    *,
    comparison: ComparisonSpec,
    index_csv: Path | str,
    data_root: Path | str,
    cache_dir: Path | str,
    reports_root: Path | str,
    variant_ids: list[str] | None = None,
    force: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    timeout_policy_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if comparison.status == ComparisonStatus.RETIRED:
        raise ValueError("Comparison execution rejected: comparison status is 'retired'.")
    if comparison.status == ComparisonStatus.DRAFT:
        raise ValueError(
            "Comparison execution rejected: comparison status is 'draft'. "
            "Lock comparison spec before execution."
        )
    compile_start = perf_counter()
    compiled_manifest = compile_comparison(
        comparison,
        index_csv=index_csv,
        variant_ids=variant_ids,
    )
    compile_duration_seconds = perf_counter() - compile_start
    return execute_compiled_comparison(
        comparison=comparison,
        compiled_manifest=compiled_manifest,
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        reports_root=reports_root,
        force=force,
        resume=resume,
        dry_run=dry_run,
        compile_duration_seconds=compile_duration_seconds,
        timeout_policy_overrides=timeout_policy_overrides,
    )
