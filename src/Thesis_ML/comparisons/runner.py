from __future__ import annotations

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
from Thesis_ML.experiments.run_states import (
    RUN_STATUS_FAILED,
    RUN_STATUS_PLANNED,
    RUN_STATUS_SKIPPED_DUE_TO_POLICY,
    RUN_STATUS_SUCCESS,
    RUN_STATUS_TIMED_OUT,
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
    run_results: list[ComparisonRunResult] = []
    reports_root_path = Path(reports_root)
    reports_root_path.mkdir(parents=True, exist_ok=True)
    run_loop_start = perf_counter()

    for spec in compiled_manifest.runs:
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
            continue

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
                "force": bool(force),
                "resume": bool(resume),
                "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
                "primary_metric_name": spec.primary_metric,
                "permutation_metric_name": spec.controls.permutation_metric,
                "methodology_policy_name": spec.methodology_policy_name.value,
                "class_weight_policy": spec.class_weight_policy.value,
                "tuning_enabled": bool(spec.tuning_enabled),
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
                run_results.append(_to_run_result_success(spec, run_payload))
                continue
            if run_status == RUN_STATUS_TIMED_OUT:
                run_results.append(
                    ComparisonRunResult(
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
                        failure_stage=str(
                            supervised_result.get("failure_stage") or "watchdog_timeout"
                        ),
                        error_details=supervised_error_details,
                        timeout_seconds=float(supervised_result.get("timeout_seconds") or 0.0),
                        elapsed_seconds=float(supervised_result.get("elapsed_seconds") or 0.0),
                        timeout_diagnostics_path=str(
                            supervised_result.get("timeout_diagnostics_path")
                        )
                        if supervised_result.get("timeout_diagnostics_path")
                        else None,
                    )
                )
                continue
            if run_status == RUN_STATUS_SKIPPED_DUE_TO_POLICY:
                run_results.append(
                    ComparisonRunResult(
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
                )
                continue
            run_results.append(
                ComparisonRunResult(
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
            )
        except Exception as exc:
            failure_payload = exception_failure_payload(exc, default_stage="runtime")
            run_results.append(
                ComparisonRunResult(
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
