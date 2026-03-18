from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.metric_policy import resolve_effective_metric_policy
from Thesis_ML.experiments.errors import exception_failure_payload
from Thesis_ML.experiments.run_experiment import run_experiment
from Thesis_ML.experiments.runtime_policies import validate_official_context_payload
from Thesis_ML.protocols.artifacts import write_protocol_artifacts
from Thesis_ML.protocols.compiler import compile_protocol
from Thesis_ML.protocols.models import (
    CompiledProtocolManifest,
    CompiledRunSpec,
    ProtocolRunResult,
    ProtocolStatus,
    ThesisProtocol,
)
from Thesis_ML.verification.official_artifacts import verify_official_artifacts


def _protocol_output_dir(protocol: ThesisProtocol, reports_root: Path | str) -> Path:
    root = Path(reports_root)
    return root / "protocol_runs" / f"{protocol.protocol_id}__{protocol.protocol_version}"


def _protocol_context_payload(
    spec: CompiledRunSpec,
    *,
    secondary_metrics: list[str],
    required_run_metadata_fields: list[str],
    evidence_policy_payload: dict[str, Any],
    data_policy_payload: dict[str, Any],
    sample_unit: str,
    label_policy: str,
    confirmatory_lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric_policy = resolve_effective_metric_policy(
        primary_metric=spec.primary_metric,
        secondary_metrics=secondary_metrics,
        decision_metric=spec.primary_metric,
        tuning_metric=spec.primary_metric,
        permutation_metric=spec.controls.permutation_metric or spec.primary_metric,
    )
    payload = {
        "framework_mode": FrameworkMode.CONFIRMATORY.value,
        "canonical_run": bool(spec.canonical_run),
        "protocol_id": spec.protocol_id,
        "protocol_version": spec.protocol_version,
        "protocol_schema_version": spec.protocol_schema_version,
        "suite_id": spec.suite_id,
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
        "sample_unit": str(sample_unit),
        "label_policy": str(label_policy),
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
    if isinstance(confirmatory_lock, dict):
        payload["confirmatory_lock"] = dict(confirmatory_lock)
        payload["target_mapping_version"] = str(
            confirmatory_lock.get("target_mapping_version", "")
        )
    return validate_official_context_payload(
        framework_mode=FrameworkMode.CONFIRMATORY,
        context_name="protocol_context",
        context=payload,
    )


def _validate_confirmatory_lock_controls(
    protocol: ThesisProtocol,
    compiled_manifest: CompiledProtocolManifest,
) -> None:
    lock_payload = protocol.confirmatory_lock
    if not isinstance(lock_payload, dict):
        return
    if str(lock_payload.get("protocol_id")) != "thesis_confirmatory_v1":
        return

    runs_by_suite: dict[str, list[CompiledRunSpec]] = {}
    for run_spec in compiled_manifest.runs:
        runs_by_suite.setdefault(run_spec.suite_id, []).append(run_spec)

    if bool(lock_payload.get("dummy_baseline_required", False)):
        suites_missing_dummy = [
            suite_id
            for suite_id, suite_runs in runs_by_suite.items()
            if all(run_spec.model != "dummy" for run_spec in suite_runs)
        ]
        if suites_missing_dummy:
            raise ValueError(
                "Confirmatory freeze hard-gate failed: dummy baseline is required but missing for suite(s): "
                + ", ".join(sorted(suites_missing_dummy))
            )

    if bool(lock_payload.get("permutation_required", False)):
        minimum_permutations = int(lock_payload.get("minimum_permutations", 0))
        invalid_runs = [
            run_spec.run_id
            for run_spec in compiled_manifest.runs
            if (
                not run_spec.controls.permutation_enabled
                or int(run_spec.controls.n_permutations) < minimum_permutations
            )
        ]
        if invalid_runs:
            raise ValueError(
                "Confirmatory freeze hard-gate failed: permutation control is required "
                f"with n_permutations >= {minimum_permutations}. "
                "Invalid run(s): " + ", ".join(sorted(invalid_runs))
            )


def _to_run_result_success(
    spec: CompiledRunSpec,
    run_payload: dict[str, Any],
) -> ProtocolRunResult:
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

    return ProtocolRunResult(
        run_id=spec.run_id,
        suite_id=spec.suite_id,
        framework_mode=FrameworkMode.CONFIRMATORY.value,
        status="completed",
        report_dir=(str(run_payload.get("report_dir")) if run_payload.get("report_dir") else None),
        metrics_path=(
            str(run_payload.get("metrics_path")) if run_payload.get("metrics_path") else None
        ),
        config_path=(
            str(run_payload.get("config_path")) if run_payload.get("config_path") else None
        ),
        metrics=metrics,
    )


def execute_compiled_protocol(
    *,
    protocol: ThesisProtocol,
    compiled_manifest: CompiledProtocolManifest,
    index_csv: Path | str,
    data_root: Path | str,
    cache_dir: Path | str,
    reports_root: Path | str,
    force: bool,
    resume: bool,
    dry_run: bool,
    compile_duration_seconds: float | None = None,
) -> dict[str, Any]:
    run_results: list[ProtocolRunResult] = []
    reports_root_path = Path(reports_root)
    reports_root_path.mkdir(parents=True, exist_ok=True)
    execute_start = perf_counter()

    for spec in compiled_manifest.runs:
        if dry_run:
            run_results.append(
                ProtocolRunResult(
                    run_id=spec.run_id,
                    suite_id=spec.suite_id,
                    framework_mode=FrameworkMode.CONFIRMATORY.value,
                    status="planned",
                )
            )
            continue

        try:
            payload = run_experiment(
                index_csv=Path(index_csv),
                data_root=Path(data_root),
                cache_dir=Path(cache_dir),
                target=spec.target,
                model=spec.model,
                cv=spec.cv_mode,
                subject=spec.subject,
                train_subject=spec.train_subject,
                test_subject=spec.test_subject,
                seed=int(spec.seed),
                filter_task=spec.filter_task,
                filter_modality=spec.filter_modality,
                n_permutations=int(spec.controls.n_permutations),
                run_id=spec.run_id,
                reports_root=reports_root_path,
                force=bool(force),
                resume=bool(resume),
                framework_mode=FrameworkMode.CONFIRMATORY,
                primary_metric_name=spec.primary_metric,
                permutation_metric_name=spec.controls.permutation_metric,
                methodology_policy_name=spec.methodology_policy_name.value,
                class_weight_policy=spec.class_weight_policy.value,
                tuning_enabled=bool(spec.tuning_enabled),
                tuning_search_space_id=spec.tuning_search_space_id,
                tuning_search_space_version=spec.tuning_search_space_version,
                tuning_inner_cv_scheme=spec.tuning_inner_cv_scheme,
                tuning_inner_group_field=spec.tuning_inner_group_field,
                subgroup_reporting_enabled=bool(spec.subgroup_reporting_enabled),
                subgroup_dimensions=list(spec.subgroup_dimensions),
                subgroup_min_samples_per_group=int(spec.subgroup_min_samples_per_group),
                interpretability_enabled_override=bool(spec.interpretability_enabled),
                protocol_context=_protocol_context_payload(
                    spec,
                    secondary_metrics=list(compiled_manifest.metric_policy.secondary_metrics),
                    required_run_metadata_fields=list(
                        compiled_manifest.required_run_metadata_fields
                    ),
                    evidence_policy_payload=compiled_manifest.evidence_policy.model_dump(
                        mode="json"
                    ),
                    data_policy_payload=compiled_manifest.data_policy.model_dump(mode="json"),
                    sample_unit=protocol.scientific_contract.sample_unit,
                    label_policy=protocol.scientific_contract.label_policy,
                    confirmatory_lock=(
                        dict(protocol.confirmatory_lock)
                        if isinstance(protocol.confirmatory_lock, dict)
                        else None
                    ),
                ),
                repeat_id=int(spec.repeat_id),
                repeat_count=int(spec.repeat_count),
                base_run_id=str(spec.base_run_id),
                evidence_run_role=spec.evidence_run_role.value,
                evidence_policy=compiled_manifest.evidence_policy.model_dump(mode="json"),
            )
            run_results.append(_to_run_result_success(spec, payload))
        except Exception as exc:
            failure_payload = exception_failure_payload(exc, default_stage="runtime")
            run_results.append(
                ProtocolRunResult(
                    run_id=spec.run_id,
                    suite_id=spec.suite_id,
                    framework_mode=FrameworkMode.CONFIRMATORY.value,
                    status="failed",
                    error=str(exc),
                    error_code=str(failure_payload["error_code"]),
                    error_type=str(failure_payload["error_type"]),
                    failure_stage=str(failure_payload["failure_stage"]),
                    error_details=dict(failure_payload["error_details"]),
                )
            )

    run_loop_duration_seconds = perf_counter() - execute_start
    artifact_write_start = perf_counter()
    protocol_output_dir = _protocol_output_dir(protocol, reports_root=reports_root_path)
    stage_timings: dict[str, float] = {
        "run_execution": float(run_loop_duration_seconds),
    }
    if compile_duration_seconds is not None:
        stage_timings["compile"] = float(compile_duration_seconds)
    artifact_paths = write_protocol_artifacts(
        protocol=protocol,
        compiled_manifest=compiled_manifest,
        run_results=run_results,
        output_dir=protocol_output_dir,
        dry_run=dry_run,
        stage_timings=stage_timings,
    )
    stage_timings["artifact_writing"] = float(perf_counter() - artifact_write_start)
    artifact_verification = verify_official_artifacts(
        output_dir=protocol_output_dir,
        mode="confirmatory",
    )
    if not bool(artifact_verification.get("passed", False)):
        raise ValueError(
            "Protocol artifact verification failed: "
            + "; ".join(
                str(issue.get("message"))
                for issue in list(artifact_verification.get("issues", []))[:5]
                if isinstance(issue, dict)
            )
        )

    n_completed = sum(result.status == "completed" for result in run_results)
    n_failed = sum(result.status == "failed" for result in run_results)
    n_planned = sum(result.status == "planned" for result in run_results)
    return {
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "protocol_output_dir": str(protocol_output_dir.resolve()),
        "compiled_manifest": compiled_manifest.model_dump(mode="json"),
        "run_results": [result.model_dump(mode="json") for result in run_results],
        "n_completed": int(n_completed),
        "n_failed": int(n_failed),
        "n_planned": int(n_planned),
        "stage_timings_seconds": {key: round(value, 6) for key, value in stage_timings.items()},
        "artifact_paths": artifact_paths,
        "artifact_verification": artifact_verification,
    }


def compile_and_run_protocol(
    *,
    protocol: ThesisProtocol,
    index_csv: Path | str,
    data_root: Path | str,
    cache_dir: Path | str,
    reports_root: Path | str,
    suite_ids: list[str] | None = None,
    force: bool = False,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if isinstance(protocol.confirmatory_lock, dict):
        analysis_status = str(protocol.confirmatory_lock.get("analysis_status", "")).strip().lower()
        if analysis_status and analysis_status != "locked":
            raise ValueError(
                "Confirmatory freeze execution requires analysis_status='locked'. "
                f"Received '{analysis_status}'."
            )
    if protocol.status == ProtocolStatus.DRAFT:
        raise ValueError(
            "Confirmatory protocol execution rejects status='draft'. "
            "Set protocol status to 'locked' or 'released' before running."
        )
    compile_start = perf_counter()
    compiled_manifest = compile_protocol(
        protocol,
        index_csv=index_csv,
        suite_ids=suite_ids,
    )
    _validate_confirmatory_lock_controls(protocol, compiled_manifest)
    compile_duration_seconds = perf_counter() - compile_start
    return execute_compiled_protocol(
        protocol=protocol,
        compiled_manifest=compiled_manifest,
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        reports_root=reports_root,
        force=force,
        resume=resume,
        dry_run=dry_run,
        compile_duration_seconds=compile_duration_seconds,
    )
