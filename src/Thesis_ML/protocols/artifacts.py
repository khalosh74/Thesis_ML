from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.metric_policy import resolve_effective_metric_policy
from Thesis_ML.experiments.evidence_statistics import (
    aggregate_repeated_runs,
    grouped_bootstrap_percentile_interval,
)
from Thesis_ML.experiments.run_states import (
    RUN_STATUS_FAILED,
    RUN_STATUS_TIMED_OUT,
    increment_run_status_count,
    initialized_run_status_counts,
    is_run_success_status,
    normalize_run_status,
)
from Thesis_ML.protocols.claim_evaluator import evaluate_claim_outcomes
from Thesis_ML.protocols.models import (
    CompiledProtocolManifest,
    ProtocolRunResult,
    ThesisProtocol,
)

_SCIENCE_CRITICAL_DEVIATION_TOKENS = (
    "confirmatory runtime target differs",
    "confirmatory runtime source column differs",
    "confirmatory runtime target mapping version differs",
    "confirmatory runtime split differs",
    "confirmatory runtime primary metric differs",
    "confirmatory runtime model differs",
    "confirmatory runtime hyperparameter policy differs",
    "confirmatory runtime class-weight policy differs",
    "confirmatory run permutations are below locked minimum",
    "confirmatory subgroup axis outside locked allowed subgroup axes",
    "confirmatory subgroup minimum samples is below locked threshold",
    "confirmatory subgroup minimum classes is below locked threshold",
    "confirmatory subgroup small-group reporting differs from locked policy",
    "confirmatory freeze hard-gate failed",
    "illegal override for official run key",
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


def _load_json(path_text: str | None) -> dict[str, Any] | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _is_science_critical_deviation(error_text: str) -> bool:
    normalized = str(error_text).strip().lower()
    if not normalized:
        return False
    return any(token in normalized for token in _SCIENCE_CRITICAL_DEVIATION_TOKENS)


def _fingerprint_signature(payload: dict[str, Any]) -> str:
    return "|".join(
        [
            str(payload.get("index_csv_sha256", "")),
            str(payload.get("selected_sample_id_sha256", "")),
            str(payload.get("target_column", "")),
            str(payload.get("cv_mode", "")),
        ]
    )


def _dataset_fingerprint_summary(run_results: list[ProtocolRunResult]) -> dict[str, Any]:
    completed = [result for result in run_results if is_run_success_status(result.status)]
    signatures: set[str] = set()
    missing_run_ids: list[str] = []
    sources: list[str] = []
    for result in completed:
        config_payload = _load_json(result.config_path)
        metrics_payload = _load_json(result.metrics_path)
        config_fingerprint = (
            config_payload.get("dataset_fingerprint") if isinstance(config_payload, dict) else None
        )
        metrics_fingerprint = (
            metrics_payload.get("dataset_fingerprint")
            if isinstance(metrics_payload, dict)
            else None
        )
        fingerprint_payload: dict[str, Any] | None = None
        if isinstance(config_fingerprint, dict):
            fingerprint_payload = config_fingerprint
            sources.append("config")
        elif isinstance(metrics_fingerprint, dict):
            fingerprint_payload = metrics_fingerprint
            sources.append("metrics")
        else:
            missing_run_ids.append(result.run_id)
            continue
        signatures.add(_fingerprint_signature(fingerprint_payload))
    n_completed = int(len(completed))
    n_present = int(len(completed) - len(missing_run_ids))
    return {
        "n_completed_runs": n_completed,
        "n_with_fingerprint": n_present,
        "n_missing_fingerprint": int(len(missing_run_ids)),
        "missing_run_ids": sorted(missing_run_ids),
        "unique_fingerprint_count": int(len(signatures)),
        "consistent_across_runs": bool(len(signatures) <= 1 if n_present > 0 else True),
        "sources": sorted(set(sources)),
    }


def _deviation_log_payload(
    protocol: ThesisProtocol,
    run_results: list[ProtocolRunResult],
) -> dict[str, Any]:
    deviations: list[dict[str, Any]] = []
    science_critical_count = 0
    for result in run_results:
        normalized_status = normalize_run_status(result.status)
        if normalized_status not in {RUN_STATUS_FAILED, RUN_STATUS_TIMED_OUT}:
            continue
        error_text = str(result.error or "")
        science_critical = _is_science_critical_deviation(error_text)
        if science_critical:
            science_critical_count += 1
        deviations.append(
            {
                "run_id": result.run_id,
                "suite_id": result.suite_id,
                "status": "deviation_detected",
                "run_status": normalized_status,
                "science_critical": bool(science_critical),
                "reason": error_text,
            }
        )
    if not deviations:
        deviations.append(
            {
                "run_id": None,
                "suite_id": None,
                "status": "no_deviation",
                "science_critical": False,
                "reason": "No protocol deviations detected.",
            }
        )
    science_critical_detected = bool(science_critical_count > 0)
    return {
        "framework_mode": FrameworkMode.CONFIRMATORY.value,
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "science_critical_deviation_detected": science_critical_detected,
        "confirmatory_status": ("downgraded" if science_critical_detected else "confirmatory"),
        "n_total_deviations": int(
            len([entry for entry in deviations if entry["status"] != "no_deviation"])
        ),
        "n_science_critical_deviations": int(science_critical_count),
        "deviations": deviations,
    }


def _controls_status(
    protocol: ThesisProtocol,
    compiled_manifest: CompiledProtocolManifest,
    run_results: list[ProtocolRunResult],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    lock_payload = (
        protocol.confirmatory_lock if isinstance(protocol.confirmatory_lock, dict) else {}
    )
    dummy_required = bool(
        lock_payload.get(
            "dummy_baseline_required",
            protocol.control_policy.dummy_baseline.enabled,
        )
    )
    permutation_required = bool(
        lock_payload.get(
            "permutation_required",
            protocol.control_policy.permutation.enabled,
        )
    )
    minimum_permutations = int(
        lock_payload.get(
            "minimum_permutations",
            protocol.control_policy.permutation.n_permutations,
        )
    )
    dummy_control_suites = set(protocol.control_policy.dummy_baseline.suites)
    permutation_control_suites = set(protocol.control_policy.permutation.suites)
    dummy_run_ids = sorted(
        spec.run_id
        for spec in compiled_manifest.runs
        if (
            spec.suite_id in dummy_control_suites
            and (bool(spec.controls.dummy_baseline_run) or str(spec.model) == "dummy")
        )
    )
    permutation_run_ids = sorted(
        spec.run_id
        for spec in compiled_manifest.runs
        if spec.suite_id in permutation_control_suites
    )
    completed_by_run = {
        result.run_id: result for result in run_results if is_run_success_status(result.status)
    }
    failed_by_run = {
        result.run_id: result
        for result in run_results
        if normalize_run_status(result.status) == RUN_STATUS_FAILED
    }
    timed_out_by_run = {
        result.run_id: result
        for result in run_results
        if normalize_run_status(result.status) == RUN_STATUS_TIMED_OUT
    }
    invalid_permutation_runs = sorted(
        spec.run_id
        for spec in compiled_manifest.runs
        if spec.suite_id in permutation_control_suites
        and (
            not bool(spec.controls.permutation_enabled)
            or int(spec.controls.n_permutations) < minimum_permutations
        )
    )
    missing_dummy_completion_run_ids: list[str] = []
    missing_permutation_completion_run_ids: list[str] = []
    missing_permutation_artifact_run_ids: list[str] = []
    permutation_runs_below_minimum_by_metrics: list[str] = []
    if dry_run:
        dummy_requirement_satisfied = bool((not dummy_required) or bool(dummy_run_ids))
        permutation_requirement_satisfied = bool(
            (not permutation_required)
            or (bool(permutation_run_ids) and not bool(invalid_permutation_runs))
        )
    else:
        missing_dummy_completion_run_ids = sorted(
            run_id for run_id in dummy_run_ids if run_id not in completed_by_run
        )
        missing_permutation_completion_run_ids = sorted(
            run_id for run_id in permutation_run_ids if run_id not in completed_by_run
        )
        for run_id in permutation_run_ids:
            run_result = completed_by_run.get(run_id)
            if run_result is None:
                continue
            metrics_payload = _load_json(run_result.metrics_path)
            permutation_payload = (
                metrics_payload.get("permutation_test")
                if isinstance(metrics_payload, dict)
                else None
            )
            if not isinstance(permutation_payload, dict):
                missing_permutation_artifact_run_ids.append(run_id)
                continue
            n_from_metrics = int(permutation_payload.get("n_permutations", 0))
            if n_from_metrics < minimum_permutations:
                permutation_runs_below_minimum_by_metrics.append(run_id)
        dummy_requirement_satisfied = bool(
            (not dummy_required)
            or (bool(dummy_run_ids) and not bool(missing_dummy_completion_run_ids))
        )
        permutation_requirement_satisfied = bool(
            (not permutation_required)
            or (
                bool(permutation_run_ids)
                and not bool(invalid_permutation_runs)
                and not bool(missing_permutation_completion_run_ids)
                and not bool(missing_permutation_artifact_run_ids)
                and not bool(permutation_runs_below_minimum_by_metrics)
            )
        )
    controls_valid_for_confirmatory = bool(
        dummy_requirement_satisfied and permutation_requirement_satisfied
    )
    return {
        "dummy_baseline_required": bool(dummy_required),
        "dummy_baseline_present": bool(dummy_run_ids),
        "dummy_baseline_run_ids": dummy_run_ids,
        "dummy_baseline_completed_run_ids": sorted(
            run_id for run_id in dummy_run_ids if run_id in completed_by_run
        ),
        "dummy_baseline_failed_run_ids": sorted(
            run_id for run_id in dummy_run_ids if run_id in failed_by_run
        ),
        "dummy_baseline_timed_out_run_ids": sorted(
            run_id for run_id in dummy_run_ids if run_id in timed_out_by_run
        ),
        "dummy_baseline_missing_completion_run_ids": missing_dummy_completion_run_ids,
        "dummy_requirement_satisfied": bool(dummy_requirement_satisfied),
        "permutation_required": bool(permutation_required),
        "minimum_permutations": int(minimum_permutations),
        "permutation_run_ids": permutation_run_ids,
        "permutation_completed_run_ids": sorted(
            run_id for run_id in permutation_run_ids if run_id in completed_by_run
        ),
        "permutation_failed_run_ids": sorted(
            run_id for run_id in permutation_run_ids if run_id in failed_by_run
        ),
        "permutation_timed_out_run_ids": sorted(
            run_id for run_id in permutation_run_ids if run_id in timed_out_by_run
        ),
        "permutation_missing_completion_run_ids": missing_permutation_completion_run_ids,
        "invalid_permutation_run_ids": invalid_permutation_runs,
        "permutation_missing_artifact_run_ids": sorted(missing_permutation_artifact_run_ids),
        "permutation_runs_below_minimum_by_metrics": sorted(
            permutation_runs_below_minimum_by_metrics
        ),
        "permutation_requirement_satisfied": bool(permutation_requirement_satisfied),
        "controls_valid_for_confirmatory": bool(controls_valid_for_confirmatory),
        "evaluated_on_completed_runs": bool(not dry_run),
    }


def _confirmatory_reporting_contract(
    protocol: ThesisProtocol,
    compiled_manifest: CompiledProtocolManifest,
    run_results: list[ProtocolRunResult],
    deviation_log_payload: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    lock_payload = (
        protocol.confirmatory_lock if isinstance(protocol.confirmatory_lock, dict) else {}
    )
    fingerprint_summary = _dataset_fingerprint_summary(run_results)
    deviations = list(deviation_log_payload.get("deviations", []))
    total_deviations = int(
        len([entry for entry in deviations if entry.get("status") != "no_deviation"])
    )
    controls_status = _controls_status(
        protocol,
        compiled_manifest,
        run_results,
        dry_run=dry_run,
    )
    science_critical_deviation_detected = bool(
        deviation_log_payload.get("science_critical_deviation_detected", False)
    )
    controls_valid = bool(controls_status.get("controls_valid_for_confirmatory", True))
    confirmatory_status = (
        "downgraded"
        if (science_critical_deviation_detected or not controls_valid)
        else "confirmatory"
    )
    return {
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "dataset_fingerprint": fingerprint_summary,
        "target_mapping_version": str(lock_payload.get("target_mapping_version", "")) or None,
        "target_mapping_hash": str(lock_payload.get("target_mapping_hash", "")) or None,
        "primary_split": str(lock_payload.get("split", protocol.split_policy.primary_mode)),
        "primary_metric": str(
            lock_payload.get("primary_metric", compiled_manifest.metric_policy.primary_metric)
        ),
        "model_family": str(
            lock_payload.get(
                "model_family",
                protocol.model_policy.models[0] if protocol.model_policy.models else "",
            )
        ),
        "controls_status": controls_status,
        "interpretation_limits": (
            dict(lock_payload.get("interpretation_limits", {}))
            if isinstance(lock_payload.get("interpretation_limits"), dict)
            else {}
        ),
        "multiplicity_policy": {
            "primary_hypotheses": int(lock_payload.get("multiplicity_primary_hypotheses", 1)),
            "primary_alpha": float(lock_payload.get("multiplicity_primary_alpha", 0.05)),
            "secondary_policy": str(
                lock_payload.get("multiplicity_secondary_policy", "descriptive_only")
            ),
            "exploratory_claims_allowed": bool(
                lock_payload.get("multiplicity_exploratory_claims_allowed", False)
            ),
        },
        "subgroup_evidence_policy": {
            "evidence_role": str(lock_payload.get("subgroup_interpretation", "descriptive_only")),
            "primary_evidence_substitution_allowed": False,
        },
        "deviations_from_protocol": {
            "n_total_deviations": int(total_deviations),
            "n_science_critical_deviations": int(
                deviation_log_payload.get("n_science_critical_deviations", 0)
            ),
            "science_critical_deviation_detected": bool(science_critical_deviation_detected),
            "controls_valid_for_confirmatory": bool(controls_valid),
            "confirmatory_status": str(confirmatory_status),
            "explicit_no_deviation_record": bool(total_deviations == 0),
        },
        "confirmatory_valid": bool(
            not dry_run and controls_valid and not science_critical_deviation_detected
        ),
    }


def _suite_summary(
    compiled_manifest: CompiledProtocolManifest,
    run_results: list[ProtocolRunResult],
    *,
    reporting_contract: dict[str, Any],
    claim_outcomes: dict[str, Any],
) -> dict[str, Any]:
    metric_policy_effective = resolve_effective_metric_policy(
        primary_metric=compiled_manifest.metric_policy.primary_metric,
        secondary_metrics=compiled_manifest.metric_policy.secondary_metrics,
        decision_metric=compiled_manifest.metric_policy.primary_metric,
        tuning_metric=compiled_manifest.metric_policy.primary_metric,
        permutation_metric=compiled_manifest.metric_policy.primary_metric,
    )
    by_suite: dict[str, dict[str, int]] = {}
    for suite_id in compiled_manifest.suite_ids:
        by_suite[suite_id] = initialized_run_status_counts()
    for result in run_results:
        status_counts = by_suite.setdefault(result.suite_id, initialized_run_status_counts())
        increment_run_status_count(status_counts, result.status)
    for counts in by_suite.values():
        counts["completed"] = int(counts.get("success", 0))
    return {
        "framework_mode": FrameworkMode.CONFIRMATORY.value,
        "protocol_id": compiled_manifest.protocol_id,
        "protocol_version": compiled_manifest.protocol_version,
        "methodology_policy_name": compiled_manifest.methodology_policy.policy_name.value,
        "primary_metric": compiled_manifest.metric_policy.primary_metric,
        "metric_policy_effective": {
            "primary_metric": metric_policy_effective.primary_metric,
            "secondary_metrics": list(metric_policy_effective.secondary_metrics),
            "decision_metric": metric_policy_effective.decision_metric,
            "tuning_metric": metric_policy_effective.tuning_metric,
            "permutation_metric": metric_policy_effective.permutation_metric,
            "higher_is_better": bool(metric_policy_effective.higher_is_better),
        },
        "subgroup_reporting_enabled": bool(compiled_manifest.subgroup_reporting_policy.enabled),
        "feature_engineering_policy": compiled_manifest.feature_engineering_policy.model_dump(
            mode="json"
        ),
        "data_policy_effective": compiled_manifest.data_policy.model_dump(mode="json"),
        "external_validation_status": {
            "enabled": bool(compiled_manifest.data_policy.external_validation.enabled),
            "mode": str(compiled_manifest.data_policy.external_validation.mode),
            "require_compatible": bool(
                compiled_manifest.data_policy.external_validation.require_compatible
            ),
        },
        "confirmatory_status": str(
            reporting_contract.get("deviations_from_protocol", {}).get(
                "confirmatory_status",
                "confirmatory",
            )
        ),
        "required_evidence_status": {
            "controls_valid_for_confirmatory": bool(
                reporting_contract.get("controls_status", {}).get(
                    "controls_valid_for_confirmatory", False
                )
            ),
            "subgroup_primary_evidence_substitution_allowed": bool(
                reporting_contract.get("subgroup_evidence_policy", {}).get(
                    "primary_evidence_substitution_allowed",
                    False,
                )
            ),
            "valid": bool(
                reporting_contract.get("deviations_from_protocol", {}).get(
                    "controls_valid_for_confirmatory",
                    False,
                )
                and not bool(
                    reporting_contract.get("deviations_from_protocol", {}).get(
                        "science_critical_deviation_detected",
                        False,
                    )
                )
            ),
        },
        "confirmatory_reporting_contract": reporting_contract,
        "suite_status_counts": by_suite,
        "n_runs": int(len(run_results)),
        "claim_outcomes_summary": {
            "primary_claim_id": claim_outcomes.get("primary_claim_id"),
            "primary_claim_verdict": claim_outcomes.get("primary_claim_verdict"),
            "n_claims": len(claim_outcomes.get("claims", []))
            if isinstance(claim_outcomes.get("claims"), list)
            else 0,
        },
        "claim_outcomes_path": "claim_outcomes.json",
    }


def _execution_status_payload(
    protocol: ThesisProtocol,
    compiled_manifest: CompiledProtocolManifest,
    run_results: list[ProtocolRunResult],
    *,
    dry_run: bool,
    reporting_contract: dict[str, Any],
    deviation_log_payload: dict[str, Any],
    stage_timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "framework_mode": FrameworkMode.CONFIRMATORY.value,
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "protocol_schema_version": protocol.protocol_schema_version,
        "compiled_schema_version": compiled_manifest.compiled_schema_version,
        "dry_run": bool(dry_run),
        "runs": [result.model_dump(mode="json") for result in run_results],
        "confirmatory_status": str(
            deviation_log_payload.get("confirmatory_status", "confirmatory")
        ),
        "science_critical_deviation_detected": bool(
            deviation_log_payload.get("science_critical_deviation_detected", False)
        ),
        "confirmatory_reporting_contract": reporting_contract,
    }
    if stage_timings:
        payload["stage_timings_seconds"] = {
            key: round(float(value), 6) for key, value in stage_timings.items()
        }
    # compute run-status counts and expose summary fields
    status_counts = initialized_run_status_counts()
    for result in run_results:
        increment_run_status_count(status_counts, result.status)
    # backward-compatible completed count
    status_counts["completed"] = int(status_counts.get("success", 0))

    payload.update(
        {
            "n_runs": int(len(run_results)),
            "n_planned": int(status_counts.get("planned", 0)),
            "n_success": int(status_counts.get("success", 0)),
            "n_failed": int(status_counts.get("failed", 0)),
            "n_timed_out": int(status_counts.get("timed_out", 0)),
            "n_skipped_due_to_policy": int(status_counts.get("skipped_due_to_policy", 0)),
            "n_completed": int(status_counts.get("success", 0)),
            "run_status_counts": status_counts,
        }
    )

    return payload


def _report_index_rows(
    protocol: ThesisProtocol,
    compiled_manifest: CompiledProtocolManifest,
    run_results: list[ProtocolRunResult],
    *,
    confirmatory_status: str,
) -> list[dict[str, Any]]:
    result_by_run_id = {result.run_id: result for result in run_results}
    lock_payload = (
        protocol.confirmatory_lock if isinstance(protocol.confirmatory_lock, dict) else {}
    )
    metric_policy_cache: dict[
        tuple[str, tuple[str, ...], str],
        dict[str, Any],
    ] = {}
    rows: list[dict[str, Any]] = []
    for spec in compiled_manifest.runs:
        cache_key = (
            spec.primary_metric,
            tuple(compiled_manifest.metric_policy.secondary_metrics),
            spec.controls.permutation_metric or spec.primary_metric,
        )
        metric_policy_effective = metric_policy_cache.get(cache_key)
        if metric_policy_effective is None:
            resolved_metric_policy = resolve_effective_metric_policy(
                primary_metric=spec.primary_metric,
                secondary_metrics=compiled_manifest.metric_policy.secondary_metrics,
                decision_metric=spec.primary_metric,
                tuning_metric=spec.primary_metric,
                permutation_metric=spec.controls.permutation_metric or spec.primary_metric,
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
                "framework_mode": FrameworkMode.CONFIRMATORY.value,
                "confirmatory_status": confirmatory_status,
                "protocol_id": compiled_manifest.protocol_id,
                "protocol_version": compiled_manifest.protocol_version,
                "suite_id": spec.suite_id,
                "claim_ids": "|".join(spec.claim_ids),
                "status": result.status if result is not None else "planned",
                "primary_split": str(lock_payload.get("split", protocol.split_policy.primary_mode)),
                "target": spec.target,
                "model_family": str(
                    lock_payload.get(
                        "model_family",
                        protocol.model_policy.models[0] if protocol.model_policy.models else "",
                    )
                ),
                "model": spec.model,
                "target_mapping_version": str(lock_payload.get("target_mapping_version", ""))
                or None,
                "target_mapping_hash": str(lock_payload.get("target_mapping_hash", "")) or None,
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
                "feature_recipe_id": str(spec.feature_recipe_id),
                "tuning_enabled": bool(spec.tuning_enabled),
                "permutation_enabled": bool(spec.controls.permutation_enabled),
                "n_permutations": int(spec.controls.n_permutations),
                "permutation_metric": spec.controls.permutation_metric,
                "higher_is_better": bool(metric_policy_effective["higher_is_better"]),
                "dummy_baseline_run": bool(spec.controls.dummy_baseline_run),
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
                "science_critical_deviation": (
                    bool(_is_science_critical_deviation(str(result.error or "")))
                    if result is not None
                    and normalize_run_status(result.status) == RUN_STATUS_FAILED
                    else False
                ),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "accuracy": metrics.get("accuracy"),
                "primary_metric_name": metrics.get("primary_metric_name"),
                "primary_metric_aggregation": metrics.get("primary_metric_aggregation"),
                "primary_metric_value": metrics.get("primary_metric_value"),
                "primary_metric_value_mean_fold": metrics.get("primary_metric_value_mean_fold"),
                "primary_metric_value_pooled": metrics.get("primary_metric_value_pooled"),
            }
        )
    return rows


def _completed_primary_rows(report_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in report_rows
        if is_run_success_status(str(row.get("status", "")))
        and str(row.get("evidence_run_role", "")).strip() == "primary"
    ]


def _build_repeated_run_outputs(
    *,
    report_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    repeated_rows, repeated_summary_frame = aggregate_repeated_runs(
        report_rows,
        metric_key="primary_metric_value",
        group_keys=["protocol_id", "protocol_version", "suite_id", "model"],
    )
    summary_payload = {
        "n_rows": int(repeated_rows.shape[0]),
        "n_groups": int(repeated_summary_frame.shape[0]),
        "groups": repeated_summary_frame.to_dict(orient="records"),
    }
    return repeated_rows, summary_payload


def _build_confidence_interval_outputs(
    *,
    protocol: ThesisProtocol,
    report_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ci_policy = protocol.evidence_policy.confidence_intervals
    interval_rows: list[dict[str, Any]] = []
    grouping: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in report_rows:
        suite_id = str(row.get("suite_id", ""))
        model = str(row.get("model", ""))
        grouping.setdefault((suite_id, model), []).append(row)
    for (suite_id, model), suite_rows in sorted(grouping.items()):
        interval = grouped_bootstrap_percentile_interval(
            suite_rows,
            value_key="primary_metric_value",
            group_key="base_run_id",
            confidence_level=float(ci_policy.confidence_level),
            n_bootstrap=int(ci_policy.n_bootstrap),
            seed=int(ci_policy.seed),
        )
        interval_rows.append(
            {
                "protocol_id": protocol.protocol_id,
                "protocol_version": protocol.protocol_version,
                "suite_id": suite_id,
                "model": model,
                "primary_metric": protocol.metric_policy.primary_metric,
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


def write_protocol_artifacts(
    *,
    protocol: ThesisProtocol,
    compiled_manifest: CompiledProtocolManifest,
    run_results: list[ProtocolRunResult],
    output_dir: Path | str,
    dry_run: bool,
    stage_timings: dict[str, float] | None = None,
) -> dict[str, str]:
    protocol_dir = Path(output_dir)
    protocol_dir.mkdir(parents=True, exist_ok=True)

    claim_outcomes_path = protocol_dir / "claim_outcomes.json"
    protocol_json_path = protocol_dir / "protocol.json"
    compiled_manifest_path = protocol_dir / "compiled_protocol_manifest.json"
    claim_map_path = protocol_dir / "claim_to_run_map.json"
    suite_summary_path = protocol_dir / "suite_summary.json"
    execution_status_path = protocol_dir / "execution_status.json"
    deviation_log_path = protocol_dir / "deviation_log.json"
    repeated_run_metrics_path = protocol_dir / "repeated_run_metrics.csv"
    repeated_run_summary_path = protocol_dir / "repeated_run_summary.json"
    confidence_intervals_path = protocol_dir / "confidence_intervals.json"
    metric_intervals_path = protocol_dir / "metric_intervals.csv"
    report_index_path = protocol_dir / "report_index.csv"

    metric_policy_effective = resolve_effective_metric_policy(
        primary_metric=compiled_manifest.metric_policy.primary_metric,
        secondary_metrics=compiled_manifest.metric_policy.secondary_metrics,
        decision_metric=compiled_manifest.metric_policy.primary_metric,
        tuning_metric=compiled_manifest.metric_policy.primary_metric,
        permutation_metric=compiled_manifest.metric_policy.primary_metric,
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

    _write_json(protocol_json_path, protocol.model_dump(mode="json"))
    _write_json(compiled_manifest_path, compiled_manifest_payload)
    _write_json(claim_map_path, compiled_manifest.claim_to_run_map)
    deviation_log_payload = _deviation_log_payload(protocol, run_results)
    reporting_contract = _confirmatory_reporting_contract(
        protocol,
        compiled_manifest,
        run_results,
        deviation_log_payload,
        dry_run=dry_run,
    )
    claim_outcomes_payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=compiled_manifest,
        run_results=run_results,
        reporting_contract=reporting_contract,
    )
    resolved_confirmatory_status = str(
        reporting_contract.get("deviations_from_protocol", {}).get(
            "confirmatory_status",
            deviation_log_payload.get("confirmatory_status", "confirmatory"),
        )
    )
    deviation_log_payload["confirmatory_status"] = resolved_confirmatory_status
    _write_json(deviation_log_path, deviation_log_payload)
    _write_json(
        suite_summary_path,
        _suite_summary(
            compiled_manifest,
            run_results,
            reporting_contract=reporting_contract,
            claim_outcomes=claim_outcomes_payload,
        ),
    )
    _write_json(
        execution_status_path,
        _execution_status_payload(
            protocol=protocol,
            compiled_manifest=compiled_manifest,
            run_results=run_results,
            dry_run=dry_run,
            reporting_contract=reporting_contract,
            deviation_log_payload=deviation_log_payload,
            stage_timings=stage_timings,
        ),
    )
    _write_json(claim_outcomes_path, claim_outcomes_payload)

    report_rows = _report_index_rows(
        protocol,
        compiled_manifest,
        run_results,
        confirmatory_status=str(resolved_confirmatory_status),
    )
    primary_rows = _completed_primary_rows(report_rows)
    repeated_run_rows, repeated_summary_payload = _build_repeated_run_outputs(
        report_rows=primary_rows
    )
    repeated_run_rows.to_csv(repeated_run_metrics_path, index=False)
    _write_json(repeated_run_summary_path, repeated_summary_payload)

    interval_rows, confidence_payload = _build_confidence_interval_outputs(
        protocol=protocol,
        report_rows=primary_rows,
    )
    _write_json(confidence_intervals_path, confidence_payload)
    pd.DataFrame(interval_rows).to_csv(metric_intervals_path, index=False)

    with report_index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(report_rows[0].keys()) if report_rows else ["run_id"],
        )
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)

    return {
        "protocol_json": str(protocol_json_path.resolve()),
        "compiled_protocol_manifest": str(compiled_manifest_path.resolve()),
        "claim_to_run_map": str(claim_map_path.resolve()),
        "suite_summary": str(suite_summary_path.resolve()),
        "execution_status": str(execution_status_path.resolve()),
        "deviation_log": str(deviation_log_path.resolve()),
        "repeated_run_metrics": str(repeated_run_metrics_path.resolve()),
        "repeated_run_summary": str(repeated_run_summary_path.resolve()),
        "confidence_intervals": str(confidence_intervals_path.resolve()),
        "metric_intervals": str(metric_intervals_path.resolve()),
        "report_index": str(report_index_path.resolve()),
        "claim_outcomes": str(claim_outcomes_path.resolve()),
    }
