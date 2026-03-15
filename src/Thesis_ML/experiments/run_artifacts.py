from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Thesis_ML.config.metric_policy import EffectiveMetricPolicy
from Thesis_ML.experiments.segment_execution import SegmentExecutionResult


def _relative_path(path: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return None


@dataclass(frozen=True)
class RunIdentity:
    protocol_id: str | None
    protocol_version: str | None
    protocol_schema_version: str | None
    suite_id: str | None
    claim_ids: list[str] | None
    comparison_id: str | None
    comparison_version: str | None
    comparison_variant_id: str | None


def resolve_run_identity(
    *,
    protocol_context: dict[str, Any],
    comparison_context: dict[str, Any],
) -> RunIdentity:
    claim_ids_raw = protocol_context.get("claim_ids")
    claim_ids = [str(value) for value in claim_ids_raw] if isinstance(claim_ids_raw, list) else None
    return RunIdentity(
        protocol_id=(
            str(protocol_context.get("protocol_id")) if protocol_context.get("protocol_id") else None
        ),
        protocol_version=(
            str(protocol_context.get("protocol_version"))
            if protocol_context.get("protocol_version")
            else None
        ),
        protocol_schema_version=(
            str(protocol_context.get("protocol_schema_version"))
            if protocol_context.get("protocol_schema_version")
            else None
        ),
        suite_id=str(protocol_context.get("suite_id")) if protocol_context.get("suite_id") else None,
        claim_ids=claim_ids,
        comparison_id=(
            str(comparison_context.get("comparison_id"))
            if comparison_context.get("comparison_id")
            else None
        ),
        comparison_version=(
            str(comparison_context.get("comparison_version"))
            if comparison_context.get("comparison_version")
            else None
        ),
        comparison_variant_id=(
            str(comparison_context.get("variant_id"))
            if comparison_context.get("variant_id")
            else None
        ),
    )


def metric_policy_effective_payload(
    metric_policy_effective: EffectiveMetricPolicy,
) -> dict[str, Any]:
    return {
        "primary_metric": metric_policy_effective.primary_metric,
        "secondary_metrics": list(metric_policy_effective.secondary_metrics),
        "decision_metric": metric_policy_effective.decision_metric,
        "tuning_metric": metric_policy_effective.tuning_metric,
        "permutation_metric": metric_policy_effective.permutation_metric,
        "higher_is_better": bool(metric_policy_effective.higher_is_better),
    }


def stamp_metrics_artifact(
    *,
    metrics_path: Path,
    canonical_run: bool,
    framework_mode: str,
    methodology_policy_name: str,
    class_weight_policy: str,
    tuning_enabled: bool,
    tuning_summary_path: Path,
    tuning_best_params_path: Path,
    subgroup_metrics_json_path: Path,
    subgroup_metrics_csv_path: Path,
    metric_policy_effective: EffectiveMetricPolicy,
    identity: RunIdentity,
) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    try:
        persisted_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(persisted_metrics, dict):
        return None

    persisted_metrics["canonical_run"] = bool(canonical_run)
    persisted_metrics["framework_mode"] = framework_mode
    persisted_metrics["methodology_policy_name"] = methodology_policy_name
    persisted_metrics["class_weight_policy"] = class_weight_policy
    persisted_metrics["tuning_enabled"] = bool(tuning_enabled)
    persisted_metrics["tuning_summary_path"] = str(tuning_summary_path.resolve())
    persisted_metrics["tuning_summary_path_relative"] = _relative_path(tuning_summary_path)
    persisted_metrics["tuning_best_params_path"] = str(tuning_best_params_path.resolve())
    persisted_metrics["tuning_best_params_path_relative"] = _relative_path(
        tuning_best_params_path
    )
    persisted_metrics["subgroup_metrics_json_path"] = str(subgroup_metrics_json_path.resolve())
    persisted_metrics["subgroup_metrics_json_path_relative"] = _relative_path(
        subgroup_metrics_json_path
    )
    persisted_metrics["subgroup_metrics_csv_path"] = str(subgroup_metrics_csv_path.resolve())
    persisted_metrics["subgroup_metrics_csv_path_relative"] = _relative_path(subgroup_metrics_csv_path)
    persisted_metrics["metric_policy_effective"] = metric_policy_effective_payload(
        metric_policy_effective
    )
    persisted_metrics["decision_metric_name"] = metric_policy_effective.decision_metric
    persisted_metrics["tuning_metric_name"] = metric_policy_effective.tuning_metric
    persisted_metrics["permutation_metric_name"] = metric_policy_effective.permutation_metric
    persisted_metrics["protocol_id"] = identity.protocol_id
    persisted_metrics["protocol_version"] = identity.protocol_version
    persisted_metrics["protocol_schema_version"] = identity.protocol_schema_version
    persisted_metrics["suite_id"] = identity.suite_id
    persisted_metrics["claim_ids"] = identity.claim_ids
    persisted_metrics["comparison_id"] = identity.comparison_id
    persisted_metrics["comparison_version"] = identity.comparison_version
    persisted_metrics["comparison_variant_id"] = identity.comparison_variant_id
    metrics_path.write_text(f"{json.dumps(persisted_metrics, indent=2)}\n", encoding="utf-8")
    return persisted_metrics


def build_run_config_payload(
    *,
    run_id: str,
    timestamp: str,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    target_column: str,
    model: str,
    cv_mode: str,
    subject: str | None,
    train_subject: str | None,
    test_subject: str | None,
    seed: int,
    primary_metric_name: str,
    permutation_metric_name: str,
    metric_policy_effective: EffectiveMetricPolicy,
    methodology_policy_name: str,
    class_weight_policy: str,
    tuning_enabled: bool,
    tuning_search_space_id: str | None,
    tuning_search_space_version: str | None,
    tuning_inner_cv_scheme: str | None,
    tuning_inner_group_field: str | None,
    tuning_summary_path: Path,
    tuning_best_params_path: Path,
    subgroup_reporting_enabled: bool,
    subgroup_dimensions: list[str],
    subgroup_min_samples_per_group: int,
    subgroup_metrics_json_path: Path,
    subgroup_metrics_csv_path: Path,
    filter_task: str | None,
    filter_modality: str | None,
    n_permutations: int,
    framework_mode: str,
    canonical_run: bool,
    identity: RunIdentity,
    protocol_context: dict[str, Any],
    comparison_context: dict[str, Any],
    start_section: str | None,
    end_section: str | None,
    base_artifact_id: str | None,
    reuse_policy: str | None,
    force: bool,
    resume: bool,
    reuse_completed_artifacts: bool,
    run_mode: str,
    segment_result: SegmentExecutionResult,
    fold_splits_path: Path,
    spatial_compatibility_status: str | None,
    spatial_compatibility_passed: bool | None,
    spatial_compatibility_n_groups_checked: int | None,
    spatial_compatibility_reference_group_id: str | None,
    spatial_compatibility_affine_atol: float | None,
    spatial_compatibility_report_path: Path,
    interpretability_enabled: bool | None,
    interpretability_performed: bool | None,
    interpretability_status: str | None,
    interpretability_fold_artifacts_path: str | None,
    interpretability_summary_path: Path,
    python_version: str,
    numpy_version: str,
    pandas_version: str,
    sklearn_version: str,
    nibabel_version: str,
    git_commit: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "index_csv": str(index_csv.resolve()),
        "index_csv_relative": _relative_path(index_csv),
        "data_root": str(data_root.resolve()),
        "data_root_relative": _relative_path(data_root),
        "cache_dir": str(cache_dir.resolve()),
        "cache_dir_relative": _relative_path(cache_dir),
        "target": target_column,
        "model": model,
        "cv": cv_mode,
        "experiment_mode": cv_mode,
        "subject": str(subject) if cv_mode == "within_subject_loso_session" else None,
        "train_subject": str(train_subject) if cv_mode == "frozen_cross_person_transfer" else None,
        "test_subject": str(test_subject) if cv_mode == "frozen_cross_person_transfer" else None,
        "seed": int(seed),
        "primary_metric_name": primary_metric_name,
        "permutation_metric_name": permutation_metric_name,
        "metric_policy_effective": metric_policy_effective_payload(metric_policy_effective),
        "decision_metric_name": metric_policy_effective.decision_metric,
        "tuning_metric_name": metric_policy_effective.tuning_metric,
        "methodology_policy_name": methodology_policy_name,
        "class_weight_policy": class_weight_policy,
        "tuning_enabled": bool(tuning_enabled),
        "tuning_search_space_id": tuning_search_space_id,
        "tuning_search_space_version": tuning_search_space_version,
        "tuning_inner_cv_scheme": tuning_inner_cv_scheme,
        "tuning_inner_group_field": tuning_inner_group_field,
        "tuning_summary_path": str(tuning_summary_path.resolve()),
        "tuning_summary_path_relative": _relative_path(tuning_summary_path),
        "tuning_best_params_path": str(tuning_best_params_path.resolve()),
        "tuning_best_params_path_relative": _relative_path(tuning_best_params_path),
        "subgroup_reporting_enabled": bool(subgroup_reporting_enabled),
        "subgroup_dimensions": list(subgroup_dimensions),
        "subgroup_min_samples_per_group": int(subgroup_min_samples_per_group),
        "subgroup_metrics_json_path": str(subgroup_metrics_json_path.resolve()),
        "subgroup_metrics_json_path_relative": _relative_path(subgroup_metrics_json_path),
        "subgroup_metrics_csv_path": str(subgroup_metrics_csv_path.resolve()),
        "subgroup_metrics_csv_path_relative": _relative_path(subgroup_metrics_csv_path),
        "filter_task": filter_task,
        "filter_modality": filter_modality,
        "n_permutations": int(n_permutations),
        "framework_mode": framework_mode,
        "canonical_run": bool(canonical_run),
        "protocol_id": identity.protocol_id,
        "protocol_version": identity.protocol_version,
        "protocol_schema_version": identity.protocol_schema_version,
        "suite_id": identity.suite_id,
        "claim_ids": identity.claim_ids,
        "comparison_id": identity.comparison_id,
        "comparison_version": identity.comparison_version,
        "comparison_variant_id": identity.comparison_variant_id,
        "protocol_context": protocol_context if protocol_context else None,
        "comparison_context": comparison_context if comparison_context else None,
        "start_section": start_section,
        "end_section": end_section,
        "base_artifact_id": base_artifact_id,
        "reuse_policy": reuse_policy,
        "force": bool(force),
        "resume": bool(resume),
        "reuse_completed_artifacts": bool(reuse_completed_artifacts),
        "run_mode": run_mode,
        "planned_sections": segment_result.planned_sections,
        "executed_sections": segment_result.executed_sections,
        "reused_sections": segment_result.reused_sections,
        "fold_splits_path": str(fold_splits_path.resolve()),
        "fold_splits_path_relative": _relative_path(fold_splits_path),
        "spatial_compatibility_status": spatial_compatibility_status,
        "spatial_compatibility_passed": spatial_compatibility_passed,
        "spatial_compatibility_n_groups_checked": spatial_compatibility_n_groups_checked,
        "spatial_compatibility_reference_group_id": spatial_compatibility_reference_group_id,
        "spatial_compatibility_affine_atol": spatial_compatibility_affine_atol,
        "spatial_compatibility_report_path": str(spatial_compatibility_report_path.resolve()),
        "spatial_compatibility_report_path_relative": _relative_path(
            spatial_compatibility_report_path
        ),
        "interpretability_enabled": interpretability_enabled,
        "interpretability_performed": interpretability_performed,
        "interpretability_status": interpretability_status,
        "interpretability_fold_artifacts_path": interpretability_fold_artifacts_path,
        "interpretability_summary_path": str(interpretability_summary_path.resolve()),
        "interpretability_summary_path_relative": _relative_path(interpretability_summary_path),
        "python_version": python_version,
        "numpy_version": numpy_version,
        "pandas_version": pandas_version,
        "sklearn_version": sklearn_version,
        "nibabel_version": nibabel_version,
        "git_commit": git_commit,
    }


def build_run_result_payload(
    *,
    run_id: str,
    report_dir: Path,
    config_path: Path,
    metrics_path: Path,
    subgroup_metrics_json_path: Path,
    subgroup_metrics_csv_path: Path,
    tuning_summary_path: Path,
    tuning_best_params_path: Path,
    fold_metrics_path: Path,
    fold_splits_path: Path,
    predictions_path: Path,
    spatial_compatibility_report_path: Path,
    interpretability_summary_path: Path,
    interpretability_fold_artifacts_path: str | None,
    artifact_registry_path: Path,
    segment_result: SegmentExecutionResult,
    artifact_ids: dict[str, str],
    metrics: dict[str, Any],
    run_status_path: Path,
    run_mode: str,
    framework_mode: str,
    canonical_run: bool,
    metric_policy_effective: EffectiveMetricPolicy,
    methodology_policy_name: str,
    class_weight_policy: str,
    tuning_enabled: bool,
    protocol_context: dict[str, Any],
    comparison_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "report_dir": str(report_dir.resolve()),
        "report_dir_relative": _relative_path(report_dir),
        "config_path": str(config_path.resolve()),
        "config_path_relative": _relative_path(config_path),
        "metrics_path": str(metrics_path.resolve()),
        "metrics_path_relative": _relative_path(metrics_path),
        "subgroup_metrics_json_path": str(subgroup_metrics_json_path.resolve()),
        "subgroup_metrics_json_path_relative": _relative_path(subgroup_metrics_json_path),
        "subgroup_metrics_csv_path": str(subgroup_metrics_csv_path.resolve()),
        "subgroup_metrics_csv_path_relative": _relative_path(subgroup_metrics_csv_path),
        "tuning_summary_path": str(tuning_summary_path.resolve()),
        "tuning_summary_path_relative": _relative_path(tuning_summary_path),
        "tuning_best_params_path": str(tuning_best_params_path.resolve()),
        "tuning_best_params_path_relative": _relative_path(tuning_best_params_path),
        "fold_metrics_path": str(fold_metrics_path.resolve()),
        "fold_metrics_path_relative": _relative_path(fold_metrics_path),
        "fold_splits_path": str(fold_splits_path.resolve()),
        "fold_splits_path_relative": _relative_path(fold_splits_path),
        "predictions_path": str(predictions_path.resolve()),
        "predictions_path_relative": _relative_path(predictions_path),
        "spatial_compatibility_report_path": str(spatial_compatibility_report_path.resolve()),
        "spatial_compatibility_report_path_relative": _relative_path(
            spatial_compatibility_report_path
        ),
        "interpretability_summary_path": str(interpretability_summary_path.resolve()),
        "interpretability_summary_path_relative": _relative_path(interpretability_summary_path),
        "interpretability_fold_artifacts_path": interpretability_fold_artifacts_path,
        "artifact_registry_path": str(artifact_registry_path.resolve()),
        "artifact_registry_path_relative": _relative_path(artifact_registry_path),
        "planned_sections": segment_result.planned_sections,
        "executed_sections": segment_result.executed_sections,
        "reused_sections": segment_result.reused_sections,
        "artifact_ids": artifact_ids,
        "metrics": metrics,
        "run_status_path": str(run_status_path.resolve()),
        "run_status_path_relative": _relative_path(run_status_path),
        "run_mode": run_mode,
        "framework_mode": framework_mode,
        "canonical_run": bool(canonical_run),
        "metric_policy_effective": metric_policy_effective_payload(metric_policy_effective),
        "methodology_policy_name": methodology_policy_name,
        "class_weight_policy": class_weight_policy,
        "tuning_enabled": bool(tuning_enabled),
        "protocol_context": protocol_context if protocol_context else None,
        "comparison_context": comparison_context if comparison_context else None,
    }
