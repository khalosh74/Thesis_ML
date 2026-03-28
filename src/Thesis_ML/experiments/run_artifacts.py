from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Thesis_ML.config.metric_policy import EffectiveMetricPolicy
from Thesis_ML.experiments.compute_policy import (
    ResolvedComputePolicy,
    stamp_compute_policy_metadata,
)
from Thesis_ML.experiments.model_admission import official_admission_summary
from Thesis_ML.experiments.model_registry import MODEL_REGISTRY_VERSION, get_model_spec
from Thesis_ML.experiments.segment_execution import SegmentExecutionResult
from Thesis_ML.experiments.stage_execution import (
    StageExecutionResult as StageExecutionMetadata,
)
from Thesis_ML.experiments.stage_execution import (
    stage_execution_payload,
)


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
            str(protocol_context.get("protocol_id"))
            if protocol_context.get("protocol_id")
            else None
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
        suite_id=str(protocol_context.get("suite_id"))
        if protocol_context.get("suite_id")
        else None,
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


def _normalized_backend_family_from_compute_policy(
    compute_policy: ResolvedComputePolicy | dict[str, Any] | None,
) -> str | None:
    if isinstance(compute_policy, ResolvedComputePolicy):
        candidate = compute_policy.assigned_backend_family or compute_policy.effective_backend_family
        return str(candidate).strip().lower()
    if isinstance(compute_policy, dict):
        candidate = compute_policy.get("assigned_backend_family") or compute_policy.get(
            "effective_backend_family"
        )
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
    return None


def _compute_backend_family_for_model_binding(backend_family: str | None) -> str | None:
    normalized = str(backend_family).strip().lower() if backend_family is not None else None
    if normalized in {"sklearn_cpu", "xgboost_cpu"}:
        return "sklearn_cpu"
    if normalized in {"torch_gpu", "xgboost_gpu"}:
        return "torch_gpu"
    return None


def _resolve_best_params_payload(tuning_best_params_path: Path) -> list[dict[str, Any]] | None:
    if not tuning_best_params_path.exists() or not tuning_best_params_path.is_file():
        return None
    resolved_rows: list[dict[str, Any]] = []
    with tuning_best_params_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            best_params_json = str(row.get("best_params_json", "")).strip()
            if not best_params_json or best_params_json == "{}":
                continue
            fold_value = row.get("fold")
            resolved_rows.append(
                {
                    "fold": int(fold_value) if str(fold_value).strip().isdigit() else fold_value,
                    "best_params_json": best_params_json,
                }
            )
    return resolved_rows if resolved_rows else None


def _model_governance_payload(
    *,
    model_name: str,
    framework_mode: str,
    feature_recipe_id: str,
    tuning_search_space_id: str | None,
    tuning_search_space_version: str | None,
    compute_policy: ResolvedComputePolicy | dict[str, Any] | None,
    compute_runtime_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = get_model_spec(model_name)
    backend_family = _normalized_backend_family_from_compute_policy(compute_policy)
    runtime_backend_family = (
        str(compute_runtime_metadata.get("actual_estimator_backend_family")).strip().lower()
        if isinstance(compute_runtime_metadata, dict)
        and isinstance(compute_runtime_metadata.get("actual_estimator_backend_family"), str)
        and str(compute_runtime_metadata.get("actual_estimator_backend_family")).strip()
        else None
    )
    resolved_backend_family = runtime_backend_family or backend_family
    compute_backend_family = _compute_backend_family_for_model_binding(resolved_backend_family)
    binding = (
        spec.backend_binding_for_compute_family(compute_backend_family)
        if compute_backend_family is not None
        else None
    )
    backend_id = (
        str(compute_runtime_metadata.get("actual_estimator_backend_id"))
        if isinstance(compute_runtime_metadata, dict)
        and compute_runtime_metadata.get("actual_estimator_backend_id") is not None
        else None
    )
    if backend_id is None and binding is not None:
        backend_id = str(binding.backend_id)
    if backend_id is None and isinstance(compute_runtime_metadata, dict):
        runtime_backend_id = compute_runtime_metadata.get("backend_id")
        if runtime_backend_id is not None:
            backend_id = str(runtime_backend_id)

    hardware_mode = "cpu_only"
    deterministic_compute = False
    allow_backend_fallback = False
    scheduler_lane_assignment = None
    if isinstance(compute_policy, ResolvedComputePolicy):
        hardware_mode = str(compute_policy.hardware_mode_requested)
        deterministic_compute = bool(compute_policy.deterministic_compute)
        allow_backend_fallback = bool(compute_policy.allow_backend_fallback)
        scheduler_lane_assignment = compute_policy.assigned_compute_lane
    elif isinstance(compute_policy, dict):
        if compute_policy.get("hardware_mode_requested") is not None:
            hardware_mode = str(compute_policy.get("hardware_mode_requested"))
        deterministic_compute = bool(compute_policy.get("deterministic_compute", False))
        allow_backend_fallback = bool(compute_policy.get("allow_backend_fallback", False))
        scheduler_lane_assignment = compute_policy.get("assigned_compute_lane")

    admission = official_admission_summary(
        framework_mode=framework_mode,
        model_name=spec.logical_name,
        backend_family=resolved_backend_family,
        hardware_mode=hardware_mode,
    )

    return {
        "logical_model_name": spec.logical_name,
        "model_family": spec.model_family,
        "backend_family": resolved_backend_family,
        "backend_id": backend_id,
        "feature_recipe_id": str(feature_recipe_id),
        "tuning_search_space_id": tuning_search_space_id,
        "tuning_search_space_version": tuning_search_space_version,
        "official_admission_summary": admission,
        "deterministic_compute_required": bool(admission.get("deterministic_compute_required")),
        "deterministic_compute": bool(deterministic_compute),
        "allow_backend_fallback": bool(allow_backend_fallback),
        "scheduler_lane_assignment": scheduler_lane_assignment,
        "model_registry_version": MODEL_REGISTRY_VERSION,
    }


def _stage_execution_payload(
    stage_execution: StageExecutionMetadata | dict[str, Any] | None,
) -> dict[str, Any] | None:
    return stage_execution_payload(stage_execution)


def stamp_metrics_artifact(
    *,
    metrics_path: Path,
    canonical_run: bool,
    framework_mode: str,
    repeat_id: int,
    repeat_count: int,
    base_run_id: str,
    evidence_run_role: str,
    evidence_policy_effective: dict[str, Any],
    methodology_policy_name: str,
    class_weight_policy: str,
    tuning_enabled: bool,
    model_cost_tier: str,
    projected_runtime_seconds: int,
    primary_metric_aggregation: str,
    preprocessing_kind: str | None,
    feature_recipe_id: str,
    tuning_summary_path: Path,
    tuning_best_params_path: Path,
    fit_timing_summary_path: Path,
    feature_qc_summary_path: Path,
    feature_qc_selected_samples_path: Path,
    subgroup_metrics_json_path: Path,
    subgroup_metrics_csv_path: Path,
    metric_policy_effective: EffectiveMetricPolicy,
    data_policy_effective: dict[str, Any] | None,
    data_artifacts: dict[str, Any] | None,
    identity: RunIdentity,
    dataset_fingerprint: dict[str, Any] | None = None,
    git_provenance: dict[str, Any] | None = None,
    stage_timings_seconds: dict[str, float] | None = None,
    resource_summary: dict[str, Any] | None = None,
    warning_summary: dict[str, Any] | None = None,
    timeout_policy_effective: dict[str, Any] | None = None,
    profiling_context: dict[str, Any] | None = None,
    compute_policy: ResolvedComputePolicy | None = None,
    compute_runtime_metadata: dict[str, Any] | None = None,
    stage_execution: StageExecutionMetadata | dict[str, Any] | None = None,
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
    persisted_metrics["repeat_id"] = int(repeat_id)
    persisted_metrics["repeat_count"] = int(repeat_count)
    persisted_metrics["base_run_id"] = str(base_run_id)
    persisted_metrics["evidence_run_role"] = str(evidence_run_role)
    persisted_metrics["evidence_policy_effective"] = dict(evidence_policy_effective)
    persisted_metrics["methodology_policy_name"] = methodology_policy_name
    persisted_metrics["class_weight_policy"] = class_weight_policy
    persisted_metrics["tuning_enabled"] = bool(tuning_enabled)
    persisted_metrics["model_cost_tier"] = str(model_cost_tier)
    persisted_metrics["projected_runtime_seconds"] = int(projected_runtime_seconds)
    persisted_metrics["primary_metric_aggregation"] = str(primary_metric_aggregation)
    persisted_metrics["preprocessing_kind"] = (
        str(preprocessing_kind) if preprocessing_kind is not None else None
    )
    persisted_metrics["feature_recipe_id"] = str(feature_recipe_id)
    persisted_metrics["tuning_summary_path"] = str(tuning_summary_path.resolve())
    persisted_metrics["tuning_summary_path_relative"] = _relative_path(tuning_summary_path)
    persisted_metrics["tuning_best_params_path"] = str(tuning_best_params_path.resolve())
    persisted_metrics["tuning_best_params_path_relative"] = _relative_path(tuning_best_params_path)
    persisted_metrics["fit_timing_summary_path"] = str(fit_timing_summary_path.resolve())
    persisted_metrics["fit_timing_summary_path_relative"] = _relative_path(fit_timing_summary_path)
    persisted_metrics["feature_qc_summary_path"] = str(feature_qc_summary_path.resolve())
    persisted_metrics["feature_qc_summary_path_relative"] = _relative_path(feature_qc_summary_path)
    persisted_metrics["feature_qc_selected_samples_path"] = str(
        feature_qc_selected_samples_path.resolve()
    )
    persisted_metrics["feature_qc_selected_samples_path_relative"] = _relative_path(
        feature_qc_selected_samples_path
    )
    persisted_metrics["subgroup_metrics_json_path"] = str(subgroup_metrics_json_path.resolve())
    persisted_metrics["subgroup_metrics_json_path_relative"] = _relative_path(
        subgroup_metrics_json_path
    )
    persisted_metrics["subgroup_metrics_csv_path"] = str(subgroup_metrics_csv_path.resolve())
    persisted_metrics["subgroup_metrics_csv_path_relative"] = _relative_path(
        subgroup_metrics_csv_path
    )
    persisted_metrics["metric_policy_effective"] = metric_policy_effective_payload(
        metric_policy_effective
    )
    persisted_metrics["data_policy_effective"] = (
        dict(data_policy_effective) if isinstance(data_policy_effective, dict) else None
    )
    persisted_metrics["data_artifacts"] = (
        dict(data_artifacts) if isinstance(data_artifacts, dict) else None
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
    resolved_model_name = (
        str(persisted_metrics.get("model")).strip().lower()
        if persisted_metrics.get("model") is not None
        else None
    )
    if resolved_model_name:
        governance_payload = _model_governance_payload(
            model_name=resolved_model_name,
            framework_mode=str(framework_mode),
            feature_recipe_id=str(feature_recipe_id),
            tuning_search_space_id=(
                str(persisted_metrics.get("tuning_search_space_id"))
                if persisted_metrics.get("tuning_search_space_id") is not None
                else None
            ),
            tuning_search_space_version=(
                str(persisted_metrics.get("tuning_search_space_version"))
                if persisted_metrics.get("tuning_search_space_version") is not None
                else None
            ),
            compute_policy=compute_policy,
            compute_runtime_metadata=compute_runtime_metadata,
        )
        persisted_metrics["model_governance"] = dict(governance_payload)
        persisted_metrics.update(governance_payload)
    resolved_best_params = _resolve_best_params_payload(tuning_best_params_path)
    if resolved_best_params is not None:
        persisted_metrics["resolved_best_params"] = resolved_best_params
    if dataset_fingerprint is not None:
        persisted_metrics["dataset_fingerprint"] = dict(dataset_fingerprint)
    if git_provenance is not None:
        persisted_metrics["git_provenance"] = dict(git_provenance)
    if stage_timings_seconds is not None:
        persisted_metrics["stage_timings_seconds"] = {
            str(key): float(value) for key, value in stage_timings_seconds.items()
        }
    if resource_summary is not None:
        persisted_metrics["resource_summary"] = dict(resource_summary)
    if warning_summary is not None:
        persisted_metrics["warning_summary"] = dict(warning_summary)
    if timeout_policy_effective is not None:
        persisted_metrics["timeout_policy_effective"] = dict(timeout_policy_effective)
    if profiling_context is not None:
        persisted_metrics["profiling_context"] = dict(profiling_context)
        persisted_metrics["profiling_only"] = bool(profiling_context.get("profiling_only", False))
    stage_execution_data = _stage_execution_payload(stage_execution)
    if stage_execution_data is not None:
        persisted_metrics["stage_execution"] = stage_execution_data
    stamp_compute_policy_metadata(
        payload=persisted_metrics,
        compute_policy=compute_policy,
        compute_runtime_metadata=compute_runtime_metadata,
    )
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
    primary_metric_aggregation: str,
    permutation_metric_name: str,
    repeat_id: int,
    repeat_count: int,
    base_run_id: str,
    evidence_run_role: str,
    evidence_policy_effective: dict[str, Any],
    data_policy_effective: dict[str, Any] | None,
    metric_policy_effective: EffectiveMetricPolicy,
    methodology_policy_name: str,
    class_weight_policy: str,
    tuning_enabled: bool,
    model_cost_tier: str,
    projected_runtime_seconds: int,
    preprocessing_kind: str | None,
    feature_recipe_id: str,
    tuning_search_space_id: str | None,
    tuning_search_space_version: str | None,
    tuning_inner_cv_scheme: str | None,
    tuning_inner_group_field: str | None,
    tuning_summary_path: Path,
    tuning_best_params_path: Path,
    fit_timing_summary_path: Path,
    feature_qc_summary_path: Path,
    feature_qc_selected_samples_path: Path,
    calibration_summary_path: Path,
    calibration_table_path: Path,
    subgroup_reporting_enabled: bool,
    subgroup_dimensions: list[str],
    subgroup_min_samples_per_group: int,
    subgroup_metrics_json_path: Path,
    subgroup_metrics_csv_path: Path,
    dataset_card_json_path: Path,
    dataset_card_md_path: Path,
    dataset_summary_json_path: Path,
    dataset_summary_csv_path: Path,
    data_quality_report_path: Path,
    class_balance_report_path: Path,
    missingness_report_path: Path,
    leakage_audit_path: Path,
    external_dataset_card_path: Path,
    external_dataset_summary_path: Path,
    external_validation_compatibility_path: Path,
    data_artifacts: dict[str, Any] | None,
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
    git_branch: str | None,
    git_dirty: bool,
    dataset_fingerprint: dict[str, Any] | None = None,
    stage_timings_seconds: dict[str, float] | None = None,
    resource_summary: dict[str, Any] | None = None,
    warning_summary: dict[str, Any] | None = None,
    timeout_policy_effective: dict[str, Any] | None = None,
    profiling_context: dict[str, Any] | None = None,
    compute_policy: ResolvedComputePolicy | None = None,
    compute_runtime_metadata: dict[str, Any] | None = None,
    stage_execution: StageExecutionMetadata | dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
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
        "repeat_id": int(repeat_id),
        "repeat_count": int(repeat_count),
        "base_run_id": str(base_run_id),
        "evidence_run_role": str(evidence_run_role),
        "evidence_policy_effective": dict(evidence_policy_effective),
        "data_policy_effective": (
            dict(data_policy_effective) if isinstance(data_policy_effective, dict) else None
        ),
        "primary_metric_name": primary_metric_name,
        "primary_metric_aggregation": str(primary_metric_aggregation),
        "permutation_metric_name": permutation_metric_name,
        "metric_policy_effective": metric_policy_effective_payload(metric_policy_effective),
        "decision_metric_name": metric_policy_effective.decision_metric,
        "tuning_metric_name": metric_policy_effective.tuning_metric,
        "methodology_policy_name": methodology_policy_name,
        "class_weight_policy": class_weight_policy,
        "tuning_enabled": bool(tuning_enabled),
        "model_cost_tier": str(model_cost_tier),
        "projected_runtime_seconds": int(projected_runtime_seconds),
        "preprocessing_kind": (str(preprocessing_kind) if preprocessing_kind is not None else None),
        "feature_recipe_id": str(feature_recipe_id),
        "tuning_search_space_id": tuning_search_space_id,
        "tuning_search_space_version": tuning_search_space_version,
        "tuning_inner_cv_scheme": tuning_inner_cv_scheme,
        "tuning_inner_group_field": tuning_inner_group_field,
        "tuning_summary_path": str(tuning_summary_path.resolve()),
        "tuning_summary_path_relative": _relative_path(tuning_summary_path),
        "tuning_best_params_path": str(tuning_best_params_path.resolve()),
        "tuning_best_params_path_relative": _relative_path(tuning_best_params_path),
        "fit_timing_summary_path": str(fit_timing_summary_path.resolve()),
        "fit_timing_summary_path_relative": _relative_path(fit_timing_summary_path),
        "feature_qc_summary_path": str(feature_qc_summary_path.resolve()),
        "feature_qc_summary_path_relative": _relative_path(feature_qc_summary_path),
        "feature_qc_selected_samples_path": str(feature_qc_selected_samples_path.resolve()),
        "feature_qc_selected_samples_path_relative": _relative_path(
            feature_qc_selected_samples_path
        ),
        "calibration_summary_path": str(calibration_summary_path.resolve()),
        "calibration_summary_path_relative": _relative_path(calibration_summary_path),
        "calibration_table_path": str(calibration_table_path.resolve()),
        "calibration_table_path_relative": _relative_path(calibration_table_path),
        "subgroup_reporting_enabled": bool(subgroup_reporting_enabled),
        "subgroup_dimensions": list(subgroup_dimensions),
        "subgroup_min_samples_per_group": int(subgroup_min_samples_per_group),
        "subgroup_metrics_json_path": str(subgroup_metrics_json_path.resolve()),
        "subgroup_metrics_json_path_relative": _relative_path(subgroup_metrics_json_path),
        "subgroup_metrics_csv_path": str(subgroup_metrics_csv_path.resolve()),
        "subgroup_metrics_csv_path_relative": _relative_path(subgroup_metrics_csv_path),
        "dataset_card_json_path": str(dataset_card_json_path.resolve()),
        "dataset_card_json_path_relative": _relative_path(dataset_card_json_path),
        "dataset_card_md_path": str(dataset_card_md_path.resolve()),
        "dataset_card_md_path_relative": _relative_path(dataset_card_md_path),
        "dataset_summary_json_path": str(dataset_summary_json_path.resolve()),
        "dataset_summary_json_path_relative": _relative_path(dataset_summary_json_path),
        "dataset_summary_csv_path": str(dataset_summary_csv_path.resolve()),
        "dataset_summary_csv_path_relative": _relative_path(dataset_summary_csv_path),
        "data_quality_report_path": str(data_quality_report_path.resolve()),
        "data_quality_report_path_relative": _relative_path(data_quality_report_path),
        "class_balance_report_path": str(class_balance_report_path.resolve()),
        "class_balance_report_path_relative": _relative_path(class_balance_report_path),
        "missingness_report_path": str(missingness_report_path.resolve()),
        "missingness_report_path_relative": _relative_path(missingness_report_path),
        "leakage_audit_path": str(leakage_audit_path.resolve()),
        "leakage_audit_path_relative": _relative_path(leakage_audit_path),
        "external_dataset_card_path": str(external_dataset_card_path.resolve()),
        "external_dataset_card_path_relative": _relative_path(external_dataset_card_path),
        "external_dataset_summary_path": str(external_dataset_summary_path.resolve()),
        "external_dataset_summary_path_relative": _relative_path(external_dataset_summary_path),
        "external_validation_compatibility_path": str(
            external_validation_compatibility_path.resolve()
        ),
        "external_validation_compatibility_path_relative": _relative_path(
            external_validation_compatibility_path
        ),
        "data_artifacts": dict(data_artifacts) if isinstance(data_artifacts, dict) else None,
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
        "git_branch": git_branch,
        "git_dirty": bool(git_dirty),
        "dataset_fingerprint": dict(dataset_fingerprint) if dataset_fingerprint else None,
        "stage_timings_seconds": (
            {str(key): float(value) for key, value in stage_timings_seconds.items()}
            if stage_timings_seconds is not None
            else None
        ),
        "resource_summary": dict(resource_summary) if resource_summary is not None else None,
        "warning_summary": dict(warning_summary) if warning_summary is not None else None,
        "timeout_policy_effective": (
            dict(timeout_policy_effective) if isinstance(timeout_policy_effective, dict) else None
        ),
        "profiling_context": (
            dict(profiling_context) if isinstance(profiling_context, dict) else None
        ),
        "profiling_only": (
            bool(profiling_context.get("profiling_only", False))
            if isinstance(profiling_context, dict)
            else False
        ),
    }
    governance_payload = _model_governance_payload(
        model_name=str(model),
        framework_mode=str(framework_mode),
        feature_recipe_id=str(feature_recipe_id),
        tuning_search_space_id=tuning_search_space_id,
        tuning_search_space_version=tuning_search_space_version,
        compute_policy=compute_policy,
        compute_runtime_metadata=compute_runtime_metadata,
    )
    payload["model_governance"] = dict(governance_payload)
    payload.update(governance_payload)
    stage_execution_data = _stage_execution_payload(stage_execution)
    if stage_execution_data is not None:
        payload["stage_execution"] = stage_execution_data
    stamp_compute_policy_metadata(
        payload=payload,
        compute_policy=compute_policy,
        compute_runtime_metadata=compute_runtime_metadata,
    )
    return payload


def build_run_result_payload(
    *,
    run_id: str,
    report_dir: Path,
    config_path: Path,
    metrics_path: Path,
    subgroup_metrics_json_path: Path,
    subgroup_metrics_csv_path: Path,
    dataset_card_json_path: Path,
    dataset_card_md_path: Path,
    dataset_summary_json_path: Path,
    dataset_summary_csv_path: Path,
    data_quality_report_path: Path,
    class_balance_report_path: Path,
    missingness_report_path: Path,
    leakage_audit_path: Path,
    external_dataset_card_path: Path,
    external_dataset_summary_path: Path,
    external_validation_compatibility_path: Path,
    tuning_summary_path: Path,
    tuning_best_params_path: Path,
    fit_timing_summary_path: Path,
    feature_qc_summary_path: Path,
    feature_qc_selected_samples_path: Path,
    calibration_summary_path: Path,
    calibration_table_path: Path,
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
    repeat_id: int,
    repeat_count: int,
    base_run_id: str,
    evidence_run_role: str,
    evidence_policy_effective: dict[str, Any],
    data_policy_effective: dict[str, Any] | None,
    data_artifacts: dict[str, Any] | None,
    metric_policy_effective: EffectiveMetricPolicy,
    methodology_policy_name: str,
    class_weight_policy: str,
    tuning_enabled: bool,
    model_cost_tier: str,
    projected_runtime_seconds: int,
    feature_recipe_id: str,
    protocol_context: dict[str, Any],
    comparison_context: dict[str, Any],
    stage_timings_seconds: dict[str, float] | None = None,
    resource_summary: dict[str, Any] | None = None,
    warning_summary: dict[str, Any] | None = None,
    dataset_fingerprint: dict[str, Any] | None = None,
    timeout_policy_effective: dict[str, Any] | None = None,
    profiling_context: dict[str, Any] | None = None,
    compute_policy: ResolvedComputePolicy | None = None,
    compute_runtime_metadata: dict[str, Any] | None = None,
    stage_execution: StageExecutionMetadata | dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
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
        "dataset_card_json_path": str(dataset_card_json_path.resolve()),
        "dataset_card_json_path_relative": _relative_path(dataset_card_json_path),
        "dataset_card_md_path": str(dataset_card_md_path.resolve()),
        "dataset_card_md_path_relative": _relative_path(dataset_card_md_path),
        "dataset_summary_json_path": str(dataset_summary_json_path.resolve()),
        "dataset_summary_json_path_relative": _relative_path(dataset_summary_json_path),
        "dataset_summary_csv_path": str(dataset_summary_csv_path.resolve()),
        "dataset_summary_csv_path_relative": _relative_path(dataset_summary_csv_path),
        "data_quality_report_path": str(data_quality_report_path.resolve()),
        "data_quality_report_path_relative": _relative_path(data_quality_report_path),
        "class_balance_report_path": str(class_balance_report_path.resolve()),
        "class_balance_report_path_relative": _relative_path(class_balance_report_path),
        "missingness_report_path": str(missingness_report_path.resolve()),
        "missingness_report_path_relative": _relative_path(missingness_report_path),
        "leakage_audit_path": str(leakage_audit_path.resolve()),
        "leakage_audit_path_relative": _relative_path(leakage_audit_path),
        "external_dataset_card_path": str(external_dataset_card_path.resolve()),
        "external_dataset_card_path_relative": _relative_path(external_dataset_card_path),
        "external_dataset_summary_path": str(external_dataset_summary_path.resolve()),
        "external_dataset_summary_path_relative": _relative_path(external_dataset_summary_path),
        "external_validation_compatibility_path": str(
            external_validation_compatibility_path.resolve()
        ),
        "external_validation_compatibility_path_relative": _relative_path(
            external_validation_compatibility_path
        ),
        "tuning_summary_path": str(tuning_summary_path.resolve()),
        "tuning_summary_path_relative": _relative_path(tuning_summary_path),
        "tuning_best_params_path": str(tuning_best_params_path.resolve()),
        "tuning_best_params_path_relative": _relative_path(tuning_best_params_path),
        "fit_timing_summary_path": str(fit_timing_summary_path.resolve()),
        "fit_timing_summary_path_relative": _relative_path(fit_timing_summary_path),
        "feature_qc_summary_path": str(feature_qc_summary_path.resolve()),
        "feature_qc_summary_path_relative": _relative_path(feature_qc_summary_path),
        "feature_qc_selected_samples_path": str(feature_qc_selected_samples_path.resolve()),
        "feature_qc_selected_samples_path_relative": _relative_path(
            feature_qc_selected_samples_path
        ),
        "calibration_summary_path": str(calibration_summary_path.resolve()),
        "calibration_summary_path_relative": _relative_path(calibration_summary_path),
        "calibration_table_path": str(calibration_table_path.resolve()),
        "calibration_table_path_relative": _relative_path(calibration_table_path),
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
        "repeat_id": int(repeat_id),
        "repeat_count": int(repeat_count),
        "base_run_id": str(base_run_id),
        "evidence_run_role": str(evidence_run_role),
        "evidence_policy_effective": dict(evidence_policy_effective),
        "data_policy_effective": (
            dict(data_policy_effective) if isinstance(data_policy_effective, dict) else None
        ),
        "data_artifacts": dict(data_artifacts) if isinstance(data_artifacts, dict) else None,
        "metric_policy_effective": metric_policy_effective_payload(metric_policy_effective),
        "methodology_policy_name": methodology_policy_name,
        "class_weight_policy": class_weight_policy,
        "tuning_enabled": bool(tuning_enabled),
        "model_cost_tier": str(model_cost_tier),
        "projected_runtime_seconds": int(projected_runtime_seconds),
        "feature_recipe_id": str(feature_recipe_id),
        "protocol_context": protocol_context if protocol_context else None,
        "comparison_context": comparison_context if comparison_context else None,
        "stage_timings_seconds": (
            {str(key): float(value) for key, value in stage_timings_seconds.items()}
            if stage_timings_seconds is not None
            else None
        ),
        "resource_summary": dict(resource_summary) if resource_summary is not None else None,
        "warning_summary": dict(warning_summary) if warning_summary is not None else None,
        "dataset_fingerprint": dict(dataset_fingerprint) if dataset_fingerprint else None,
        "timeout_policy_effective": (
            dict(timeout_policy_effective) if isinstance(timeout_policy_effective, dict) else None
        ),
        "profiling_context": (
            dict(profiling_context) if isinstance(profiling_context, dict) else None
        ),
        "profiling_only": (
            bool(profiling_context.get("profiling_only", False))
            if isinstance(profiling_context, dict)
            else False
        ),
    }
    stage_execution_data = _stage_execution_payload(stage_execution)
    if stage_execution_data is not None:
        payload["stage_execution"] = stage_execution_data
    stamp_compute_policy_metadata(
        payload=payload,
        compute_policy=compute_policy,
        compute_runtime_metadata=compute_runtime_metadata,
    )
    return payload
