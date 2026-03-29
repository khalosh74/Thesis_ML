from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from Thesis_ML.config.metric_policy import validate_metric_name


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _json_text(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, list):
        return json.dumps(value)
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _required_metric_name(value: Any, *, context: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{context} is missing required primary_metric_name.")
    return validate_metric_name(str(value).strip())


def _build_dataset_subset_label(params: dict[str, Any]) -> str:
    subject = str(params.get("subject") or "").strip()
    train_subject = str(params.get("train_subject") or "").strip()
    test_subject = str(params.get("test_subject") or "").strip()
    task = str(params.get("filter_task") or "all_tasks").strip() or "all_tasks"
    modality = str(params.get("filter_modality") or "all_modalities").strip() or "all_modalities"
    feature_space = (
        str(params.get("feature_space") or "whole_brain_masked").strip()
        or "whole_brain_masked"
    )
    dimensionality_strategy = (
        str(params.get("dimensionality_strategy") or "none").strip() or "none"
    )
    if train_subject and test_subject:
        subject_part = f"train={train_subject}|test={test_subject}"
    elif subject:
        subject_part = f"subject={subject}"
    else:
        subject_part = "subject=pooled"
    return (
        f"{subject_part};task={task};modality={modality};feature_space={feature_space};"
        f"dimensionality={dimensionality_strategy}"
    )


def _feature_set_label(params: dict[str, Any]) -> str:
    feature_space = str(params.get("feature_space") or "whole_brain_masked").strip().lower()
    dimensionality_strategy = (
        str(params.get("dimensionality_strategy") or "none").strip().lower()
    )
    pca_variance_ratio = params.get("pca_variance_ratio")
    pca_n_components = params.get("pca_n_components")
    pca_suffix = ""
    if dimensionality_strategy == "pca":
        if pca_variance_ratio is not None:
            pca_suffix = f" + PCA(var={pca_variance_ratio})"
        elif pca_n_components is not None:
            pca_suffix = f" + PCA(n={pca_n_components})"
        else:
            pca_suffix = " + PCA"
    if feature_space == "roi_mean_predefined":
        roi_spec_path = str(params.get("roi_spec_path") or "").strip()
        if roi_spec_path:
            return f"predefined ROI means ({Path(roi_spec_path).name}){pca_suffix}"
        return f"predefined ROI means{pca_suffix}"
    return f"masked whole-brain voxel cache (current pipeline){pca_suffix}"


def status_for_machine_sheet(variant_records: list[dict[str, Any]]) -> str:
    statuses = {str(row.get("status", "unknown")) for row in variant_records}
    if "failed" in statuses:
        return "Monitoring"
    if statuses and statuses.issubset({"completed", "blocked", "dry_run"}):
        return "Closed"
    return "Open"


def build_machine_status_rows(
    *,
    campaign_id: str,
    source_workbook_path: Path,
    variant_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    total = len(variant_records)
    completed = sum(1 for row in variant_records if str(row.get("status")) == "completed")
    failed = sum(1 for row in variant_records if str(row.get("status")) == "failed")
    blocked = sum(1 for row in variant_records if str(row.get("status")) == "blocked")
    dry_run = sum(1 for row in variant_records if str(row.get("status")) == "dry_run")
    notes = (
        f"Workbook source={source_workbook_path.resolve().name}; "
        f"campaign_id={campaign_id}; total={total}; completed={completed}; "
        f"failed={failed}; blocked={blocked}; dry_run={dry_run}"
    )
    return [
        {
            "machine_id": f"campaign_{campaign_id}",
            "hostname": platform.node(),
            "environment_name": "decision_support_orchestrator",
            "python_version": platform.python_version(),
            "gpu": "not_recorded",
            "status": status_for_machine_sheet(variant_records),
            "last_checked": _utc_timestamp(),
            "notes": notes,
        }
    ]


def build_trial_results_rows(variant_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in variant_records:
        notes = str(row.get("blocked_reason") or row.get("error") or "")
        rows.append(
            {
                "trial_id": row.get("variant_id"),
                "experiment_id": row.get("experiment_id"),
                "run_id": row.get("run_id"),
                "status": row.get("status"),
                "primary_metric_name": _required_metric_name(
                    row.get("primary_metric_name"),
                    context=f"trial_result row '{row.get('variant_id')}'",
                ),
                "primary_metric_value": _safe_float(row.get("primary_metric_value")),
                "report_path": row.get("report_dir"),
                "metrics_path": row.get("metrics_path"),
                "artifact_bundle": row.get("orchestrator_artifact_id") or row.get("manifest_path"),
                "notes": notes,
                "study_id": row.get("study_id"),
                "cell_id": row.get("cell_id"),
                "factor_settings_json": _json_text(row.get("factor_settings")),
                "resolved_params_json": _json_text(
                    row.get("resolved_params") or row.get("params_snapshot")
                ),
            }
        )
    return rows


def build_generated_design_rows(variant_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_trials: set[str] = set()
    for row in variant_records:
        study_id = str(row.get("study_id") or "").strip()
        if not study_id:
            continue
        trial_id = str(row.get("trial_id") or row.get("variant_id") or "").strip()
        if not trial_id or trial_id in seen_trials:
            continue
        seen_trials.add(trial_id)
        rows.append(
            {
                "study_id": study_id,
                "trial_id": trial_id,
                "cell_id": row.get("cell_id"),
                "factor_settings_json": _json_text(row.get("factor_settings")),
                "start_section": row.get("start_section") or "dataset_selection",
                "end_section": row.get("end_section") or "evaluation",
                "base_artifact_id": row.get("base_artifact_id") or "",
                "resolved_params_json": _json_text(
                    row.get("resolved_params") or row.get("params_snapshot")
                ),
                "status": row.get("status"),
            }
        )
    return rows


def build_effect_summary_rows(aggregation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    factorial = aggregation.get("factorial", {})

    for row in factorial.get("best_by_study", []):
        rows.append(
            {
                "study_id": row.get("study_id"),
                "summary_type": "best_by_study",
                "summary_key": str(row.get("study_id") or ""),
                "factor_level_key": "",
                "interaction_key": "",
                "primary_metric_name": _required_metric_name(
                    row.get("primary_metric_name"),
                    context=f"effect_summary best_by_study '{row.get('study_id')}'",
                ),
                "mean_primary_metric_value": _safe_float(row.get("mean_primary_metric_value")),
                "best_primary_metric_value": _safe_float(row.get("best_primary_metric_value")),
                "best_trial_id": row.get("best_trial_id"),
                "notes": "descriptive_only=true",
            }
        )

    for row in factorial.get("by_factor_level", []):
        rows.append(
            {
                "study_id": row.get("study_id"),
                "summary_type": "factor_level",
                "summary_key": str(row.get("summary_key") or ""),
                "factor_level_key": str(row.get("factor_level_key") or ""),
                "interaction_key": "",
                "primary_metric_name": _required_metric_name(
                    row.get("primary_metric_name"),
                    context=f"effect_summary factor_level '{row.get('summary_key')}'",
                ),
                "mean_primary_metric_value": _safe_float(row.get("mean_primary_metric_value")),
                "best_primary_metric_value": _safe_float(row.get("best_primary_metric_value")),
                "best_trial_id": row.get("best_trial_id"),
                "notes": "descriptive_only=true",
            }
        )

    for row in factorial.get("by_factor_combination", []):
        rows.append(
            {
                "study_id": row.get("study_id"),
                "summary_type": "factor_combination",
                "summary_key": str(row.get("summary_key") or ""),
                "factor_level_key": str(row.get("factor_level_key") or ""),
                "interaction_key": "",
                "primary_metric_name": _required_metric_name(
                    row.get("primary_metric_name"),
                    context=f"effect_summary factor_combination '{row.get('summary_key')}'",
                ),
                "mean_primary_metric_value": _safe_float(row.get("mean_primary_metric_value")),
                "best_primary_metric_value": _safe_float(row.get("best_primary_metric_value")),
                "best_trial_id": row.get("best_trial_id"),
                "notes": "descriptive_only=true",
            }
        )

    for row in factorial.get("interaction_descriptive", []):
        rows.append(
            {
                "study_id": row.get("study_id"),
                "summary_type": "interaction_descriptive",
                "summary_key": str(row.get("summary_key") or ""),
                "factor_level_key": "",
                "interaction_key": str(row.get("interaction_key") or ""),
                "primary_metric_name": _required_metric_name(
                    row.get("primary_metric_name"),
                    context=f"effect_summary interaction '{row.get('summary_key')}'",
                ),
                "mean_primary_metric_value": _safe_float(row.get("mean_primary_metric_value")),
                "best_primary_metric_value": _safe_float(row.get("best_primary_metric_value")),
                "best_trial_id": row.get("best_trial_id"),
                "notes": "descriptive_only=true",
            }
        )
    return rows


def build_study_review_rows(
    *,
    study_reviews: list[dict[str, Any]],
    variant_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    status_by_study: dict[str, dict[str, int]] = {}
    for record in variant_records:
        study_id = str(record.get("study_id") or "").strip()
        if not study_id:
            continue
        status = str(record.get("status") or "unknown")
        status_map = status_by_study.setdefault(study_id, {})
        status_map[status] = status_map.get(status, 0) + 1

    rows: list[dict[str, Any]] = []
    for review in study_reviews:
        study_id = str(review.get("study_id") or "").strip()
        if not study_id:
            continue
        observed = status_by_study.get(study_id, {})
        warnings = list(review.get("warnings", []) or [])
        errors = list(review.get("errors", []) or [])
        notes_parts: list[str] = []
        if warnings:
            notes_parts.append("warnings=" + "; ".join(str(item) for item in warnings))
        if errors:
            notes_parts.append("errors=" + "; ".join(str(item) for item in errors))
        rows.append(
            {
                "study_id": study_id,
                "study_name": review.get("study_name"),
                "intent": review.get("intent"),
                "execution_disposition": review.get("execution_disposition"),
                "execution_eligibility_status": review.get("execution_eligibility_status"),
                "warning_count": _int_value(review.get("warning_count")),
                "error_count": _int_value(review.get("error_count")),
                "missing_fields_json": _json_text(review.get("missing_fields")),
                "warnings_json": _json_text(warnings),
                "errors_json": _json_text(errors),
                "question": review.get("question"),
                "generalization_claim": review.get("generalization_claim"),
                "start_section": review.get("start_section"),
                "end_section": review.get("end_section"),
                "factors_json": _json_text(review.get("factors")),
                "fixed_controls_json": _json_text(review.get("fixed_controls")),
                "constraints_json": _json_text(review.get("blocked_constraints")),
                "excluded_combination_count": _int_value(review.get("excluded_combination_count")),
                "expected_design_cells": _int_value(review.get("expected_design_cells")),
                "expected_trials": _int_value(review.get("expected_trials")),
                "primary_metric": review.get("primary_metric"),
                "secondary_metrics": review.get("secondary_metrics"),
                "cv_scheme": review.get("cv_scheme"),
                "nested_cv": review.get("nested_cv"),
                "external_validation_planned": review.get("external_validation_planned"),
                "blocking_strategy": review.get("blocking_strategy"),
                "randomization_strategy": review.get("randomization_strategy"),
                "replication_strategy": review.get("replication_strategy"),
                "replication_mode": review.get("replication_mode"),
                "num_repeats": _int_value(review.get("num_repeats")),
                "random_seed_policy": review.get("random_seed_policy"),
                "rigor_checklist_status": review.get("rigor_checklist_status"),
                "analysis_plan_status": review.get("analysis_plan_status"),
                "completed_trials": int(observed.get("completed", 0)),
                "failed_trials": int(observed.get("failed", 0)),
                "blocked_trials": int(observed.get("blocked", 0)),
                "dry_run_trials": int(observed.get("dry_run", 0)),
                "notes": " | ".join(notes_parts),
            }
        )
    return rows


def _derive_transfer_direction(row: dict[str, Any]) -> str:
    train_subject = str(row.get("train_subject") or "").strip()
    test_subject = str(row.get("test_subject") or "").strip()
    if train_subject and test_subject:
        if train_subject == test_subject:
            return "none_or_within_subject"
        return "custom"
    return "none_or_within_subject"


def build_run_log_writeback_rows(
    *,
    variant_records: list[dict[str, Any]],
    dataset_name: str,
    seed: int,
    commit: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in variant_records:
        notes = str(row.get("blocked_reason") or row.get("error") or "")
        rows.append(
            {
                "Run_ID": row.get("run_id"),
                "Experiment_ID": row.get("experiment_id"),
                "Run_Date": str(row.get("started_at", ""))[:10],
                "Dataset_Name": dataset_name,
                "Data_Subset": _build_dataset_subset_label(row),
                "Data_Slice_ID": "",
                "Grouping_Strategy_ID": "",
                "Code_Commit_or_Version": commit or "",
                "Config_File_or_Path": row.get("config_path") or "",
                "Random_Seed": seed,
                "Target": row.get("target"),
                "Split_ID_or_Fold_Definition": row.get("cv"),
                "Train_Group_Rule": row.get("start_section") or "",
                "Test_Group_Rule": row.get("end_section") or "",
                "Transfer_Direction": _derive_transfer_direction(row),
                "Session_Coverage": "",
                "Task_Coverage": "",
                "Modality_Coverage": "",
                "Model": row.get("model"),
                "Feature_Set": _feature_set_label(row),
                "Run_Type": "Decision-support",
                "Affects_Frozen_Pipeline": "Yes",
                "Eligible_for_Method_Decision": (
                    "Yes" if row.get("status") == "completed" else "No"
                ),
                "Sample_Count": "",
                "Class_Counts": "",
                "Imbalance_Status": "unknown",
                "Leakage_Check_Status": "not_checked",
                "Primary_Metric_Value": _safe_float(row.get("primary_metric_value")),
                "Secondary_Metric_1": _safe_float(row.get("macro_f1")),
                "Secondary_Metric_2": _safe_float(row.get("accuracy")),
                "Robustness_Output_Summary": "",
                "Result_Summary": row.get("status"),
                "Preliminary_Interpretation": "",
                "Reviewed": "No",
                "Used_in_Thesis": "No",
                "Artifact_Path": row.get("report_dir") or row.get("manifest_path"),
                "Notes": notes,
            }
        )
    return rows


__all__ = [
    "build_effect_summary_rows",
    "build_generated_design_rows",
    "build_machine_status_rows",
    "build_run_log_writeback_rows",
    "build_study_review_rows",
    "build_trial_results_rows",
    "status_for_machine_sheet",
]
