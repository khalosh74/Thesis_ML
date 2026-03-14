from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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


def _build_dataset_subset_label(params: dict[str, Any]) -> str:
    subject = str(params.get("subject") or "").strip()
    train_subject = str(params.get("train_subject") or "").strip()
    test_subject = str(params.get("test_subject") or "").strip()
    task = str(params.get("filter_task") or "all_tasks").strip() or "all_tasks"
    modality = str(params.get("filter_modality") or "all_modalities").strip() or "all_modalities"
    if train_subject and test_subject:
        subject_part = f"train={train_subject}|test={test_subject}"
    elif subject:
        subject_part = f"subject={subject}"
    else:
        subject_part = "subject=pooled"
    return f"{subject_part};task={task};modality={modality}"


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
                "primary_metric_name": row.get("primary_metric_name") or "balanced_accuracy",
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
                "primary_metric_name": row.get("primary_metric_name") or "balanced_accuracy",
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
                "primary_metric_name": row.get("primary_metric_name") or "balanced_accuracy",
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
                "primary_metric_name": row.get("primary_metric_name") or "balanced_accuracy",
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
                "primary_metric_name": row.get("primary_metric_name") or "balanced_accuracy",
                "mean_primary_metric_value": _safe_float(row.get("mean_primary_metric_value")),
                "best_primary_metric_value": _safe_float(row.get("best_primary_metric_value")),
                "best_trial_id": row.get("best_trial_id"),
                "notes": "descriptive_only=true",
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
                "Feature_Set": "masked whole-brain voxel cache (current pipeline)",
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
    "build_trial_results_rows",
    "status_for_machine_sheet",
]
