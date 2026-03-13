from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.orchestration.experiment_selection import STAGE_ORDER

STAGE_SUMMARY_FILENAMES = {
    "Stage 1 - Target lock": "stage1_target_lock_summary",
    "Stage 2 - Split lock": "stage2_split_lock_summary",
    "Stage 3 - Model lock": "stage3_model_lock_summary",
    "Stage 4 - Feature/preprocessing lock": "stage4_feature_lock_summary",
    "Stage 5 - Confirmatory analysis": "stage5_confirmatory_summary",
    "Stage 6 - Robustness analysis": "stage6_robustness_summary",
    "Stage 7 - Exploratory extension": "stage7_exploratory_summary",
}

RUN_LOG_EXPORT_COLUMNS = [
    "Run_ID",
    "Experiment_ID",
    "Run_Date",
    "Dataset_Name",
    "Data_Subset",
    "Code_Commit_or_Version",
    "Config_File_or_Path",
    "Random_Seed",
    "Target",
    "Split_ID_or_Fold_Definition",
    "Model",
    "Feature_Set",
    "Run_Type",
    "Affects_Frozen_Pipeline",
    "Eligible_for_Method_Decision",
    "Primary_Metric_Value",
    "Secondary_Metric_1",
    "Secondary_Metric_2",
    "Robustness_Output_Summary",
    "Result_Summary",
    "Preliminary_Interpretation",
    "Reviewed",
    "Used_in_Thesis",
    "Artifact_Path",
    "Notes",
]

SUMMARY_COLUMNS = [
    "experiment_id",
    "title",
    "stage",
    "decision_id",
    "manipulated_factor",
    "primary_metric",
    "total_variants",
    "completed_variants",
    "failed_variants",
    "blocked_variants",
    "dry_run_variants",
    "best_variant_id",
    "best_primary_metric_value",
    "mean_primary_metric_value",
    "status",
    "notes",
]

VARIANT_EXPORT_COLUMNS = [
    "experiment_id",
    "title",
    "stage",
    "decision_id",
    "template_id",
    "variant_id",
    "variant_label",
    "status",
    "target",
    "cv",
    "model",
    "subject",
    "train_subject",
    "test_subject",
    "filter_task",
    "filter_modality",
    "start_section",
    "end_section",
    "base_artifact_id",
    "reuse_policy",
    "search_space_id",
    "search_assignment",
    "primary_metric_name",
    "primary_metric_value",
    "balanced_accuracy",
    "macro_f1",
    "accuracy",
    "run_id",
    "report_dir",
    "manifest_path",
    "blocked_reason",
    "error",
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def build_dataset_subset_label(params: dict[str, Any]) -> str:
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


def write_experiment_outputs(
    experiment: dict[str, Any],
    experiment_root: Path,
    variant_records: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    export_rows = []
    for row in variant_records:
        export_rows.append({column: row.get(column) for column in VARIANT_EXPORT_COLUMNS})
    pd.DataFrame(export_rows, columns=VARIANT_EXPORT_COLUMNS).to_csv(
        experiment_root / "experiment_variants.csv",
        index=False,
    )

    experiment_manifest = {
        "experiment_id": str(experiment.get("experiment_id")),
        "title": str(experiment.get("title", "")),
        "stage": str(experiment.get("stage", "")),
        "decision_id": str(experiment.get("decision_id", "")),
        "manipulated_factor": str(experiment.get("manipulated_factor", "")),
        "primary_metric": str(experiment.get("primary_metric", "balanced_accuracy")),
        "warnings": warnings,
        "variant_count": int(len(variant_records)),
        "completed_count": int(sum(1 for row in variant_records if row["status"] == "completed")),
        "failed_count": int(sum(1 for row in variant_records if row["status"] == "failed")),
        "blocked_count": int(sum(1 for row in variant_records if row["status"] == "blocked")),
        "dry_run_count": int(sum(1 for row in variant_records if row["status"] == "dry_run")),
        "variant_manifest_paths": [row["manifest_path"] for row in variant_records],
    }
    (experiment_root / "experiment_manifest.json").write_text(
        f"{json.dumps(experiment_manifest, indent=2)}\n",
        encoding="utf-8",
    )


def summarize_by_experiment(
    experiments: list[dict[str, Any]],
    variant_records: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        experiment_id = str(experiment["experiment_id"])
        metric_name = str(experiment.get("primary_metric", "balanced_accuracy"))
        records = [row for row in variant_records if row["experiment_id"] == experiment_id]

        completed = [row for row in records if row["status"] == "completed"]
        failed = [row for row in records if row["status"] == "failed"]
        blocked = [row for row in records if row["status"] == "blocked"]
        dry_run = [row for row in records if row["status"] == "dry_run"]

        best_variant_id = None
        best_metric_value = None
        mean_metric_value = None
        if completed:
            metric_pairs_raw = [
                (
                    row["variant_id"],
                    _safe_float(row.get("primary_metric_value")),
                )
                for row in completed
            ]
            metric_pairs: list[tuple[str, float]] = [
                (variant_id, metric_value)
                for variant_id, metric_value in metric_pairs_raw
                if metric_value is not None
            ]
            if metric_pairs:
                best_variant_id, best_metric_value = max(metric_pairs, key=lambda item: item[1])
                mean_metric_value = sum(value for _, value in metric_pairs) / len(metric_pairs)

        if completed and not failed and not blocked:
            status = "completed"
        elif completed and (failed or blocked):
            status = "partial"
        elif dry_run and blocked and not completed:
            status = "dry_run_partial_blocked"
        elif dry_run and not completed:
            status = "dry_run"
        elif blocked and not completed:
            status = "blocked"
        elif failed and not completed:
            status = "failed"
        else:
            status = "not_executed"

        notes: list[str] = []
        if blocked:
            blocked_reasons = sorted(
                {str(row.get("blocked_reason")) for row in blocked if row.get("blocked_reason")}
            )
            if blocked_reasons:
                notes.append("blocked: " + "; ".join(blocked_reasons))
        if failed:
            notes.append(f"failed_variants={len(failed)}")

        rows.append(
            {
                "experiment_id": experiment_id,
                "title": str(experiment.get("title", "")),
                "stage": str(experiment.get("stage", "")),
                "decision_id": str(experiment.get("decision_id", "")),
                "manipulated_factor": str(experiment.get("manipulated_factor", "")),
                "primary_metric": metric_name,
                "total_variants": int(len(records)),
                "completed_variants": int(len(completed)),
                "failed_variants": int(len(failed)),
                "blocked_variants": int(len(blocked)),
                "dry_run_variants": int(len(dry_run)),
                "best_variant_id": best_variant_id,
                "best_primary_metric_value": best_metric_value,
                "mean_primary_metric_value": mean_metric_value,
                "status": status,
                "notes": " | ".join(notes),
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def write_stage_summaries(
    campaign_root: Path,
    variant_records: list[dict[str, Any]],
) -> list[Path]:
    created_files: list[Path] = []
    stage_columns = [
        "experiment_id",
        "title",
        "decision_id",
        "stage",
        "template_id",
        "variant_id",
        "variant_label",
        "status",
        "primary_metric_name",
        "primary_metric_value",
        "balanced_accuracy",
        "macro_f1",
        "accuracy",
        "report_dir",
        "blocked_reason",
        "error",
    ]
    df = pd.DataFrame(variant_records)
    if df.empty:
        df = pd.DataFrame(columns=stage_columns)

    for stage in STAGE_ORDER:
        stage_name = STAGE_SUMMARY_FILENAMES.get(stage, stage.replace(" ", "_").lower())
        csv_path = campaign_root / f"{stage_name}.csv"
        markdown_path = campaign_root / f"{stage_name}.md"
        stage_df = df[df.get("stage", pd.Series(dtype=str)) == stage].copy()
        if stage_df.empty:
            stage_df = pd.DataFrame(columns=stage_columns)
        else:
            stage_df = stage_df.reindex(columns=stage_columns)
        stage_df.to_csv(csv_path, index=False)

        lines = [f"# {stage}", ""]
        if stage_df.empty:
            lines.append("No variants were selected for this stage in this campaign.")
        else:
            lines.append("## What was compared")
            experiment_ids = sorted(stage_df["experiment_id"].astype(str).unique().tolist())
            lines.append(f"- Experiments: {', '.join(experiment_ids)}")
            lines.append(f"- Variants evaluated: {len(stage_df)}")
            lines.append("")
            lines.append("## Status summary")
            status_counts = stage_df["status"].value_counts().to_dict()
            for status_name in sorted(status_counts):
                lines.append(f"- {status_name}: {status_counts[status_name]}")
            lines.append("")
            lines.append("## Metric focus")
            lines.append(
                "- Primary metric tracked per experiment: balanced_accuracy (registry-defined)."
            )
            lines.append(
                "- Use corresponding CSV for exact per-variant evidence before locking decisions."
            )
            lines.append("")
            lines.append("## Decision linkage")
            decision_ids = sorted(stage_df["decision_id"].astype(str).unique().tolist())
            lines.append(f"- Informs Decision_Log IDs: {', '.join(decision_ids)}")

        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        created_files.extend([csv_path, markdown_path])

    return created_files


def write_run_log_export(
    campaign_root: Path,
    variant_records: list[dict[str, Any]],
    dataset_name: str,
    seed: int,
    commit: str | None,
) -> Path:
    rows: list[dict[str, Any]] = []
    for row in variant_records:
        primary_value = _safe_float(row.get("primary_metric_value"))
        run_date = str(row.get("started_at", ""))[:10]
        notes = row.get("blocked_reason") or row.get("error") or ""
        rows.append(
            {
                "Run_ID": row.get("run_id"),
                "Experiment_ID": row.get("experiment_id"),
                "Run_Date": run_date,
                "Dataset_Name": dataset_name,
                "Data_Subset": build_dataset_subset_label(row),
                "Code_Commit_or_Version": commit or "",
                "Config_File_or_Path": row.get("config_path") or "",
                "Random_Seed": seed,
                "Target": row.get("target"),
                "Split_ID_or_Fold_Definition": row.get("cv"),
                "Model": row.get("model"),
                "Feature_Set": "masked whole-brain voxel cache (current pipeline)",
                "Run_Type": "Decision-support",
                "Affects_Frozen_Pipeline": "Yes",
                "Eligible_for_Method_Decision": (
                    "Yes" if row.get("status") == "completed" else "No"
                ),
                "Primary_Metric_Value": primary_value,
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

    df = pd.DataFrame(rows, columns=RUN_LOG_EXPORT_COLUMNS)
    out_path = campaign_root / "run_log_export.csv"
    df.to_csv(out_path, index=False)
    return out_path


def status_snapshot(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


__all__ = [
    "STAGE_SUMMARY_FILENAMES",
    "SUMMARY_COLUMNS",
    "VARIANT_EXPORT_COLUMNS",
    "build_dataset_subset_label",
    "status_snapshot",
    "summarize_by_experiment",
    "write_experiment_outputs",
    "write_run_log_export",
    "write_stage_summaries",
]
