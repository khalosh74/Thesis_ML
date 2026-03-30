from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.config.metric_policy import validate_metric_name
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
    "feature_space",
    "roi_spec_path",
    "preprocessing_strategy",
    "dimensionality_strategy",
    "pca_n_components",
    "pca_variance_ratio",
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
    feature_space = (
        str(params.get("feature_space") or "whole_brain_masked").strip() or "whole_brain_masked"
    )
    preprocessing_strategy = (
        str(params.get("preprocessing_strategy") or "model_default").strip() or "model_default"
    )
    dimensionality_strategy = str(params.get("dimensionality_strategy") or "none").strip() or "none"
    if train_subject and test_subject:
        subject_part = f"train={train_subject}|test={test_subject}"
    elif subject:
        subject_part = f"subject={subject}"
    else:
        subject_part = "subject=pooled"
    return (
        f"{subject_part};task={task};modality={modality};feature_space={feature_space};"
        f"preprocessing={preprocessing_strategy};"
        f"dimensionality={dimensionality_strategy}"
    )


def _feature_set_label(params: dict[str, Any]) -> str:
    feature_space = str(params.get("feature_space") or "whole_brain_masked").strip().lower()
    preprocessing_strategy = str(params.get("preprocessing_strategy") or "").strip().lower()
    dimensionality_strategy = str(params.get("dimensionality_strategy") or "none").strip().lower()
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
            base_label = f"predefined ROI means ({Path(roi_spec_path).name}){pca_suffix}"
        else:
            base_label = f"predefined ROI means{pca_suffix}"
    else:
        base_label = f"masked whole-brain voxel cache (current pipeline){pca_suffix}"
    if preprocessing_strategy:
        return f"{base_label}; preprocess={preprocessing_strategy}"
    return base_label


def write_experiment_outputs(
    experiment: dict[str, Any],
    experiment_root: Path,
    variant_records: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    raw_primary_metric = experiment.get("primary_metric")
    if raw_primary_metric is None or not str(raw_primary_metric).strip():
        raise ValueError(
            f"Experiment '{experiment.get('experiment_id')}' is missing required primary_metric."
        )
    primary_metric_name = validate_metric_name(str(raw_primary_metric))
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
        "primary_metric": primary_metric_name,
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
    warnings_by_experiment: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    warnings_lookup = warnings_by_experiment or {}
    for experiment in experiments:
        experiment_id = str(experiment["experiment_id"])
        raw_metric_name = experiment.get("primary_metric")
        if raw_metric_name is None or not str(raw_metric_name).strip():
            raise ValueError(f"Experiment '{experiment_id}' is missing required primary_metric.")
        metric_name = validate_metric_name(str(raw_metric_name))
        records = [row for row in variant_records if row["experiment_id"] == experiment_id]
        warnings = list(warnings_lookup.get(experiment_id, []))

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
        elif not records and warnings:
            status = "skipped"
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
        if warnings:
            notes.append("warnings: " + "; ".join(sorted({str(item) for item in warnings if str(item)})))

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
            metric_names = sorted(
                {
                    validate_metric_name(str(value).strip())
                    for value in stage_df["primary_metric_name"].tolist()
                    if isinstance(value, str) and value.strip()
                }
            )
            if metric_names:
                lines.append(
                    "- Primary metric tracked per experiment: "
                    + ", ".join(metric_names)
                    + " (registry-defined)."
                )
            else:
                lines.append("- Primary metric tracked per experiment: missing_in_stage_rows.")
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
                "Feature_Set": _feature_set_label(row),
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


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _existing_relative_paths(campaign_root: Path, patterns: list[str]) -> list[str]:
    rel_paths: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(campaign_root.glob(pattern)):
            rel = str(path.relative_to(campaign_root))
            if rel in seen:
                continue
            seen.add(rel)
            rel_paths.append(rel)
    return rel_paths


def write_campaign_execution_report(
    campaign_root: Path,
    campaign_id: str,
) -> tuple[Path, Path]:
    campaign_root = Path(campaign_root)
    live_status_path = campaign_root / "campaign_live_status.json"
    live_status = _read_optional_json(live_status_path) or {}
    eta_state = _read_optional_json(campaign_root / "eta_state.json")
    eta_calibration = _read_optional_json(campaign_root / "campaign_eta_calibration.json")
    anomaly_report = _read_optional_json(campaign_root / "campaign_anomaly_report.json")

    stage_summary_paths = _existing_relative_paths(
        campaign_root,
        ["stage*_summary.csv", "stage*_summary.md"],
    )
    stage_decision_note_paths = _existing_relative_paths(
        campaign_root,
        ["stage*_decision_notes.md"],
    )
    phase_artifact_paths = _existing_relative_paths(
        campaign_root,
        [
            "stage*_lock.json",
            "final_confirmatory_pipeline.json",
            "phase_skip_summary.json",
        ],
    )
    key_artifacts: dict[str, Any] = {
        "execution_events": (
            "execution_events.jsonl"
            if (campaign_root / "execution_events.jsonl").exists()
            else None
        ),
        "campaign_live_status": (
            "campaign_live_status.json"
            if (campaign_root / "campaign_live_status.json").exists()
            else None
        ),
        "decision_support_summary": (
            "decision_support_summary.csv"
            if (campaign_root / "decision_support_summary.csv").exists()
            else None
        ),
        "decision_recommendations": (
            "decision_recommendations.md"
            if (campaign_root / "decision_recommendations.md").exists()
            else None
        ),
        "result_aggregation": (
            "result_aggregation.json" if (campaign_root / "result_aggregation.json").exists() else None
        ),
        "summary_outputs_export": (
            "summary_outputs_export.csv"
            if (campaign_root / "summary_outputs_export.csv").exists()
            else None
        ),
        "stage_summaries": stage_summary_paths,
        "stage_decision_notes": stage_decision_note_paths,
        "phase_artifacts": phase_artifact_paths,
    }
    if eta_state is not None:
        key_artifacts["eta_state"] = "eta_state.json"
    if eta_calibration is not None:
        key_artifacts["campaign_eta_calibration"] = "campaign_eta_calibration.json"
    if anomaly_report is not None:
        key_artifacts["campaign_anomaly_report"] = "campaign_anomaly_report.json"

    counts = live_status.get("counts") if isinstance(live_status.get("counts"), dict) else {}
    report_payload = {
        "campaign_id": str(campaign_id),
        "generated_at_utc": _utc_now(),
        "campaign_root": str(campaign_root.resolve()),
        "final_execution_status": {
            "status": live_status.get("status"),
            "started_at_utc": live_status.get("started_at_utc"),
            "last_updated_at_utc": live_status.get("last_updated_at_utc"),
        },
        "phase_summary": {
            "current_phase": live_status.get("current_phase"),
            "current_phase_status": live_status.get("current_phase_status"),
            "blocked_experiments": list(live_status.get("blocked_experiments", [])),
        },
        "run_summary": {
            "runs_planned": int(counts.get("runs_planned", 0)),
            "runs_dispatched": int(counts.get("runs_dispatched", 0)),
            "runs_started": int(counts.get("runs_started", 0)),
            "runs_completed": int(counts.get("runs_completed", 0)),
            "runs_failed": int(counts.get("runs_failed", 0)),
            "runs_blocked": int(counts.get("runs_blocked", 0)),
            "runs_dry_run": int(counts.get("runs_dry_run", 0)),
            "active_runs": len(live_status.get("active_runs", []))
            if isinstance(live_status.get("active_runs"), list)
            else 0,
        },
        "eta_summary": {
            "campaign_eta": eta_state.get("campaign_eta") if isinstance(eta_state, dict) else None,
            "phase_eta": eta_state.get("phase_eta") if isinstance(eta_state, dict) else None,
            "calibration": eta_calibration,
        },
        "anomaly_summary": anomaly_report
        if isinstance(anomaly_report, dict)
        else {
            "anomaly_counts": live_status.get("anomaly_counts"),
            "latest_anomaly": live_status.get("latest_anomaly"),
        },
        "key_generated_artifacts": key_artifacts,
        "operator_notes": {
            "blocked_experiments": list(live_status.get("blocked_experiments", [])),
            "failed_runs": list(live_status.get("failed_runs", [])),
        },
    }

    report_json_path = campaign_root / "campaign_execution_report.json"
    report_json_path.write_text(f"{json.dumps(report_payload, indent=2)}\n", encoding="utf-8")

    status_value = str(report_payload["final_execution_status"].get("status") or "unknown")
    lines = [
        "# Campaign Execution Report",
        "",
        "## Campaign overview",
        f"- Campaign ID: {campaign_id}",
        f"- Campaign root: {campaign_root.resolve()}",
        f"- Generated at (UTC): {report_payload['generated_at_utc']}",
        "",
        "## Final execution status",
        f"- Status: {status_value}",
        f"- Started at: {report_payload['final_execution_status'].get('started_at_utc')}",
        f"- Last updated at: {report_payload['final_execution_status'].get('last_updated_at_utc')}",
        "",
        "## Phase summary",
        f"- Current phase: {report_payload['phase_summary'].get('current_phase')}",
        f"- Current phase status: {report_payload['phase_summary'].get('current_phase_status')}",
        f"- Blocked experiments: {', '.join(report_payload['phase_summary'].get('blocked_experiments', [])) or 'none'}",
        "",
        "## Run summary",
        f"- Runs planned: {report_payload['run_summary']['runs_planned']}",
        f"- Runs dispatched: {report_payload['run_summary']['runs_dispatched']}",
        f"- Runs started: {report_payload['run_summary']['runs_started']}",
        f"- Runs completed: {report_payload['run_summary']['runs_completed']}",
        f"- Runs failed: {report_payload['run_summary']['runs_failed']}",
        f"- Runs blocked: {report_payload['run_summary']['runs_blocked']}",
        f"- Runs dry-run: {report_payload['run_summary']['runs_dry_run']}",
        f"- Active runs: {report_payload['run_summary']['active_runs']}",
        "",
        "## ETA summary and calibration",
        f"- Campaign ETA: {json.dumps(report_payload['eta_summary'].get('campaign_eta'))}",
        f"- Phase ETA: {json.dumps(report_payload['eta_summary'].get('phase_eta'))}",
        f"- Calibration available: {'yes' if eta_calibration is not None else 'no'}",
        "",
        "## Anomaly summary",
        f"- Anomaly report available: {'yes' if anomaly_report is not None else 'no'}",
        f"- Latest anomaly: {json.dumps((live_status.get('latest_anomaly') if isinstance(live_status, dict) else None))}",
        "",
        "## Key generated artifacts",
    ]
    for key, value in key_artifacts.items():
        if value is None:
            continue
        if isinstance(value, list):
            if not value:
                continue
            lines.append(f"- {key}:")
            for item in value:
                lines.append(f"  - {item}")
            continue
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Operator notes / blocked items",
            f"- Blocked experiments: {', '.join(report_payload['operator_notes'].get('blocked_experiments', [])) or 'none'}",
            f"- Failed runs: {', '.join(report_payload['operator_notes'].get('failed_runs', [])) or 'none'}",
            "",
        ]
    )

    report_md_path = campaign_root / "campaign_execution_report.md"
    report_md_path.write_text("\n".join(lines), encoding="utf-8")
    return report_md_path, report_json_path


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
    "write_campaign_execution_report",
]
