from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _design_metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("design_metadata")
    return dict(value) if isinstance(value, dict) else {}


def _load_metrics_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    metrics_path_text = _safe_text(record.get("metrics_path"))
    if not metrics_path_text:
        return None
    metrics_path = Path(metrics_path_text)
    if not metrics_path.exists() or not metrics_path.is_file():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _is_e13_confirmatory_dummy_baseline(record: dict[str, Any]) -> bool:
    if _safe_text(record.get("experiment_id")) != "E13":
        return False
    metadata = _design_metadata(record)
    return _safe_text(metadata.get("special_cell_kind")) == "confirmatory_dummy_baseline"


def build_e13_table_ready_rows(*, reporting_variant_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in reporting_variant_records:
        if not _is_e13_confirmatory_dummy_baseline(record):
            continue
        metadata = _design_metadata(record)
        cv_mode = _safe_text(record.get("cv"))
        subject = _safe_text(record.get("subject"))
        train_subject = _safe_text(record.get("train_subject"))
        test_subject = _safe_text(record.get("test_subject"))
        analysis_label = _safe_text(metadata.get("anchor_analysis_label"))
        if not analysis_label:
            if cv_mode == "within_subject_loso_session" and subject:
                analysis_label = f"within_subject_loso_session:{subject}"
            elif cv_mode == "frozen_cross_person_transfer" and train_subject and test_subject:
                analysis_label = f"frozen_cross_person_transfer:{train_subject}->{test_subject}"
            else:
                analysis_label = _safe_text(record.get("variant_id"))

        metrics_payload = _load_metrics_payload(record) or {}
        primary_metric_name = _safe_text(record.get("primary_metric_name")) or _safe_text(
            metrics_payload.get("primary_metric_name")
        )
        primary_metric_value = _safe_float(record.get("primary_metric_value"))
        if primary_metric_value is None and primary_metric_name:
            primary_metric_value = _safe_float(metrics_payload.get(primary_metric_name))
        if primary_metric_value is None:
            primary_metric_value = _safe_float(metrics_payload.get("balanced_accuracy"))

        rows.append(
            {
                "analysis_label": analysis_label,
                "anchor_experiment_id": _safe_text(metadata.get("anchor_experiment_id")),
                "anchor_variant_id": _safe_text(metadata.get("anchor_variant_id")),
                "anchor_analysis_type": _safe_text(metadata.get("anchor_analysis_type")),
                "target": _safe_text(record.get("target")),
                "cv": cv_mode,
                "subject": subject or None,
                "train_subject": train_subject or None,
                "test_subject": test_subject or None,
                "transfer_direction": (
                    f"{train_subject}->{test_subject}" if train_subject and test_subject else None
                ),
                "model": _safe_text(record.get("model")),
                "metric_name": primary_metric_name or None,
                "observed_baseline_score": primary_metric_value,
                "majority_class_metadata": metrics_payload.get("majority_class_metadata"),
                "status": _safe_text(record.get("status")),
                "run_id": _safe_text(record.get("run_id")),
                "variant_id": _safe_text(record.get("variant_id")),
                "baseline_group_id": _safe_text(metadata.get("baseline_group_id")),
                "metrics_path": _safe_text(record.get("metrics_path")),
                "report_dir": _safe_text(record.get("report_dir")),
            }
        )

    rows.sort(
        key=lambda row: (
            str(row.get("analysis_label") or ""),
            str(row.get("anchor_experiment_id") or ""),
            str(row.get("anchor_variant_id") or ""),
        )
    )
    return rows


__all__ = ["build_e13_table_ready_rows"]
