from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Thesis_ML.config.schema_versions import SUMMARY_RESULT_SCHEMA_VERSION

SECTION_DEFAULT_START = "dataset_selection"
SECTION_DEFAULT_END = "evaluation"

XAI_METHOD_REGISTRY: dict[str, dict[str, str]] = {
    "linear": {
        "xai_method": "linear_coefficients_stability",
        "notes": (
            "Use fold-wise coefficient stability and sign consistency. "
            "Interpret as model-behavior evidence, not localization."
        ),
    },
    "tree_ensemble": {
        "xai_method": "tree_importance_with_permutation_check",
        "notes": "Use gain/split importances with permutation sanity-checks.",
    },
    "kernel": {
        "xai_method": "permutation_importance",
        "notes": "Use held-out perturbation attribution because coefficients are unavailable.",
    },
    "neural": {
        "xai_method": "gradient_or_occlusion",
        "notes": "Use gradient- or occlusion-based saliency with stability checks.",
    },
    "unknown": {
        "xai_method": "manual_xai_plan_required",
        "notes": "Model family not recognized; define XAI method before claim-level reporting.",
    },
}


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def normalize_section(value: Any, *, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def is_full_pipeline(start_section: str, end_section: str) -> bool:
    return start_section == SECTION_DEFAULT_START and end_section == SECTION_DEFAULT_END


def model_family(model_name: Any) -> str:
    model = str(model_name or "").strip().lower()
    if model in {"ridge", "logreg", "linearsvc", "linear_svc", "lasso", "elasticnet"}:
        return "linear"
    if (
        "forest" in model
        or model.startswith("rf")
        or model.startswith("xgb")
        or model.startswith("lgbm")
    ):
        return "tree_ensemble"
    if "svm" in model or "svc" in model:
        return "kernel"
    if model.startswith("mlp") or "cnn" in model or "transformer" in model:
        return "neural"
    return "unknown"


def xai_method_for_model(model_name: Any) -> str:
    family = model_family(model_name)
    return XAI_METHOD_REGISTRY.get(family, XAI_METHOD_REGISTRY["unknown"])["xai_method"]


def _load_metrics_payload(metrics_path: Any) -> dict[str, Any]:
    text = str(metrics_path or "").strip()
    if not text:
        return {}
    path = Path(text)
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _xai_execution_status(row: dict[str, Any]) -> tuple[str, bool | None]:
    metrics_payload = _load_metrics_payload(row.get("metrics_path"))
    interpretability = metrics_payload.get("interpretability")
    if isinstance(interpretability, dict):
        status = str(interpretability.get("status", "unknown"))
        performed = interpretability.get("performed")
        if isinstance(performed, bool):
            return status, performed
        return status, None
    return "unknown", None


def completed_metric_rows(variant_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in variant_records:
        if str(row.get("status")) != "completed":
            continue
        metric = safe_float(row.get("primary_metric_value"))
        if metric is None:
            metric = (
                safe_float(row.get("balanced_accuracy"))
                or safe_float(row.get("macro_f1"))
                or safe_float(row.get("accuracy"))
            )
        if metric is None:
            continue
        start_section = normalize_section(
            row.get("start_section"),
            default=SECTION_DEFAULT_START,
        )
        end_section = normalize_section(
            row.get("end_section"),
            default=SECTION_DEFAULT_END,
        )
        xai_status, xai_performed = _xai_execution_status(row)
        rows.append(
            {
                **row,
                "primary_metric_value_float": metric,
                "start_section_norm": start_section,
                "end_section_norm": end_section,
                "section_key": f"{start_section}->{end_section}",
                "is_full_pipeline": is_full_pipeline(start_section, end_section),
                "model_family": model_family(row.get("model")),
                "xai_method": xai_method_for_model(row.get("model")),
                "xai_status": xai_status,
                "xai_performed": xai_performed,
            }
        )
    return rows


def top_runs(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda item: float(item["primary_metric_value_float"]),
        reverse=True,
    )
    return ordered[: max(0, int(top_k))]


def aggregate_by_key(rows: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(key_name, "") or "")
        if not key:
            continue
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        best_row = max(group_rows, key=lambda item: float(item["primary_metric_value_float"]))
        mean_metric = sum(float(item["primary_metric_value_float"]) for item in group_rows) / len(
            group_rows
        )
        summary.append(
            {
                "key": key,
                "n_runs": int(len(group_rows)),
                "mean_primary_metric_value": float(mean_metric),
                "best_primary_metric_value": float(best_row["primary_metric_value_float"]),
                "best_run_id": best_row.get("run_id"),
                "best_experiment_id": best_row.get("experiment_id"),
            }
        )
    return summary


def aggregate_by_section(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row["section_key"])
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        best_row = max(group_rows, key=lambda item: float(item["primary_metric_value_float"]))
        mean_metric = sum(float(item["primary_metric_value_float"]) for item in group_rows) / len(
            group_rows
        )
        summary.append(
            {
                "section_key": key,
                "start_section": best_row["start_section_norm"],
                "end_section": best_row["end_section_norm"],
                "n_runs": int(len(group_rows)),
                "mean_primary_metric_value": float(mean_metric),
                "best_primary_metric_value": float(best_row["primary_metric_value_float"]),
                "best_run_id": best_row.get("run_id"),
                "best_experiment_id": best_row.get("experiment_id"),
                "best_model": best_row.get("model"),
                "best_cv": best_row.get("cv"),
                "best_target": best_row.get("target"),
                "best_report_path": best_row.get("report_dir"),
            }
        )
    return summary


def aggregate_xai_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        method = str(row.get("xai_method") or "manual_xai_plan_required")
        grouped.setdefault(method, []).append(row)

    summary: list[dict[str, Any]] = []
    for method in sorted(grouped):
        group_rows = grouped[method]
        best_row = max(group_rows, key=lambda item: float(item["primary_metric_value_float"]))
        mean_metric = sum(float(item["primary_metric_value_float"]) for item in group_rows) / len(
            group_rows
        )
        performed_count = sum(1 for item in group_rows if item.get("xai_performed") is True)
        not_performed_count = sum(1 for item in group_rows if item.get("xai_performed") is False)
        unknown_count = sum(1 for item in group_rows if item.get("xai_performed") is None)
        summary.append(
            {
                "xai_method": method,
                "n_runs": int(len(group_rows)),
                "performed_count": int(performed_count),
                "not_performed_count": int(not_performed_count),
                "unknown_count": int(unknown_count),
                "mean_primary_metric_value": float(mean_metric),
                "best_primary_metric_value": float(best_row["primary_metric_value_float"]),
                "best_run_id": best_row.get("run_id"),
                "best_experiment_id": best_row.get("experiment_id"),
            }
        )
    return summary


def aggregate_variant_records(
    variant_records: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    completed = completed_metric_rows(variant_records)
    full_pipeline = [row for row in completed if bool(row["is_full_pipeline"])]
    segmented = [row for row in completed if not bool(row["is_full_pipeline"])]

    return {
        "summary_result_schema_version": SUMMARY_RESULT_SCHEMA_VERSION,
        "top_k": int(top_k),
        "total_variant_records": int(len(variant_records)),
        "completed_with_metric_count": int(len(completed)),
        "full_pipeline_count": int(len(full_pipeline)),
        "segment_count": int(len(segmented)),
        "best_full_pipeline_runs": top_runs(full_pipeline, top_k=top_k),
        "best_segment_runs": top_runs(segmented, top_k=top_k),
        "section_level_effects": aggregate_by_section(completed),
        "cross_run_patterns": {
            "by_model": aggregate_by_key(completed, "model"),
            "by_cv": aggregate_by_key(completed, "cv"),
            "by_target": aggregate_by_key(completed, "target"),
        },
        "xai": {
            "registry": XAI_METHOD_REGISTRY,
            "method_effects": aggregate_xai_methods(completed),
        },
    }


__all__ = [
    "SECTION_DEFAULT_END",
    "SECTION_DEFAULT_START",
    "XAI_METHOD_REGISTRY",
    "aggregate_variant_records",
    "safe_float",
    "xai_method_for_model",
]
