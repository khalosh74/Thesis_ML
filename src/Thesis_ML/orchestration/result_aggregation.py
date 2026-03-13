from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SECTION_DEFAULT_START = "dataset_selection"
SECTION_DEFAULT_END = "evaluation"
SUMMARY_COLUMNS = [
    "summary_type",
    "summary_key",
    "primary_metric_name",
    "primary_metric_value",
    "run_id",
    "experiment_id",
    "start_section",
    "end_section",
    "model",
    "cv",
    "target",
    "xai_method",
    "report_path",
    "notes",
]

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


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_section(value: Any, *, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _is_full_pipeline(start_section: str, end_section: str) -> bool:
    return (
        start_section == SECTION_DEFAULT_START
        and end_section == SECTION_DEFAULT_END
    )


def _model_family(model_name: Any) -> str:
    model = str(model_name or "").strip().lower()
    if model in {"ridge", "logreg", "linearsvc", "linear_svc", "lasso", "elasticnet"}:
        return "linear"
    if "forest" in model or model.startswith("rf") or model.startswith("xgb") or model.startswith("lgbm"):
        return "tree_ensemble"
    if "svm" in model or "svc" in model:
        return "kernel"
    if model.startswith("mlp") or "cnn" in model or "transformer" in model:
        return "neural"
    return "unknown"


def _xai_method_for_model(model_name: Any) -> str:
    family = _model_family(model_name)
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


def _completed_metric_rows(variant_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in variant_records:
        if str(row.get("status")) != "completed":
            continue
        metric = _safe_float(row.get("primary_metric_value"))
        if metric is None:
            metric = (
                _safe_float(row.get("balanced_accuracy"))
                or _safe_float(row.get("macro_f1"))
                or _safe_float(row.get("accuracy"))
            )
        if metric is None:
            continue
        start_section = _normalize_section(
            row.get("start_section"),
            default=SECTION_DEFAULT_START,
        )
        end_section = _normalize_section(
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
                "is_full_pipeline": _is_full_pipeline(start_section, end_section),
                "model_family": _model_family(row.get("model")),
                "xai_method": _xai_method_for_model(row.get("model")),
                "xai_status": xai_status,
                "xai_performed": xai_performed,
            }
        )
    return rows


def _top_runs(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda item: float(item["primary_metric_value_float"]),
        reverse=True,
    )
    return ordered[: max(0, int(top_k))]


def _aggregate_by_key(rows: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
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
        mean_metric = sum(float(item["primary_metric_value_float"]) for item in group_rows) / len(group_rows)
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


def _aggregate_by_section(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row["section_key"])
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        best_row = max(group_rows, key=lambda item: float(item["primary_metric_value_float"]))
        mean_metric = sum(float(item["primary_metric_value_float"]) for item in group_rows) / len(group_rows)
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


def _aggregate_xai_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        method = str(row.get("xai_method") or "manual_xai_plan_required")
        grouped.setdefault(method, []).append(row)

    summary: list[dict[str, Any]] = []
    for method in sorted(grouped):
        group_rows = grouped[method]
        best_row = max(group_rows, key=lambda item: float(item["primary_metric_value_float"]))
        mean_metric = sum(float(item["primary_metric_value_float"]) for item in group_rows) / len(group_rows)
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
    completed = _completed_metric_rows(variant_records)
    full_pipeline = [row for row in completed if bool(row["is_full_pipeline"])]
    segmented = [row for row in completed if not bool(row["is_full_pipeline"])]

    return {
        "top_k": int(top_k),
        "total_variant_records": int(len(variant_records)),
        "completed_with_metric_count": int(len(completed)),
        "full_pipeline_count": int(len(full_pipeline)),
        "segment_count": int(len(segmented)),
        "best_full_pipeline_runs": _top_runs(full_pipeline, top_k=top_k),
        "best_segment_runs": _top_runs(segmented, top_k=top_k),
        "section_level_effects": _aggregate_by_section(completed),
        "cross_run_patterns": {
            "by_model": _aggregate_by_key(completed, "model"),
            "by_cv": _aggregate_by_key(completed, "cv"),
            "by_target": _aggregate_by_key(completed, "target"),
        },
        "xai": {
            "registry": XAI_METHOD_REGISTRY,
            "method_effects": _aggregate_xai_methods(completed),
        },
    }


def _base_summary_row(
    *,
    summary_type: str,
    summary_key: str,
    primary_metric_name: str,
    primary_metric_value: float | None,
    run_id: Any = None,
    experiment_id: Any = None,
    start_section: Any = None,
    end_section: Any = None,
    model: Any = None,
    cv: Any = None,
    target: Any = None,
    xai_method: Any = None,
    report_path: Any = None,
    notes: str = "",
) -> dict[str, Any]:
    row = {
        "summary_type": summary_type,
        "summary_key": summary_key,
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": primary_metric_value,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "start_section": start_section,
        "end_section": end_section,
        "model": model,
        "cv": cv,
        "target": target,
        "xai_method": xai_method,
        "report_path": report_path,
        "notes": notes,
    }
    return {column: row.get(column) for column in SUMMARY_COLUMNS}


def build_summary_output_rows(aggregation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for rank, row in enumerate(aggregation.get("best_full_pipeline_runs", []), start=1):
        rows.append(
            _base_summary_row(
                summary_type="best_full_pipeline",
                summary_key=f"rank_{rank}",
                primary_metric_name=str(row.get("primary_metric_name") or "balanced_accuracy"),
                primary_metric_value=_safe_float(row.get("primary_metric_value_float")),
                run_id=row.get("run_id"),
                experiment_id=row.get("experiment_id"),
                start_section=row.get("start_section_norm"),
                end_section=row.get("end_section_norm"),
                model=row.get("model"),
                cv=row.get("cv"),
                target=row.get("target"),
                xai_method=row.get("xai_method"),
                report_path=row.get("report_dir"),
                notes=f"xai_status={row.get('xai_status', 'unknown')}",
            )
        )

    for rank, row in enumerate(aggregation.get("best_segment_runs", []), start=1):
        rows.append(
            _base_summary_row(
                summary_type="best_segment_run",
                summary_key=f"rank_{rank}",
                primary_metric_name=str(row.get("primary_metric_name") or "balanced_accuracy"),
                primary_metric_value=_safe_float(row.get("primary_metric_value_float")),
                run_id=row.get("run_id"),
                experiment_id=row.get("experiment_id"),
                start_section=row.get("start_section_norm"),
                end_section=row.get("end_section_norm"),
                model=row.get("model"),
                cv=row.get("cv"),
                target=row.get("target"),
                xai_method=row.get("xai_method"),
                report_path=row.get("report_dir"),
                notes=f"xai_status={row.get('xai_status', 'unknown')}",
            )
        )

    for row in aggregation.get("section_level_effects", []):
        rows.append(
            _base_summary_row(
                summary_type="section_effect",
                summary_key=str(row.get("section_key") or ""),
                primary_metric_name="mean_primary_metric_value",
                primary_metric_value=_safe_float(row.get("mean_primary_metric_value")),
                run_id=row.get("best_run_id"),
                experiment_id=row.get("best_experiment_id"),
                start_section=row.get("start_section"),
                end_section=row.get("end_section"),
                model=row.get("best_model"),
                cv=row.get("best_cv"),
                target=row.get("best_target"),
                xai_method=_xai_method_for_model(row.get("best_model")),
                report_path=row.get("best_report_path"),
                notes=(
                    f"n_runs={row.get('n_runs')}; "
                    f"best_primary_metric={row.get('best_primary_metric_value')}"
                ),
            )
        )

    patterns = aggregation.get("cross_run_patterns", {})
    for row in patterns.get("by_model", []):
        rows.append(
            _base_summary_row(
                summary_type="pattern_by_model",
                summary_key=str(row.get("key") or ""),
                primary_metric_name="mean_primary_metric_value",
                primary_metric_value=_safe_float(row.get("mean_primary_metric_value")),
                run_id=row.get("best_run_id"),
                experiment_id=row.get("best_experiment_id"),
                model=row.get("key"),
                xai_method=_xai_method_for_model(row.get("key")),
                notes=(
                    f"n_runs={row.get('n_runs')}; "
                    f"best_primary_metric={row.get('best_primary_metric_value')}"
                ),
            )
        )
    for row in patterns.get("by_cv", []):
        rows.append(
            _base_summary_row(
                summary_type="pattern_by_cv",
                summary_key=str(row.get("key") or ""),
                primary_metric_name="mean_primary_metric_value",
                primary_metric_value=_safe_float(row.get("mean_primary_metric_value")),
                run_id=row.get("best_run_id"),
                experiment_id=row.get("best_experiment_id"),
                cv=row.get("key"),
                notes=(
                    f"n_runs={row.get('n_runs')}; "
                    f"best_primary_metric={row.get('best_primary_metric_value')}"
                ),
            )
        )

    xai = aggregation.get("xai", {})
    for row in xai.get("method_effects", []):
        rows.append(
            _base_summary_row(
                summary_type="xai_method_effect",
                summary_key=str(row.get("xai_method") or ""),
                primary_metric_name="mean_primary_metric_value",
                primary_metric_value=_safe_float(row.get("mean_primary_metric_value")),
                run_id=row.get("best_run_id"),
                experiment_id=row.get("best_experiment_id"),
                xai_method=row.get("xai_method"),
                notes=(
                    f"n_runs={row.get('n_runs')}; performed={row.get('performed_count')}; "
                    f"not_performed={row.get('not_performed_count')}; unknown={row.get('unknown_count')}"
                ),
            )
        )

    return rows
