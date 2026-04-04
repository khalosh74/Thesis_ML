from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.experiments.model_factory import model_supports_linear_interpretability


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


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except Exception:
        return None


def _design_metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("design_metadata")
    return dict(value) if isinstance(value, dict) else {}


def _load_json_dict(path_text: Any) -> dict[str, Any] | None:
    path = Path(_safe_text(path_text))
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _analysis_label(record: dict[str, Any]) -> str:
    cv_mode = _safe_text(record.get("cv"))
    subject = _safe_text(record.get("subject"))
    train_subject = _safe_text(record.get("train_subject"))
    test_subject = _safe_text(record.get("test_subject"))
    if cv_mode == "within_subject_loso_session" and subject:
        return f"{cv_mode}:{subject}"
    if cv_mode == "frozen_cross_person_transfer" and train_subject and test_subject:
        return f"{cv_mode}:{train_subject}->{test_subject}"
    return _safe_text(record.get("analysis_label"))


def _is_e14_special_cell(record: dict[str, Any]) -> bool:
    if _safe_text(record.get("experiment_id")) != "E14":
        return False
    metadata = _design_metadata(record)
    return _safe_text(metadata.get("special_cell_kind")) == "interpretability_stability"


def _is_e16_completed_record(record: dict[str, Any]) -> bool:
    return (
        _safe_text(record.get("experiment_id")) == "E16"
        and _safe_text(record.get("status")) == "completed"
    )


def _candidate_anchor_records(
    reporting_variant_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [record for record in reporting_variant_records if _is_e16_completed_record(record)]


def _resolve_anchor_record(
    *,
    anchor_candidates: list[dict[str, Any]],
    anchor_variant_id: str,
    anchor_analysis_label: str,
) -> dict[str, Any] | None:
    if anchor_variant_id:
        for row in anchor_candidates:
            if _safe_text(row.get("variant_id")) == anchor_variant_id:
                return row
    if anchor_analysis_label:
        for row in anchor_candidates:
            if _analysis_label(row) == anchor_analysis_label:
                return row
    return None


def _resolve_interpretability_payload(anchor_record: dict[str, Any]) -> dict[str, Any]:
    metrics_payload = _load_json_dict(anchor_record.get("metrics_path")) or {}
    metrics_interpretability = (
        dict(metrics_payload.get("interpretability"))
        if isinstance(metrics_payload.get("interpretability"), dict)
        else {}
    )

    summary_path = _safe_text(metrics_interpretability.get("summary_path"))
    if not summary_path:
        report_dir = Path(_safe_text(anchor_record.get("report_dir")))
        if report_dir.exists():
            summary_path = str((report_dir / "interpretability_summary.json").resolve())

    summary_payload = _load_json_dict(summary_path) if summary_path else None
    summary_payload = summary_payload if isinstance(summary_payload, dict) else {}

    fold_artifacts_path = _safe_text(summary_payload.get("fold_artifacts_path")) or _safe_text(
        metrics_interpretability.get("fold_artifacts_path")
    )
    fold_artifacts_frame = None
    if fold_artifacts_path:
        fold_path = Path(fold_artifacts_path)
        if fold_path.exists() and fold_path.is_file():
            try:
                fold_artifacts_frame = pd.read_csv(fold_path)
            except Exception:
                fold_artifacts_frame = None

    coefficient_paths: list[str] = []
    if fold_artifacts_frame is not None and "coefficient_file" in set(fold_artifacts_frame.columns):
        for value in fold_artifacts_frame["coefficient_file"].tolist():
            text = _safe_text(value)
            if text:
                coefficient_paths.append(text)

    stability_payload = (
        dict(summary_payload.get("stability"))
        if isinstance(summary_payload.get("stability"), dict)
        else (
            dict(metrics_interpretability.get("stability"))
            if isinstance(metrics_interpretability.get("stability"), dict)
            else {}
        )
    )

    return {
        "summary_path": summary_path,
        "summary_payload": summary_payload,
        "fold_artifacts_path": fold_artifacts_path,
        "coefficient_paths": coefficient_paths,
        "stability": stability_payload,
    }


def _valid_e14_conditions(
    *,
    anchor_record: dict[str, Any],
    interpretability_payload: dict[str, Any],
) -> tuple[bool, str | None]:
    if _safe_text(anchor_record.get("cv")) != "within_subject_loso_session":
        return False, "E14 applies only to within_subject_loso_session anchors."
    model_name = _safe_text(anchor_record.get("model"))
    if not model_name or not model_supports_linear_interpretability(model_name):
        return False, "E14 requires a linear-interpretability model family anchor."

    summary_payload = interpretability_payload.get("summary_payload")
    if not isinstance(summary_payload, dict) or not summary_payload:
        return False, "interpretability_summary.json is missing for E14 anchor."
    if _safe_text(summary_payload.get("status")) != "performed":
        return False, "interpretability summary is not in performed status."

    summary_path = _safe_text(interpretability_payload.get("summary_path"))
    if not summary_path or not Path(summary_path).exists():
        return False, "interpretability_summary.json path is missing."

    fold_artifacts_path = _safe_text(interpretability_payload.get("fold_artifacts_path"))
    if not fold_artifacts_path or not Path(fold_artifacts_path).exists():
        return False, "interpretability fold artifacts are missing."

    coefficient_paths = list(interpretability_payload.get("coefficient_paths") or [])
    if not coefficient_paths:
        return False, "foldwise coefficient artifacts are missing."
    if not all(Path(path).exists() for path in coefficient_paths):
        return False, "one or more foldwise coefficient artifact files are missing."

    stability = interpretability_payload.get("stability")
    if not isinstance(stability, dict) or not stability:
        return False, "interpretability stability payload is missing."
    return True, None


def _build_e14_summary_row(
    *,
    e14_record: dict[str, Any],
    anchor_record: dict[str, Any] | None,
    interpretability_payload: dict[str, Any] | None,
    status: str,
    completion_status: str,
    status_reason: str | None,
) -> dict[str, Any]:
    metadata = _design_metadata(e14_record)
    anchor = anchor_record or {}
    payload = interpretability_payload or {}
    stability = dict(payload.get("stability") or {})
    if status != "completed":
        stability = {}

    caution = (
        _safe_text((payload.get("summary_payload") or {}).get("caution"))
        or "Model-behavior evidence only; not direct neural localization evidence."
    )

    return {
        "analysis_label": _safe_text(metadata.get("anchor_analysis_label"))
        or _analysis_label(anchor),
        "anchor_experiment_id": _safe_text(metadata.get("anchor_experiment_id"))
        or _safe_text(anchor.get("experiment_id")),
        "anchor_variant_id": _safe_text(metadata.get("anchor_variant_id"))
        or _safe_text(anchor.get("variant_id")),
        "subject": _safe_text(metadata.get("anchor_subject"))
        or _safe_text(anchor.get("subject"))
        or None,
        "cv": _safe_text(anchor.get("cv")) or _safe_text(e14_record.get("cv")) or None,
        "model": _safe_text(anchor.get("model")) or _safe_text(e14_record.get("model")) or None,
        "feature_space": _safe_text(anchor.get("feature_space"))
        or _safe_text(e14_record.get("feature_space"))
        or None,
        "mean_pairwise_coef_correlation": _safe_float(stability.get("mean_pairwise_correlation")),
        "mean_sign_consistency": _safe_float(stability.get("mean_sign_consistency")),
        "mean_topk_overlap": _safe_float(stability.get("mean_top_k_overlap")),
        "n_folds": _safe_int(stability.get("n_folds")),
        "interpretability_summary_path": _safe_text(payload.get("summary_path")) or None,
        "interpretability_fold_artifacts_path": _safe_text(payload.get("fold_artifacts_path"))
        or None,
        "coefficient_artifact_paths": list(payload.get("coefficient_paths") or []),
        "status": status,
        "completion_status": completion_status,
        "status_reason": status_reason,
        "scientific_caution": caution,
    }


def build_e14_reporting_records(
    *,
    reporting_variant_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    anchor_candidates = _candidate_anchor_records(reporting_variant_records)
    e14_rows = [row for row in reporting_variant_records if _is_e14_special_cell(row)]
    if not e14_rows:
        return list(reporting_variant_records), [], {"rows": [], "errors": []}

    converted_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for row in e14_rows:
        metadata = _design_metadata(row)
        anchor_variant_id = _safe_text(metadata.get("anchor_variant_id"))
        anchor_analysis_label = _safe_text(metadata.get("anchor_analysis_label"))
        anchor_record = _resolve_anchor_record(
            anchor_candidates=anchor_candidates,
            anchor_variant_id=anchor_variant_id,
            anchor_analysis_label=anchor_analysis_label,
        )
        if anchor_record is None:
            reason = "No completed E16 anchor run was found for this E14 cell."
            summary_rows.append(
                _build_e14_summary_row(
                    e14_record=row,
                    anchor_record=None,
                    interpretability_payload=None,
                    status="not_applicable",
                    completion_status="missing_anchor",
                    status_reason=reason,
                )
            )
            derived_row = dict(row)
            derived_row["status"] = "blocked"
            derived_row["blocked_reason"] = reason
            converted_rows.append(derived_row)
            errors.append(reason)
            continue

        interpretability_payload = _resolve_interpretability_payload(anchor_record)
        valid, invalid_reason = _valid_e14_conditions(
            anchor_record=anchor_record,
            interpretability_payload=interpretability_payload,
        )
        if not valid:
            summary_rows.append(
                _build_e14_summary_row(
                    e14_record=row,
                    anchor_record=anchor_record,
                    interpretability_payload=interpretability_payload,
                    status="not_applicable",
                    completion_status="ineligible_or_missing_artifacts",
                    status_reason=invalid_reason,
                )
            )
            derived_row = dict(row)
            derived_row["status"] = "blocked"
            derived_row["blocked_reason"] = invalid_reason
            converted_rows.append(derived_row)
            if invalid_reason:
                errors.append(invalid_reason)
            continue

        summary_rows.append(
            _build_e14_summary_row(
                e14_record=row,
                anchor_record=anchor_record,
                interpretability_payload=interpretability_payload,
                status="completed",
                completion_status="completed",
                status_reason=None,
            )
        )

        derived_row = dict(row)
        derived_row.update(
            {
                "status": "completed",
                "blocked_reason": None,
                "error": None,
                "run_id": f"{_safe_text(anchor_record.get('run_id'))}__e14_stability",
                "seed": anchor_record.get("seed"),
                "report_dir": anchor_record.get("report_dir"),
                "metrics_path": anchor_record.get("metrics_path"),
                "config_path": anchor_record.get("config_path"),
                "primary_metric_name": anchor_record.get("primary_metric_name")
                or "balanced_accuracy",
                "primary_metric_value": anchor_record.get("primary_metric_value"),
                "balanced_accuracy": anchor_record.get("balanced_accuracy"),
                "macro_f1": anchor_record.get("macro_f1"),
                "accuracy": anchor_record.get("accuracy"),
                "notes": (
                    "Derived from E16 interpretability artifacts; model-behavior evidence only, "
                    "not direct neural localization."
                ),
                "n_folds": _safe_int(
                    (interpretability_payload.get("stability") or {}).get("n_folds")
                ),
            }
        )
        converted_rows.append(derived_row)

    converted_by_variant_id = {
        _safe_text(row.get("variant_id")): row
        for row in converted_rows
        if _safe_text(row.get("variant_id"))
    }
    output_records: list[dict[str, Any]] = []
    for row in reporting_variant_records:
        if not _is_e14_special_cell(row):
            output_records.append(row)
            continue
        variant_id = _safe_text(row.get("variant_id"))
        output_records.append(converted_by_variant_id.get(variant_id, row))

    summary_rows.sort(
        key=lambda row: (
            _safe_text(row.get("analysis_label")),
            _safe_text(row.get("anchor_experiment_id")),
            _safe_text(row.get("anchor_variant_id")),
        )
    )
    summary_payload = {
        "schema_version": "e14-interpretability-stability-v1",
        "rows": summary_rows,
        "errors": sorted(set(item for item in errors if _safe_text(item))),
    }
    return output_records, summary_rows, summary_payload


__all__ = ["build_e14_reporting_records"]
