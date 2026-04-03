from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


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


def _factor_settings(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("factor_settings")
    return dict(value) if isinstance(value, dict) else {}


def _is_e12_chunk_record(record: dict[str, Any]) -> bool:
    if _safe_text(record.get("experiment_id")) != "E12":
        return False
    design_metadata = _design_metadata(record)
    if _safe_text(design_metadata.get("special_cell_kind")) == "permutation_chunk":
        return True
    return bool(_safe_text(design_metadata.get("permutation_group_id")))


def _permutation_group_id(record: dict[str, Any]) -> str:
    design_metadata = _design_metadata(record)
    group_id = _safe_text(design_metadata.get("permutation_group_id"))
    if group_id:
        return group_id
    factor_settings = _factor_settings(record)
    group_id = _safe_text(factor_settings.get("permutation_group_id"))
    if group_id:
        return group_id
    return f"E12::{_safe_text(record.get('template_id')) or 'template'}::{_safe_text(record.get('variant_id'))}"


def _load_permutation_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    metrics_path_text = _safe_text(record.get("metrics_path"))
    if not metrics_path_text:
        return None
    metrics_path = Path(metrics_path_text)
    if not metrics_path.exists() or not metrics_path.is_file():
        return None
    try:
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(metrics_payload, dict):
        return None
    permutation_payload = metrics_payload.get("permutation_test")
    if not isinstance(permutation_payload, dict):
        return None
    null_scores_raw = permutation_payload.get("null_scores")
    if not isinstance(null_scores_raw, list):
        return None
    return dict(permutation_payload)


def _build_null_summary(null_scores: list[float]) -> dict[str, float]:
    values = np.asarray(null_scores, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
    }


def _merge_permutation_payload(
    *,
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    merged_null_scores: list[float] = []
    for payload in payloads:
        for value in list(payload.get("null_scores", [])):
            maybe_float = _safe_float(value)
            if maybe_float is not None:
                merged_null_scores.append(float(maybe_float))
    if not merged_null_scores:
        raise ValueError("No permutation null_scores were available for merged E12 payload.")

    first_payload = payloads[0]
    observed_score = _safe_float(first_payload.get("observed_score"))
    if observed_score is None:
        observed_score = _safe_float(first_payload.get("observed_metric"))
    if observed_score is None:
        raise ValueError("Merged E12 payload requires observed_score.")

    alpha = _safe_float(first_payload.get("alpha"))
    minimum_required = _safe_int(first_payload.get("minimum_required")) or 0
    metric_name = _safe_text(first_payload.get("metric_name")) or "balanced_accuracy"
    primary_metric_aggregation = (
        _safe_text(first_payload.get("primary_metric_aggregation")) or "mean_fold_scores"
    )
    permutation_seed = _safe_int(first_payload.get("permutation_seed"))

    ge_count = int(sum(1 for score in merged_null_scores if float(score) >= float(observed_score)))
    n_permutations = int(len(merged_null_scores))
    p_value = float((float(ge_count) + 1.0) / (float(n_permutations) + 1.0))
    null_summary = _build_null_summary(merged_null_scores)
    meets_minimum = bool(n_permutations >= int(minimum_required))
    passes_threshold = bool(alpha is not None and p_value <= float(alpha))

    merged_payload = dict(first_payload)
    merged_payload.update(
        {
            "n_permutations": int(n_permutations),
            "metric_name": str(metric_name),
            "observed_score": float(observed_score),
            "observed_metric": float(observed_score),
            "primary_metric_aggregation": str(primary_metric_aggregation),
            "permutation_seed": int(permutation_seed) if permutation_seed is not None else None,
            "p_value": float(p_value),
            "permutation_p_value": float(p_value),
            "null_summary": null_summary,
            "null_scores": [float(value) for value in merged_null_scores],
            "permutation_metric_mean": float(null_summary["mean"]),
            "permutation_metric_std": float(null_summary["std"]),
            "minimum_required": int(minimum_required),
            "meets_minimum": bool(meets_minimum),
            "alpha": float(alpha) if alpha is not None else None,
            "passes_threshold": bool(passes_threshold),
            "interpretation_status": (
                "passes_threshold" if bool(passes_threshold) else "fails_threshold"
            ),
        }
    )
    return merged_payload


def _merge_output_path(*, campaign_root: Path, group_id: str) -> Path:
    digest = hashlib.sha256(str(group_id).encode("utf-8")).hexdigest()[:16]
    merge_dir = campaign_root / "permutation_chunk_merges"
    merge_dir.mkdir(parents=True, exist_ok=True)
    return merge_dir / f"e12_permutation_merge_{digest}.json"


def _build_merged_record(
    *,
    campaign_root: Path,
    group_id: str,
    chunk_records: list[dict[str, Any]],
    completed_records: list[dict[str, Any]],
    merged_permutation_payload: dict[str, Any],
) -> dict[str, Any]:
    reference_record = dict(completed_records[0] if completed_records else chunk_records[0])
    design_metadata = _design_metadata(reference_record)
    factor_settings = _factor_settings(reference_record)

    expected_chunk_count = _safe_int(design_metadata.get("expected_chunk_count"))
    if expected_chunk_count is None:
        expected_chunk_count = _safe_int(factor_settings.get("expected_chunk_count"))
    if expected_chunk_count is None:
        expected_chunk_count = int(len(chunk_records))

    total_permutations_requested = _safe_int(design_metadata.get("total_permutations_requested"))
    if total_permutations_requested is None:
        total_permutations_requested = _safe_int(
            factor_settings.get("total_permutations_requested")
        )
    if total_permutations_requested is None:
        total_permutations_requested = int(merged_permutation_payload.get("n_permutations", 0))

    metrics_payload = {
        "primary_metric_name": reference_record.get("primary_metric_name"),
        "primary_metric_value": reference_record.get("primary_metric_value"),
        "balanced_accuracy": reference_record.get("balanced_accuracy"),
        "macro_f1": reference_record.get("macro_f1"),
        "accuracy": reference_record.get("accuracy"),
        "permutation_test": merged_permutation_payload,
        "chunk_merge": {
            "permutation_group_id": str(group_id),
            "expected_chunk_count": int(expected_chunk_count),
            "completed_chunk_count": int(len(completed_records)),
            "all_chunks_present": bool(len(completed_records) == int(expected_chunk_count)),
            "source_run_ids": [
                _safe_text(row.get("run_id"))
                for row in chunk_records
                if _safe_text(row.get("run_id"))
            ],
        },
    }
    merge_path = _merge_output_path(campaign_root=campaign_root, group_id=group_id)
    merge_path.write_text(f"{json.dumps(metrics_payload, indent=2)}\n", encoding="utf-8")

    merged_record = dict(reference_record)
    merged_record.update(
        {
            "variant_id": f"{_safe_text(reference_record.get('variant_id'))}__perm_merged",
            "trial_id": f"{_safe_text(reference_record.get('trial_id'))}__perm_merged",
            "cell_id": f"{_safe_text(reference_record.get('cell_id'))}__perm_merged",
            "run_id": f"{_safe_text(reference_record.get('run_id'))}__perm_merged",
            "status": "completed",
            "n_permutations": int(merged_permutation_payload.get("n_permutations", 0)),
            "metrics_path": str(merge_path.resolve()),
            "manifest_path": str(merge_path.resolve()),
            "error": None,
            "blocked_reason": None,
            "design_metadata": {
                **design_metadata,
                "special_cell_kind": "permutation_chunk_merged",
                "permutation_group_id": str(group_id),
                "expected_chunk_count": int(expected_chunk_count),
                "completed_chunk_count": int(len(completed_records)),
                "all_chunks_present": bool(len(completed_records) == int(expected_chunk_count)),
                "total_permutations_requested": int(total_permutations_requested),
            },
            "factor_settings": {
                **factor_settings,
                "permutation_group_id": str(group_id),
                "expected_chunk_count": int(expected_chunk_count),
                "completed_chunk_count": int(len(completed_records)),
                "all_chunks_present": bool(len(completed_records) == int(expected_chunk_count)),
                "total_permutations_requested": int(total_permutations_requested),
            },
            "notes": "; ".join(
                [
                    _safe_text(reference_record.get("notes")),
                    "e12_permutation_chunk_merge=true",
                ]
            ).strip("; "),
        }
    )
    return merged_record


def build_reporting_variant_records(
    *,
    campaign_root: Path,
    variant_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunk_records = [row for row in variant_records if _is_e12_chunk_record(row)]
    if not chunk_records:
        return list(variant_records), {"groups": [], "errors": []}

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in chunk_records:
        groups.setdefault(_permutation_group_id(row), []).append(row)

    merged_by_group: dict[str, dict[str, Any]] = {}
    group_summaries: list[dict[str, Any]] = []
    errors: list[str] = []
    for group_id, rows in sorted(groups.items()):
        completed_rows = [row for row in rows if _safe_text(row.get("status")) == "completed"]
        payloads: list[dict[str, Any]] = []
        for row in completed_rows:
            payload = _load_permutation_payload(row)
            if payload is not None:
                payloads.append(payload)
        if not payloads:
            errors.append(
                f"E12 group '{group_id}' could not be merged because no completed chunk permutation payloads were available."
            )
            continue
        try:
            merged_payload = _merge_permutation_payload(payloads=payloads)
            merged_record = _build_merged_record(
                campaign_root=campaign_root,
                group_id=group_id,
                chunk_records=rows,
                completed_records=completed_rows,
                merged_permutation_payload=merged_payload,
            )
        except Exception as exc:
            errors.append(f"E12 group '{group_id}' merge failed: {exc}")
            continue
        merged_by_group[str(group_id)] = merged_record
        group_summaries.append(
            {
                "permutation_group_id": str(group_id),
                "chunk_count": int(len(rows)),
                "completed_chunk_count": int(len(completed_rows)),
                "expected_chunk_count": int(
                    _safe_int(_design_metadata(merged_record).get("expected_chunk_count"))
                    or len(rows)
                ),
                "all_chunks_present": bool(
                    _design_metadata(merged_record).get("all_chunks_present", False)
                ),
                "merged_n_permutations": int(merged_payload.get("n_permutations", 0)),
                "merged_p_value": _safe_float(merged_payload.get("p_value")),
                "meets_minimum": bool(merged_payload.get("meets_minimum", False)),
                "minimum_required": _safe_int(merged_payload.get("minimum_required")),
                "anchor_experiment_id": _safe_text(
                    _design_metadata(merged_record).get("anchor_experiment_id")
                ),
            }
        )

    reporting_records: list[dict[str, Any]] = []
    emitted_groups: set[str] = set()
    for row in variant_records:
        if not _is_e12_chunk_record(row):
            reporting_records.append(row)
            continue
        group_id = _permutation_group_id(row)
        merged = merged_by_group.get(group_id)
        if merged is None:
            reporting_records.append(row)
            continue
        if group_id in emitted_groups:
            continue
        reporting_records.append(merged)
        emitted_groups.add(group_id)

    summary_payload = {
        "schema_version": "e12-permutation-chunk-merge-v1",
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "groups": group_summaries,
        "errors": errors,
    }
    return reporting_records, summary_payload


def _is_e12_merged_record(record: dict[str, Any]) -> bool:
    if _safe_text(record.get("experiment_id")) != "E12":
        return False
    design_metadata = _design_metadata(record)
    return _safe_text(design_metadata.get("special_cell_kind")) == "permutation_chunk_merged"


def build_e12_table_ready_rows(*, reporting_variant_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in reporting_variant_records:
        if not _is_e12_merged_record(record):
            continue
        metrics_payload = _load_permutation_payload(record)
        if not isinstance(metrics_payload, dict):
            continue
        design_metadata = _design_metadata(record)
        cv_mode = _safe_text(record.get("cv"))
        subject = _safe_text(record.get("subject"))
        train_subject = _safe_text(record.get("train_subject"))
        test_subject = _safe_text(record.get("test_subject"))
        if cv_mode == "within_subject_loso_session" and subject:
            analysis_label = f"within_subject_loso_session:{subject}"
        elif cv_mode == "frozen_cross_person_transfer" and train_subject and test_subject:
            analysis_label = f"frozen_cross_person_transfer:{train_subject}->{test_subject}"
        else:
            analysis_label = (
                _safe_text(design_metadata.get("anchor_analysis_label"))
                or _safe_text(design_metadata.get("permutation_group_id"))
                or _safe_text(record.get("variant_id"))
            )
        null_summary = metrics_payload.get("null_summary")
        null_summary_payload = dict(null_summary) if isinstance(null_summary, dict) else {}
        rows.append(
            {
                "analysis_label": str(analysis_label),
                "observed_balanced_accuracy": _safe_float(metrics_payload.get("observed_score")),
                "null_mean": _safe_float(null_summary_payload.get("mean")),
                "null_min": _safe_float(null_summary_payload.get("min")),
                "null_max": _safe_float(null_summary_payload.get("max")),
                "null_q25": _safe_float(null_summary_payload.get("q25")),
                "null_q75": _safe_float(null_summary_payload.get("q75")),
                "empirical_p": _safe_float(metrics_payload.get("p_value")),
                "n_permutations": _safe_int(metrics_payload.get("n_permutations")),
                "minimum_required": _safe_int(metrics_payload.get("minimum_required")),
                "meets_minimum": bool(metrics_payload.get("meets_minimum", False)),
                "anchor_experiment_id": _safe_text(design_metadata.get("anchor_experiment_id")),
                "anchor_template_id": _safe_text(design_metadata.get("anchor_template_id")),
                "subject": subject or None,
                "transfer_direction": (
                    f"{train_subject}->{test_subject}" if train_subject and test_subject else None
                ),
                "all_chunks_present": bool(design_metadata.get("all_chunks_present", False)),
                "completed_chunk_count": _safe_int(design_metadata.get("completed_chunk_count")),
                "expected_chunk_count": _safe_int(design_metadata.get("expected_chunk_count")),
                "permutation_group_id": _safe_text(design_metadata.get("permutation_group_id")),
                "run_id": _safe_text(record.get("run_id")),
                "variant_id": _safe_text(record.get("variant_id")),
                "metrics_path": _safe_text(record.get("metrics_path")),
                "report_dir": _safe_text(record.get("report_dir")),
            }
        )
    rows.sort(key=lambda row: str(row.get("analysis_label") or ""))
    return rows


__all__ = ["build_e12_table_ready_rows", "build_reporting_variant_records"]
