from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_METRICS_BUNDLE,
    compute_config_hash,
    register_artifact,
)
from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.methodology import MethodologyPolicyName
from Thesis_ML.experiments.compute_policy import resolve_compute_policy
from Thesis_ML.experiments.compute_scheduler import (
    ComputeRunAssignment,
    ComputeRunRequest,
    plan_compute_schedule,
)
from Thesis_ML.experiments.model_catalog import (
    get_model_cost_entry,
)
from Thesis_ML.experiments.model_catalog import (
    projected_runtime_seconds as resolve_projected_runtime_seconds,
)
from Thesis_ML.observability import (
    AnomalyEngine,
    EtaEstimator,
    ExecutionEventBus,
)
from Thesis_ML.observability.console_reporter import build_progress_reporter
from Thesis_ML.orchestration.contracts import CompiledStudyManifest
from Thesis_ML.orchestration.decision_reports import (
    write_decision_reports as _write_decision_reports,
)
from Thesis_ML.orchestration.dummy_baseline_aggregation import (
    build_e13_table_ready_rows as _build_e13_table_ready_rows,
)
from Thesis_ML.orchestration.execution_bridge import (
    apply_feature_matrix_reuse_variant as _apply_feature_matrix_reuse_variant,
)
from Thesis_ML.orchestration.execution_bridge import (
    build_variant_official_job as _build_variant_official_job,
)
from Thesis_ML.orchestration.execution_bridge import (
    execute_official_jobs as _execute_official_jobs,
)
from Thesis_ML.orchestration.execution_bridge import execute_variant as _execute_variant
from Thesis_ML.orchestration.execution_bridge import (
    extract_artifact_registry_path as _extract_artifact_registry_path,
)
from Thesis_ML.orchestration.execution_bridge import (
    extract_feature_matrix_artifact_id as _extract_feature_matrix_artifact_id,
)
from Thesis_ML.orchestration.execution_bridge import (
    plan_sibling_feature_matrix_reuse as _plan_sibling_feature_matrix_reuse,
)
from Thesis_ML.orchestration.execution_bridge import (
    resolve_variant_id as _resolve_variant_id,
)
from Thesis_ML.orchestration.execution_bridge import (
    resolve_variant_run_id as _resolve_variant_run_id,
)
from Thesis_ML.orchestration.experiment_selection import (
    collect_dataset_scope as _collect_dataset_scope,
)
from Thesis_ML.orchestration.experiment_selection import (
    select_experiments as _select_experiments,
)
from Thesis_ML.orchestration.interpretability_stability_aggregation import (
    build_e14_reporting_records as _build_e14_reporting_records,
)
from Thesis_ML.orchestration.permutation_chunk_aggregation import (
    build_e12_table_ready_rows as _build_e12_table_ready_rows,
)
from Thesis_ML.orchestration.permutation_chunk_aggregation import (
    build_reporting_variant_records as _build_reporting_variant_records,
)
from Thesis_ML.orchestration.reporting import (
    status_snapshot as _status_snapshot,
)
from Thesis_ML.orchestration.reporting import (
    summarize_by_experiment as _summarize_by_experiment,
)
from Thesis_ML.orchestration.reporting import (
    write_campaign_execution_report as _write_campaign_execution_report,
)
from Thesis_ML.orchestration.reporting import (
    write_experiment_outputs as _write_experiment_outputs,
)
from Thesis_ML.orchestration.reporting import (
    write_run_log_export as _write_run_log_export,
)
from Thesis_ML.orchestration.reporting import (
    write_stage_summaries as _write_stage_summaries,
)
from Thesis_ML.orchestration.result_aggregation import (
    aggregate_variant_records,
    build_summary_output_rows,
)
from Thesis_ML.orchestration.search_space import build_search_space_map
from Thesis_ML.orchestration.study_loading import (
    read_registry_manifest,
    read_workbook_manifest,
)
from Thesis_ML.orchestration.variant_expansion import (
    expand_experiment_variants as _expand_experiment_variants,
)
from Thesis_ML.orchestration.variant_expansion import (
    materialize_experiment_cells as _materialize_experiment_cells,
)
from Thesis_ML.orchestration.workbook_bridge import (
    build_effect_summary_rows as _build_effect_summary_rows,
)
from Thesis_ML.orchestration.workbook_bridge import (
    build_generated_design_rows as _build_generated_design_rows,
)
from Thesis_ML.orchestration.workbook_bridge import (
    build_machine_status_rows as _build_machine_status_rows,
)
from Thesis_ML.orchestration.workbook_bridge import (
    build_run_log_writeback_rows as _build_run_log_writeback_rows,
)
from Thesis_ML.orchestration.workbook_bridge import (
    build_study_review_rows as _build_study_review_rows,
)
from Thesis_ML.orchestration.workbook_bridge import (
    build_trial_results_rows as _build_trial_results_rows,
)
from Thesis_ML.orchestration.workbook_writeback import write_workbook_results
from Thesis_ML.verification.confirmatory_scope_runtime_alignment import (
    build_confirmatory_control_coverage_rows,
    collect_runtime_confirmatory_anchors,
)


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_e14_derived_reporting_only_block(row: dict[str, Any]) -> bool:
    if _safe_text(row.get("experiment_id")) != "E14":
        return False
    reason = _safe_text(row.get("blocked_reason")).lower()
    return "derived from e16 artifacts" in reason and "not executed as a separate fit" in reason


def _design_metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("design_metadata")
    return dict(value) if isinstance(value, dict) else {}


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _analysis_label_from_identity(row: dict[str, Any]) -> str:
    cv_mode = _safe_text(row.get("cv"))
    subject = _safe_text(row.get("subject"))
    train_subject = _safe_text(row.get("train_subject"))
    test_subject = _safe_text(row.get("test_subject"))
    if cv_mode == "within_subject_loso_session" and subject:
        return f"{cv_mode}:{subject}"
    if cv_mode == "frozen_cross_person_transfer" and train_subject and test_subject:
        return f"{cv_mode}:{train_subject}->{test_subject}"
    return _safe_text(row.get("analysis_label"))


_THESIS_LOCKED_CORE_KEYS: tuple[str, ...] = (
    "target",
    "model",
    "feature_space",
    "preprocessing_strategy",
    "dimensionality_strategy",
    "methodology_policy_name",
    "class_weight_policy",
)


def _normalized_lock_value(value: Any) -> str:
    return _safe_text(value).lower()


def _derive_within_family_locked_core(runtime_anchor_rows: list[dict[str, Any]]) -> dict[str, str]:
    within_rows = [
        row
        for row in runtime_anchor_rows
        if _safe_text(row.get("experiment_id")) == "E16"
        and _safe_text(row.get("cv")) == "within_subject_loso_session"
    ]
    if not within_rows:
        return {}
    resolved: dict[str, str] = {}
    for key in _THESIS_LOCKED_CORE_KEYS:
        values = {
            _normalized_lock_value(row.get(key))
            for row in within_rows
            if _normalized_lock_value(row.get(key))
        }
        if len(values) == 1:
            resolved[key] = next(iter(values))
    return resolved


def _locked_core_mismatch_keys(
    *,
    row: dict[str, Any],
    locked_core: dict[str, str],
) -> list[str]:
    mismatches: list[str] = []
    for key, expected in locked_core.items():
        if not expected:
            continue
        observed = _normalized_lock_value(row.get(key))
        if observed != expected:
            mismatches.append(str(key))
    return mismatches


def _extract_external_template_id(run_manifest: dict[str, Any]) -> str:
    template_id = _safe_text(run_manifest.get("template_id"))
    if template_id:
        return template_id
    variant_id = _safe_text(run_manifest.get("variant_id"))
    if variant_id:
        return variant_id.split("__")[0]
    run_id = _safe_text(run_manifest.get("run_id"))
    if not run_id:
        return ""
    # run_id format: ds_<experiment_id>_<template_id>__<index>_<campaign_id>
    prefix = "ds_E16_"
    if not run_id.startswith(prefix):
        return ""
    suffix = run_id[len(prefix) :]
    marker = "__"
    if marker not in suffix:
        return ""
    return suffix.split(marker, 1)[0]


def _collect_external_e16_anchor_records(
    *,
    output_root: Path,
    runtime_anchor_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    e16_root = output_root / "E16"
    if not e16_root.exists():
        return []
    expected_labels = {
        _safe_text(row.get("analysis_label"))
        for row in runtime_anchor_rows
        if _safe_text(row.get("experiment_id")) == "E16"
        and _safe_text(row.get("cv")) == "within_subject_loso_session"
        and _safe_text(row.get("analysis_label"))
    }
    manifest_paths = sorted(e16_root.glob("*/run_manifests/*.json"))
    candidates_by_label: dict[str, tuple[tuple[str, float], dict[str, Any]]] = {}

    for manifest_path in manifest_paths:
        manifest_payload = _safe_load_json(manifest_path)
        if not isinstance(manifest_payload, dict):
            continue
        if _safe_text(manifest_payload.get("experiment_id")) != "E16":
            continue
        status = _safe_text(manifest_payload.get("status")).lower()
        if status not in {"completed", "success"}:
            continue

        config_used = (
            dict(manifest_payload.get("config_used"))
            if isinstance(manifest_payload.get("config_used"), dict)
            else {}
        )
        params_payload = (
            dict(config_used.get("params_snapshot"))
            if isinstance(config_used.get("params_snapshot"), dict)
            else (
                dict(config_used.get("params"))
                if isinstance(config_used.get("params"), dict)
                else {}
            )
        )
        cv_mode = _safe_text(params_payload.get("cv"))
        subject = _safe_text(params_payload.get("subject"))
        if cv_mode != "within_subject_loso_session" or not subject:
            continue
        analysis_label = f"{cv_mode}:{subject}"
        if expected_labels and analysis_label not in expected_labels:
            continue

        artifacts = (
            dict(manifest_payload.get("artifacts"))
            if isinstance(manifest_payload.get("artifacts"), dict)
            else {}
        )
        report_dir = _safe_text(artifacts.get("report_dir"))
        metrics_path = _safe_text(artifacts.get("metrics_path"))
        config_path = _safe_text(artifacts.get("config_path"))
        if not metrics_path:
            continue
        metrics_payload = _safe_load_json(Path(metrics_path)) or {}
        primary_metric_name = (
            _safe_text(metrics_payload.get("primary_metric_name")) or "balanced_accuracy"
        )
        primary_metric_value = _safe_float(metrics_payload.get(primary_metric_name))
        if primary_metric_value is None:
            primary_metric_value = _safe_float(metrics_payload.get("balanced_accuracy"))

        template_id = _extract_external_template_id(manifest_payload)
        if not template_id:
            continue

        candidate = {
            "experiment_id": "E16",
            "variant_id": template_id,
            "status": "completed",
            "cv": cv_mode,
            "model": _safe_text(params_payload.get("model")) or "ridge",
            "subject": subject,
            "target": _safe_text(params_payload.get("target")) or None,
            "filter_modality": _safe_text(params_payload.get("filter_modality")) or None,
            "feature_space": _safe_text(params_payload.get("feature_space")) or None,
            "preprocessing_strategy": _safe_text(params_payload.get("preprocessing_strategy"))
            or None,
            "dimensionality_strategy": _safe_text(params_payload.get("dimensionality_strategy"))
            or None,
            "methodology_policy_name": _safe_text(params_payload.get("methodology_policy_name"))
            or None,
            "class_weight_policy": _safe_text(params_payload.get("class_weight_policy")) or None,
            "run_id": _safe_text(manifest_payload.get("run_id")) or None,
            "report_dir": report_dir or None,
            "metrics_path": metrics_path,
            "config_path": config_path or None,
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": primary_metric_value,
            "balanced_accuracy": _safe_float(metrics_payload.get("balanced_accuracy")),
            "macro_f1": _safe_float(metrics_payload.get("macro_f1")),
            "accuracy": _safe_float(metrics_payload.get("accuracy")),
            "anchor_source": "external_output_root_fallback",
            "anchor_source_manifest_path": str(manifest_path.resolve()),
            "analysis_label": analysis_label,
        }
        finished_at = _safe_text(manifest_payload.get("finished_at"))
        sort_key = (
            finished_at,
            float(manifest_path.stat().st_mtime),
        )
        current = candidates_by_label.get(analysis_label)
        if current is None or sort_key > current[0]:
            candidates_by_label[analysis_label] = (sort_key, candidate)

    rows = [payload for _, payload in sorted(candidates_by_label.values(), key=lambda item: item[0])]
    rows.sort(key=lambda row: _safe_text(row.get("analysis_label")))
    return rows


def _build_confirmatory_model_rows(
    *,
    reporting_variant_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_label: dict[str, dict[str, Any]] = {}
    for record in reporting_variant_records:
        if str(record.get("experiment_id") or "") not in {"E16", "E17"}:
            continue
        design_metadata = _design_metadata(record)
        params = {
            "cv": record.get("cv"),
            "subject": record.get("subject"),
            "train_subject": record.get("train_subject"),
            "test_subject": record.get("test_subject"),
        }
        analysis_label = _analysis_label_from_identity(params)
        if not analysis_label:
            continue
        metrics_payload = _safe_load_json(Path(str(record.get("metrics_path") or ""))) or {}
        metric_name = (
            str(record.get("primary_metric_name") or "").strip()
            or str(metrics_payload.get("primary_metric_name") or "balanced_accuracy").strip()
        )
        observed_score = _safe_float(record.get("primary_metric_value"))
        if observed_score is None and metric_name:
            observed_score = _safe_float(metrics_payload.get(metric_name))
        if observed_score is None:
            observed_score = _safe_float(metrics_payload.get("balanced_accuracy"))
        row = {
            "analysis_label": analysis_label,
            "family": (
                "E16_within_person_confirmatory"
                if str(record.get("experiment_id") or "") == "E16"
                else "E17_cross_person_transfer_confirmatory"
            ),
            "experiment_id": str(record.get("experiment_id") or ""),
            "variant_id": str(record.get("variant_id") or ""),
            "target": str(record.get("target") or "").strip() or None,
            "cv": str(record.get("cv") or "").strip() or None,
            "subject": str(record.get("subject") or "").strip() or None,
            "train_subject": str(record.get("train_subject") or "").strip() or None,
            "test_subject": str(record.get("test_subject") or "").strip() or None,
            "model": str(record.get("model") or "").strip() or None,
            "feature_space": str(record.get("feature_space") or "").strip() or None,
            "filter_modality": str(record.get("filter_modality") or "").strip() or None,
            "preprocessing_strategy": str(record.get("preprocessing_strategy") or "").strip()
            or None,
            "dimensionality_strategy": str(record.get("dimensionality_strategy") or "").strip()
            or None,
            "methodology_policy_name": str(record.get("methodology_policy_name") or "").strip()
            or None,
            "class_weight_policy": str(record.get("class_weight_policy") or "").strip() or None,
            "metric_name": metric_name or None,
            "observed_score": observed_score,
            "status": str(record.get("status") or "").strip() or None,
            "run_id": str(record.get("run_id") or "").strip() or None,
            "metrics_path": str(record.get("metrics_path") or "").strip() or None,
            "report_dir": str(record.get("report_dir") or "").strip() or None,
            "legacy_diagnostic": bool(
                bool(record.get("legacy_diagnostic"))
                or bool(design_metadata.get("legacy_diagnostic"))
                or _safe_text(record.get("thesis_selection_status")) == "legacy_diagnostic_excluded"
            ),
            "legacy_reason": _safe_text(record.get("legacy_reason"))
            or _safe_text(design_metadata.get("legacy_reason"))
            or None,
        }
        if analysis_label not in rows_by_label:
            rows_by_label[analysis_label] = row
    return [rows_by_label[key] for key in sorted(rows_by_label.keys())]


def _write_confirmatory_thesis_artifacts(
    *,
    campaign_root: Path,
    runtime_anchor_rows: list[dict[str, Any]],
    confirmatory_model_rows: list[dict[str, Any]],
    e12_table_rows: list[dict[str, Any]],
    e13_table_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> dict[str, str | None]:
    if not runtime_anchor_rows:
        return {}

    import pandas as pd

    out_root = campaign_root / "special_aggregations" / "confirmatory"
    out_root.mkdir(parents=True, exist_ok=True)

    def _map_by_label(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        mapped: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = str(row.get("analysis_label") or "").strip()
            if not label or label in mapped:
                continue
            mapped[label] = dict(row)
        return mapped

    confirmatory_by_label = _map_by_label(confirmatory_model_rows)
    e12_by_label = _map_by_label(e12_table_rows)
    e13_by_label = _map_by_label(e13_table_rows)
    coverage_by_label = _map_by_label(coverage_rows)
    locked_core = _derive_within_family_locked_core(runtime_anchor_rows)

    anchor_manifest_rows: list[dict[str, Any]] = []
    within_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    e12_thesis_rows: list[dict[str, Any]] = []
    e13_thesis_rows: list[dict[str, Any]] = []

    for anchor in runtime_anchor_rows:
        analysis_label = str(anchor.get("analysis_label") or "").strip()
        if not analysis_label:
            continue
        cv_mode = str(anchor.get("cv") or "").strip()
        family = (
            "within_person_confirmatory"
            if cv_mode == "within_subject_loso_session"
            else "cross_person_transfer_confirmatory"
        )
        confirmatory_row = confirmatory_by_label.get(analysis_label, {})
        e12_row = e12_by_label.get(analysis_label, {})
        e13_row = e13_by_label.get(analysis_label, {})
        coverage_row = coverage_by_label.get(analysis_label, {})
        anchor_lock_mismatches = _locked_core_mismatch_keys(row=anchor, locked_core=locked_core)
        confirmatory_lock_mismatches = (
            _locked_core_mismatch_keys(row=confirmatory_row, locked_core=locked_core)
            if confirmatory_row
            else []
        )
        explicit_legacy = bool(
            bool(anchor.get("legacy_diagnostic")) or bool(confirmatory_row.get("legacy_diagnostic"))
        )
        legacy_reason = ""
        if explicit_legacy:
            legacy_reason = (
                _safe_text(confirmatory_row.get("legacy_reason"))
                or _safe_text(anchor.get("legacy_reason"))
                or "explicit_legacy_diagnostic"
            )
        elif anchor_lock_mismatches:
            legacy_reason = (
                "runtime_anchor_locked_core_mismatch:"
                + ",".join(sorted(anchor_lock_mismatches))
            )
        elif confirmatory_lock_mismatches:
            legacy_reason = (
                "resolved_confirmatory_locked_core_mismatch:"
                + ",".join(sorted(confirmatory_lock_mismatches))
            )
        selected_for_thesis = bool(confirmatory_row) and not bool(legacy_reason)

        anchor_manifest_rows.append(
            {
                "analysis_label": analysis_label,
                "family": family,
                "runtime_anchor_experiment_id": str(anchor.get("experiment_id") or "") or None,
                "runtime_anchor_template_id": str(anchor.get("template_id") or "") or None,
                "cv": cv_mode or None,
                "subject": str(anchor.get("subject") or "") or None,
                "train_subject": str(anchor.get("train_subject") or "") or None,
                "test_subject": str(anchor.get("test_subject") or "") or None,
                "target": str(anchor.get("target") or "") or None,
                "feature_space": str(anchor.get("feature_space") or "") or None,
                "preprocessing_strategy": str(anchor.get("preprocessing_strategy") or "") or None,
                "dimensionality_strategy": str(anchor.get("dimensionality_strategy") or "") or None,
                "methodology_policy_name": str(anchor.get("methodology_policy_name") or "") or None,
                "class_weight_policy": str(anchor.get("class_weight_policy") or "") or None,
                "confirmatory_present": bool(confirmatory_row),
                "locked_core_expected": dict(locked_core),
                "locked_core_match": bool(not anchor_lock_mismatches and not confirmatory_lock_mismatches),
                "legacy_diagnostic": bool(legacy_reason),
                "legacy_reason": (legacy_reason or None),
                "thesis_selection_status": (
                    "selected" if selected_for_thesis else "legacy_diagnostic_excluded"
                ),
                "task_scope_semantics": "advisory_only_not_execution_filtered",
                "e12_covered": bool(coverage_row.get("e12_covered", False)),
                "e13_covered": bool(coverage_row.get("e13_covered", False)),
            }
        )

        baseline_line = {
            "analysis_label": analysis_label,
            "family": family,
            "row_type": "majority_baseline",
            "subject": str(anchor.get("subject") or "") or None,
            "train_subject": str(anchor.get("train_subject") or "") or None,
            "test_subject": str(anchor.get("test_subject") or "") or None,
            "model": str(e13_row.get("model") or "") or None,
            "metric_name": str(e13_row.get("metric_name") or "") or None,
            "observed_score": _safe_float(e13_row.get("observed_baseline_score")),
            "status": str(e13_row.get("status") or "") or None,
            "run_id": str(e13_row.get("run_id") or "") or None,
            "metrics_path": str(e13_row.get("metrics_path") or "") or None,
        }
        confirmatory_line = {
            "analysis_label": analysis_label,
            "family": family,
            "row_type": "confirmatory_model",
            "subject": str(anchor.get("subject") or "") or None,
            "train_subject": str(anchor.get("train_subject") or "") or None,
            "test_subject": str(anchor.get("test_subject") or "") or None,
            "model": str(confirmatory_row.get("model") or "") or None,
            "metric_name": str(confirmatory_row.get("metric_name") or "") or None,
            "observed_score": _safe_float(confirmatory_row.get("observed_score")),
            "status": str(confirmatory_row.get("status") or "") or None,
            "run_id": str(confirmatory_row.get("run_id") or "") or None,
            "metrics_path": str(confirmatory_row.get("metrics_path") or "") or None,
        }
        if selected_for_thesis:
            if family == "within_person_confirmatory":
                within_rows.append(confirmatory_line)
                within_rows.append(baseline_line)
            else:
                transfer_rows.append(confirmatory_line)
                transfer_rows.append(baseline_line)

        if selected_for_thesis:
            permutation_rows.append(
                {
                    "analysis_label": analysis_label,
                    "family": family,
                    "observed_score": _safe_float(e12_row.get("observed_balanced_accuracy")),
                    "null_mean": _safe_float(e12_row.get("null_mean")),
                    "null_min": _safe_float(e12_row.get("null_min")),
                    "null_max": _safe_float(e12_row.get("null_max")),
                    "null_q25": _safe_float(e12_row.get("null_q25")),
                    "null_q75": _safe_float(e12_row.get("null_q75")),
                    "p_value": _safe_float(e12_row.get("empirical_p")),
                    "n_permutations": _optional_int(e12_row.get("n_permutations")),
                    "coverage_status": bool(coverage_row.get("e12_covered", False)),
                    "completion_status": (
                        "completed"
                        if bool(coverage_row.get("e12_covered", False))
                        and bool(e12_row.get("meets_minimum", False))
                        else "missing_or_incomplete"
                    ),
                    "metrics_path": str(e12_row.get("metrics_path") or "") or None,
                    "run_id": str(e12_row.get("run_id") or "") or None,
                }
            )

        if selected_for_thesis:
            e12_thesis_rows.append(
                {
                    "analysis_label": analysis_label,
                    "family": family,
                    **dict(e12_row),
                    "covered": bool(coverage_row.get("e12_covered", False)),
                }
            )
            e13_thesis_rows.append(
                {
                    "analysis_label": analysis_label,
                    "family": family,
                    **dict(e13_row),
                    "covered": bool(coverage_row.get("e13_covered", False)),
                }
            )

    files = {
        "confirmatory_anchor_manifest": "confirmatory_anchor_manifest",
        "thesis_e16_within_person_summary": "thesis_e16_within_person_summary",
        "thesis_e17_transfer_summary": "thesis_e17_transfer_summary",
        "thesis_e12_permutation_summary": "thesis_e12_permutation_summary",
        "thesis_e13_baseline_summary": "thesis_e13_baseline_summary",
        "thesis_permutation_robustness_summary": "thesis_permutation_robustness_summary",
    }
    payloads: dict[str, list[dict[str, Any]]] = {
        "confirmatory_anchor_manifest": sorted(
            anchor_manifest_rows, key=lambda row: str(row.get("analysis_label") or "")
        ),
        "thesis_e16_within_person_summary": sorted(
            within_rows,
            key=lambda row: (str(row.get("analysis_label") or ""), str(row.get("row_type") or "")),
        ),
        "thesis_e17_transfer_summary": sorted(
            transfer_rows,
            key=lambda row: (str(row.get("analysis_label") or ""), str(row.get("row_type") or "")),
        ),
        "thesis_e12_permutation_summary": sorted(
            e12_thesis_rows, key=lambda row: str(row.get("analysis_label") or "")
        ),
        "thesis_e13_baseline_summary": sorted(
            e13_thesis_rows, key=lambda row: str(row.get("analysis_label") or "")
        ),
        "thesis_permutation_robustness_summary": sorted(
            permutation_rows, key=lambda row: str(row.get("analysis_label") or "")
        ),
    }

    output_paths: dict[str, str | None] = {}
    for key, stem in files.items():
        rows = payloads.get(key, [])
        csv_path = out_root / f"{stem}.csv"
        json_path = out_root / f"{stem}.json"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        json_path.write_text(f"{json.dumps(rows, indent=2)}\n", encoding="utf-8")
        output_paths[f"{key}_csv"] = str(csv_path.resolve())
        output_paths[f"{key}_json"] = str(json_path.resolve())
    return output_paths


def _extract_stage_execution_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    stage_execution = record.get("stage_execution")
    if isinstance(stage_execution, dict):
        return dict(stage_execution)
    config_path_raw = record.get("config_path")
    if not isinstance(config_path_raw, str) or not config_path_raw:
        return None
    config_payload = _safe_load_json(Path(config_path_raw))
    if not isinstance(config_payload, dict):
        return None
    stage_execution_payload = config_payload.get("stage_execution")
    if isinstance(stage_execution_payload, dict):
        return dict(stage_execution_payload)
    return None


def _build_stage_evidence_rows(variant_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in variant_records:
        stage_execution = _extract_stage_execution_payload(record)
        if not isinstance(stage_execution, dict):
            continue
        telemetry_rows = stage_execution.get("telemetry")
        if not isinstance(telemetry_rows, list):
            continue
        run_id = str(record.get("run_id") or "")
        experiment_id = str(record.get("experiment_id") or "")
        variant_id = str(record.get("variant_id") or "")
        for telemetry in telemetry_rows:
            if not isinstance(telemetry, dict):
                continue
            stage_key = str(telemetry.get("stage") or "")
            if not stage_key:
                continue
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "variant_id": variant_id,
                    "run_id": run_id,
                    "stage_key": stage_key,
                    "status": str(telemetry.get("status") or ""),
                    "duration_seconds": _safe_float(telemetry.get("duration_seconds")),
                    "duration_source": telemetry.get("duration_source"),
                    "resource_coverage": telemetry.get("resource_coverage"),
                    "evidence_quality": telemetry.get("evidence_quality"),
                    "fallback_used": bool(telemetry.get("fallback_used", False)),
                    "fallback_reason": telemetry.get("fallback_reason"),
                    "planned_backend_family": telemetry.get("planned_backend_family"),
                    "observed_backend_family": telemetry.get("observed_backend_family"),
                    "planned_compute_lane": telemetry.get("planned_compute_lane"),
                    "observed_compute_lane": telemetry.get("observed_compute_lane"),
                    "planning_match": telemetry.get("planning_match"),
                    "backend_match": telemetry.get("backend_match"),
                    "lane_match": telemetry.get("lane_match"),
                    "executor_match": telemetry.get("executor_match"),
                    "mean_cpu_percent": _safe_float(telemetry.get("mean_cpu_percent")),
                    "peak_cpu_percent": _safe_float(telemetry.get("peak_cpu_percent")),
                    "peak_rss_mb": _safe_float(telemetry.get("peak_rss_mb")),
                    "peak_vms_mb": _safe_float(telemetry.get("peak_vms_mb")),
                    "read_bytes_delta": telemetry.get("read_bytes_delta"),
                    "write_bytes_delta": telemetry.get("write_bytes_delta"),
                    "peak_gpu_memory_mb": _safe_float(telemetry.get("peak_gpu_memory_mb")),
                    "peak_gpu_utilization_percent": _safe_float(
                        telemetry.get("peak_gpu_utilization_percent")
                    ),
                    "lease_required": bool(telemetry.get("lease_required", False)),
                    "lease_class": telemetry.get("lease_class"),
                    "lease_owner_identity": telemetry.get("lease_owner_identity"),
                    "lease_acquired": telemetry.get("lease_acquired"),
                    "lease_wait_seconds": _safe_float(telemetry.get("lease_wait_seconds")),
                    "lease_queue_depth_at_acquire": telemetry.get("lease_queue_depth_at_acquire"),
                    "lease_acquired_at_utc": telemetry.get("lease_acquired_at_utc"),
                    "lease_released_at_utc": telemetry.get("lease_released_at_utc"),
                    "lease_held_seconds": _safe_float(telemetry.get("lease_held_seconds")),
                    "lease_released": telemetry.get("lease_released"),
                }
            )
    return rows


def _write_stage_evidence_summaries(
    *,
    campaign_root: Path,
    variant_records: list[dict[str, Any]],
) -> dict[str, str]:
    stage_rows = _build_stage_evidence_rows(variant_records)

    def _parse_utc(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _percentile(values: list[float], quantile: float) -> float | None:
        if not values:
            return None
        sorted_values = sorted(float(item) for item in values)
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        clipped = min(max(float(quantile), 0.0), 1.0)
        position = (len(sorted_values) - 1) * clipped
        lower = int(position)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = float(position - lower)
        return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)

    stage_totals: dict[str, float] = {}
    stage_counts: Counter[str] = Counter()
    stage_missing: Counter[str] = Counter()
    stage_fallbacks: Counter[str] = Counter()
    stage_coverage: dict[str, Counter[str]] = {}
    stage_lease_required_counts: Counter[str] = Counter()
    stage_lease_acquired_counts: Counter[str] = Counter()
    stage_lease_missing_counts: Counter[str] = Counter()
    stage_lease_waits: dict[str, list[float]] = {}
    gpu_lease_intervals: list[dict[str, Any]] = []
    stage_gpu_hold_totals: dict[str, float] = {}
    for row in stage_rows:
        stage_key = str(row.get("stage_key") or "")
        if not stage_key:
            continue
        stage_counts[stage_key] += 1
        duration_seconds = _safe_float(row.get("duration_seconds"))
        if duration_seconds is not None:
            stage_totals[stage_key] = float(stage_totals.get(stage_key, 0.0)) + float(
                duration_seconds
            )
        if str(row.get("status")) == "missing":
            stage_missing[stage_key] += 1
        if bool(row.get("fallback_used", False)):
            stage_fallbacks[stage_key] += 1
        coverage = str(row.get("resource_coverage") or "none")
        stage_coverage.setdefault(stage_key, Counter())[coverage] += 1

        lease_required = bool(row.get("lease_required", False))
        lease_acquired = bool(row.get("lease_acquired", False))
        if lease_required:
            stage_lease_required_counts[stage_key] += 1
            if lease_acquired:
                stage_lease_acquired_counts[stage_key] += 1
            else:
                stage_lease_missing_counts[stage_key] += 1
            wait_seconds = _safe_float(row.get("lease_wait_seconds"))
            if wait_seconds is not None:
                stage_lease_waits.setdefault(stage_key, []).append(float(wait_seconds))

        lease_class = str(row.get("lease_class") or "").strip().lower()
        lease_started = _parse_utc(row.get("lease_acquired_at_utc"))
        lease_ended = _parse_utc(row.get("lease_released_at_utc"))
        if lease_class == "gpu" and lease_started is not None and lease_ended is not None:
            hold_seconds = max(0.0, float((lease_ended - lease_started).total_seconds()))
            gpu_lease_intervals.append(
                {
                    "stage_key": stage_key,
                    "start": lease_started,
                    "end": lease_ended,
                    "hold_seconds": hold_seconds,
                }
            )
            stage_gpu_hold_totals[stage_key] = float(
                stage_gpu_hold_totals.get(stage_key, 0.0)
            ) + float(hold_seconds)

    dominant_stages = sorted(
        [
            {
                "stage_key": stage_key,
                "total_duration_seconds": float(total_duration),
                "observation_count": int(stage_counts.get(stage_key, 0)),
            }
            for stage_key, total_duration in stage_totals.items()
        ],
        key=lambda item: _safe_float(item.get("total_duration_seconds")) or 0.0,
        reverse=True,
    )
    fallback_hotspots = sorted(
        [
            {
                "stage_key": stage_key,
                "fallback_count": int(count),
                "observation_count": int(stage_counts.get(stage_key, 0)),
            }
            for stage_key, count in stage_fallbacks.items()
            if int(count) > 0
        ],
        key=lambda item: _optional_int(item.get("fallback_count")) or 0,
        reverse=True,
    )

    stage_execution_summary = {
        "schema_version": "stage-execution-summary-v1",
        "generated_at_utc": _utc_timestamp(),
        "n_stage_rows": int(len(stage_rows)),
        "stage_totals": {
            stage_key: {
                "observation_count": int(stage_counts.get(stage_key, 0)),
                "missing_count": int(stage_missing.get(stage_key, 0)),
                "fallback_count": int(stage_fallbacks.get(stage_key, 0)),
                "total_duration_seconds": float(stage_totals.get(stage_key, 0.0)),
                "resource_coverage_counts": {
                    str(key): int(value)
                    for key, value in sorted(stage_coverage.get(stage_key, Counter()).items())
                },
            }
            for stage_key in sorted(stage_counts)
        },
        "dominant_stages": dominant_stages[:10],
        "fallback_hotspots": fallback_hotspots[:10],
    }
    stage_execution_summary_path = campaign_root / "stage_execution_summary.json"
    stage_execution_summary_path.write_text(
        f"{json.dumps(stage_execution_summary, indent=2)}\\n",
        encoding="utf-8",
    )

    stage_resource_summary_path = campaign_root / "stage_resource_summary.csv"
    import pandas as pd

    pd.DataFrame(stage_rows).to_csv(stage_resource_summary_path, index=False)

    stage_lease_summary = {
        "schema_version": "stage-lease-summary-v1",
        "generated_at_utc": _utc_timestamp(),
        "n_stage_rows": int(len(stage_rows)),
        "stage_lease_totals": {
            stage_key: {
                "observation_count": int(stage_counts.get(stage_key, 0)),
                "lease_required_count": int(stage_lease_required_counts.get(stage_key, 0)),
                "lease_acquired_count": int(stage_lease_acquired_counts.get(stage_key, 0)),
                "lease_missing_count": int(stage_lease_missing_counts.get(stage_key, 0)),
                "wait_mean_seconds": (
                    float(
                        sum(stage_lease_waits.get(stage_key, []))
                        / len(stage_lease_waits[stage_key])
                    )
                    if stage_lease_waits.get(stage_key)
                    else None
                ),
                "wait_p50_seconds": _percentile(stage_lease_waits.get(stage_key, []), 0.50),
                "wait_p95_seconds": _percentile(stage_lease_waits.get(stage_key, []), 0.95),
            }
            for stage_key in sorted(stage_counts)
        },
    }
    stage_lease_summary_path = campaign_root / "stage_lease_summary.json"
    stage_lease_summary_path.write_text(
        f"{json.dumps(stage_lease_summary, indent=2)}\\n",
        encoding="utf-8",
    )

    queue_rows = [
        {
            "experiment_id": str(row.get("experiment_id") or ""),
            "variant_id": str(row.get("variant_id") or ""),
            "run_id": str(row.get("run_id") or ""),
            "stage_key": str(row.get("stage_key") or ""),
            "lease_class": row.get("lease_class"),
            "lease_required": bool(row.get("lease_required", False)),
            "lease_acquired": bool(row.get("lease_acquired", False)),
            "lease_wait_seconds": _safe_float(row.get("lease_wait_seconds")),
            "lease_queue_depth_at_acquire": row.get("lease_queue_depth_at_acquire"),
            "lease_owner_identity": row.get("lease_owner_identity"),
            "lease_acquired_at_utc": row.get("lease_acquired_at_utc"),
            "lease_released_at_utc": row.get("lease_released_at_utc"),
        }
        for row in stage_rows
        if bool(row.get("lease_required", False))
    ]
    stage_queue_summary_path = campaign_root / "stage_queue_summary.csv"
    pd.DataFrame(queue_rows).to_csv(stage_queue_summary_path, index=False)

    campaign_window_seconds: float | None = None
    aggregate_gpu_held_seconds = float(
        sum(float(item.get("hold_seconds", 0.0)) for item in gpu_lease_intervals)
    )
    if gpu_lease_intervals:
        min_start = min(item["start"] for item in gpu_lease_intervals)
        max_end = max(item["end"] for item in gpu_lease_intervals)
        campaign_window_seconds = max(0.0, float((max_end - min_start).total_seconds()))
    gpu_stage_utilization_summary = {
        "schema_version": "gpu-stage-utilization-summary-v1",
        "generated_at_utc": _utc_timestamp(),
        "gpu_stage_interval_count": int(len(gpu_lease_intervals)),
        "campaign_window_seconds": campaign_window_seconds,
        "aggregate_gpu_held_seconds": float(aggregate_gpu_held_seconds),
        "aggregate_gpu_duty_cycle": (
            float(aggregate_gpu_held_seconds / campaign_window_seconds)
            if campaign_window_seconds is not None and campaign_window_seconds > 0.0
            else None
        ),
        "stage_gpu_hold_seconds": {
            stage_key: float(stage_gpu_hold_totals[stage_key])
            for stage_key in sorted(stage_gpu_hold_totals)
        },
    }
    gpu_stage_utilization_summary_path = campaign_root / "gpu_stage_utilization_summary.json"
    gpu_stage_utilization_summary_path.write_text(
        f"{json.dumps(gpu_stage_utilization_summary, indent=2)}\\n",
        encoding="utf-8",
    )

    backend_fallback_summary = {
        "schema_version": "backend-fallback-summary-v1",
        "generated_at_utc": _utc_timestamp(),
        "rows": [
            {
                "stage_key": str(row.get("stage_key") or ""),
                "run_id": str(row.get("run_id") or ""),
                "experiment_id": str(row.get("experiment_id") or ""),
                "variant_id": str(row.get("variant_id") or ""),
                "planned_backend_family": row.get("planned_backend_family"),
                "observed_backend_family": row.get("observed_backend_family"),
                "planned_compute_lane": row.get("planned_compute_lane"),
                "observed_compute_lane": row.get("observed_compute_lane"),
                "fallback_used": bool(row.get("fallback_used", False)),
                "fallback_reason": row.get("fallback_reason"),
                "backend_match": row.get("backend_match"),
                "lane_match": row.get("lane_match"),
                "executor_match": row.get("executor_match"),
            }
            for row in stage_rows
            if bool(row.get("fallback_used", False))
            or row.get("backend_match") is False
            or row.get("lane_match") is False
            or row.get("executor_match") is False
        ],
    }
    backend_fallback_summary_path = campaign_root / "backend_fallback_summary.json"
    backend_fallback_summary_path.write_text(
        f"{json.dumps(backend_fallback_summary, indent=2)}\\n",
        encoding="utf-8",
    )

    return {
        "stage_execution_summary": str(stage_execution_summary_path.resolve()),
        "stage_resource_summary": str(stage_resource_summary_path.resolve()),
        "backend_fallback_summary": str(backend_fallback_summary_path.resolve()),
        "stage_lease_summary": str(stage_lease_summary_path.resolve()),
        "stage_queue_summary": str(stage_queue_summary_path.resolve()),
        "gpu_stage_utilization_summary": str(gpu_stage_utilization_summary_path.resolve()),
    }


def _resolve_tuning_enabled(params: dict[str, Any]) -> bool:
    methodology = str(params.get("methodology_policy_name") or "").strip().lower()
    if methodology == MethodologyPolicyName.GROUPED_NESTED_TUNING.value:
        return True
    explicit = params.get("tuning_enabled")
    if isinstance(explicit, bool):
        return bool(explicit)
    if explicit is None:
        return False
    return str(explicit).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_planned_projected_runtime_seconds(
    *,
    params: dict[str, Any],
    n_permutations: int,
) -> float | None:
    explicit_projected = _safe_float(params.get("projected_runtime_seconds"))
    if explicit_projected is not None and explicit_projected > 0.0:
        return float(explicit_projected)
    model_name = str(params.get("model") or "").strip().lower()
    if not model_name:
        return None
    try:
        projected = resolve_projected_runtime_seconds(
            model_name=model_name,
            framework_mode=FrameworkMode.EXPLORATORY,
            methodology_policy=str(params.get("methodology_policy_name") or "").strip() or None,
            tuning_enabled=_resolve_tuning_enabled(params),
        )
    except Exception:
        return None
    if int(n_permutations) > 0:
        projected = int(projected) + int(max(0, int(n_permutations)))
    return float(max(1, int(projected)))


def _build_eta_planning_metadata(
    *,
    campaign_id: str,
    phase_name: str,
    experiment_id: str,
    run_id: str,
    params: dict[str, Any],
    effective_n_permutations: int,
) -> dict[str, Any]:
    model_name = str(params.get("model") or "").strip().lower()
    model_cost_tier: str | None = None
    if model_name:
        try:
            model_cost_tier = str(get_model_cost_entry(model_name).cost_tier.value)
        except Exception:
            model_cost_tier = None
    return {
        "campaign_id": str(campaign_id),
        "phase_name": str(phase_name),
        "experiment_id": str(experiment_id),
        "run_id": str(run_id),
        "framework_mode": FrameworkMode.EXPLORATORY.value,
        "model": model_name if model_name else None,
        "model_cost_tier": model_cost_tier,
        "feature_space": (
            str(params.get("feature_space"))
            if params.get("feature_space") not in (None, "")
            else "whole_brain_masked"
        ),
        "preprocessing_strategy": (
            str(params.get("preprocessing_strategy"))
            if params.get("preprocessing_strategy") not in (None, "")
            else "none"
        ),
        "dimensionality_strategy": (
            str(params.get("dimensionality_strategy"))
            if params.get("dimensionality_strategy") not in (None, "")
            else "none"
        ),
        "tuning_enabled": bool(_resolve_tuning_enabled(params)),
        "cv_mode": str(params.get("cv")) if params.get("cv") not in (None, "") else None,
        "n_permutations": int(max(0, int(effective_n_permutations))),
        "subject": str(params.get("subject")) if params.get("subject") else None,
        "train_subject": str(params.get("train_subject")) if params.get("train_subject") else None,
        "test_subject": str(params.get("test_subject")) if params.get("test_subject") else None,
        "task": str(params.get("filter_task")) if params.get("filter_task") else None,
        "modality": str(params.get("filter_modality")) if params.get("filter_modality") else None,
        "projected_runtime_seconds": _resolve_planned_projected_runtime_seconds(
            params=params,
            n_permutations=int(max(0, int(effective_n_permutations))),
        ),
    }


def _extract_actual_runtime_seconds(record: dict[str, Any]) -> float | None:
    stage_timings = record.get("stage_timings_seconds")
    if isinstance(stage_timings, dict):
        total = _safe_float(stage_timings.get("total"))
        if total is not None and total > 0.0:
            return float(total)
    return None


def _git_commit() -> str | None:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = process.stdout.strip()
    return commit or None


_STUDY_GUARDRAIL_POLICY = {
    "exploratory": {
        "core_fields_required": ["question", "generalization_claim", "primary_metric", "cv_scheme"],
        "non_core_gaps": "warnings",
        "disposition_when_core_present": ["allowed", "warning"],
    },
    "confirmatory": {
        "core_fields_required": ["question", "generalization_claim", "primary_metric", "cv_scheme"],
        "strict_requirements": [
            "leakage_risk_reviewed",
            "unit_of_analysis_defined",
            "data_hierarchy_defined",
            "primary_contrast",
            "interpretation_rules",
            "confirmatory_lock_applied",
            "multiplicity_handling",
        ],
        "non_compliance": "blocked",
    },
}

_PHASE_BLUEPRINT_AUTO: list[dict[str, Any]] = [
    {"phase_name": "Preflight", "groups": []},
    {"phase_name": "Stage 1 target/scope lock", "groups": [["E01"], ["E02", "E03"]]},
    {"phase_name": "Stage 2 split/transfer lock", "groups": [["E04"], ["E05"]]},
    {"phase_name": "Stage 3 model lock", "groups": [["E06"], ["E07"], ["E08"]]},
    {
        "phase_name": "Stage 4 representation/preprocessing lock",
        "groups": [["E09"], ["E10"], ["E11"]],
    },
    {"phase_name": "Freeze final confirmatory pipeline", "groups": []},
    {"phase_name": "Confirmatory", "groups": [["E16", "E17"]]},
    {"phase_name": "Primary robustness", "groups": [["E12", "E13"], ["E14", "E15"]]},
    {"phase_name": "Blocking robustness", "groups": [["E20"]]},
    {"phase_name": "Context robustness", "groups": [["E21", "E22", "E23"]]},
    {"phase_name": "Reproducibility audit", "groups": [["E24"]]},
]

_PHASE_ARTIFACTS: dict[str, str] = {
    "Stage 1 target/scope lock": "stage1_lock.json",
    "Stage 2 split/transfer lock": "stage2_lock.json",
    "Stage 3 model lock": "stage3_lock.json",
    "Stage 4 representation/preprocessing lock": "stage4_lock.json",
    "Freeze final confirmatory pipeline": "final_confirmatory_pipeline.json",
}


def _resolve_phase_plan(phase_plan: str) -> str:
    resolved = str(phase_plan or "auto").strip().lower()
    if resolved not in {"auto", "flat"}:
        raise ValueError("phase_plan must be one of: auto, flat")
    return resolved


def _build_phase_batches(
    *,
    selected_experiments: list[dict[str, Any]],
    phase_plan: str,
) -> list[dict[str, Any]]:
    selected_by_id = {str(exp["experiment_id"]): exp for exp in selected_experiments}
    if phase_plan == "flat":
        flat_groups = [[selected_by_id[key]] for key in selected_by_id]
        return [
            {
                "phase_name": "Flat selected sequence",
                "phase_order_index": 0,
                "groups": flat_groups,
                "expected_experiment_ids": sorted(selected_by_id.keys()),
            }
        ]

    phases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(_PHASE_BLUEPRINT_AUTO):
        phase_name = str(entry["phase_name"])
        expected_ids: list[str] = []
        groups: list[list[dict[str, Any]]] = []
        for group_ids in entry["groups"]:
            present_group: list[dict[str, Any]] = []
            for experiment_id in group_ids:
                expected_ids.append(str(experiment_id))
                experiment = selected_by_id.get(str(experiment_id))
                if experiment is None:
                    continue
                present_group.append(experiment)
                seen_ids.add(str(experiment_id))
            if present_group:
                groups.append(present_group)
        phases.append(
            {
                "phase_name": phase_name,
                "phase_order_index": int(index),
                "groups": groups,
                "expected_experiment_ids": expected_ids,
            }
        )

    remaining = [exp for exp in selected_experiments if str(exp["experiment_id"]) not in seen_ids]
    if remaining:
        phases.append(
            {
                "phase_name": "Unmapped selected experiments",
                "phase_order_index": len(phases),
                "groups": [[exp] for exp in remaining],
                "expected_experiment_ids": [str(exp["experiment_id"]) for exp in remaining],
            }
        )
    return phases


def _coerce_experiment_ids(group: list[dict[str, Any]]) -> list[str]:
    return [str(exp["experiment_id"]) for exp in group]


def _phase_status_from_records(records: list[dict[str, Any]]) -> str:
    if not records:
        return "no_runs"
    statuses = {str(record.get("status")) for record in records}
    if statuses.issubset({"dry_run"}):
        return "dry_run"
    if statuses.issubset({"completed"}):
        return "completed"
    if statuses.issubset({"blocked"}):
        return "blocked"
    if "failed" in statuses:
        return "failed"
    if "completed" in statuses:
        return "partial"
    return "mixed"


def _write_phase_artifact(
    *,
    campaign_root: Path,
    filename: str,
    payload: dict[str, Any],
) -> Path:
    output_path = campaign_root / filename
    output_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    return output_path


_PREFLIGHT_EXPERIMENT_IDS = frozenset(
    {"E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11"}
)


def _load_preflight_review_payload(
    *,
    campaign_root: Path,
    experiment_id: str,
) -> dict[str, Any] | None:
    exp_id = str(experiment_id).strip().upper()
    if exp_id not in _PREFLIGHT_EXPERIMENT_IDS:
        return None
    review_path = campaign_root / "preflight_reviews" / f"{exp_id}_review.json"
    if not review_path.exists() or not review_path.is_file():
        return None
    payload = _safe_load_json(review_path)
    return payload if isinstance(payload, dict) else None


def _phase_review_fields(
    *,
    campaign_root: Path,
    experiment_ids: list[str],
) -> dict[str, Any]:
    review_by_experiment: dict[str, dict[str, Any]] = {}
    for experiment_id in sorted({str(value).strip().upper() for value in experiment_ids}):
        review_payload = _load_preflight_review_payload(
            campaign_root=campaign_root,
            experiment_id=experiment_id,
        )
        if isinstance(review_payload, dict):
            review_by_experiment[experiment_id] = review_payload

    if not review_by_experiment:
        return {
            "lock_status": "not_reviewed",
            "candidate_winner": None,
            "candidate_winner_metric": None,
            "consistency_pass": None,
            "min_margin_pass": None,
            "baseline_delta_pass": None,
            "manual_review_required": None,
            "review_artifacts": [],
            "dependency_reruns_required": False,
            "preflight_review_by_experiment": {},
        }

    review_items = list(review_by_experiment.values())
    manual_review_required = any(bool(item.get("manual_review_required")) for item in review_items)
    consistency_pass = all(bool(item.get("consistency_pass")) for item in review_items)
    min_margin_pass = all(bool(item.get("min_margin_pass")) for item in review_items)
    baseline_delta_pass = all(bool(item.get("baseline_delta_pass")) for item in review_items)
    dependency_reruns_required = any(
        bool(item.get("dependency_reruns_required")) for item in review_items
    )
    review_artifacts = []
    for payload in review_items:
        artifacts_payload = payload.get("review_artifacts")
        if isinstance(artifacts_payload, dict):
            review_artifacts.extend(
                [str(value) for value in artifacts_payload.values() if str(value).strip()]
            )

    return {
        "lock_status": ("manual_review_required" if manual_review_required else "auto_lock_passed"),
        "candidate_winner": {
            exp_id: payload.get("candidate_winner")
            for exp_id, payload in sorted(review_by_experiment.items())
        },
        "candidate_winner_metric": {
            exp_id: payload.get("candidate_winner_metric")
            for exp_id, payload in sorted(review_by_experiment.items())
        },
        "consistency_pass": bool(consistency_pass),
        "min_margin_pass": bool(min_margin_pass),
        "baseline_delta_pass": bool(baseline_delta_pass),
        "manual_review_required": bool(manual_review_required),
        "review_artifacts": sorted(set(review_artifacts)),
        "dependency_reruns_required": bool(dependency_reruns_required),
        "preflight_review_by_experiment": {
            exp_id: {
                "lock_status": payload.get("lock_status"),
                "candidate_winner": payload.get("candidate_winner"),
                "candidate_winner_metric": payload.get("candidate_winner_metric"),
                "consistency_pass": payload.get("consistency_pass"),
                "min_margin_pass": payload.get("min_margin_pass"),
                "baseline_delta_pass": payload.get("baseline_delta_pass"),
                "manual_review_required": payload.get("manual_review_required"),
                "review_artifacts": payload.get("review_artifacts"),
                "dependency_reruns_required": payload.get("dependency_reruns_required"),
            }
            for exp_id, payload in sorted(review_by_experiment.items())
        },
    }


def _is_sequential_only_group(
    *,
    group_experiment_ids: list[str],
    cells: list[dict[str, Any]],
) -> bool:
    if "E24" in group_experiment_ids:
        return True
    for cell in cells:
        if bool(cell.get("sequential_only")):
            return True
        design_metadata = cell.get("design_metadata")
        if isinstance(design_metadata, dict) and bool(design_metadata.get("sequential_only")):
            return True
    return False


def run_decision_support_campaign(
    *,
    registry_path: Path,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    output_root: Path,
    experiment_id: str | None,
    stage: str | None,
    run_all: bool,
    seed: int,
    n_permutations: int,
    dry_run: bool,
    subjects_filter: list[str] | None = None,
    tasks_filter: list[str] | None = None,
    modalities_filter: list[str] | None = None,
    max_runs_per_experiment: int | None = None,
    dataset_name: str = "Internal BAS2",
    registry_manifest: CompiledStudyManifest | None = None,
    write_back_to_workbook: bool = False,
    workbook_source_path: Path | None = None,
    workbook_output_dir: Path | None = None,
    append_workbook_run_log: bool = True,
    search_mode: str = "deterministic",
    optuna_trials: int | None = None,
    max_parallel_runs: int = 1,
    max_parallel_gpu_runs: int = 1,
    hardware_mode: str = "cpu_only",
    gpu_device_id: int | None = None,
    deterministic_compute: bool = False,
    allow_backend_fallback: bool = False,
    phase_plan: str = "auto",
    runtime_profile_summary: Path | None = None,
    quiet_progress: bool = False,
    progress_interval_seconds: float = 15.0,
    progress_ui: str = "auto",
    progress_detail: str = "experiment_stage",
    run_experiment_fn: Callable[..., dict[str, Any]] | None = None,
    experiment_ids: list[str] | None = None,
) -> dict[str, Any]:
    if run_experiment_fn is None:
        from Thesis_ML.experiments.run_experiment import run_experiment as _run_experiment

        run_experiment_fn = _run_experiment

    registry = (
        registry_manifest
        if registry_manifest is not None
        else read_registry_manifest(registry_path)
    )
    selected_experiments = _select_experiments(
        registry=registry,
        experiment_id=experiment_id,
        experiment_ids=experiment_ids,
        stage=stage,
        run_all=run_all,
    )
    registry_experiments_payload = [
        experiment.model_dump(mode="python") for experiment in list(registry.experiments)
    ]
    if experiment_id and len(selected_experiments) == 1:
        selected = selected_experiments[0]
        if not bool(selected.get("executable_now", True)):
            reasons = selected.get("blocked_reasons", [])
            reason_text = "; ".join(str(reason) for reason in reasons) if reasons else "unspecified"
            raise RuntimeError(f"Experiment '{experiment_id}' is not executable now: {reason_text}")
    dataset_scope = _collect_dataset_scope(
        index_csv=index_csv,
        subjects_filter=subjects_filter,
        tasks_filter=tasks_filter,
        modalities_filter=modalities_filter,
    )

    campaign_id = _now_timestamp()
    campaign_root = output_root / "campaigns" / campaign_id
    campaign_root.mkdir(parents=True, exist_ok=False)
    history_path = campaign_root.parent / "runtime_history.jsonl"
    anomaly_engine: AnomalyEngine | None = None
    try:
        anomaly_engine = AnomalyEngine(
            campaign_root=campaign_root,
            campaign_id=campaign_id,
        )
    except Exception:
        anomaly_engine = None
    eta_estimator: EtaEstimator | None = None
    try:
        eta_estimator = EtaEstimator(
            campaign_root=campaign_root,
            campaign_id=campaign_id,
            history_path=history_path,
            runtime_profile_summary_path=(
                Path(runtime_profile_summary) if runtime_profile_summary is not None else None
            ),
        )
    except Exception:
        eta_estimator = None
    progress_reporter: Any | None = None
    try:
        progress_reporter = build_progress_reporter(
            interval_seconds=float(progress_interval_seconds),
            quiet=bool(quiet_progress),
            progress_ui=str(progress_ui),
            progress_detail=str(progress_detail),
        )
    except Exception:
        progress_reporter = None
    event_bus: ExecutionEventBus | None = None
    try:
        event_bus = ExecutionEventBus(
            campaign_root=campaign_root,
            campaign_id=campaign_id,
            eta_estimator=eta_estimator,
            anomaly_engine=anomaly_engine,
            console_reporter=progress_reporter,
        )
    except Exception:
        event_bus = None

    def _emit_campaign_event(**kwargs: Any) -> None:
        if event_bus is None:
            return
        try:
            event_bus.emit_event(**kwargs)
        except Exception:
            return

    _emit_campaign_event(
        event_name="campaign_started",
        scope="campaign",
        status="running",
        stage="campaign",
        message="campaign started",
        metadata={
            "selected_experiments": [str(exp["experiment_id"]) for exp in selected_experiments],
            "experiments_total": int(len(selected_experiments)),
            "dry_run": bool(dry_run),
        },
    )

    study_review_summary_path = campaign_root / "study_review_summary.json"
    study_reviews_payload = [review.model_dump(mode="python") for review in registry.study_reviews]
    study_review_summary = {
        "generated_at": _utc_timestamp(),
        "guardrail_policy": _STUDY_GUARDRAIL_POLICY,
        "studies": study_reviews_payload,
    }
    study_review_summary_path.write_text(
        f"{json.dumps(study_review_summary, indent=2)}\n",
        encoding="utf-8",
    )

    commit = _git_commit()
    artifact_registry_path = output_root / "artifact_registry.sqlite3"
    search_space_map = build_search_space_map(list(registry.search_spaces))
    search_mode_value = str(search_mode).strip().lower()
    if search_mode_value not in {"deterministic", "optuna"}:
        raise ValueError("search_mode must be one of: deterministic, optuna")
    if int(max_parallel_runs) <= 0:
        raise ValueError("max_parallel_runs must be >= 1.")
    if int(max_parallel_gpu_runs) < 0:
        raise ValueError("max_parallel_gpu_runs must be >= 0.")
    if int(max_parallel_gpu_runs) > int(max_parallel_runs):
        raise ValueError("max_parallel_gpu_runs cannot exceed max_parallel_runs.")
    if float(progress_interval_seconds) <= 0.0:
        raise ValueError("progress_interval_seconds must be > 0.")
    phase_plan_value = _resolve_phase_plan(phase_plan)
    optuna_enabled = search_mode_value == "optuna"
    base_compute_policy = resolve_compute_policy(
        framework_mode=FrameworkMode.EXPLORATORY,
        hardware_mode=hardware_mode,
        gpu_device_id=gpu_device_id,
        deterministic_compute=bool(deterministic_compute),
        allow_backend_fallback=bool(allow_backend_fallback),
    )
    all_variant_records: list[dict[str, Any]] = []
    blocked_experiments: list[dict[str, Any]] = []
    phase_skip_rows: list[dict[str, Any]] = []
    phase_artifact_paths: list[str] = []
    experiment_roots: dict[str, str] = {}
    experiment_records: dict[str, list[dict[str, Any]]] = {}
    experiment_warnings: dict[str, list[str]] = {}

    for experiment in selected_experiments:
        exp_id = str(experiment["experiment_id"])
        experiment_root = output_root / exp_id / campaign_id
        experiment_root.mkdir(parents=True, exist_ok=False)
        experiment_roots[exp_id] = str(experiment_root.resolve())
        experiment_records[exp_id] = []
        experiment_warnings[exp_id] = []

    phase_batches = _build_phase_batches(
        selected_experiments=selected_experiments,
        phase_plan=phase_plan_value,
    )

    global_order_index = 0
    selected_by_id = {str(experiment["experiment_id"]) for experiment in selected_experiments}
    experiment_phase_by_id: dict[str, str] = {}
    experiment_started_ids: set[str] = set()
    experiment_planned_terminal_counts: dict[str, int] = {
        str(experiment["experiment_id"]): 0 for experiment in selected_experiments
    }
    experiment_terminal_counts: dict[str, int] = {
        str(experiment["experiment_id"]): 0 for experiment in selected_experiments
    }
    experiment_finished_emitted_ids: set[str] = set()

    def _maybe_emit_experiment_finished_event(exp_id: str) -> None:
        exp_id_key = str(exp_id)
        if exp_id_key in experiment_finished_emitted_ids:
            return
        planned_count = int(experiment_planned_terminal_counts.get(exp_id_key, 0))
        terminal_count = int(experiment_terminal_counts.get(exp_id_key, 0))
        if planned_count <= 0 or terminal_count < planned_count:
            return
        variant_records = list(experiment_records.get(exp_id_key, []))
        warnings = list(experiment_warnings.get(exp_id_key, []))
        experiment_status = _phase_status_from_records(variant_records)
        if not variant_records and warnings:
            experiment_status = "skipped"
        _emit_campaign_event(
            event_name="experiment_finished",
            scope="experiment",
            status=experiment_status,
            stage="campaign",
            phase_name=experiment_phase_by_id.get(exp_id_key),
            experiment_id=exp_id_key,
            message="experiment finished",
            metadata={"warnings": list(warnings)},
        )
        experiment_finished_emitted_ids.add(exp_id_key)

    for phase in phase_batches:
        phase_name = str(phase["phase_name"])
        # compute expected ids and groups first; skip emitting phase events when
        # the phase has no groups to process (avoids spurious preflight phase events)
        expected_ids = [str(value) for value in phase.get("expected_experiment_ids", [])]
        groups = phase.get("groups", [])
        # If the phase has no groups, do not emit phase_started/phase_finished events
        # (keeps ordering of emitted experiment events stable), but still write the
        # phase artifact file so downstream consumers and tests find the expected
        # phase artifact payloads.
        if not groups:
            artifact_filename = _PHASE_ARTIFACTS.get(phase_name)
            if artifact_filename:
                phase_payload = {
                    "campaign_id": campaign_id,
                    "generated_at": _utc_timestamp(),
                    "phase_name": phase_name,
                    "experiment_ids": list(expected_ids),
                    "status": "no_runs",
                    "selected_or_completed_cells": [],
                    "skipped_experiments": [
                        row for row in phase_skip_rows if str(row.get("phase_name")) == phase_name
                    ],
                    "decision_note": None,
                }
                path = _write_phase_artifact(
                    campaign_root=campaign_root, filename=artifact_filename, payload=phase_payload
                )
                phase_artifact_paths.append(str(path.resolve()))
            continue
        phase_records: list[dict[str, Any]] = []
        phase_experiment_ids: list[str] = []
        _emit_campaign_event(
            event_name="phase_started",
            scope="phase",
            status="running",
            stage="campaign",
            phase_name=phase_name,
            message="phase started",
            metadata={
                "dry_run": bool(dry_run),
                "expected_experiment_ids": list(expected_ids),
            },
        )

        expected_ids = [str(value) for value in phase.get("expected_experiment_ids", [])]
        missing_expected = [value for value in expected_ids if value not in selected_by_id]
        for missing_id in missing_expected:
            if missing_id in {"E09", "E10", "E11"}:
                phase_skip_rows.append(
                    {
                        "phase_name": phase_name,
                        "experiment_id": missing_id,
                        "reason": "experiment not present in executable registry selection",
                    }
                )

        for group in phase.get("groups", []):
            group_ids = _coerce_experiment_ids(group)
            phase_experiment_ids.extend(group_ids)
            group_cells: list[tuple[dict[str, Any], dict[str, Any]]] = []

            for experiment in group:
                exp_id = str(experiment["experiment_id"])
                experiment_phase_by_id.setdefault(exp_id, phase_name)
                if exp_id == "E22" and len(list(dataset_scope.get("modalities", []))) <= 1:
                    phase_skip_rows.append(
                        {
                            "phase_name": phase_name,
                            "experiment_id": exp_id,
                            "reason": "not applicable under single-modality dataset scope",
                        }
                    )
                    continue

                variants, warnings = _expand_experiment_variants(
                    experiment=experiment,
                    dataset_scope=dataset_scope,
                    search_space_map=search_space_map,
                    search_seed=seed,
                    optuna_enabled=optuna_enabled,
                    optuna_trials=optuna_trials,
                    max_runs_per_experiment=max_runs_per_experiment,
                )
                cells, materialization_warnings = _materialize_experiment_cells(
                    experiment=experiment,
                    variants=variants,
                    dataset_scope=dataset_scope,
                    n_permutations=n_permutations,
                    seed=seed,
                    registry_experiments=registry_experiments_payload,
                )
                combined_warnings = list(warnings) + list(materialization_warnings)
                if combined_warnings:
                    experiment_warnings[exp_id].extend(combined_warnings)
                if not cells:
                    reason = (
                        "; ".join(combined_warnings)
                        if combined_warnings
                        else "no runnable cells were materialized"
                    )
                    phase_skip_rows.append(
                        {
                            "phase_name": phase_name,
                            "experiment_id": exp_id,
                            "reason": reason,
                        }
                    )
                    _emit_campaign_event(
                        event_name="experiment_skipped",
                        scope="experiment",
                        status="skipped",
                        stage="campaign",
                        phase_name=phase_name,
                        experiment_id=exp_id,
                        message="experiment selected but produced no materialized cells",
                        metadata={
                            "reason": reason,
                            "dry_run": bool(dry_run),
                        },
                    )
                    continue
                if exp_id not in experiment_started_ids:
                    _emit_campaign_event(
                        event_name="experiment_started",
                        scope="experiment",
                        status="running",
                        stage="campaign",
                        phase_name=phase_name,
                        experiment_id=exp_id,
                        message="experiment started",
                    )
                    experiment_started_ids.add(exp_id)
                for cell in cells:
                    group_cells.append((experiment, cell))

            if not group_cells:
                continue

            for experiment, cell in group_cells:
                exp_id = str(experiment["experiment_id"])
                variant_id = _resolve_variant_id(cell)
                run_id = _resolve_variant_run_id(
                    experiment_id=exp_id,
                    variant=cell,
                    campaign_id=campaign_id,
                )
                params_for_eta = (
                    dict(cell.get("params", {})) if isinstance(cell.get("params"), dict) else {}
                )
                resolved_permutation_override = _optional_int(cell.get("n_permutations_override"))
                effective_n_permutations = (
                    int(resolved_permutation_override)
                    if resolved_permutation_override is not None
                    else int(n_permutations)
                )
                eta_planning_metadata = _build_eta_planning_metadata(
                    campaign_id=campaign_id,
                    phase_name=phase_name,
                    experiment_id=exp_id,
                    run_id=run_id,
                    params=params_for_eta,
                    effective_n_permutations=int(effective_n_permutations),
                )
                _emit_campaign_event(
                    event_name="run_planned",
                    scope="run",
                    status="planned",
                    stage="campaign",
                    phase_name=phase_name,
                    experiment_id=exp_id,
                    variant_id=variant_id,
                    run_id=run_id,
                    message="run planned",
                    metadata={
                        "dry_run": bool(dry_run),
                        "supported": bool(cell.get("supported", False)),
                        "blocked_reason": cell.get("blocked_reason"),
                        **eta_planning_metadata,
                    },
                )
                experiment_planned_terminal_counts[exp_id] = (
                    int(experiment_planned_terminal_counts.get(exp_id, 0)) + 1
                )

            runnable_cells = [
                (experiment, cell)
                for experiment, cell in group_cells
                if bool(cell.get("supported", False))
            ]

            assignments_by_run_id: dict[str, dict[str, Any]] = {}
            job_results_by_run_id: dict[str, dict[str, Any]] = {}
            job_builder_blocked: dict[str, str] = {}
            cells_for_execution_by_run_id: dict[str, dict[str, Any]] = {}
            sequential_only_group = _is_sequential_only_group(
                group_experiment_ids=group_ids,
                cells=[cell for _, cell in group_cells],
            )

            if not dry_run and runnable_cells:
                run_requests: list[ComputeRunRequest] = []
                request_cells: list[tuple[dict[str, Any], dict[str, Any], str, int]] = []
                run_id_by_group_index: dict[int, str] = {}
                for group_index, (experiment, cell) in enumerate(group_cells):
                    if not bool(cell.get("supported", False)):
                        continue
                    exp_id = str(experiment["experiment_id"])
                    run_id = _resolve_variant_run_id(
                        experiment_id=exp_id,
                        variant=cell,
                        campaign_id=campaign_id,
                    )
                    run_requests.append(
                        ComputeRunRequest(
                            order_index=int(global_order_index),
                            run_id=str(run_id),
                            model_name=str(cell.get("params", {}).get("model", "")),
                        )
                    )
                    run_id_by_group_index[int(group_index)] = str(run_id)
                    request_cells.append((experiment, cell, str(run_id), int(group_index)))
                    cells_for_execution_by_run_id[str(run_id)] = dict(cell)
                    global_order_index += 1

                schedule = plan_compute_schedule(
                    run_requests=run_requests,
                    base_compute_policy=base_compute_policy,
                    max_parallel_runs=(1 if sequential_only_group else int(max_parallel_runs)),
                    max_parallel_gpu_runs=(
                        0 if sequential_only_group else int(max_parallel_gpu_runs)
                    ),
                )
                assignments_by_run_id = {
                    str(assignment.run_id): assignment.to_payload() for assignment in schedule
                }

                request_cells_by_run_id = {
                    str(run_id): (experiment, cell, group_index)
                    for experiment, cell, run_id, group_index in request_cells
                }
                reuse_dependency_by_group_index = _plan_sibling_feature_matrix_reuse(
                    variants=[cell for _, cell in group_cells],
                    cache_dir=cache_dir,
                    parent_experiment_ids=[
                        str(experiment["experiment_id"]) for experiment, _ in group_cells
                    ],
                )
                reuse_dependency_by_run_id: dict[str, str] = {}
                for dependent_index, anchor_index in reuse_dependency_by_group_index.items():
                    dependent_group_index = int(dependent_index)
                    anchor_group_index = int(anchor_index)
                    if (
                        dependent_group_index < 0
                        or anchor_group_index < 0
                        or dependent_group_index >= len(group_cells)
                        or anchor_group_index >= len(group_cells)
                    ):
                        continue
                    dependent_experiment_id = str(
                        group_cells[dependent_group_index][0]["experiment_id"]
                    )
                    anchor_experiment_id = str(group_cells[anchor_group_index][0]["experiment_id"])
                    if dependent_experiment_id != anchor_experiment_id:
                        continue
                    dependent_run_id = run_id_by_group_index.get(dependent_group_index)
                    anchor_run_id = run_id_by_group_index.get(anchor_group_index)
                    if dependent_run_id is None or anchor_run_id is None:
                        continue
                    if dependent_run_id == anchor_run_id:
                        continue
                    reuse_dependency_by_run_id[str(dependent_run_id)] = str(anchor_run_id)

                def _dispatch_requested_runs(
                    *,
                    requested_run_ids: set[str],
                    variant_overrides: dict[str, dict[str, Any]] | None = None,
                    request_cells_bound: list[
                        tuple[dict[str, Any], dict[str, Any], str, int]
                    ] = request_cells,
                    cells_for_execution_by_run_id_bound: dict[
                        str, dict[str, Any]
                    ] = cells_for_execution_by_run_id,
                    assignments_by_run_id_bound: dict[str, dict[str, Any]] = assignments_by_run_id,
                    phase_name_bound: str = phase_name,
                    job_builder_blocked_bound: dict[str, str] = job_builder_blocked,
                    sequential_only_group_bound: bool = sequential_only_group,
                    job_results_by_run_id_bound: dict[str, dict[str, Any]] = job_results_by_run_id,
                ) -> None:
                    if not requested_run_ids:
                        return

                    jobs: list[Any] = []
                    for order_index, (experiment, cell, run_id, _) in enumerate(
                        request_cells_bound
                    ):
                        if run_id not in requested_run_ids:
                            continue
                        variant_for_job = (
                            dict(variant_overrides[run_id])
                            if isinstance(variant_overrides, dict) and run_id in variant_overrides
                            else dict(cell)
                        )
                        cells_for_execution_by_run_id_bound[run_id] = dict(variant_for_job)
                        assignment_payload = assignments_by_run_id_bound.get(run_id)
                        assigned_order_index_override = (
                            _optional_int(assignment_payload.get("order_index"))
                            if isinstance(assignment_payload, dict)
                            else None
                        )
                        assigned_order_index = (
                            int(assigned_order_index_override)
                            if assigned_order_index_override is not None
                            else int(order_index)
                        )
                        assignment = (
                            None
                            if assignment_payload is None
                            else ComputeRunAssignment.from_payload(
                                dict(assignment_payload),
                                default_order_index=int(assigned_order_index),
                                default_run_id=str(run_id),
                                default_model_name=str(
                                    variant_for_job.get("params", {}).get("model", "")
                                ),
                            )
                        )
                        job, blocked_reason, _ = _build_variant_official_job(
                            experiment=experiment,
                            variant=variant_for_job,
                            campaign_id=campaign_id,
                            experiment_root=output_root
                            / str(experiment["experiment_id"])
                            / campaign_id,
                            index_csv=index_csv,
                            data_root=data_root,
                            cache_dir=cache_dir,
                            seed=seed,
                            n_permutations=n_permutations,
                            phase_name=phase_name_bound,
                            order_index=int(assigned_order_index),
                            hardware_mode=hardware_mode,
                            gpu_device_id=gpu_device_id,
                            deterministic_compute=bool(deterministic_compute),
                            allow_backend_fallback=bool(allow_backend_fallback),
                            scheduled_compute_assignment=assignment,
                            worker_execution_mode="native_worker",
                        )
                        if job is None:
                            job_builder_blocked_bound[run_id] = str(
                                blocked_reason or "job_build_failed"
                            )
                            continue
                        jobs.append(job)
                        _emit_campaign_event(
                            event_name="run_dispatched",
                            scope="run",
                            status="dispatched",
                            stage="campaign",
                            phase_name=phase_name_bound,
                            experiment_id=str(experiment["experiment_id"]),
                            variant_id=_resolve_variant_id(variant_for_job),
                            run_id=str(run_id),
                            message="run dispatched",
                        )
                        _emit_campaign_event(
                            event_name="run_started",
                            scope="run",
                            status="running",
                            stage="campaign",
                            phase_name=phase_name_bound,
                            experiment_id=str(experiment["experiment_id"]),
                            variant_id=_resolve_variant_id(variant_for_job),
                            run_id=str(run_id),
                            message="run started",
                        )

                    if not jobs:
                        return
                    effective_parallelism = (
                        1 if sequential_only_group_bound else int(max_parallel_runs)
                    )
                    effective_gpu_parallelism = (
                        0 if sequential_only_group_bound else int(max_parallel_gpu_runs)
                    )
                    job_payloads = _execute_official_jobs(
                        jobs=jobs,
                        max_parallel_runs=effective_parallelism,
                        max_parallel_gpu_runs=effective_gpu_parallelism,
                        run_experiment_fn=run_experiment_fn,
                    )
                    for payload in job_payloads:
                        if "run_id" not in payload:
                            continue
                        job_results_by_run_id_bound[str(payload["run_id"])] = payload

                dependent_run_ids = set(reuse_dependency_by_run_id.keys())
                first_wave_run_ids = {
                    str(run_id)
                    for _, _, run_id, _ in request_cells
                    if str(run_id) not in dependent_run_ids
                }
                _dispatch_requested_runs(requested_run_ids=first_wave_run_ids)

                if dependent_run_ids:
                    second_wave_overrides: dict[str, dict[str, Any]] = {}
                    for dependent_run_id, anchor_run_id in reuse_dependency_by_run_id.items():
                        anchor_result = job_results_by_run_id.get(str(anchor_run_id))
                        base_feature_matrix_id = _extract_feature_matrix_artifact_id(anchor_result)
                        source_registry_path = _extract_artifact_registry_path(anchor_result)
                        if base_feature_matrix_id is None:
                            continue
                        request_row = request_cells_by_run_id.get(str(dependent_run_id))
                        if request_row is None:
                            continue
                        _, original_variant, _ = request_row
                        second_wave_overrides[str(dependent_run_id)] = (
                            _apply_feature_matrix_reuse_variant(
                                variant=original_variant,
                                base_artifact_id=str(base_feature_matrix_id),
                                source_run_id=str(anchor_run_id),
                                source_registry_path=source_registry_path,
                            )
                        )
                    _dispatch_requested_runs(
                        requested_run_ids=dependent_run_ids,
                        variant_overrides=second_wave_overrides,
                    )

            for experiment, cell in group_cells:
                exp_id = str(experiment["experiment_id"])
                variant_id = _resolve_variant_id(cell)
                run_id = _resolve_variant_run_id(
                    experiment_id=exp_id,
                    variant=cell,
                    campaign_id=campaign_id,
                )

                planned_cell = cells_for_execution_by_run_id.get(run_id, cell)
                cell_for_record = dict(planned_cell)
                if run_id in job_builder_blocked:
                    cell_for_record["supported"] = False
                    cell_for_record["blocked_reason"] = str(job_builder_blocked[run_id])
                job_execution_override = job_results_by_run_id.get(run_id)
                if (
                    not dry_run
                    and bool(cell_for_record.get("supported", False))
                    and run_id not in job_builder_blocked
                    and job_execution_override is None
                ):
                    job_execution_override = {
                        "watchdog_result": None,
                        "execution_error": {
                            "error": "official_job_result_missing_for_scheduled_run"
                        },
                    }

                record = _execute_variant(
                    experiment=experiment,
                    variant=cell_for_record,
                    campaign_id=campaign_id,
                    experiment_root=output_root / exp_id / campaign_id,
                    index_csv=index_csv,
                    data_root=data_root,
                    cache_dir=cache_dir,
                    seed=seed,
                    n_permutations=n_permutations,
                    dry_run=dry_run,
                    run_experiment_fn=run_experiment_fn,
                    hardware_mode=hardware_mode,
                    gpu_device_id=gpu_device_id,
                    deterministic_compute=bool(deterministic_compute),
                    allow_backend_fallback=bool(allow_backend_fallback),
                    max_parallel_runs=int(max_parallel_runs),
                    max_parallel_gpu_runs=int(max_parallel_gpu_runs),
                    scheduled_compute_assignment=assignments_by_run_id.get(run_id),
                    job_execution_result=job_execution_override,
                    progress_callback=(
                        event_bus.build_progress_callback(
                            phase_name=phase_name,
                            experiment_id=exp_id,
                            variant_id=variant_id,
                            run_id=run_id,
                        )
                        if event_bus is not None
                        else None
                    ),
                    artifact_registry_path=artifact_registry_path,
                    code_ref=commit,
                )
                experiment_records[exp_id].append(record)
                phase_records.append(record)
                all_variant_records.append(record)
                record_status = str(record.get("status"))
                terminal_params = (
                    dict(cell_for_record.get("params", {}))
                    if isinstance(cell_for_record.get("params"), dict)
                    else {}
                )
                terminal_override = _optional_int(cell_for_record.get("n_permutations_override"))
                terminal_n_permutations = (
                    int(terminal_override) if terminal_override is not None else int(n_permutations)
                )
                eta_terminal_metadata = _build_eta_planning_metadata(
                    campaign_id=campaign_id,
                    phase_name=phase_name,
                    experiment_id=exp_id,
                    run_id=run_id,
                    params=terminal_params,
                    effective_n_permutations=int(terminal_n_permutations),
                )
                eta_terminal_metadata["dry_run"] = bool(dry_run)
                eta_terminal_metadata["actual_runtime_seconds"] = _extract_actual_runtime_seconds(
                    record
                )
                if record.get("framework_mode") not in (None, ""):
                    eta_terminal_metadata["framework_mode"] = str(record.get("framework_mode"))
                if record.get("model_cost_tier") not in (None, ""):
                    eta_terminal_metadata["model_cost_tier"] = str(record.get("model_cost_tier"))
                projected_runtime = _safe_float(record.get("projected_runtime_seconds"))
                if projected_runtime is not None and projected_runtime > 0.0:
                    eta_terminal_metadata["projected_runtime_seconds"] = float(projected_runtime)
                if record.get("cv") not in (None, ""):
                    eta_terminal_metadata["cv_mode"] = str(record.get("cv"))
                if record.get("feature_space") not in (None, ""):
                    eta_terminal_metadata["feature_space"] = str(record.get("feature_space"))
                if record.get("preprocessing_strategy") not in (None, ""):
                    eta_terminal_metadata["preprocessing_strategy"] = str(
                        record.get("preprocessing_strategy")
                    )
                if record.get("dimensionality_strategy") not in (None, ""):
                    eta_terminal_metadata["dimensionality_strategy"] = str(
                        record.get("dimensionality_strategy")
                    )
                if record.get("tuning_enabled") is not None:
                    eta_terminal_metadata["tuning_enabled"] = bool(record.get("tuning_enabled"))
                anomaly_terminal_metadata = dict(eta_terminal_metadata)
                anomaly_terminal_metadata.update(
                    {
                        "status": str(record_status),
                        "roi_spec_path": record.get("roi_spec_path"),
                        "stage_timings_seconds": record.get("stage_timings_seconds"),
                        "process_profile_summary": record.get("process_profile_summary"),
                    }
                )
                if anomaly_engine is not None:
                    try:
                        anomaly_engine.inspect_terminal_run(anomaly_terminal_metadata)
                    except Exception:
                        pass
                if record_status == "completed":
                    _emit_campaign_event(
                        event_name="run_finished",
                        scope="run",
                        status="completed",
                        stage="campaign",
                        phase_name=phase_name,
                        experiment_id=exp_id,
                        variant_id=variant_id,
                        run_id=run_id,
                        message="run finished",
                        metadata=eta_terminal_metadata,
                    )
                elif record_status == "failed":
                    _emit_campaign_event(
                        event_name="run_failed",
                        scope="run",
                        status="failed",
                        stage="campaign",
                        phase_name=phase_name,
                        experiment_id=exp_id,
                        variant_id=variant_id,
                        run_id=run_id,
                        message="run failed",
                        metadata={
                            "error": record.get("error"),
                            **eta_terminal_metadata,
                        },
                    )
                elif record_status == "blocked":
                    _emit_campaign_event(
                        event_name="run_blocked",
                        scope="run",
                        status="blocked",
                        stage="campaign",
                        phase_name=phase_name,
                        experiment_id=exp_id,
                        variant_id=variant_id,
                        run_id=run_id,
                        message="run blocked",
                        metadata={
                            "dry_run": bool(dry_run),
                            "blocked_reason": record.get("blocked_reason"),
                            **eta_terminal_metadata,
                        },
                    )
                elif record_status == "dry_run":
                    _emit_campaign_event(
                        event_name="run_dry_run",
                        scope="run",
                        status="dry_run",
                        stage="campaign",
                        phase_name=phase_name,
                        experiment_id=exp_id,
                        variant_id=variant_id,
                        run_id=run_id,
                        message="run dry-run",
                        metadata=eta_terminal_metadata,
                    )
                if record_status in {"completed", "failed", "blocked", "dry_run"}:
                    experiment_terminal_counts[exp_id] = (
                        int(experiment_terminal_counts.get(exp_id, 0)) + 1
                    )
                    _maybe_emit_experiment_finished_event(exp_id)

        phase_payload = {
            "campaign_id": campaign_id,
            "generated_at": _utc_timestamp(),
            "phase_name": phase_name,
            "experiment_ids": sorted(set(phase_experiment_ids)),
            "status": _phase_status_from_records(phase_records),
            "selected_or_completed_cells": [
                {
                    "experiment_id": str(record.get("experiment_id")),
                    "variant_id": str(record.get("variant_id")),
                    "status": str(record.get("status")),
                }
                for record in phase_records
            ],
            "skipped_experiments": [
                row for row in phase_skip_rows if str(row.get("phase_name")) == phase_name
            ],
            "decision_note": None,
        }
        phase_payload.update(
            _phase_review_fields(
                campaign_root=campaign_root,
                experiment_ids=list(sorted(set(phase_experiment_ids))),
            )
        )
        artifact_filename = _PHASE_ARTIFACTS.get(phase_name)
        if artifact_filename:
            path = _write_phase_artifact(
                campaign_root=campaign_root,
                filename=artifact_filename,
                payload=phase_payload,
            )
            phase_artifact_paths.append(str(path.resolve()))
        selected_or_completed_cells_payload = phase_payload.get("selected_or_completed_cells")
        selected_or_completed_cells_metadata = (
            list(selected_or_completed_cells_payload)
            if isinstance(selected_or_completed_cells_payload, list)
            else []
        )
        _emit_campaign_event(
            event_name="phase_finished",
            scope="phase",
            status=str(phase_payload["status"]),
            stage="campaign",
            phase_name=phase_name,
            message="phase finished",
            metadata={
                "dry_run": bool(dry_run),
                "experiment_ids": list(sorted(set(phase_experiment_ids))),
                "selected_or_completed_cells": selected_or_completed_cells_metadata,
            },
        )

    phase_skip_summary_payload = {
        "campaign_id": campaign_id,
        "generated_at": _utc_timestamp(),
        "phase_name": "phase_skip_summary",
        "status": "created",
        "skipped_experiments": phase_skip_rows,
    }
    phase_skip_summary_path = _write_phase_artifact(
        campaign_root=campaign_root,
        filename="phase_skip_summary.json",
        payload=phase_skip_summary_payload,
    )
    phase_artifact_paths.append(str(phase_skip_summary_path.resolve()))

    for experiment in selected_experiments:
        exp_id = str(experiment["experiment_id"])
        variant_records = list(experiment_records.get(exp_id, []))
        warnings = list(experiment_warnings.get(exp_id, []))
        experiment_status = _phase_status_from_records(variant_records)
        if not variant_records and warnings:
            experiment_status = "skipped"
        _write_experiment_outputs(
            experiment=experiment,
            experiment_root=output_root / exp_id / campaign_id,
            variant_records=variant_records,
            warnings=warnings,
        )
        if exp_id not in experiment_finished_emitted_ids:
            _emit_campaign_event(
                event_name="experiment_finished",
                scope="experiment",
                status=experiment_status,
                stage="campaign",
                phase_name=experiment_phase_by_id.get(exp_id),
                experiment_id=exp_id,
                message="experiment finished",
                metadata={"warnings": list(warnings)},
            )
            experiment_finished_emitted_ids.add(exp_id)
        if not variant_records and warnings:
            blocked_experiments.append(
                {
                    "experiment_id": exp_id,
                    "reasons": sorted({str(item) for item in warnings if str(item)}),
                }
            )
        if variant_records and all(row["status"] == "blocked" for row in variant_records):
            blocked_reasons = sorted(
                {
                    str(row.get("blocked_reason"))
                    for row in variant_records
                    if row.get("blocked_reason")
                }
            )
            blocked_experiments.append(
                {
                    "experiment_id": exp_id,
                    "reasons": blocked_reasons,
                }
            )

    if experiment_id:
        selected_records = [
            row
            for row in all_variant_records
            if str(row.get("experiment_id")) == str(experiment_id)
        ]
        if selected_records and all(
            str(row.get("status")) == "blocked" for row in selected_records
        ):
            if not all(_is_e14_derived_reporting_only_block(row) for row in selected_records):
                reasons = sorted(
                    {
                        str(row.get("blocked_reason"))
                        for row in selected_records
                        if row.get("blocked_reason")
                    }
                )
                reason_text = "; ".join(reasons) if reasons else "unspecified"
                raise RuntimeError(
                    f"Experiment '{experiment_id}' is not executable now: {reason_text}"
                )

    stage_summary_paths = _write_stage_summaries(
        campaign_root=campaign_root,
        variant_records=all_variant_records,
    )
    stage_evidence_summary_paths = _write_stage_evidence_summaries(
        campaign_root=campaign_root,
        variant_records=all_variant_records,
    )
    reporting_variant_records, permutation_chunk_merge_summary = _build_reporting_variant_records(
        campaign_root=campaign_root,
        variant_records=all_variant_records,
    )
    permutation_chunk_merge_summary_path: str | None = None
    e12_table_ready_summary_csv_path: str | None = None
    e12_table_ready_summary_json_path: str | None = None
    e13_table_ready_summary_csv_path: str | None = None
    e13_table_ready_summary_json_path: str | None = None
    e14_stability_summary_csv_path: str | None = None
    e14_stability_summary_json_path: str | None = None
    confirmatory_anchor_control_coverage_csv_path: str | None = None
    confirmatory_anchor_control_coverage_json_path: str | None = None
    thesis_confirmatory_artifact_paths: dict[str, str | None] = {}
    e12_table_rows: list[dict[str, Any]] = []
    e13_table_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    confirmatory_model_rows = _build_confirmatory_model_rows(
        reporting_variant_records=reporting_variant_records
    )
    runtime_anchor_rows: list[dict[str, Any]] = []
    try:
        runtime_registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
        if isinstance(runtime_registry_payload, dict):
            runtime_anchor_rows = collect_runtime_confirmatory_anchors(runtime_registry_payload)
    except Exception:
        runtime_anchor_rows = []
    external_e16_anchor_records = _collect_external_e16_anchor_records(
        output_root=output_root,
        runtime_anchor_rows=runtime_anchor_rows,
    )
    if isinstance(permutation_chunk_merge_summary, dict):
        merge_groups = permutation_chunk_merge_summary.get("groups")
        merge_errors = permutation_chunk_merge_summary.get("errors")
        has_merge_groups = isinstance(merge_groups, list) and bool(merge_groups)
        has_merge_errors = isinstance(merge_errors, list) and bool(merge_errors)
        if has_merge_groups or has_merge_errors:
            merge_summary_path = campaign_root / "e12_permutation_chunk_merge_summary.json"
            merge_payload = dict(permutation_chunk_merge_summary)
            merge_payload["generated_at_utc"] = _utc_timestamp()
            merge_summary_path.write_text(
                f"{json.dumps(merge_payload, indent=2)}\n",
                encoding="utf-8",
            )
            permutation_chunk_merge_summary_path = str(merge_summary_path.resolve())
        e12_table_rows = _build_e12_table_ready_rows(
            reporting_variant_records=reporting_variant_records
        )
        if e12_table_rows:
            special_aggregation_root = campaign_root / "special_aggregations" / "E12"
            special_aggregation_root.mkdir(parents=True, exist_ok=True)
            e12_csv_path = special_aggregation_root / "e12_permutation_analysis_summary.csv"
            e12_json_path = special_aggregation_root / "e12_permutation_analysis_summary.json"
            import pandas as pd

            pd.DataFrame(e12_table_rows).to_csv(e12_csv_path, index=False)
            e12_json_path.write_text(
                f"{json.dumps(e12_table_rows, indent=2)}\n",
                encoding="utf-8",
            )
            e12_table_ready_summary_csv_path = str(e12_csv_path.resolve())
            e12_table_ready_summary_json_path = str(e12_json_path.resolve())

    e13_table_rows = _build_e13_table_ready_rows(
        reporting_variant_records=reporting_variant_records
    )
    if e13_table_rows:
        special_aggregation_root = campaign_root / "special_aggregations" / "E13"
        special_aggregation_root.mkdir(parents=True, exist_ok=True)
        e13_csv_path = special_aggregation_root / "e13_dummy_baseline_analysis_summary.csv"
        e13_json_path = special_aggregation_root / "e13_dummy_baseline_analysis_summary.json"
        import pandas as pd

        pd.DataFrame(e13_table_rows).to_csv(e13_csv_path, index=False)
        e13_json_path.write_text(
            f"{json.dumps(e13_table_rows, indent=2)}\n",
            encoding="utf-8",
        )
        e13_table_ready_summary_csv_path = str(e13_csv_path.resolve())
        e13_table_ready_summary_json_path = str(e13_json_path.resolve())

    reporting_variant_records, e14_summary_rows, e14_summary_payload = _build_e14_reporting_records(
        reporting_variant_records=reporting_variant_records,
        runtime_anchor_rows=runtime_anchor_rows,
        runtime_anchor_records=external_e16_anchor_records,
    )
    if e14_summary_rows:
        special_aggregation_root = campaign_root / "special_aggregations" / "E14"
        special_aggregation_root.mkdir(parents=True, exist_ok=True)
        e14_csv_path = special_aggregation_root / "e14_explanation_stability_summary.csv"
        e14_json_path = special_aggregation_root / "e14_explanation_stability_summary.json"
        import pandas as pd

        pd.DataFrame(e14_summary_rows).to_csv(e14_csv_path, index=False)
        e14_json_payload = dict(e14_summary_payload)
        e14_json_payload["generated_at_utc"] = _utc_timestamp()
        e14_json_payload["n_rows"] = int(len(e14_summary_rows))
        e14_json_path.write_text(
            f"{json.dumps(e14_json_payload, indent=2)}\n",
            encoding="utf-8",
        )
        e14_stability_summary_csv_path = str(e14_csv_path.resolve())
        e14_stability_summary_json_path = str(e14_json_path.resolve())

    if runtime_anchor_rows:
        e14_table_rows = [
            row
            for row in reporting_variant_records
            if str(row.get("experiment_id")) == "E14"
            and str(row.get("analysis_label") or "").strip()
        ]
        coverage_rows = build_confirmatory_control_coverage_rows(
            runtime_anchors=runtime_anchor_rows,
            e12_table_rows=e12_table_rows,
            e13_table_rows=e13_table_rows,
            e14_table_rows=e14_table_rows,
            e12_summary_json_path=e12_table_ready_summary_json_path,
            e13_summary_json_path=e13_table_ready_summary_json_path,
            e14_summary_json_path=e14_stability_summary_json_path,
        )
        coverage_root = campaign_root / "special_aggregations" / "confirmatory"
        coverage_root.mkdir(parents=True, exist_ok=True)
        coverage_csv = coverage_root / "confirmatory_anchor_control_coverage.csv"
        coverage_json = coverage_root / "confirmatory_anchor_control_coverage.json"
        import pandas as pd

        pd.DataFrame(coverage_rows).to_csv(coverage_csv, index=False)
        coverage_payload = {
            "generated_at_utc": _utc_timestamp(),
            "runtime_registry_path": str(registry_path.resolve()),
            "rows": coverage_rows,
            "summary": {
                "n_runtime_anchors": int(len(coverage_rows)),
                "n_e12_covered": int(
                    sum(1 for row in coverage_rows if bool(row.get("e12_covered")))
                ),
                "n_e13_covered": int(
                    sum(1 for row in coverage_rows if bool(row.get("e13_covered")))
                ),
                "n_e14_expected": int(
                    sum(1 for row in coverage_rows if bool(row.get("e14_expected")))
                ),
                "n_e14_covered": int(
                    sum(1 for row in coverage_rows if bool(row.get("e14_covered")))
                ),
            },
        }
        coverage_json.write_text(f"{json.dumps(coverage_payload, indent=2)}\n", encoding="utf-8")
        confirmatory_anchor_control_coverage_csv_path = str(coverage_csv.resolve())
        confirmatory_anchor_control_coverage_json_path = str(coverage_json.resolve())
        thesis_confirmatory_artifact_paths = _write_confirmatory_thesis_artifacts(
            campaign_root=campaign_root,
            runtime_anchor_rows=runtime_anchor_rows,
            confirmatory_model_rows=confirmatory_model_rows,
            e12_table_rows=e12_table_rows,
            e13_table_rows=e13_table_rows,
            coverage_rows=coverage_rows,
        )

    summary_df = _summarize_by_experiment(
        experiments=selected_experiments,
        variant_records=reporting_variant_records,
        warnings_by_experiment=experiment_warnings,
    )
    decision_summary_path = campaign_root / "decision_support_summary.csv"
    summary_df.to_csv(decision_summary_path, index=False)

    run_log_path = _write_run_log_export(
        campaign_root=campaign_root,
        variant_records=all_variant_records,
        dataset_name=dataset_name,
        seed=seed,
        commit=commit,
    )

    decision_report_path, stage_decision_paths = _write_decision_reports(
        campaign_root=campaign_root,
        experiments=selected_experiments,
        variant_records=reporting_variant_records,
    )

    aggregation = aggregate_variant_records(reporting_variant_records, top_k=5)
    aggregation_path = campaign_root / "result_aggregation.json"
    aggregation_path.write_text(
        f"{json.dumps(aggregation, indent=2)}\n",
        encoding="utf-8",
    )
    summary_output_rows = build_summary_output_rows(aggregation)
    summary_output_path = campaign_root / "summary_outputs_export.csv"
    import pandas as pd

    pd.DataFrame(summary_output_rows).to_csv(summary_output_path, index=False)

    campaign_metrics_artifact = register_artifact(
        registry_path=artifact_registry_path,
        artifact_type=ARTIFACT_TYPE_METRICS_BUNDLE,
        run_id=campaign_id,
        upstream_artifact_ids=[],
        config_hash=compute_config_hash(
            {
                "campaign_id": campaign_id,
                "seed": int(seed),
                "n_permutations": int(n_permutations),
                "dry_run": bool(dry_run),
                "phase_plan": str(phase_plan_value),
                "max_parallel_runs": int(max_parallel_runs),
                "max_parallel_gpu_runs": int(max_parallel_gpu_runs),
                "hardware_mode": str(hardware_mode),
                "gpu_device_id": int(gpu_device_id) if gpu_device_id is not None else None,
                "deterministic_compute": bool(deterministic_compute),
                "allow_backend_fallback": bool(allow_backend_fallback),
                "quiet_progress": bool(quiet_progress),
                "progress_interval_seconds": float(progress_interval_seconds),
                "progress_ui": str(progress_ui),
                "progress_detail": str(progress_detail),
                "selected_experiments": [str(exp["experiment_id"]) for exp in selected_experiments],
            }
        ),
        code_ref=commit,
        path=decision_summary_path,
        status="created",
    )

    workbook_output_path: Path | None = None
    if write_back_to_workbook:
        if workbook_source_path is None:
            raise ValueError("write_back_to_workbook=True requires workbook_source_path.")
        machine_rows = _build_machine_status_rows(
            campaign_id=campaign_id,
            source_workbook_path=workbook_source_path,
            variant_records=all_variant_records,
        )
        trial_rows = _build_trial_results_rows(all_variant_records)
        generated_design_rows = _build_generated_design_rows(all_variant_records)
        effect_rows = _build_effect_summary_rows(aggregation)
        summary_rows = list(summary_output_rows)
        run_log_rows = _build_run_log_writeback_rows(
            variant_records=all_variant_records,
            dataset_name=dataset_name,
            seed=seed,
            commit=commit,
        )
        study_review_rows = _build_study_review_rows(
            study_reviews=study_reviews_payload,
            variant_records=all_variant_records,
        )
        workbook_output_path = write_workbook_results(
            source_workbook_path=workbook_source_path,
            version_tag=campaign_id,
            machine_status_rows=machine_rows,
            trial_result_rows=trial_rows,
            summary_output_rows=summary_rows,
            generated_design_rows=generated_design_rows,
            effect_summary_rows=effect_rows,
            study_review_rows=study_review_rows,
            run_log_rows=run_log_rows,
            append_run_log=append_workbook_run_log,
            output_dir=workbook_output_dir,
        )

    eta_calibration_path: str | None = None
    if eta_estimator is not None:
        try:
            eta_estimator.finalize()
            eta_calibration_path = str((campaign_root / "campaign_eta_calibration.json").resolve())
        except Exception:
            eta_calibration_path = None

    anomaly_report_path: str | None = None
    if anomaly_engine is not None:
        try:
            anomaly_engine.finalize()
            anomaly_report_path = str((campaign_root / "campaign_anomaly_report.json").resolve())
        except Exception:
            anomaly_report_path = None

    execution_report_md_path: str | None = None
    execution_report_json_path: str | None = None
    try:
        execution_report_md, execution_report_json = _write_campaign_execution_report(
            campaign_root=campaign_root,
            campaign_id=campaign_id,
        )
        execution_report_md_path = str(execution_report_md.resolve())
        execution_report_json_path = str(execution_report_json.resolve())
    except Exception:
        execution_report_md_path = None
        execution_report_json_path = None

    campaign_manifest = {
        "campaign_id": campaign_id,
        "created_at": _utc_timestamp(),
        "registry_path": str(registry_path.resolve()),
        "selected_experiments": [str(exp["experiment_id"]) for exp in selected_experiments],
        "dataset_scope": dataset_scope,
        "seed": int(seed),
        "n_permutations": int(n_permutations),
        "dry_run": bool(dry_run),
        "phase_plan": str(phase_plan_value),
        "max_parallel_runs": int(max_parallel_runs),
        "max_parallel_gpu_runs": int(max_parallel_gpu_runs),
        "hardware_mode": str(hardware_mode),
        "gpu_device_id": int(gpu_device_id) if gpu_device_id is not None else None,
        "deterministic_compute": bool(deterministic_compute),
        "allow_backend_fallback": bool(allow_backend_fallback),
        "quiet_progress": bool(quiet_progress),
        "progress_interval_seconds": float(progress_interval_seconds),
        "progress_ui": str(progress_ui),
        "progress_detail": str(progress_detail),
        "search_mode": search_mode_value,
        "optuna_trials": int(optuna_trials) if optuna_trials is not None else None,
        "search_space_ids": sorted(search_space_map.keys()),
        "status_counts": _status_snapshot(all_variant_records),
        "experiment_roots": experiment_roots,
        "exports": {
            "run_log_export": str(run_log_path.resolve()),
            "decision_support_summary": str(decision_summary_path.resolve()),
            "decision_recommendations": str(decision_report_path.resolve()),
            "study_review_summary": str(study_review_summary_path.resolve()),
            "result_aggregation": str(aggregation_path.resolve()),
            "summary_outputs_export": str(summary_output_path.resolve()),
            "stage_summaries": [str(path.resolve()) for path in stage_summary_paths],
            "stage_execution_summary": (
                stage_evidence_summary_paths.get("stage_execution_summary")
                if isinstance(stage_evidence_summary_paths, dict)
                else None
            ),
            "stage_resource_summary": (
                stage_evidence_summary_paths.get("stage_resource_summary")
                if isinstance(stage_evidence_summary_paths, dict)
                else None
            ),
            "backend_fallback_summary": (
                stage_evidence_summary_paths.get("backend_fallback_summary")
                if isinstance(stage_evidence_summary_paths, dict)
                else None
            ),
            "stage_lease_summary": (
                stage_evidence_summary_paths.get("stage_lease_summary")
                if isinstance(stage_evidence_summary_paths, dict)
                else None
            ),
            "stage_queue_summary": (
                stage_evidence_summary_paths.get("stage_queue_summary")
                if isinstance(stage_evidence_summary_paths, dict)
                else None
            ),
            "gpu_stage_utilization_summary": (
                stage_evidence_summary_paths.get("gpu_stage_utilization_summary")
                if isinstance(stage_evidence_summary_paths, dict)
                else None
            ),
            "e12_permutation_chunk_merge_summary": permutation_chunk_merge_summary_path,
            "e12_permutation_analysis_summary_csv": e12_table_ready_summary_csv_path,
            "e12_permutation_analysis_summary_json": e12_table_ready_summary_json_path,
            "e13_dummy_baseline_analysis_summary_csv": e13_table_ready_summary_csv_path,
            "e13_dummy_baseline_analysis_summary_json": e13_table_ready_summary_json_path,
            "e14_explanation_stability_summary_csv": e14_stability_summary_csv_path,
            "e14_explanation_stability_summary_json": e14_stability_summary_json_path,
            "confirmatory_anchor_control_coverage_csv": confirmatory_anchor_control_coverage_csv_path,
            "confirmatory_anchor_control_coverage_json": confirmatory_anchor_control_coverage_json_path,
            "confirmatory_anchor_manifest_csv": thesis_confirmatory_artifact_paths.get(
                "confirmatory_anchor_manifest_csv"
            ),
            "confirmatory_anchor_manifest_json": thesis_confirmatory_artifact_paths.get(
                "confirmatory_anchor_manifest_json"
            ),
            "thesis_e16_within_person_summary_csv": thesis_confirmatory_artifact_paths.get(
                "thesis_e16_within_person_summary_csv"
            ),
            "thesis_e16_within_person_summary_json": thesis_confirmatory_artifact_paths.get(
                "thesis_e16_within_person_summary_json"
            ),
            "thesis_e17_transfer_summary_csv": thesis_confirmatory_artifact_paths.get(
                "thesis_e17_transfer_summary_csv"
            ),
            "thesis_e17_transfer_summary_json": thesis_confirmatory_artifact_paths.get(
                "thesis_e17_transfer_summary_json"
            ),
            "thesis_e12_permutation_summary_csv": thesis_confirmatory_artifact_paths.get(
                "thesis_e12_permutation_summary_csv"
            ),
            "thesis_e12_permutation_summary_json": thesis_confirmatory_artifact_paths.get(
                "thesis_e12_permutation_summary_json"
            ),
            "thesis_e13_baseline_summary_csv": thesis_confirmatory_artifact_paths.get(
                "thesis_e13_baseline_summary_csv"
            ),
            "thesis_e13_baseline_summary_json": thesis_confirmatory_artifact_paths.get(
                "thesis_e13_baseline_summary_json"
            ),
            "thesis_permutation_robustness_summary_csv": thesis_confirmatory_artifact_paths.get(
                "thesis_permutation_robustness_summary_csv"
            ),
            "thesis_permutation_robustness_summary_json": thesis_confirmatory_artifact_paths.get(
                "thesis_permutation_robustness_summary_json"
            ),
            "stage_decision_notes": [str(path.resolve()) for path in stage_decision_paths],
            "phase_artifacts": list(phase_artifact_paths),
            "phase_skip_summary": str(phase_skip_summary_path.resolve()),
            "eta_state": str((campaign_root / "eta_state.json").resolve()),
            "eta_calibration": eta_calibration_path,
            "runtime_history": str(history_path.resolve()),
            "anomalies": str((campaign_root / "anomalies.jsonl").resolve()),
            "anomaly_report": anomaly_report_path,
            "campaign_execution_report_md": execution_report_md_path,
            "campaign_execution_report_json": execution_report_json_path,
            "workbook_output_path": (
                str(workbook_output_path.resolve()) if workbook_output_path is not None else None
            ),
        },
        "artifact_registry_path": str(artifact_registry_path.resolve()),
        "campaign_metrics_artifact_id": campaign_metrics_artifact.artifact_id,
        "blocked_experiments": blocked_experiments,
    }
    manifest_path = campaign_root / "campaign_manifest.json"
    manifest_path.write_text(f"{json.dumps(campaign_manifest, indent=2)}\n", encoding="utf-8")
    status_counts_payload = campaign_manifest.get("status_counts")
    status_counts_metadata = (
        dict(status_counts_payload) if isinstance(status_counts_payload, dict) else {}
    )
    _emit_campaign_event(
        event_name="campaign_finished",
        scope="campaign",
        status="finished",
        stage="campaign",
        message="campaign finished",
        metadata={"status_counts": status_counts_metadata},
    )

    return {
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root.resolve()),
        "campaign_manifest_path": str(manifest_path.resolve()),
        "run_log_export_path": str(run_log_path.resolve()),
        "decision_support_summary_path": str(decision_summary_path.resolve()),
        "decision_recommendations_path": str(decision_report_path.resolve()),
        "study_review_summary_path": str(study_review_summary_path.resolve()),
        "result_aggregation_path": str(aggregation_path.resolve()),
        "summary_outputs_export_path": str(summary_output_path.resolve()),
        "phase_skip_summary_path": str(phase_skip_summary_path.resolve()),
        "stage_execution_summary_path": (
            stage_evidence_summary_paths.get("stage_execution_summary")
            if isinstance(stage_evidence_summary_paths, dict)
            else None
        ),
        "stage_resource_summary_path": (
            stage_evidence_summary_paths.get("stage_resource_summary")
            if isinstance(stage_evidence_summary_paths, dict)
            else None
        ),
        "backend_fallback_summary_path": (
            stage_evidence_summary_paths.get("backend_fallback_summary")
            if isinstance(stage_evidence_summary_paths, dict)
            else None
        ),
        "stage_lease_summary_path": (
            stage_evidence_summary_paths.get("stage_lease_summary")
            if isinstance(stage_evidence_summary_paths, dict)
            else None
        ),
        "stage_queue_summary_path": (
            stage_evidence_summary_paths.get("stage_queue_summary")
            if isinstance(stage_evidence_summary_paths, dict)
            else None
        ),
        "gpu_stage_utilization_summary_path": (
            stage_evidence_summary_paths.get("gpu_stage_utilization_summary")
            if isinstance(stage_evidence_summary_paths, dict)
            else None
        ),
        "e12_permutation_chunk_merge_summary_path": permutation_chunk_merge_summary_path,
        "e12_permutation_analysis_summary_csv_path": e12_table_ready_summary_csv_path,
        "e12_permutation_analysis_summary_json_path": e12_table_ready_summary_json_path,
        "e13_dummy_baseline_analysis_summary_csv_path": e13_table_ready_summary_csv_path,
        "e13_dummy_baseline_analysis_summary_json_path": e13_table_ready_summary_json_path,
        "confirmatory_anchor_control_coverage_csv_path": confirmatory_anchor_control_coverage_csv_path,
        "confirmatory_anchor_control_coverage_json_path": confirmatory_anchor_control_coverage_json_path,
        "eta_state_path": str((campaign_root / "eta_state.json").resolve()),
        "eta_calibration_path": eta_calibration_path,
        "runtime_history_path": str(history_path.resolve()),
        "anomalies_path": str((campaign_root / "anomalies.jsonl").resolve()),
        "anomaly_report_path": anomaly_report_path,
        "campaign_execution_report_md_path": execution_report_md_path,
        "campaign_execution_report_json_path": execution_report_json_path,
        "selected_experiments": [str(exp["experiment_id"]) for exp in selected_experiments],
        "status_counts": _status_snapshot(all_variant_records),
        "blocked_experiments": blocked_experiments,
        "workbook_output_path": (
            str(workbook_output_path.resolve()) if workbook_output_path is not None else None
        ),
    }


def run_workbook_decision_support_campaign(
    *,
    workbook_path: Path,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    output_root: Path,
    experiment_id: str | None,
    stage: str | None,
    run_all: bool,
    seed: int,
    n_permutations: int,
    dry_run: bool,
    subjects_filter: list[str] | None = None,
    tasks_filter: list[str] | None = None,
    modalities_filter: list[str] | None = None,
    max_runs_per_experiment: int | None = None,
    dataset_name: str = "Internal BAS2",
    write_back_to_workbook: bool = True,
    workbook_output_dir: Path | None = None,
    append_workbook_run_log: bool = True,
    search_mode: str = "deterministic",
    optuna_trials: int | None = None,
    max_parallel_runs: int = 1,
    max_parallel_gpu_runs: int = 1,
    hardware_mode: str = "cpu_only",
    gpu_device_id: int | None = None,
    deterministic_compute: bool = False,
    allow_backend_fallback: bool = False,
    phase_plan: str = "auto",
    runtime_profile_summary: Path | None = None,
    quiet_progress: bool = False,
    progress_interval_seconds: float = 15.0,
    progress_ui: str = "auto",
    progress_detail: str = "experiment_stage",
    run_experiment_fn: Callable[..., dict[str, Any]] | None = None,
    experiment_ids: list[str] | None = None,
) -> dict[str, Any]:
    workbook_manifest = read_workbook_manifest(workbook_path)
    return run_decision_support_campaign(
        registry_path=workbook_path,
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        output_root=output_root,
        experiment_id=experiment_id,
        experiment_ids=experiment_ids,
        stage=stage,
        run_all=run_all,
        seed=seed,
        n_permutations=n_permutations,
        dry_run=dry_run,
        subjects_filter=subjects_filter,
        tasks_filter=tasks_filter,
        modalities_filter=modalities_filter,
        max_runs_per_experiment=max_runs_per_experiment,
        dataset_name=dataset_name,
        registry_manifest=workbook_manifest,
        write_back_to_workbook=write_back_to_workbook,
        workbook_source_path=workbook_path,
        workbook_output_dir=workbook_output_dir,
        append_workbook_run_log=append_workbook_run_log,
        search_mode=search_mode,
        optuna_trials=optuna_trials,
        max_parallel_runs=max_parallel_runs,
        max_parallel_gpu_runs=max_parallel_gpu_runs,
        hardware_mode=hardware_mode,
        gpu_device_id=gpu_device_id,
        deterministic_compute=deterministic_compute,
        allow_backend_fallback=allow_backend_fallback,
        phase_plan=phase_plan,
        runtime_profile_summary=runtime_profile_summary,
        quiet_progress=bool(quiet_progress),
        progress_interval_seconds=float(progress_interval_seconds),
        progress_ui=str(progress_ui),
        progress_detail=str(progress_detail),
        run_experiment_fn=run_experiment_fn,
    )


__all__ = [
    "run_decision_support_campaign",
    "run_workbook_decision_support_campaign",
]
