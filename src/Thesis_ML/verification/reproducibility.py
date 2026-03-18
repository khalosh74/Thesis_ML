from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

_NON_DETERMINISTIC_KEYS = {
    "timestamp",
    "updated_at_utc",
    "stage_timings_seconds",
    "resource_summary",
    "warnings",
    "warning_summary",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, raw in value.items():
            key_text = str(key)
            if key_text in _NON_DETERMINISTIC_KEYS:
                continue
            if key_text == "data_artifacts":
                continue
            if key_text.endswith("_path") or key_text.endswith("_path_relative"):
                continue
            if key_text in {
                "report_dir",
                "report_dir_relative",
                "run_status_path",
                "artifact_registry_path",
                "cache_dir",
                "data_root",
                "index_csv",
            }:
                continue
            normalized[key_text] = _normalize_payload(raw)
        return normalized
    if isinstance(value, list):
        return [_normalize_payload(item) for item in value]
    return value


def _read_report_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{str(k): str(v) for k, v in row.items()} for row in reader]


def _resolve_report_dir(root: Path, report_dir_text: str) -> Path:
    report_dir = Path(report_dir_text)
    if report_dir.is_absolute():
        return report_dir
    return root / report_dir


def collect_official_invariants(output_dir: Path | str) -> dict[str, Any]:
    root = Path(output_dir)
    execution_status = _load_json(root / "execution_status.json")
    report_rows = _read_report_index(root / "report_index.csv")

    run_statuses = sorted(
        {
            str(run.get("run_id", "")): str(run.get("status", ""))
            for run in execution_status.get("runs", [])
            if isinstance(run, dict)
        }.items(),
        key=lambda item: item[0],
    )

    invariants: dict[str, Any] = {
        "framework_mode": execution_status.get("framework_mode"),
        "run_statuses": run_statuses,
        "mode_artifacts": {},
        "run_artifacts": {},
    }

    framework_mode = str(execution_status.get("framework_mode", ""))
    if framework_mode == "confirmatory":
        suite_summary_path = root / "suite_summary.json"
        if suite_summary_path.exists():
            invariants["mode_artifacts"]["suite_summary"] = _normalize_payload(
                _load_json(suite_summary_path)
            )
    if framework_mode == "locked_comparison":
        decision_path = root / "comparison_decision.json"
        if decision_path.exists():
            invariants["mode_artifacts"]["comparison_decision"] = _normalize_payload(
                _load_json(decision_path)
            )
    repeated_summary_path = root / "repeated_run_summary.json"
    if repeated_summary_path.exists():
        invariants["mode_artifacts"]["repeated_run_summary"] = _normalize_payload(
            _load_json(repeated_summary_path)
        )
    repeated_metrics_path = root / "repeated_run_metrics.csv"
    if repeated_metrics_path.exists():
        invariants["mode_artifacts"]["repeated_run_metrics_sha256"] = _file_sha256(
            repeated_metrics_path
        )
    confidence_intervals_path = root / "confidence_intervals.json"
    if confidence_intervals_path.exists():
        invariants["mode_artifacts"]["confidence_intervals"] = _normalize_payload(
            _load_json(confidence_intervals_path)
        )
    metric_intervals_path = root / "metric_intervals.csv"
    if metric_intervals_path.exists():
        invariants["mode_artifacts"]["metric_intervals_sha256"] = _file_sha256(
            metric_intervals_path
        )
    if framework_mode == "locked_comparison":
        paired_path = root / "paired_model_comparisons.json"
        if paired_path.exists():
            invariants["mode_artifacts"]["paired_model_comparisons"] = _normalize_payload(
                _load_json(paired_path)
            )
        paired_csv_path = root / "paired_model_comparisons.csv"
        if paired_csv_path.exists():
            invariants["mode_artifacts"]["paired_model_comparisons_csv_sha256"] = _file_sha256(
                paired_csv_path
            )

    for row in report_rows:
        if str(row.get("status", "")).strip().lower() != "completed":
            continue
        run_id = str(row.get("run_id", "")).strip()
        report_dir_text = str(row.get("report_dir", "")).strip()
        if not run_id or not report_dir_text:
            continue
        report_dir = _resolve_report_dir(root, report_dir_text)
        if not report_dir.exists():
            continue

        config_path = report_dir / "config.json"
        metrics_path = report_dir / "metrics.json"
        fold_splits_path = report_dir / "fold_splits.csv"
        predictions_path = report_dir / "predictions.csv"
        calibration_summary_path = report_dir / "calibration_summary.json"
        calibration_table_path = report_dir / "calibration_table.csv"
        dataset_card_path = report_dir / "dataset_card.json"
        dataset_summary_path = report_dir / "dataset_summary.json"
        data_quality_report_path = report_dir / "data_quality_report.json"
        leakage_audit_path = report_dir / "leakage_audit.json"
        external_validation_path = report_dir / "external_validation_compatibility.json"
        class_balance_report_path = report_dir / "class_balance_report.csv"
        missingness_report_path = report_dir / "missingness_report.csv"
        dataset_summary_csv_path = report_dir / "dataset_summary.csv"

        run_payload: dict[str, Any] = {}
        if config_path.exists():
            run_payload["config"] = _normalize_payload(_load_json(config_path))
        if metrics_path.exists():
            run_payload["metrics"] = _normalize_payload(_load_json(metrics_path))
        if calibration_summary_path.exists():
            run_payload["calibration_summary"] = _normalize_payload(
                _load_json(calibration_summary_path)
            )
        if dataset_card_path.exists():
            run_payload["dataset_card"] = _normalize_payload(_load_json(dataset_card_path))
        if dataset_summary_path.exists():
            run_payload["dataset_summary"] = _normalize_payload(_load_json(dataset_summary_path))
        if data_quality_report_path.exists():
            run_payload["data_quality_report"] = _normalize_payload(
                _load_json(data_quality_report_path)
            )
        if leakage_audit_path.exists():
            run_payload["leakage_audit"] = _normalize_payload(_load_json(leakage_audit_path))
        if external_validation_path.exists():
            run_payload["external_validation_compatibility"] = _normalize_payload(
                _load_json(external_validation_path)
            )
        if fold_splits_path.exists():
            run_payload["fold_splits_sha256"] = _file_sha256(fold_splits_path)
        if predictions_path.exists():
            run_payload["predictions_sha256"] = _file_sha256(predictions_path)
        if calibration_table_path.exists():
            run_payload["calibration_table_sha256"] = _file_sha256(calibration_table_path)
        if class_balance_report_path.exists():
            run_payload["class_balance_report_sha256"] = _file_sha256(class_balance_report_path)
        if missingness_report_path.exists():
            run_payload["missingness_report_sha256"] = _file_sha256(missingness_report_path)
        if dataset_summary_csv_path.exists():
            run_payload["dataset_summary_csv_sha256"] = _file_sha256(dataset_summary_csv_path)

        invariants["run_artifacts"][run_id] = run_payload

    return invariants


def compare_official_outputs(*, left_dir: Path | str, right_dir: Path | str) -> dict[str, Any]:
    left = collect_official_invariants(left_dir)
    right = collect_official_invariants(right_dir)

    mismatches: list[dict[str, Any]] = []
    if left.get("framework_mode") != right.get("framework_mode"):
        mismatches.append(
            {
                "code": "framework_mode_mismatch",
                "left": left.get("framework_mode"),
                "right": right.get("framework_mode"),
            }
        )

    if left.get("run_statuses") != right.get("run_statuses"):
        mismatches.append(
            {
                "code": "run_statuses_mismatch",
                "left": left.get("run_statuses"),
                "right": right.get("run_statuses"),
            }
        )

    if left.get("mode_artifacts") != right.get("mode_artifacts"):
        mismatches.append(
            {
                "code": "mode_artifacts_mismatch",
                "left": left.get("mode_artifacts"),
                "right": right.get("mode_artifacts"),
            }
        )

    if left.get("run_artifacts") != right.get("run_artifacts"):
        mismatches.append(
            {
                "code": "run_artifacts_mismatch",
                "left": left.get("run_artifacts"),
                "right": right.get("run_artifacts"),
            }
        )

    return {
        "passed": not mismatches,
        "left_dir": str(Path(left_dir).resolve()),
        "right_dir": str(Path(right_dir).resolve()),
        "mismatches": mismatches,
        "left": left,
        "right": right,
    }


__all__ = ["collect_official_invariants", "compare_official_outputs"]
