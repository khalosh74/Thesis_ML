from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.methodology import DataPolicy
from Thesis_ML.data.affect_labels import with_coarse_affect

_BASE_REQUIRED_INDEX_COLUMNS = {
    "sample_id",
    "subject",
    "session",
    "task",
    "modality",
    "beta_path",
    "mask_path",
    "regressor_label",
}


def _distribution(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    counts = series.astype(str).value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _scope_summary(frame: pd.DataFrame, *, scope: str, target_column: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "scope": scope,
            "n_rows": 0,
            "n_subjects": 0,
            "n_sessions": 0,
            "rows_per_subject": {},
            "rows_per_session": {},
            "task_distribution": {},
            "modality_distribution": {},
            "target_distribution": {},
        }
    rows_per_subject = _distribution(frame["subject"])
    rows_per_session = _distribution(frame["session"])
    return {
        "scope": scope,
        "n_rows": int(len(frame)),
        "n_subjects": int(frame["subject"].astype(str).nunique(dropna=False)),
        "n_sessions": int(frame["session"].astype(str).nunique(dropna=False)),
        "rows_per_subject": rows_per_subject,
        "rows_per_session": rows_per_session,
        "task_distribution": _distribution(frame["task"]),
        "modality_distribution": _distribution(frame["modality"]),
        "target_distribution": _distribution(frame[target_column]),
    }


def _issue(
    *,
    code: str,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": str(code),
        "severity": str(severity),
        "message": str(message),
    }
    if details:
        payload["details"] = details
    return payload


def _resolve_external_index_path(path_text: str, *, index_csv: Path) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    cwd_resolved = (Path.cwd() / candidate).resolve()
    if cwd_resolved.exists():
        return cwd_resolved
    return (index_csv.parent / candidate).resolve()


def evaluate_official_data_policy(
    *,
    framework_mode: FrameworkMode,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    full_index_df: pd.DataFrame,
    selected_index_df: pd.DataFrame,
    target_column: str,
    cv_mode: str,
    subject: str | None,
    train_subject: str | None,
    test_subject: str | None,
    filter_task: str | None,
    filter_modality: str | None,
    official_context: dict[str, Any],
) -> dict[str, Any]:
    policy_payload = official_context.get("data_policy")
    if isinstance(policy_payload, dict):
        data_policy = DataPolicy.model_validate(policy_payload)
    else:
        data_policy = DataPolicy()
    policy_effective = data_policy.model_dump(mode="json")

    warnings_list: list[dict[str, Any]] = []
    blocking_list: list[dict[str, Any]] = []

    required_columns = set(_BASE_REQUIRED_INDEX_COLUMNS) | {target_column}
    required_columns.update(str(value) for value in data_policy.required_index_columns)
    missing_required_columns = sorted(required_columns - set(full_index_df.columns))
    if missing_required_columns:
        blocking_list.append(
            _issue(
                code="required_columns_missing",
                severity="blocking",
                message="Dataset index is missing required columns from data_policy.",
                details={"missing_columns": missing_required_columns},
            )
        )

    summary_full = _scope_summary(full_index_df, scope="full_index", target_column=target_column)
    summary_selected = _scope_summary(
        selected_index_df,
        scope="selected_subset",
        target_column=target_column,
    )
    dataset_summary = {
        "framework_mode": framework_mode.value,
        "target_column": target_column,
        "cv_mode": cv_mode,
        "subject": subject,
        "train_subject": train_subject,
        "test_subject": test_subject,
        "filter_task": filter_task,
        "filter_modality": filter_modality,
        "full_index": summary_full,
        "selected_subset": summary_selected,
    }
    dataset_summary_rows = [
        {
            "scope": row["scope"],
            "n_rows": row["n_rows"],
            "n_subjects": row["n_subjects"],
            "n_sessions": row["n_sessions"],
        }
        for row in (summary_full, summary_selected)
    ]

    class_balance_rows: list[dict[str, Any]] = []
    if data_policy.class_balance.enabled:
        warning_threshold = data_policy.class_balance.min_class_fraction_warning
        blocking_threshold = data_policy.class_balance.min_class_fraction_blocking
        for axis in data_policy.class_balance.axes:
            if axis == "overall":
                grouped = [("__overall__", selected_index_df)]
            elif axis not in selected_index_df.columns:
                warnings_list.append(
                    _issue(
                        code="class_balance_axis_missing",
                        severity="warning",
                        message=f"Requested class-balance axis '{axis}' is not present in selected subset.",
                        details={"axis": axis},
                    )
                )
                continue
            else:
                grouped = [
                    (group_value, subset.copy())
                    for group_value, subset in selected_index_df.groupby(axis, dropna=False)
                ]
            for group_value, subset in grouped:
                counts = subset[target_column].astype(str).value_counts(dropna=False).sort_index()
                n_samples = int(len(subset))
                n_classes = int(counts.shape[0])
                min_fraction = float((counts / max(n_samples, 1)).min()) if n_samples > 0 else 0.0
                status = "ok"
                if blocking_threshold is not None and min_fraction < float(blocking_threshold):
                    status = "blocking"
                    blocking_list.append(
                        _issue(
                            code="class_balance_blocking_threshold",
                            severity="blocking",
                            message="Class-balance blocking threshold was violated.",
                            details={
                                "axis": axis,
                                "group_value": str(group_value),
                                "min_class_fraction": min_fraction,
                                "blocking_threshold": float(blocking_threshold),
                            },
                        )
                    )
                elif warning_threshold is not None and min_fraction < float(warning_threshold):
                    status = "warning"
                    warnings_list.append(
                        _issue(
                            code="class_balance_warning_threshold",
                            severity="warning",
                            message="Class-balance warning threshold was violated.",
                            details={
                                "axis": axis,
                                "group_value": str(group_value),
                                "min_class_fraction": min_fraction,
                                "warning_threshold": float(warning_threshold),
                            },
                        )
                    )
                class_balance_rows.append(
                    {
                        "scope": "selected_subset",
                        "axis": axis,
                        "group_value": str(group_value),
                        "n_samples": n_samples,
                        "n_classes": n_classes,
                        "min_class_fraction": min_fraction,
                        "status": status,
                    }
                )

    missingness_rows: list[dict[str, Any]] = []
    if data_policy.missingness.enabled:
        warning_threshold = data_policy.missingness.max_missing_fraction_warning
        blocking_threshold = data_policy.missingness.max_missing_fraction_blocking
        for column_name in sorted(selected_index_df.columns):
            missing_count = int(selected_index_df[column_name].isna().sum())
            missing_fraction = float(
                missing_count / max(int(len(selected_index_df)), 1)
                if len(selected_index_df) > 0
                else 0.0
            )
            status = "ok"
            if blocking_threshold is not None and missing_fraction > float(blocking_threshold):
                status = "blocking"
                blocking_list.append(
                    _issue(
                        code="missingness_blocking_threshold",
                        severity="blocking",
                        message="Missingness blocking threshold was violated.",
                        details={
                            "column": column_name,
                            "missing_fraction": missing_fraction,
                            "blocking_threshold": float(blocking_threshold),
                        },
                    )
                )
            elif warning_threshold is not None and missing_fraction > float(warning_threshold):
                status = "warning"
                warnings_list.append(
                    _issue(
                        code="missingness_warning_threshold",
                        severity="warning",
                        message="Missingness warning threshold was violated.",
                        details={
                            "column": column_name,
                            "missing_fraction": missing_fraction,
                            "warning_threshold": float(warning_threshold),
                        },
                    )
                )
            missingness_rows.append(
                {
                    "scope": "selected_subset",
                    "column": column_name,
                    "missing_count": missing_count,
                    "missing_fraction": missing_fraction,
                    "status": status,
                }
            )

    leakage_checks: list[dict[str, Any]] = []
    duplicate_sample_ids = int(selected_index_df["sample_id"].astype(str).duplicated().sum())
    duplicate_sample_status = (
        "fail"
        if duplicate_sample_ids > 0 and data_policy.leakage.fail_on_duplicate_sample_id
        else "warning"
        if duplicate_sample_ids > 0
        else "pass"
    )
    leakage_checks.append(
        {
            "check": "duplicate_sample_id",
            "status": duplicate_sample_status,
            "count": duplicate_sample_ids,
            "blocking_policy": bool(data_policy.leakage.fail_on_duplicate_sample_id),
        }
    )
    if duplicate_sample_status == "fail":
        blocking_list.append(
            _issue(
                code="leakage_duplicate_sample_id",
                severity="blocking",
                message="Duplicate sample_id values detected in selected subset.",
                details={"count": duplicate_sample_ids},
            )
        )
    elif duplicate_sample_status == "warning":
        warnings_list.append(
            _issue(
                code="leakage_duplicate_sample_id_warning",
                severity="warning",
                message="Duplicate sample_id values detected in selected subset.",
                details={"count": duplicate_sample_ids},
            )
        )

    duplicate_beta_paths = int(selected_index_df["beta_path"].astype(str).duplicated().sum())
    duplicate_beta_status = "pass"
    if duplicate_beta_paths > 0:
        if data_policy.leakage.fail_on_duplicate_beta_path:
            duplicate_beta_status = "fail"
            blocking_list.append(
                _issue(
                    code="leakage_duplicate_beta_path",
                    severity="blocking",
                    message="Duplicate beta_path values detected in selected subset.",
                    details={"count": duplicate_beta_paths},
                )
            )
        elif data_policy.leakage.warn_on_duplicate_beta_path:
            duplicate_beta_status = "warning"
            warnings_list.append(
                _issue(
                    code="leakage_duplicate_beta_path_warning",
                    severity="warning",
                    message="Duplicate beta_path values detected in selected subset.",
                    details={"count": duplicate_beta_paths},
                )
            )
    leakage_checks.append(
        {
            "check": "duplicate_beta_path",
            "status": duplicate_beta_status,
            "count": duplicate_beta_paths,
            "blocking_policy": bool(data_policy.leakage.fail_on_duplicate_beta_path),
        }
    )

    transfer_subject_overlap = bool(
        cv_mode == "frozen_cross_person_transfer"
        and train_subject is not None
        and test_subject is not None
        and str(train_subject) == str(test_subject)
    )
    transfer_subject_status = (
        "fail"
        if transfer_subject_overlap and data_policy.leakage.fail_on_subject_overlap_for_transfer
        else "pass"
    )
    leakage_checks.append(
        {
            "check": "transfer_subject_overlap",
            "status": transfer_subject_status,
            "overlap_detected": bool(transfer_subject_overlap),
            "blocking_policy": bool(data_policy.leakage.fail_on_subject_overlap_for_transfer),
        }
    )
    if transfer_subject_status == "fail":
        blocking_list.append(
            _issue(
                code="leakage_transfer_subject_overlap",
                severity="blocking",
                message="Train/test subject overlap detected for frozen transfer run.",
                details={
                    "train_subject": train_subject,
                    "test_subject": test_subject,
                },
            )
        )

    group_overlap_detected = False
    if data_policy.leakage.fail_on_cv_group_overlap:
        if cv_mode == "within_subject_loso_session":
            n_sessions = int(selected_index_df["session"].astype(str).nunique(dropna=False))
            group_overlap_detected = bool(n_sessions < 2)
        elif cv_mode == "frozen_cross_person_transfer" and train_subject and test_subject:
            scoped = selected_index_df.copy()
            scoped["subject"] = scoped["subject"].astype(str)
            train_sessions = set(
                scoped.loc[scoped["subject"] == str(train_subject), "session"]
                .astype(str)
                .tolist()
            )
            test_sessions = set(
                scoped.loc[scoped["subject"] == str(test_subject), "session"].astype(str).tolist()
            )
            group_overlap_detected = False if train_sessions and test_sessions else True
    group_overlap_status = "fail" if group_overlap_detected else "pass"
    leakage_checks.append(
        {
            "check": "cv_group_overlap_guardrail",
            "status": group_overlap_status,
            "overlap_detected": bool(group_overlap_detected),
            "blocking_policy": bool(data_policy.leakage.fail_on_cv_group_overlap),
        }
    )
    if group_overlap_status == "fail":
        blocking_list.append(
            _issue(
                code="leakage_cv_group_overlap",
                severity="blocking",
                message="CV group-overlap guardrail failed for selected subset.",
                details={"cv_mode": cv_mode},
            )
        )

    leakage_verdict = (
        "fail"
        if any(check["status"] == "fail" for check in leakage_checks)
        else "warning"
        if any(check["status"] == "warning" for check in leakage_checks)
        else "pass"
    )
    leakage_audit = {
        "framework_mode": framework_mode.value,
        "cv_mode": cv_mode,
        "target_column": target_column,
        "verdict": leakage_verdict,
        "checks": leakage_checks,
    }

    external_card: dict[str, Any] = {
        "mode": str(data_policy.external_validation.mode),
        "enabled": bool(data_policy.external_validation.enabled),
        "datasets": [],
    }
    external_summary: dict[str, Any] = {
        "enabled": bool(data_policy.external_validation.enabled),
        "n_datasets": 0,
        "n_compatible": 0,
        "n_incompatible": 0,
        "datasets": [],
    }
    external_compatibility: dict[str, Any] = {
        "enabled": bool(data_policy.external_validation.enabled),
        "mode": str(data_policy.external_validation.mode),
        "require_compatible": bool(data_policy.external_validation.require_compatible),
        "require_for_official_runs": bool(
            data_policy.external_validation.require_for_official_runs
        ),
        "status": "not_configured",
        "datasets": [],
    }

    external_policy = data_policy.external_validation
    if external_policy.enabled:
        external_compatibility["status"] = "checked"
        required_external_columns = (
            set(_BASE_REQUIRED_INDEX_COLUMNS)
            | {target_column}
            | {str(value) for value in data_policy.required_index_columns}
        )
        if external_policy.require_for_official_runs and not external_policy.datasets:
            blocking_list.append(
                _issue(
                    code="external_validation_required_missing",
                    severity="blocking",
                    message=(
                        "external_validation.require_for_official_runs=true requires at least one external dataset."
                    ),
                )
            )
        for dataset_spec in external_policy.datasets:
            dataset_path = _resolve_external_index_path(dataset_spec.index_csv, index_csv=index_csv)
            dataset_status = "compatible"
            details: dict[str, Any] = {
                "dataset_id": dataset_spec.dataset_id,
                "index_csv": str(dataset_path),
                "required": bool(dataset_spec.required),
            }
            dataset_rows = 0
            dataset_subjects = 0
            dataset_sessions = 0
            missing_columns: list[str] = []
            if not dataset_path.exists() or not dataset_path.is_file():
                dataset_status = "incompatible"
                details["reason"] = "index_csv_missing"
            else:
                external_df = pd.read_csv(dataset_path)
                if target_column == "coarse_affect":
                    external_df = with_coarse_affect(
                        external_df,
                        emotion_column="emotion",
                        coarse_column="coarse_affect",
                    )
                required_columns = set(required_external_columns)
                required_columns.update(str(value) for value in dataset_spec.required_columns)
                if dataset_spec.target_column:
                    required_columns.add(str(dataset_spec.target_column))
                missing_columns = sorted(required_columns - set(external_df.columns))
                if missing_columns:
                    dataset_status = "incompatible"
                    details["reason"] = "missing_required_columns"
                    details["missing_columns"] = missing_columns
                else:
                    dataset_rows = int(len(external_df))
                    dataset_subjects = int(external_df["subject"].astype(str).nunique(dropna=False))
                    dataset_sessions = int(external_df["session"].astype(str).nunique(dropna=False))
                    target_series = external_df[target_column].dropna().astype(str)
                    if int(target_series.nunique(dropna=False)) < 2:
                        dataset_status = "incompatible"
                        details["reason"] = "insufficient_target_classes"
            if dataset_status != "compatible":
                if dataset_spec.required or external_policy.require_compatible:
                    blocking_list.append(
                        _issue(
                            code="external_dataset_incompatible",
                            severity="blocking",
                            message="External dataset compatibility check failed.",
                            details=details,
                        )
                    )
                else:
                    warnings_list.append(
                        _issue(
                            code="external_dataset_incompatible_warning",
                            severity="warning",
                            message="External dataset compatibility check failed (non-blocking).",
                            details=details,
                        )
                    )

            external_row = {
                "dataset_id": dataset_spec.dataset_id,
                "index_csv": str(dataset_path),
                "status": dataset_status,
                "n_rows": dataset_rows,
                "n_subjects": dataset_subjects,
                "n_sessions": dataset_sessions,
                "required": bool(dataset_spec.required),
                "notes": dataset_spec.notes,
                "missing_columns": missing_columns,
            }
            external_compatibility["datasets"].append(external_row)
            external_summary["datasets"].append(external_row)
            external_card["datasets"].append(
                {
                    "dataset_id": dataset_spec.dataset_id,
                    "status": dataset_status,
                    "index_csv": str(dataset_path),
                    "required_columns": sorted(
                        set(required_external_columns)
                        | {str(value) for value in dataset_spec.required_columns}
                    ),
                    "notes": dataset_spec.notes,
                }
            )
        external_summary["n_datasets"] = int(len(external_summary["datasets"]))
        external_summary["n_compatible"] = int(
            len([row for row in external_summary["datasets"] if row["status"] == "compatible"])
        )
        external_summary["n_incompatible"] = int(
            len([row for row in external_summary["datasets"] if row["status"] != "compatible"])
        )

    data_quality_report = {
        "framework_mode": framework_mode.value,
        "target_column": target_column,
        "cv_mode": cv_mode,
        "status": "failed" if blocking_list else "warning" if warnings_list else "pass",
        "n_blocking_issues": int(len(blocking_list)),
        "n_warnings": int(len(warnings_list)),
        "blocking_issues": blocking_list,
        "warnings": warnings_list,
        "threshold_policy": {
            "class_balance": policy_effective["class_balance"],
            "missingness": policy_effective["missingness"],
        },
        "leakage_audit_verdict": leakage_verdict,
        "external_validation_status": external_compatibility["status"],
    }

    return {
        "data_policy_effective": policy_effective,
        "dataset_summary": dataset_summary,
        "dataset_summary_rows": dataset_summary_rows,
        "class_balance_rows": class_balance_rows,
        "missingness_rows": missingness_rows,
        "leakage_audit": leakage_audit,
        "data_quality_report": data_quality_report,
        "external_dataset_card": external_card,
        "external_dataset_summary": external_summary,
        "external_validation_compatibility": external_compatibility,
        "blocking_issues": blocking_list,
        "warning_issues": warnings_list,
        "required_columns": sorted(required_columns),
    }


def _dataset_card_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Dataset Card",
        "",
        f"- Framework mode: `{payload.get('framework_mode')}`",
        f"- Target: `{payload.get('target_definition', {}).get('target_column')}`",
        f"- Unit of analysis: `{payload.get('unit_of_analysis')}`",
        f"- Dataset fingerprint present: `{bool(payload.get('dataset_identity', {}).get('dataset_fingerprint'))}`",
        "",
        "## Coverage",
    ]
    coverage = payload.get("coverage", {})
    selected = coverage.get("selected_subset", {})
    lines.extend(
        [
            f"- Rows: `{selected.get('n_rows', 0)}`",
            f"- Subjects: `{selected.get('n_subjects', 0)}`",
            f"- Sessions: `{selected.get('n_sessions', 0)}`",
        ]
    )
    lines.extend(
        [
            "",
            "## Usage",
            f"- Intended use: {payload.get('intended_use')}",
            "- Not intended use:",
        ]
    )
    not_intended = payload.get("not_intended_use", [])
    for entry in not_intended:
        lines.append(f"  - {entry}")
    limitations = payload.get("known_limitations", [])
    if limitations:
        lines.extend(["", "## Known limitations"])
        for entry in limitations:
            lines.append(f"- {entry}")
    return "\n".join(lines) + "\n"


def write_official_data_artifacts(
    *,
    report_dir: Path,
    assessment: dict[str, Any],
    framework_mode: FrameworkMode,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    target_column: str,
    cv_mode: str,
    subject: str | None,
    train_subject: str | None,
    test_subject: str | None,
    filter_task: str | None,
    filter_modality: str | None,
    sample_unit: str | None,
    label_policy: str | None,
    target_mapping_version: str | None,
    target_mapping_hash: str | None,
    dataset_fingerprint: dict[str, Any] | None,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset_card_json_path = report_dir / "dataset_card.json"
    dataset_card_md_path = report_dir / "dataset_card.md"
    dataset_summary_json_path = report_dir / "dataset_summary.json"
    dataset_summary_csv_path = report_dir / "dataset_summary.csv"
    data_quality_report_path = report_dir / "data_quality_report.json"
    class_balance_csv_path = report_dir / "class_balance_report.csv"
    missingness_csv_path = report_dir / "missingness_report.csv"
    leakage_audit_path = report_dir / "leakage_audit.json"
    external_dataset_card_path = report_dir / "external_dataset_card.json"
    external_dataset_summary_path = report_dir / "external_dataset_summary.json"
    external_validation_compatibility_path = report_dir / "external_validation_compatibility.json"

    dataset_summary = dict(assessment.get("dataset_summary", {}))
    dataset_summary_json_path.write_text(
        f"{json.dumps(dataset_summary, indent=2)}\n",
        encoding="utf-8",
    )
    pd.DataFrame(list(assessment.get("dataset_summary_rows", []))).to_csv(
        dataset_summary_csv_path,
        index=False,
    )
    pd.DataFrame(list(assessment.get("class_balance_rows", []))).to_csv(
        class_balance_csv_path,
        index=False,
    )
    pd.DataFrame(list(assessment.get("missingness_rows", []))).to_csv(
        missingness_csv_path,
        index=False,
    )
    leakage_payload = dict(assessment.get("leakage_audit", {}))
    leakage_audit_path.write_text(f"{json.dumps(leakage_payload, indent=2)}\n", encoding="utf-8")
    quality_payload = dict(assessment.get("data_quality_report", {}))
    data_quality_report_path.write_text(
        f"{json.dumps(quality_payload, indent=2)}\n",
        encoding="utf-8",
    )

    external_dataset_card = dict(assessment.get("external_dataset_card", {}))
    external_dataset_summary = dict(assessment.get("external_dataset_summary", {}))
    external_compatibility = dict(assessment.get("external_validation_compatibility", {}))
    external_dataset_card_path.write_text(
        f"{json.dumps(external_dataset_card, indent=2)}\n",
        encoding="utf-8",
    )
    external_dataset_summary_path.write_text(
        f"{json.dumps(external_dataset_summary, indent=2)}\n",
        encoding="utf-8",
    )
    external_validation_compatibility_path.write_text(
        f"{json.dumps(external_compatibility, indent=2)}\n",
        encoding="utf-8",
    )

    data_policy_effective = dict(assessment.get("data_policy_effective", {}))
    dataset_card_payload = {
        "schema_version": "official-dataset-card-v1",
        "framework_mode": framework_mode.value,
        "dataset_identity": {
            "index_csv": str(index_csv.resolve()),
            "data_root": str(data_root.resolve()),
            "cache_dir": str(cache_dir.resolve()),
            "dataset_fingerprint": dict(dataset_fingerprint) if dataset_fingerprint else None,
        },
        "unit_of_analysis": sample_unit or "beta_event",
        "target_definition": {
            "target_column": target_column,
            "label_policy": label_policy,
            "target_mapping_version": target_mapping_version,
            "target_mapping_hash": target_mapping_hash,
        },
        "selection_scope": {
            "cv_mode": cv_mode,
            "subject": subject,
            "train_subject": train_subject,
            "test_subject": test_subject,
            "filter_task": filter_task,
            "filter_modality": filter_modality,
        },
        "coverage": {
            "full_index": dataset_summary.get("full_index", {}),
            "selected_subset": dataset_summary.get("selected_subset", {}),
        },
        "leakage_sensitive_structure": {
            "verdict": leakage_payload.get("verdict"),
            "checks": leakage_payload.get("checks", []),
        },
        "external_validation": external_compatibility,
        "intended_use": data_policy_effective.get("intended_use"),
        "not_intended_use": data_policy_effective.get("not_intended_use", []),
        "known_limitations": data_policy_effective.get("known_limitations", []),
    }
    dataset_card_json_path.write_text(
        f"{json.dumps(dataset_card_payload, indent=2)}\n",
        encoding="utf-8",
    )
    dataset_card_md_path.write_text(_dataset_card_markdown(dataset_card_payload), encoding="utf-8")

    data_artifacts = {
        "dataset_card_json": str(dataset_card_json_path.resolve()),
        "dataset_card_md": str(dataset_card_md_path.resolve()),
        "dataset_summary_json": str(dataset_summary_json_path.resolve()),
        "dataset_summary_csv": str(dataset_summary_csv_path.resolve()),
        "data_quality_report_json": str(data_quality_report_path.resolve()),
        "class_balance_report_csv": str(class_balance_csv_path.resolve()),
        "missingness_report_csv": str(missingness_csv_path.resolve()),
        "leakage_audit_json": str(leakage_audit_path.resolve()),
        "external_dataset_card_json": str(external_dataset_card_path.resolve()),
        "external_dataset_summary_json": str(external_dataset_summary_path.resolve()),
        "external_validation_compatibility_json": str(
            external_validation_compatibility_path.resolve()
        ),
    }
    return {
        "data_policy_effective": data_policy_effective,
        "data_artifacts": data_artifacts,
        "data_quality_report": quality_payload,
    }


__all__ = [
    "evaluate_official_data_policy",
    "write_official_data_artifacts",
]
