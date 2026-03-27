from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from Thesis_ML.experiments.cv_split_plan import build_cv_split_plan
from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.methodology import DataPolicy
from Thesis_ML.data.affect_labels import with_coarse_affect
from Thesis_ML.data.index_validation import (
    CANONICAL_BETA_PATH_COLUMN,
    DatasetIndexValidationError,
    canonicalize_index_paths,
)
from Thesis_ML.features.preprocessing import BASELINE_STANDARD_SCALER_RECIPE_ID

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


def _stable_json_sha256(payload: Any) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


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


def _build_exact_cv_split_audit(
    *,
    selected_index_df: pd.DataFrame,
    target_column: str,
    cv_mode: str,
    subject: str | None,
    train_subject: str | None,
    test_subject: str | None,
) -> dict[str, Any]:
    split_plan = build_cv_split_plan(
        metadata_df=selected_index_df,
        target_column=target_column,
        cv_mode=cv_mode,
        subject=subject,
        train_subject=train_subject,
        test_subject=test_subject,
        seed=0 if cv_mode == "record_random_split" else None,
    )

    n_rows = int(len(selected_index_df))
    test_counts = np.zeros(n_rows, dtype=int)
    fold_rows: list[dict[str, Any]] = []
    global_failures: list[str] = []

    for fold in split_plan.folds:
        overlap_count = int(np.intersect1d(fold.train_idx, fold.test_idx).size)
        test_counts[fold.test_idx] += 1

        fold_failures: list[str] = []
        if overlap_count > 0:
            fold_failures.append("train_test_index_overlap")

        if cv_mode == "within_subject_loso_session":
            if fold.train_subjects != (str(subject),):
                fold_failures.append("train_subject_mismatch")
            if fold.test_subjects != (str(subject),):
                fold_failures.append("test_subject_mismatch")
            if len(fold.test_sessions) != 1:
                fold_failures.append("expected_single_test_session")
            if set(fold.train_sessions).intersection(set(fold.test_sessions)):
                fold_failures.append("session_overlap")

        elif cv_mode == "frozen_cross_person_transfer":
            if fold.train_subjects != (str(train_subject),):
                fold_failures.append("train_subject_mismatch")
            if fold.test_subjects != (str(test_subject),):
                fold_failures.append("test_subject_mismatch")
            if set(fold.train_subjects).intersection(set(fold.test_subjects)):
                fold_failures.append("subject_overlap")

        elif cv_mode == "loso_session":
            if set(fold.train_groups).intersection(set(fold.test_groups)):
                fold_failures.append("group_overlap")

        fold_rows.append(
            {
                "fold": int(fold.fold),
                "status": "fail" if fold_failures else "pass",
                "failure_codes": "|".join(fold_failures),
                "train_sample_count": int(len(fold.train_idx)),
                "test_sample_count": int(len(fold.test_idx)),
                "train_subjects": "|".join(fold.train_subjects),
                "test_subjects": "|".join(fold.test_subjects),
                "train_sessions": "|".join(fold.train_sessions),
                "test_sessions": "|".join(fold.test_sessions),
                "train_groups": "|".join(fold.train_groups),
                "test_groups": "|".join(fold.test_groups),
                "train_test_index_overlap_count": overlap_count,
            }
        )
        global_failures.extend(fold_failures)

    if cv_mode == "frozen_cross_person_transfer":
        expected_test_mask = (
            selected_index_df["subject"].astype(str) == str(test_subject)
        ).to_numpy(dtype=bool)
    else:
        expected_test_mask = np.ones(n_rows, dtype=bool)

    missing_expected_test_rows = int(np.sum(expected_test_mask & (test_counts == 0)))
    unexpected_test_rows = int(np.sum((~expected_test_mask) & (test_counts > 0)))
    duplicate_test_coverage_rows = int(np.sum(test_counts > 1))

    if missing_expected_test_rows > 0:
        global_failures.append("missing_expected_test_rows")
    if unexpected_test_rows > 0:
        global_failures.append("unexpected_test_rows")
    if duplicate_test_coverage_rows > 0:
        global_failures.append("duplicate_test_coverage_rows")

    if cv_mode == "within_subject_loso_session":
        expected_n_folds = int(selected_index_df["session"].astype(str).nunique(dropna=False))
    elif cv_mode == "frozen_cross_person_transfer":
        expected_n_folds = 1
    else:
        expected_n_folds = int(len(split_plan.folds))

    if int(len(split_plan.folds)) != int(expected_n_folds):
        global_failures.append("unexpected_fold_count")

    return {
        "status": "fail" if global_failures else "pass",
        "cv_mode": cv_mode,
        "n_rows": n_rows,
        "n_folds": int(len(split_plan.folds)),
        "expected_n_folds": int(expected_n_folds),
        "missing_expected_test_rows": missing_expected_test_rows,
        "unexpected_test_rows": unexpected_test_rows,
        "duplicate_test_coverage_rows": duplicate_test_coverage_rows,
        "failure_codes": sorted(set(global_failures)),
        "fold_rows": fold_rows,
    }


def _build_cv_split_manifest(
    *,
    selected_index_df: pd.DataFrame,
    target_column: str,
    cv_mode: str,
    subject: str | None,
    train_subject: str | None,
    test_subject: str | None,
) -> dict[str, Any]:
    split_plan = build_cv_split_plan(
        metadata_df=selected_index_df,
        target_column=target_column,
        cv_mode=cv_mode,
        subject=subject,
        train_subject=train_subject,
        test_subject=test_subject,
        seed=0 if cv_mode == "record_random_split" else None,
    )
    if CANONICAL_BETA_PATH_COLUMN in selected_index_df.columns:
        beta_path_column = CANONICAL_BETA_PATH_COLUMN
    else:
        beta_path_column = "beta_path"

    rows: list[dict[str, Any]] = []
    for fold in split_plan.folds:
        for partition, index_values in (("train", fold.train_idx), ("test", fold.test_idx)):
            fold_frame = selected_index_df.iloc[np.asarray(index_values, dtype=int)]
            for _, row in fold_frame.iterrows():
                rows.append(
                    {
                        "fold": int(fold.fold),
                        "partition": str(partition),
                        "sample_id": str(row.get("sample_id", "")),
                        "beta_path": str(row.get(beta_path_column, "")),
                        "subject": str(row.get("subject", "")),
                        "session": str(row.get("session", "")),
                        "bas": str(row.get("bas", "")),
                        "task": str(row.get("task", "")),
                        "modality": str(row.get("modality", "")),
                        "target_label": str(row.get(target_column, "")),
                    }
                )

    sort_keys = [
        "fold",
        "partition",
        "sample_id",
        "beta_path",
        "subject",
        "session",
        "bas",
        "task",
        "modality",
        "target_label",
    ]
    rows = sorted(rows, key=lambda item: tuple(item.get(key, "") for key in sort_keys))
    return {
        "status": "pass",
        "schema_version": "cv-split-manifest-v1",
        "cv_mode": str(cv_mode),
        "target_column": str(target_column),
        "row_count": int(len(rows)),
        "rows": rows,
        "sha256": _stable_json_sha256(rows),
    }

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
    target_derivation_audit_df: pd.DataFrame | None = None,
    derived_label_inconsistency_audit_df: pd.DataFrame | None = None,
    selection_exclusion_manifest_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    policy_payload = official_context.get("data_policy")
    if isinstance(policy_payload, dict):
        data_policy = DataPolicy.model_validate(policy_payload)
    else:
        data_policy = DataPolicy()
    policy_effective = data_policy.model_dump(mode="json")

    feature_engineering_context = official_context.get("feature_engineering")
    context_feature_recipe = official_context.get("feature_recipe_id")
    if context_feature_recipe is None and isinstance(feature_engineering_context, dict):
        context_feature_recipe = feature_engineering_context.get("feature_recipe_id")
    context_emit_feature_qc = official_context.get("emit_feature_qc_artifacts")
    if context_emit_feature_qc is None and isinstance(feature_engineering_context, dict):
        context_emit_feature_qc = feature_engineering_context.get("emit_feature_qc_artifacts")
    feature_engineering_summary = {
        "feature_recipe_id": str(
            context_feature_recipe or BASELINE_STANDARD_SCALER_RECIPE_ID
        ),
        "emit_feature_qc_artifacts": bool(
            True if context_emit_feature_qc is None else context_emit_feature_qc
        ),
    }

    warnings_list: list[dict[str, Any]] = []
    blocking_list: list[dict[str, Any]] = []

    if selection_exclusion_manifest_df is None:
        selection_exclusion_manifest_df = pd.DataFrame()

    selection_exclusion_rows = (
        selection_exclusion_manifest_df.to_dict(orient="records")
        if not selection_exclusion_manifest_df.empty
        else []
    )

    selection_exclusion_summary: dict[str, Any] = {
        "n_rows": int(len(selection_exclusion_manifest_df)),
        "by_stage": {},
        "by_reason": {},
    }

    if not selection_exclusion_manifest_df.empty:
        if "exclusion_stage" in selection_exclusion_manifest_df.columns:
            by_stage = (
                selection_exclusion_manifest_df["exclusion_stage"]
                .astype(str)
                .value_counts(dropna=False)
                .sort_index()
                .to_dict()
            )
            selection_exclusion_summary["by_stage"] = {
                str(key): int(value) for key, value in by_stage.items()
            }
        if "exclusion_reason" in selection_exclusion_manifest_df.columns:
            by_reason = (
                selection_exclusion_manifest_df["exclusion_reason"]
                .astype(str)
                .value_counts(dropna=False)
                .sort_index()
                .to_dict()
            )
            selection_exclusion_summary["by_reason"] = {
                str(key): int(value) for key, value in by_reason.items()
            }

    if target_derivation_audit_df is None:
        target_derivation_audit_df = pd.DataFrame()

    target_derivation_audit_rows = (
        target_derivation_audit_df.to_dict(orient="records")
        if not target_derivation_audit_df.empty
        else []
    )

    target_derivation_summary: dict[str, Any] = {
        "n_rows": int(len(target_derivation_audit_df)),
        "by_category": {},
    }
    if not target_derivation_audit_df.empty and "drop_category" in target_derivation_audit_df.columns:
        counts = (
            target_derivation_audit_df["drop_category"]
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
        target_derivation_summary["by_category"] = {
            str(key): int(value) for key, value in counts.items()
        }

    intended_exclusion_count = int(
        (
            target_derivation_audit_df["drop_category"].astype(str) == "intended_target_exclusion"
        ).sum()
    ) if not target_derivation_audit_df.empty and "drop_category" in target_derivation_audit_df.columns else 0

    if intended_exclusion_count > 0:
        warnings_list.append(
            _issue(
                code="target_derivation_intended_exclusion",
                severity="warning",
                message="Rows were intentionally excluded during derived-target construction.",
                details={
                    "target_column": target_column,
                    "count": intended_exclusion_count,
                },
            )
        )

    if derived_label_inconsistency_audit_df is None:
        derived_label_inconsistency_audit_df = pd.DataFrame()

    derived_label_inconsistency_rows = (
        derived_label_inconsistency_audit_df.to_dict(orient="records")
        if not derived_label_inconsistency_audit_df.empty
        else []
    )
    derived_label_inconsistency_summary: dict[str, Any] = {
        "n_rows": int(len(derived_label_inconsistency_audit_df)),
        "by_category": {},
    }
    if (
        not derived_label_inconsistency_audit_df.empty
        and "inconsistency_category" in derived_label_inconsistency_audit_df.columns
    ):
        counts = (
            derived_label_inconsistency_audit_df["inconsistency_category"]
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
        derived_label_inconsistency_summary["by_category"] = {
            str(key): int(value) for key, value in counts.items()
        }

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

    selected_index_for_checks = selected_index_df.copy()
    beta_path_canonicalization_status = "pass"
    beta_path_canonicalization_error: str | None = None
    try:
        selected_index_for_checks = canonicalize_index_paths(
            selected_index_for_checks,
            data_root=data_root,
            require_exists=False,
        )
    except DatasetIndexValidationError as exc:
        beta_path_canonicalization_status = "fail"
        beta_path_canonicalization_error = str(exc)
        blocking_list.append(
            _issue(
                code="leakage_beta_path_canonicalization_failed",
                severity="blocking",
                message="Failed to canonicalize selected beta/mask paths for leakage checks.",
                details={"error": str(exc)},
            )
        )
        selected_index_for_checks[CANONICAL_BETA_PATH_COLUMN] = (
            selected_index_for_checks["beta_path"].astype(str).str.strip()
        )

    if CANONICAL_BETA_PATH_COLUMN in selected_index_for_checks.columns:
        selected_beta_records = (
            selected_index_for_checks[["sample_id", CANONICAL_BETA_PATH_COLUMN]]
            .astype(str)
            .rename(columns={CANONICAL_BETA_PATH_COLUMN: "beta_path"})
            .sort_values(by=["sample_id", "beta_path"], kind="mergesort")
            .to_dict(orient="records")
        )
    else:
        selected_beta_records = (
            selected_index_for_checks[["sample_id", "beta_path"]]
            .astype(str)
            .sort_values(by=["sample_id", "beta_path"], kind="mergesort")
            .to_dict(orient="records")
        )
    selected_beta_path_sha256 = _stable_json_sha256(selected_beta_records)

    mapping_integrity_summary = {
        "coarse_affect_mapping_versions": sorted(
            {
                str(value).strip()
                for value in selected_index_df.get("coarse_affect_mapping_version", pd.Series(dtype=object))
                if str(value).strip() and str(value).strip().lower() != "nan"
            }
        ),
        "coarse_affect_mapping_hashes": sorted(
            {
                str(value).strip()
                for value in selected_index_df.get("coarse_affect_mapping_sha256", pd.Series(dtype=object))
                if str(value).strip() and str(value).strip().lower() != "nan"
            }
        ),
        "binary_valence_mapping_versions": sorted(
            {
                str(value).strip()
                for value in selected_index_df.get("binary_valence_mapping_version", pd.Series(dtype=object))
                if str(value).strip() and str(value).strip().lower() != "nan"
            }
        ),
        "binary_valence_mapping_hashes": sorted(
            {
                str(value).strip()
                for value in selected_index_df.get("binary_valence_mapping_sha256", pd.Series(dtype=object))
                if str(value).strip() and str(value).strip().lower() != "nan"
            }
        ),
    }

    glm_unknown_regressor_summary = {
        "n_rows_with_unknown_regressors": 0,
        "total_unknown_regressor_count": 0,
        "sample_ids_head": [],
        "unknown_regressor_labels_head": [],
    }
    if {
        "glm_has_unknown_regressors",
        "glm_unknown_regressor_count",
        "glm_unknown_regressor_labels_json",
    } <= set(selected_index_df.columns):
        try:
            unknown_mask = selected_index_df["glm_has_unknown_regressors"].map(_normalize_bool)
        except ValueError as exc:
            blocking_list.append(
                _issue(
                    code="glm_unknown_regressors_boolean_invalid",
                    severity="blocking",
                    message="glm_has_unknown_regressors contains invalid boolean values.",
                    details={"error": str(exc)},
                )
            )
            unknown_mask = pd.Series(False, index=selected_index_df.index)
        unknown_rows = selected_index_df.loc[unknown_mask].copy()
        unknown_labels: list[str] = []
        for raw in unknown_rows.get("glm_unknown_regressor_labels_json", pd.Series(dtype=object)):
            try:
                parsed = json.loads(str(raw))
            except Exception:
                parsed = []
            if isinstance(parsed, list):
                unknown_labels.extend(str(value) for value in parsed)
        glm_unknown_regressor_summary = {
            "n_rows_with_unknown_regressors": int(len(unknown_rows)),
            "total_unknown_regressor_count": int(
                pd.to_numeric(
                    unknown_rows.get("glm_unknown_regressor_count", pd.Series(dtype=float)),
                    errors="coerce",
                ).fillna(0).sum()
            ),
            "sample_ids_head": unknown_rows.get("sample_id", pd.Series(dtype=object))
            .astype(str)
            .tolist()[:10],
            "unknown_regressor_labels_head": sorted(set(unknown_labels))[:20],
        }
        if int(len(unknown_rows)) > 0:
            blocking_list.append(
                _issue(
                    code="glm_unknown_regressors_present",
                    severity="blocking",
                    message="Selected subset contains rows with unknown GLM regressors.",
                    details=glm_unknown_regressor_summary,
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
        "mapping_integrity": mapping_integrity_summary,
        "glm_unknown_regressor_summary": glm_unknown_regressor_summary,
        "derived_label_inconsistency_summary": derived_label_inconsistency_summary,
        "feature_engineering": feature_engineering_summary,
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
    duplicate_sample_ids = int(
        selected_index_for_checks["sample_id"].astype(str).duplicated().sum()
    )
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

    canonical_beta_column = (
        CANONICAL_BETA_PATH_COLUMN
        if CANONICAL_BETA_PATH_COLUMN in selected_index_for_checks.columns
        else "beta_path"
    )
    duplicate_beta_paths = int(
        selected_index_for_checks[canonical_beta_column].astype(str).duplicated().sum()
    )
    duplicate_beta_status = "pass"
    if duplicate_beta_paths > 0:
        if data_policy.leakage.fail_on_duplicate_beta_path:
            duplicate_beta_status = "fail"
            blocking_list.append(
                _issue(
                    code="leakage_duplicate_beta_path",
                    severity="blocking",
                    message="Duplicate canonical beta_path values detected in selected subset.",
                    details={"count": duplicate_beta_paths},
                )
            )
        elif data_policy.leakage.warn_on_duplicate_beta_path:
            duplicate_beta_status = "warning"
            warnings_list.append(
                _issue(
                    code="leakage_duplicate_beta_path_warning",
                    severity="warning",
                    message="Duplicate canonical beta_path values detected in selected subset.",
                    details={"count": duplicate_beta_paths},
                )
            )
    leakage_checks.append(
        {
            "check": "duplicate_beta_path",
            "status": duplicate_beta_status,
            "count": duplicate_beta_paths,
            "canonicalization_status": beta_path_canonicalization_status,
            "canonicalization_error": beta_path_canonicalization_error,
            "blocking_policy": bool(data_policy.leakage.fail_on_duplicate_beta_path),
        }
    )

    duplicate_beta_hash_status = "pass"
    duplicate_beta_hash_count = 0
    missing_beta_hash_count = 0
    beta_hash_column = "beta_file_sha256"
    if beta_hash_column in selected_index_for_checks.columns:
        beta_hashes = selected_index_for_checks[beta_hash_column].astype(str).str.strip().str.lower()
        valid_hashes = beta_hashes[
            beta_hashes.str.fullmatch(r"[0-9a-f]{64}", na=False)
        ].copy()
        duplicate_beta_hash_count = int(valid_hashes.duplicated().sum())
        missing_beta_hash_count = int(len(beta_hashes) - len(valid_hashes))
    else:
        missing_beta_hash_count = int(len(selected_index_for_checks))

    if missing_beta_hash_count > 0:
        duplicate_beta_hash_status = "warning"
        warnings_list.append(
            _issue(
                code="leakage_duplicate_beta_content_hash_missing_warning",
                severity="warning",
                message=(
                    "Selected subset is missing valid beta_file_sha256 integrity values; "
                    "duplicate beta-content-hash checks are incomplete."
                ),
                details={"count": missing_beta_hash_count},
            )
        )
    elif duplicate_beta_hash_count > 0:
        if data_policy.leakage.fail_on_duplicate_beta_content_hash:
            duplicate_beta_hash_status = "fail"
            blocking_list.append(
                _issue(
                    code="leakage_duplicate_beta_content_hash",
                    severity="blocking",
                    message="Duplicate beta_file_sha256 values detected in selected subset.",
                    details={"count": duplicate_beta_hash_count},
                )
            )
        elif data_policy.leakage.warn_on_duplicate_beta_content_hash:
            duplicate_beta_hash_status = "warning"
            warnings_list.append(
                _issue(
                    code="leakage_duplicate_beta_content_hash_warning",
                    severity="warning",
                    message="Duplicate beta_file_sha256 values detected in selected subset.",
                    details={"count": duplicate_beta_hash_count},
                )
            )
    leakage_checks.append(
        {
            "check": "duplicate_beta_content_hash",
            "status": duplicate_beta_hash_status,
            "count": int(duplicate_beta_hash_count),
            "missing_hash_count": int(missing_beta_hash_count),
            "blocking_policy": bool(data_policy.leakage.fail_on_duplicate_beta_content_hash),
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

    try:
        cv_split_audit = _build_exact_cv_split_audit(
            selected_index_df=selected_index_df,
            target_column=target_column,
            cv_mode=cv_mode,
            subject=subject,
            train_subject=train_subject,
            test_subject=test_subject,
        )
    except ValueError as exc:
        cv_split_audit = {
            "status": "fail",
            "cv_mode": cv_mode,
            "planner_error": str(exc),
            "n_rows": int(len(selected_index_df)),
            "n_folds": 0,
            "expected_n_folds": 0,
            "missing_expected_test_rows": int(len(selected_index_df)),
            "unexpected_test_rows": 0,
            "duplicate_test_coverage_rows": 0,
            "failure_codes": ["split_planner_error"],
            "fold_rows": [],
        }

    try:
        cv_split_manifest = _build_cv_split_manifest(
            selected_index_df=selected_index_for_checks,
            target_column=target_column,
            cv_mode=cv_mode,
            subject=subject,
            train_subject=train_subject,
            test_subject=test_subject,
        )
    except ValueError as exc:
        cv_split_manifest = {
            "status": "fail",
            "schema_version": "cv-split-manifest-v1",
            "cv_mode": cv_mode,
            "target_column": target_column,
            "row_count": 0,
            "rows": [],
            "sha256": None,
            "planner_error": str(exc),
        }

    cv_split_status = (
        "fail"
        if cv_split_audit.get("status") == "fail" and data_policy.leakage.fail_on_cv_group_overlap
        else "pass"
    )
    leakage_checks.append(
        {
            "check": "cv_split_plan_exact",
            "status": cv_split_status,
            "blocking_policy": bool(data_policy.leakage.fail_on_cv_group_overlap),
            "n_folds": int(cv_split_audit.get("n_folds", 0)),
            "expected_n_folds": int(cv_split_audit.get("expected_n_folds", 0)),
            "missing_expected_test_rows": int(cv_split_audit.get("missing_expected_test_rows", 0)),
            "unexpected_test_rows": int(cv_split_audit.get("unexpected_test_rows", 0)),
            "duplicate_test_coverage_rows": int(
                cv_split_audit.get("duplicate_test_coverage_rows", 0)
            ),
            "failure_codes": list(cv_split_audit.get("failure_codes", [])),
        }
    )
    cv_split_manifest_status = (
        "fail"
        if cv_split_manifest.get("status") == "fail" and data_policy.leakage.fail_on_cv_group_overlap
        else "pass"
    )
    leakage_checks.append(
        {
            "check": "cv_split_manifest",
            "status": cv_split_manifest_status,
            "blocking_policy": bool(data_policy.leakage.fail_on_cv_group_overlap),
            "row_count": int(cv_split_manifest.get("row_count", 0)),
            "sha256": cv_split_manifest.get("sha256"),
            "planner_error": cv_split_manifest.get("planner_error"),
        }
    )
    if cv_split_status == "fail":
        blocking_list.append(
            _issue(
                code="leakage_cv_split_plan_invalid",
                severity="blocking",
                message="Exact CV split audit failed for selected subset.",
                details={
                    "cv_mode": cv_mode,
                    "planner_error": cv_split_audit.get("planner_error"),
                    "failure_codes": cv_split_audit.get("failure_codes", []),
                    "n_folds": int(cv_split_audit.get("n_folds", 0)),
                    "expected_n_folds": int(cv_split_audit.get("expected_n_folds", 0)),
                    "missing_expected_test_rows": int(
                        cv_split_audit.get("missing_expected_test_rows", 0)
                    ),
                    "unexpected_test_rows": int(cv_split_audit.get("unexpected_test_rows", 0)),
                    "duplicate_test_coverage_rows": int(
                        cv_split_audit.get("duplicate_test_coverage_rows", 0)
                    ),
                },
            )
        )
    if cv_split_manifest_status == "fail":
        blocking_list.append(
            _issue(
                code="leakage_cv_split_manifest_invalid",
                severity="blocking",
                message="CV split manifest generation failed for selected subset.",
                details={
                    "cv_mode": cv_mode,
                    "planner_error": cv_split_manifest.get("planner_error"),
                },
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
        "cv_split_audit": cv_split_audit,
        "cv_split_manifest_sha256": cv_split_manifest.get("sha256"),
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
        "derived_label_inconsistency_summary": derived_label_inconsistency_summary,
        "mapping_integrity": mapping_integrity_summary,
        "glm_unknown_regressor_summary": glm_unknown_regressor_summary,
        "cv_split_manifest_sha256": cv_split_manifest.get("sha256"),
        "feature_engineering": feature_engineering_summary,
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
        "target_derivation_summary": target_derivation_summary,
        "target_derivation_audit_rows": target_derivation_audit_rows,
        "derived_label_inconsistency_summary": derived_label_inconsistency_summary,
        "derived_label_inconsistency_audit_rows": derived_label_inconsistency_rows,
        "cv_split_audit": cv_split_audit,
        "cv_split_audit_rows": list(cv_split_audit.get("fold_rows", [])),
        "cv_split_manifest": cv_split_manifest,
        "cv_split_manifest_rows": list(cv_split_manifest.get("rows", [])),
        "cv_split_manifest_sha256": cv_split_manifest.get("sha256"),
        "selected_beta_path_sha256": selected_beta_path_sha256,
        "selection_exclusion_summary": selection_exclusion_summary,
        "selection_exclusion_rows": selection_exclusion_rows,
        "feature_engineering": feature_engineering_summary,
    }


def _dataset_card_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Dataset Card",
        "",
        f"- Framework mode: `{payload.get('framework_mode')}`",
        f"- Target: `{payload.get('target_definition', {}).get('target_column')}`",
        f"- Feature recipe: `{payload.get('feature_engineering', {}).get('feature_recipe_id')}`",
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
    cv_split_audit_json_path = report_dir / "cv_split_audit.json"
    cv_split_audit_csv_path = report_dir / "cv_split_audit.csv"
    cv_split_manifest_json_path = report_dir / "cv_split_manifest.json"
    cv_split_manifest_csv_path = report_dir / "cv_split_manifest.csv"
    external_dataset_card_path = report_dir / "external_dataset_card.json"
    external_dataset_summary_path = report_dir / "external_dataset_summary.json"
    external_validation_compatibility_path = report_dir / "external_validation_compatibility.json"
    target_derivation_audit_json_path = report_dir / "target_derivation_audit.json"
    target_derivation_audit_csv_path = report_dir / "target_derivation_audit.csv"
    selection_exclusion_summary_path = report_dir / "selection_exclusion_summary.json"
    selection_exclusion_manifest_csv_path = report_dir / "selection_exclusion_manifest.csv"

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
    selection_exclusion_summary = dict(assessment.get("selection_exclusion_summary", {}))
    selection_exclusion_summary_path.write_text(
        f"{json.dumps(selection_exclusion_summary, indent=2)}\n",
        encoding="utf-8",
    )
    pd.DataFrame(list(assessment.get("selection_exclusion_rows", []))).to_csv(
        selection_exclusion_manifest_csv_path,
        index=False,
    )
    cv_split_payload = dict(assessment.get("cv_split_audit", {}))
    cv_split_audit_json_path.write_text(
        f"{json.dumps(cv_split_payload, indent=2)}\n",
        encoding="utf-8",
    )
    pd.DataFrame(list(assessment.get("cv_split_audit_rows", []))).to_csv(
        cv_split_audit_csv_path,
        index=False,
    )
    cv_split_manifest_payload = dict(assessment.get("cv_split_manifest", {}))
    cv_split_manifest_json_path.write_text(
        f"{json.dumps(cv_split_manifest_payload, indent=2)}\n",
        encoding="utf-8",
    )
    pd.DataFrame(list(assessment.get("cv_split_manifest_rows", []))).to_csv(
        cv_split_manifest_csv_path,
        index=False,
    )
    quality_payload = dict(assessment.get("data_quality_report", {}))
    data_quality_report_path.write_text(
        f"{json.dumps(quality_payload, indent=2)}\n",
        encoding="utf-8",
    )

    target_derivation_rows = list(assessment.get("target_derivation_audit_rows", []))
    target_derivation_audit_json_path.write_text(
        f"{json.dumps(target_derivation_rows, indent=2)}\n",
        encoding="utf-8",
    )
    pd.DataFrame(target_derivation_rows).to_csv(
        target_derivation_audit_csv_path,
        index=False,
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
    dataset_fingerprint_payload = (
        dict(dataset_fingerprint) if isinstance(dataset_fingerprint, dict) else {}
    )
    selected_beta_path_sha256 = assessment.get("selected_beta_path_sha256")
    cv_split_manifest_sha256 = assessment.get("cv_split_manifest_sha256")
    if selected_beta_path_sha256:
        dataset_fingerprint_payload["selected_beta_path_sha256"] = str(
            selected_beta_path_sha256
        )
    if cv_split_manifest_sha256:
        dataset_fingerprint_payload["cv_split_manifest_sha256"] = str(
            cv_split_manifest_sha256
        )

    dataset_card_payload = {
        "schema_version": "official-dataset-card-v1",
        "framework_mode": framework_mode.value,
        "dataset_identity": {
            "index_csv": str(index_csv.resolve()),
            "data_root": str(data_root.resolve()),
            "cache_dir": str(cache_dir.resolve()),
            "dataset_fingerprint": dataset_fingerprint_payload or None,
        },
        "unit_of_analysis": sample_unit or "beta_event",
        "target_definition": {
            "target_column": target_column,
            "label_policy": label_policy,
            "target_mapping_version": target_mapping_version,
            "target_mapping_hash": target_mapping_hash,
            "mapping_integrity": assessment.get("dataset_summary", {})
            .get("mapping_integrity", {}),
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
            "cv_split_audit_summary": {
                "status": cv_split_payload.get("status"),
                "n_folds": cv_split_payload.get("n_folds"),
                "expected_n_folds": cv_split_payload.get("expected_n_folds"),
                "missing_expected_test_rows": cv_split_payload.get(
                    "missing_expected_test_rows"
                ),
                "unexpected_test_rows": cv_split_payload.get("unexpected_test_rows"),
                "duplicate_test_coverage_rows": cv_split_payload.get(
                    "duplicate_test_coverage_rows"
                ),
                "failure_codes": cv_split_payload.get("failure_codes", []),
            },
            "cv_split_manifest_sha256": cv_split_manifest_payload.get("sha256"),
        },
        "glm_unknown_regressor_summary": assessment.get("dataset_summary", {}).get(
            "glm_unknown_regressor_summary", {}
        ),
        "derived_label_inconsistency_summary": assessment.get(
            "derived_label_inconsistency_summary", {}
        ),
        "feature_engineering": assessment.get("feature_engineering", {}),
        "external_validation": external_compatibility,
        "intended_use": data_policy_effective.get("intended_use"),
        "not_intended_use": data_policy_effective.get("not_intended_use", []),
        "known_limitations": data_policy_effective.get("known_limitations", []),
        "selection_scope": {
            "cv_mode": cv_mode,
            "subject": subject,
            "train_subject": train_subject,
            "test_subject": test_subject,
            "filter_task": filter_task,
            "filter_modality": filter_modality,
            "selection_exclusions": dict(assessment.get("selection_exclusion_summary", {})),
        },
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
        "cv_split_audit_json": str(cv_split_audit_json_path.resolve()),
        "cv_split_audit_csv": str(cv_split_audit_csv_path.resolve()),
        "cv_split_manifest_json": str(cv_split_manifest_json_path.resolve()),
        "cv_split_manifest_csv": str(cv_split_manifest_csv_path.resolve()),
        "cv_split_manifest_sha256": str(cv_split_manifest_payload.get("sha256", "")),
        "external_dataset_card_json": str(external_dataset_card_path.resolve()),
        "external_dataset_summary_json": str(external_dataset_summary_path.resolve()),
        "external_validation_compatibility_json": str(
            external_validation_compatibility_path.resolve()
        ),
        "target_derivation_audit_json": str(target_derivation_audit_json_path.resolve()),
        "target_derivation_audit_csv": str(target_derivation_audit_csv_path.resolve()),
        "selection_exclusion_summary_json": str(selection_exclusion_summary_path.resolve()),
        "selection_exclusion_manifest_csv": str(selection_exclusion_manifest_csv_path.resolve()),
    }
    return {
        "data_policy_effective": data_policy_effective,
        "data_artifacts": data_artifacts,
        "data_quality_report": quality_payload,
        "dataset_fingerprint": dataset_fingerprint_payload or None,
    }


__all__ = [
    "evaluate_official_data_policy",
    "write_official_data_artifacts",
]
