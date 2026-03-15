from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.metric_policy import validate_metric_name
from Thesis_ML.data.affect_labels import with_coarse_affect
from Thesis_ML.experiments.errors import (
    OfficialArtifactContractError,
    OfficialContractValidationError,
)

_REQUIRED_INDEX_COLUMNS = {
    "sample_id",
    "subject",
    "session",
    "task",
    "modality",
    "beta_path",
    "mask_path",
    "regressor_label",
    "emotion",
}
_SUBGROUP_COLUMN_MAP = {
    "label": None,
    "task": "task",
    "modality": "modality",
    "session": "session",
    "subject": "subject",
}
_REQUIRED_RUN_ARTIFACT_MINIMUM = {"config.json", "metrics.json"}
_REQUIRED_RUN_METADATA_MINIMUM = {"framework_mode", "canonical_run"}


@dataclass(frozen=True)
class OfficialPreflightResult:
    selected_index_df: pd.DataFrame
    index_row_count: int
    required_run_artifacts: list[str]
    required_run_metadata_fields: list[str]


def _require_columns(frame: pd.DataFrame, *, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise OfficialContractValidationError(
            f"{label} is missing required columns: {', '.join(missing)}.",
            details={"missing_columns": missing},
        )


def _resolve_official_requirements(
    *,
    official_context: dict[str, Any],
) -> tuple[list[str], list[str]]:
    raw_artifacts = official_context.get("artifact_requirements", [])
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise OfficialContractValidationError(
            "Official run context must provide non-empty artifact_requirements.",
            details={"artifact_requirements": raw_artifacts},
        )
    artifact_requirements = [str(value) for value in raw_artifacts]
    missing_artifacts = sorted(_REQUIRED_RUN_ARTIFACT_MINIMUM - set(artifact_requirements))
    if missing_artifacts:
        raise OfficialContractValidationError(
            "Official run artifact_requirements are missing required entries: "
            + ", ".join(missing_artifacts),
            details={"missing_artifacts": missing_artifacts},
        )

    raw_metadata_fields = official_context.get("required_run_metadata_fields", [])
    if not isinstance(raw_metadata_fields, list) or not raw_metadata_fields:
        raise OfficialContractValidationError(
            "Official run context must provide non-empty required_run_metadata_fields.",
            details={"required_run_metadata_fields": raw_metadata_fields},
        )
    metadata_fields = [str(value) for value in raw_metadata_fields]
    missing_metadata = sorted(_REQUIRED_RUN_METADATA_MINIMUM - set(metadata_fields))
    if missing_metadata:
        raise OfficialContractValidationError(
            "Official run required_run_metadata_fields are missing required entries: "
            + ", ".join(missing_metadata),
            details={"missing_metadata_fields": missing_metadata},
        )

    return artifact_requirements, metadata_fields


def validate_official_preflight(
    *,
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
    n_permutations: int,
    primary_metric_name: str,
    permutation_metric_name: str,
    methodology_policy_name: str,
    model: str,
    tuning_enabled: bool,
    tuning_search_space_id: str | None,
    tuning_search_space_version: str | None,
    tuning_inner_group_field: str | None,
    subgroup_reporting_enabled: bool,
    subgroup_dimensions: list[str],
    official_context: dict[str, Any],
) -> OfficialPreflightResult:
    if framework_mode not in {FrameworkMode.CONFIRMATORY, FrameworkMode.LOCKED_COMPARISON}:
        raise OfficialContractValidationError(
            f"validate_official_preflight called with non-official framework_mode='{framework_mode.value}'.",
            details={"framework_mode": framework_mode.value},
        )

    if not index_csv.exists() or not index_csv.is_file():
        raise OfficialContractValidationError(
            f"index_csv must exist as a file for official runs: {index_csv}",
            details={"index_csv": str(index_csv)},
        )
    if not data_root.exists() or not data_root.is_dir():
        raise OfficialContractValidationError(
            f"data_root must exist as a directory for official runs: {data_root}",
            details={"data_root": str(data_root)},
        )
    if cache_dir.exists() and not cache_dir.is_dir():
        raise OfficialContractValidationError(
            f"cache_dir exists but is not a directory: {cache_dir}",
            details={"cache_dir": str(cache_dir)},
        )

    artifact_requirements, metadata_fields = _resolve_official_requirements(
        official_context=official_context,
    )

    frame = pd.read_csv(index_csv)
    index_row_count = int(frame.shape[0])
    if frame.empty:
        raise OfficialContractValidationError(
            f"Dataset index is empty for official run: {index_csv}",
            details={"index_csv": str(index_csv)},
        )

    frame = with_coarse_affect(frame, emotion_column="emotion", coarse_column="coarse_affect")
    _require_columns(frame, required=_REQUIRED_INDEX_COLUMNS | {target_column}, label="Dataset index")

    selected = frame.copy()
    if filter_task is not None:
        selected = selected[selected["task"].astype(str) == str(filter_task)].copy()
    if filter_modality is not None:
        selected = selected[selected["modality"].astype(str) == str(filter_modality)].copy()

    if cv_mode == "within_subject_loso_session":
        selected = selected[selected["subject"].astype(str) == str(subject)].copy()
    elif cv_mode == "frozen_cross_person_transfer":
        if train_subject is None or test_subject is None:
            raise OfficialContractValidationError(
                "frozen_cross_person_transfer requires train_subject and test_subject.",
                details={"train_subject": train_subject, "test_subject": test_subject},
            )
        if str(train_subject) == str(test_subject):
            raise OfficialContractValidationError(
                "frozen_cross_person_transfer requires train_subject and test_subject to differ.",
                details={"train_subject": train_subject, "test_subject": test_subject},
            )
        selected = selected[
            selected["subject"].astype(str).isin({str(train_subject), str(test_subject)})
        ].copy()

    selected = selected.dropna(subset=[target_column]).copy()
    selected[target_column] = selected[target_column].astype(str)
    if selected.empty:
        raise OfficialContractValidationError(
            "Official run filtering produced an empty dataset subset.",
            details={
                "cv_mode": cv_mode,
                "subject": subject,
                "train_subject": train_subject,
                "test_subject": test_subject,
                "filter_task": filter_task,
                "filter_modality": filter_modality,
            },
        )

    if cv_mode == "within_subject_loso_session":
        unique_subjects = sorted(selected["subject"].astype(str).unique().tolist())
        if len(unique_subjects) != 1:
            raise OfficialContractValidationError(
                "within_subject_loso_session official subset must contain exactly one subject.",
                details={"subjects": unique_subjects},
            )
        n_sessions = int(selected["session"].astype(str).nunique(dropna=False))
        if n_sessions < 2:
            raise OfficialContractValidationError(
                "within_subject_loso_session requires at least two sessions in selected subset.",
                details={"n_sessions": n_sessions},
            )
    if cv_mode == "frozen_cross_person_transfer":
        subjects_present = set(selected["subject"].astype(str).unique().tolist())
        missing_subjects = [
            value
            for value in (str(train_subject), str(test_subject))
            if value not in subjects_present
        ]
        if missing_subjects:
            raise OfficialContractValidationError(
                "frozen_cross_person_transfer selected subset is missing required subject(s): "
                + ", ".join(sorted(missing_subjects)),
                details={"subjects_present": sorted(subjects_present)},
            )

    n_classes = int(selected[target_column].astype(str).nunique(dropna=False))
    if n_classes < 2:
        raise OfficialContractValidationError(
            "Official run subset must contain at least two target classes.",
            details={"target_column": target_column, "n_classes": n_classes},
        )

    if subgroup_reporting_enabled:
        missing_subgroup_columns: list[str] = []
        for subgroup_dimension in subgroup_dimensions:
            mapped_column = _SUBGROUP_COLUMN_MAP.get(subgroup_dimension)
            if mapped_column is None:
                continue
            if mapped_column not in selected.columns:
                missing_subgroup_columns.append(mapped_column)
        if missing_subgroup_columns:
            raise OfficialContractValidationError(
                "Official subgroup reporting is enabled but required columns are missing: "
                + ", ".join(sorted(set(missing_subgroup_columns))),
                details={"subgroup_dimensions": subgroup_dimensions},
            )

    resolved_primary_metric = validate_metric_name(primary_metric_name)
    resolved_permutation_metric = validate_metric_name(permutation_metric_name)
    if int(n_permutations) > 0 and resolved_primary_metric != resolved_permutation_metric:
        raise OfficialContractValidationError(
            "Official runs require permutation metric to match primary metric.",
            details={
                "primary_metric": resolved_primary_metric,
                "permutation_metric": resolved_permutation_metric,
            },
        )

    if tuning_enabled and methodology_policy_name == "grouped_nested_tuning" and model != "dummy":
        missing_tuning_fields = [
            key
            for key, value in {
                "tuning_search_space_id": tuning_search_space_id,
                "tuning_search_space_version": tuning_search_space_version,
                "tuning_inner_group_field": tuning_inner_group_field,
            }.items()
            if value is None or str(value).strip() == ""
        ]
        if missing_tuning_fields:
            raise OfficialContractValidationError(
                "grouped_nested_tuning official run is missing required tuning fields: "
                + ", ".join(sorted(missing_tuning_fields)),
                details={"missing_tuning_fields": missing_tuning_fields},
            )
        inner_group = str(tuning_inner_group_field)
        if inner_group not in selected.columns:
            raise OfficialContractValidationError(
                "grouped_nested_tuning inner group field is missing from selected dataset subset.",
                details={
                    "inner_group_field": inner_group,
                    "available_columns": sorted(selected.columns.tolist()),
                },
            )

    return OfficialPreflightResult(
        selected_index_df=selected,
        index_row_count=index_row_count,
        required_run_artifacts=artifact_requirements,
        required_run_metadata_fields=metadata_fields,
    )


def validate_run_artifact_contract(
    *,
    report_dir: Path,
    required_run_artifacts: list[str],
    required_run_metadata_fields: list[str],
    framework_mode: FrameworkMode,
    canonical_run: bool,
    config_payload: dict[str, Any],
    metrics_payload: dict[str, Any],
) -> None:
    missing_files = [
        artifact_name
        for artifact_name in required_run_artifacts
        if not (report_dir / artifact_name).exists()
    ]
    if missing_files:
        raise OfficialArtifactContractError(
            "Official run artifact contract failed: missing required files: "
            + ", ".join(sorted(missing_files)),
            details={"missing_files": sorted(missing_files), "report_dir": str(report_dir)},
        )

    missing_metadata_keys = [
        key
        for key in required_run_metadata_fields
        if key not in config_payload or key not in metrics_payload
    ]
    if missing_metadata_keys:
        raise OfficialArtifactContractError(
            "Official run artifact contract failed: missing required metadata keys in config/metrics: "
            + ", ".join(sorted(missing_metadata_keys)),
            details={"missing_metadata_keys": sorted(missing_metadata_keys)},
        )

    expected_mode = framework_mode.value
    if str(config_payload.get("framework_mode")) != expected_mode:
        raise OfficialArtifactContractError(
            "Official run config framework_mode drift detected.",
            details={
                "expected": expected_mode,
                "actual": config_payload.get("framework_mode"),
            },
        )
    if str(metrics_payload.get("framework_mode")) != expected_mode:
        raise OfficialArtifactContractError(
            "Official run metrics framework_mode drift detected.",
            details={
                "expected": expected_mode,
                "actual": metrics_payload.get("framework_mode"),
            },
        )
    if bool(config_payload.get("canonical_run")) is not bool(canonical_run):
        raise OfficialArtifactContractError(
            "Official run config canonical_run drift detected.",
            details={"expected": bool(canonical_run), "actual": config_payload.get("canonical_run")},
        )
    if bool(metrics_payload.get("canonical_run")) is not bool(canonical_run):
        raise OfficialArtifactContractError(
            "Official run metrics canonical_run drift detected.",
            details={"expected": bool(canonical_run), "actual": metrics_payload.get("canonical_run")},
        )

    config_metric_policy = config_payload.get("metric_policy_effective")
    metrics_metric_policy = metrics_payload.get("metric_policy_effective")
    if not isinstance(config_metric_policy, dict) or not isinstance(metrics_metric_policy, dict):
        raise OfficialArtifactContractError(
            "Official run artifacts must include metric_policy_effective in both config and metrics.",
            details={
                "config_metric_policy_type": type(config_metric_policy).__name__,
                "metrics_metric_policy_type": type(metrics_metric_policy).__name__,
            },
        )

    primary_metric = config_metric_policy.get("primary_metric")
    permutation_payload = metrics_payload.get("permutation_test")
    if isinstance(permutation_payload, dict) and primary_metric is not None:
        metric_name = permutation_payload.get("metric_name")
        if isinstance(metric_name, str) and validate_metric_name(metric_name) != validate_metric_name(
            str(primary_metric)
        ):
            raise OfficialArtifactContractError(
                "Permutation metric in metrics artifact does not match effective primary metric.",
                details={
                    "primary_metric": primary_metric,
                    "permutation_metric": metric_name,
                },
            )


__all__ = [
    "OfficialPreflightResult",
    "validate_official_preflight",
    "validate_run_artifact_contract",
]
