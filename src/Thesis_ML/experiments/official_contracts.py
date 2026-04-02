from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.metric_policy import (
    classification_metric_score as policy_metric_score,
)
from Thesis_ML.config.metric_policy import (
    validate_metric_name,
)
from Thesis_ML.data.affect_labels import (
    blocking_derived_label_inconsistency_rows,
    blocking_target_derivation_audit_rows,
    build_derived_label_inconsistency_audit,
    build_target_derivation_audit,
    summarize_derived_label_inconsistency_audit,
    summarize_target_derivation_audit,
    with_binary_valence_like,
    with_coarse_affect,
)
from Thesis_ML.data.index_validation import (
    DatasetIndexValidationError,
    validate_dataset_index_strict,
)
from Thesis_ML.experiments.data_reporting import evaluate_official_data_policy
from Thesis_ML.experiments.errors import (
    OfficialArtifactContractError,
    OfficialContractValidationError,
)
from Thesis_ML.experiments.model_admission import model_is_official
from Thesis_ML.experiments.model_registry import get_model_spec, official_model_names
from Thesis_ML.experiments.selection_manifest import apply_dataset_selection_filters

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
_TARGET_SOURCE_COLUMN_MAP = {
    "coarse_affect": "emotion",
    "binary_valence_like": "coarse_affect",
}
_REQUIRED_RUN_ARTIFACT_MINIMUM = {"config.json", "metrics.json"}
_REQUIRED_RUN_METADATA_MINIMUM = {"framework_mode", "canonical_run"}
_PRIMARY_METRIC_AGGREGATIONS = {"mean_fold_scores", "pooled_held_out_predictions"}
_PRIMARY_METRIC_MATCH_ATOL = 1e-9


@dataclass(frozen=True)
class OfficialPreflightResult:
    selected_index_df: pd.DataFrame
    index_row_count: int
    required_run_artifacts: list[str]
    required_run_metadata_fields: list[str]
    data_policy_effective: dict[str, Any]
    data_assessment: dict[str, Any]


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
    feature_space: str | None = None,
    roi_spec_path: Path | None = None,
    dimensionality_strategy: str | None = None,
    pca_n_components: int | None = None,
    pca_variance_ratio: float | None = None,
    n_permutations: int,
    primary_metric_name: str,
    permutation_metric_name: str,
    methodology_policy_name: str,
    class_weight_policy: str,
    model: str,
    tuning_enabled: bool,
    tuning_search_space_id: str | None,
    tuning_search_space_version: str | None,
    tuning_inner_group_field: str | None,
    subgroup_reporting_enabled: bool,
    subgroup_dimensions: list[str],
    subgroup_min_samples_per_group: int,
    subgroup_min_classes_per_group: int,
    subgroup_report_small_groups: bool,
    official_context: dict[str, Any],
) -> OfficialPreflightResult:
    if framework_mode not in {FrameworkMode.CONFIRMATORY, FrameworkMode.LOCKED_COMPARISON}:
        raise OfficialContractValidationError(
            f"validate_official_preflight called with non-official framework_mode='{framework_mode.value}'.",
            details={"framework_mode": framework_mode.value},
        )
    if not model_is_official(model):
        raise OfficialContractValidationError(
            f"Model '{model}' is exploratory-only and not admitted for official runs.",
            details={
                "model": str(model),
                "allowed_official_models": sorted(official_model_names()),
            },
        )
    model_spec = get_model_spec(model)
    normalized_class_weight_policy = str(class_weight_policy).strip().lower()
    if normalized_class_weight_policy not in set(model_spec.supported_class_weight_policies):
        raise OfficialContractValidationError(
            f"Model '{model_spec.logical_name}' does not support class_weight_policy="
            f"'{class_weight_policy}' on official paths.",
            details={
                "model": model_spec.logical_name,
                "class_weight_policy": normalized_class_weight_policy,
                "supported_class_weight_policies": list(
                    model_spec.supported_class_weight_policies
                ),
            },
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

    try:
        frame = validate_dataset_index_strict(
            frame,
            data_root=data_root,
            required_columns=_REQUIRED_INDEX_COLUMNS | {target_column},
            require_integrity_columns=True,
        )
    except DatasetIndexValidationError as exc:
        raise OfficialContractValidationError(
            f"Official strict dataset-index validation failed: {exc}",
            details={"error": str(exc), "index_csv": str(index_csv)},
        ) from exc

    derived_label_inconsistency_audit_df = build_derived_label_inconsistency_audit(
        frame,
        emotion_column="emotion",
        coarse_column="coarse_affect",
        binary_column="binary_valence_like",
    )
    blocking_derived_label_inconsistency_df = blocking_derived_label_inconsistency_rows(
        derived_label_inconsistency_audit_df
    )
    if not blocking_derived_label_inconsistency_df.empty:
        summary = summarize_derived_label_inconsistency_audit(
            blocking_derived_label_inconsistency_df
        )
        raise OfficialContractValidationError(
            "Official run dataset index contains inconsistent stored derived labels.",
            details={
                "n_problem_rows": summary["n_rows"],
                "by_category": summary["by_category"],
                "sample_ids_head": summary["sample_ids_head"],
            },
        )

    frame = with_coarse_affect(
        frame,
        emotion_column="emotion",
        coarse_column="coarse_affect",
        strict_recompute=True,
        attach_mapping_metadata=False,
    )
    frame = with_binary_valence_like(
        frame,
        coarse_column="coarse_affect",
        binary_column="binary_valence_like",
        strict_recompute=True,
        attach_mapping_metadata=False,
    )

    unknown_column = "glm_has_unknown_regressors"
    if unknown_column in frame.columns:
        normalized_unknown = (
            frame[unknown_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
        )
        if bool(normalized_unknown.isna().any()):
            raise OfficialContractValidationError(
                "Official run dataset index has invalid glm_has_unknown_regressors values.",
                details={"column": unknown_column},
            )
        if bool(normalized_unknown.any()):
            affected = frame.loc[normalized_unknown, "sample_id"].astype(str).tolist()[:10]
            raise OfficialContractValidationError(
                "Official run blocked: dataset index indicates unknown GLM regressors.",
                details={
                    "n_rows_with_unknown_regressors": int(normalized_unknown.sum()),
                    "sample_ids_head": affected,
                },
            )

    _require_columns(
        frame,
        required=_REQUIRED_INDEX_COLUMNS | {target_column},
        label="Dataset index",
    )

    target_derivation_audit_df = build_target_derivation_audit(
        frame,
        target_column=target_column,
    )
    blocking_target_audit_df = blocking_target_derivation_audit_rows(target_derivation_audit_df)
    if not blocking_target_audit_df.empty:
        summary = summarize_target_derivation_audit(blocking_target_audit_df)
        raise OfficialContractValidationError(
            "Official run has unsupported or missing source labels for a derived target.",
            details={
                "target_column": target_column,
                "n_problem_rows": summary["n_rows"],
                "by_category": summary["by_category"],
                "sample_ids_head": summary["sample_ids_head"],
            },
        )

    scope_frame = frame.copy()
    if filter_task is not None:
        scope_frame = scope_frame[scope_frame["task"].astype(str) == str(filter_task)].copy()
    if filter_modality is not None:
        scope_frame = scope_frame[
            scope_frame["modality"].astype(str) == str(filter_modality)
        ].copy()

    selection_result = apply_dataset_selection_filters(
        frame,
        target_column=target_column,
        cv_mode=cv_mode,
        subject=subject,
        train_subject=train_subject,
        test_subject=test_subject,
        filter_task=filter_task,
        filter_modality=filter_modality,
    )
    selected = selection_result.selected_index_df

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
                "selection_summary": selection_result.selection_summary,
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

    if framework_mode == FrameworkMode.CONFIRMATORY:
        confirmatory_lock = official_context.get("confirmatory_lock")
        if isinstance(confirmatory_lock, dict):
            analysis_status = str(confirmatory_lock.get("analysis_status", "")).strip().lower()
            if analysis_status != "locked":
                raise OfficialContractValidationError(
                    "Confirmatory freeze requires analysis_status='locked'.",
                    details={"analysis_status": analysis_status},
                )

            expected_target = str(confirmatory_lock.get("target_name", "")).strip()
            if expected_target and expected_target != str(target_column):
                raise OfficialContractValidationError(
                    "Confirmatory runtime target differs from locked protocol target.",
                    details={"expected_target": expected_target, "actual_target": target_column},
                )

            expected_source_column = str(confirmatory_lock.get("target_source_column", "")).strip()
            actual_source_column = _TARGET_SOURCE_COLUMN_MAP.get(str(target_column))
            if expected_source_column and actual_source_column != expected_source_column:
                raise OfficialContractValidationError(
                    "Confirmatory runtime source column differs from locked protocol source column.",
                    details={
                        "expected_source_column": expected_source_column,
                        "actual_source_column": actual_source_column,
                    },
                )
            if expected_source_column and expected_source_column not in frame.columns:
                raise OfficialContractValidationError(
                    "Confirmatory locked source column is missing from dataset index.",
                    details={"source_column": expected_source_column},
                )

            expected_mapping_version = str(
                confirmatory_lock.get("target_mapping_version", "")
            ).strip()
            context_mapping_version = str(
                official_context.get("target_mapping_version", "")
            ).strip()
            if expected_mapping_version and context_mapping_version != expected_mapping_version:
                raise OfficialContractValidationError(
                    "Confirmatory runtime target mapping version differs from locked protocol mapping version.",
                    details={
                        "expected_target_mapping_version": expected_mapping_version,
                        "actual_target_mapping_version": context_mapping_version,
                    },
                )
            expected_mapping_hash = str(confirmatory_lock.get("target_mapping_hash", "")).strip().lower()
            if str(target_column).strip() == "coarse_affect":
                required_mapping_columns = [
                    "coarse_affect_mapping_version",
                    "coarse_affect_mapping_sha256",
                ]
                missing_mapping_columns = [
                    column_name
                    for column_name in required_mapping_columns
                    if column_name not in selected.columns
                ]
                if missing_mapping_columns:
                    raise OfficialContractValidationError(
                        "Confirmatory coarse_affect selected rows are missing mapping metadata columns.",
                        details={"missing_mapping_columns": missing_mapping_columns},
                    )

                version_series = selected["coarse_affect_mapping_version"].map(
                    lambda value: str(value).strip() if not pd.isna(value) else ""
                )
                hash_series = selected["coarse_affect_mapping_sha256"].map(
                    lambda value: str(value).strip().lower() if not pd.isna(value) else ""
                )
                if bool((version_series == "").any()):
                    raise OfficialContractValidationError(
                        "Confirmatory coarse_affect selected rows must provide coarse_affect_mapping_version.",
                        details={
                            "missing_version_rows": int((version_series == "").sum()),
                        },
                    )
                if bool((hash_series == "").any()):
                    raise OfficialContractValidationError(
                        "Confirmatory coarse_affect selected rows must provide coarse_affect_mapping_sha256.",
                        details={
                            "missing_hash_rows": int((hash_series == "").sum()),
                        },
                    )

                selected_versions = sorted(set(version_series.tolist()))
                selected_hashes = sorted(set(hash_series.tolist()))
                if len(selected_versions) != 1:
                    raise OfficialContractValidationError(
                        "Confirmatory coarse_affect selected rows must have exactly one unique mapping version.",
                        details={"selected_mapping_versions": selected_versions},
                    )
                if len(selected_hashes) != 1:
                    raise OfficialContractValidationError(
                        "Confirmatory coarse_affect selected rows must have exactly one unique mapping hash.",
                        details={"selected_mapping_hashes": selected_hashes},
                    )

                selected_mapping_version = selected_versions[0]
                selected_mapping_hash = selected_hashes[0]
                if (
                    expected_mapping_version
                    and selected_mapping_version != expected_mapping_version
                ):
                    raise OfficialContractValidationError(
                        "Confirmatory coarse_affect selected mapping version differs from locked protocol mapping version.",
                        details={
                            "expected_target_mapping_version": expected_mapping_version,
                            "selected_target_mapping_version": selected_mapping_version,
                        },
                    )
                if expected_mapping_hash and selected_mapping_hash != expected_mapping_hash:
                    raise OfficialContractValidationError(
                        "Confirmatory coarse_affect selected mapping hash differs from locked protocol mapping hash.",
                        details={
                            "expected_target_mapping_hash": expected_mapping_hash,
                            "selected_target_mapping_hash": selected_mapping_hash,
                        },
                    )

            expected_split = str(confirmatory_lock.get("split", "")).strip()
            if expected_split and expected_split != str(cv_mode):
                raise OfficialContractValidationError(
                    "Confirmatory runtime split differs from locked protocol split.",
                    details={"expected_split": expected_split, "actual_split": cv_mode},
                )

            expected_primary_metric = str(confirmatory_lock.get("primary_metric", "")).strip()
            if expected_primary_metric and validate_metric_name(
                expected_primary_metric
            ) != validate_metric_name(primary_metric_name):
                raise OfficialContractValidationError(
                    "Confirmatory runtime primary metric differs from locked protocol metric.",
                    details={
                        "expected_primary_metric": expected_primary_metric,
                        "actual_primary_metric": primary_metric_name,
                    },
                )

            controls_payload = official_context.get("controls")
            dummy_baseline_run = isinstance(controls_payload, dict) and bool(
                controls_payload.get("dummy_baseline_run", False)
            )
            expected_model = str(confirmatory_lock.get("model_family", "")).strip()
            if dummy_baseline_run:
                if str(model) != "dummy":
                    raise OfficialContractValidationError(
                        "Confirmatory dummy baseline control run must use model='dummy'.",
                        details={"actual_model": model},
                    )
            elif expected_model and expected_model != str(model):
                raise OfficialContractValidationError(
                    "Confirmatory runtime model differs from locked protocol model family.",
                    details={"expected_model": expected_model, "actual_model": model},
                )

            expected_hyperparameter_policy = str(
                confirmatory_lock.get("hyperparameter_policy", "")
            ).strip()
            if expected_hyperparameter_policy == "fixed":
                if methodology_policy_name != "fixed_baselines_only" or tuning_enabled:
                    raise OfficialContractValidationError(
                        "Confirmatory runtime hyperparameter policy differs from locked policy.",
                        details={
                            "expected_hyperparameter_policy": expected_hyperparameter_policy,
                            "methodology_policy_name": methodology_policy_name,
                            "tuning_enabled": bool(tuning_enabled),
                        },
                    )
            if expected_hyperparameter_policy == "grouped_nested_tuning":
                if methodology_policy_name != "grouped_nested_tuning" or not tuning_enabled:
                    raise OfficialContractValidationError(
                        "Confirmatory runtime hyperparameter policy differs from locked policy.",
                        details={
                            "expected_hyperparameter_policy": expected_hyperparameter_policy,
                            "methodology_policy_name": methodology_policy_name,
                            "tuning_enabled": bool(tuning_enabled),
                        },
                    )

            expected_class_weight_policy = str(
                confirmatory_lock.get("class_weight_policy", "")
            ).strip()
            if expected_class_weight_policy and expected_class_weight_policy != str(
                class_weight_policy
            ):
                raise OfficialContractValidationError(
                    "Confirmatory runtime class-weight policy differs from locked protocol.",
                    details={
                        "expected_class_weight_policy": expected_class_weight_policy,
                        "actual_class_weight_policy": class_weight_policy,
                    },
                )

            expected_required_columns = [
                str(value) for value in list(confirmatory_lock.get("required_index_columns", []))
            ]
            if expected_required_columns:
                _require_columns(
                    frame,
                    required=set(expected_required_columns),
                    label="Confirmatory dataset index",
                )

            minimum_subjects = int(confirmatory_lock.get("minimum_subjects", 1))
            subjects_in_scope = sorted(scope_frame["subject"].astype(str).unique().tolist())
            if len(subjects_in_scope) < minimum_subjects:
                raise OfficialContractValidationError(
                    "Confirmatory dataset scope does not meet minimum_subjects.",
                    details={
                        "minimum_subjects": minimum_subjects,
                        "subjects_in_scope": subjects_in_scope,
                    },
                )

            minimum_sessions_per_subject = int(
                confirmatory_lock.get("minimum_sessions_per_subject", 1)
            )
            session_counts = (
                scope_frame.groupby(scope_frame["subject"].astype(str))["session"]
                .nunique(dropna=False)
                .astype(int)
                .to_dict()
            )
            subjects_below_min = [
                subject_id
                for subject_id, count in session_counts.items()
                if int(count) < minimum_sessions_per_subject
            ]
            if subjects_below_min:
                raise OfficialContractValidationError(
                    "Confirmatory dataset scope does not meet minimum_sessions_per_subject.",
                    details={
                        "minimum_sessions_per_subject": minimum_sessions_per_subject,
                        "subjects_below_min_sessions": sorted(subjects_below_min),
                    },
                )

            if bool(confirmatory_lock.get("permutation_required", False)):
                minimum_permutations = int(confirmatory_lock.get("minimum_permutations", 0))
                if int(n_permutations) < minimum_permutations:
                    raise OfficialContractValidationError(
                        "Confirmatory run permutations are below locked minimum.",
                        details={
                            "minimum_permutations": minimum_permutations,
                            "actual_n_permutations": int(n_permutations),
                        },
                    )

            if subgroup_reporting_enabled:
                allowed_axes = {
                    str(value) for value in list(confirmatory_lock.get("allowed_subgroup_axes", []))
                }
                if allowed_axes:
                    invalid_axes = [
                        axis for axis in subgroup_dimensions if axis not in allowed_axes
                    ]
                    if invalid_axes:
                        raise OfficialContractValidationError(
                            "Confirmatory subgroup axis outside locked allowed subgroup axes.",
                            details={
                                "invalid_subgroup_axes": sorted(set(invalid_axes)),
                                "allowed_subgroup_axes": sorted(allowed_axes),
                            },
                        )
                locked_min_samples = int(confirmatory_lock.get("subgroup_min_samples_per_group", 1))
                context_min_samples = int(subgroup_min_samples_per_group)
                if context_min_samples < locked_min_samples:
                    raise OfficialContractValidationError(
                        "Confirmatory subgroup minimum samples is below locked threshold.",
                        details={
                            "locked_min_samples": locked_min_samples,
                            "actual_min_samples": context_min_samples,
                        },
                    )
                locked_min_classes = int(confirmatory_lock.get("subgroup_min_classes_per_group", 1))
                context_min_classes = int(subgroup_min_classes_per_group)
                if context_min_classes < locked_min_classes:
                    raise OfficialContractValidationError(
                        "Confirmatory subgroup minimum classes is below locked threshold.",
                        details={
                            "locked_min_classes": locked_min_classes,
                            "actual_min_classes": context_min_classes,
                        },
                    )
                locked_report_small_groups = bool(
                    confirmatory_lock.get("subgroup_report_small_groups", False)
                )
                if bool(subgroup_report_small_groups) is not locked_report_small_groups:
                    raise OfficialContractValidationError(
                        "Confirmatory subgroup small-group reporting differs from locked policy.",
                        details={
                            "locked_report_small_groups": locked_report_small_groups,
                            "actual_report_small_groups": bool(subgroup_report_small_groups),
                        },
                    )

            multiplicity_primary_hypotheses = int(
                confirmatory_lock.get("multiplicity_primary_hypotheses", 1)
            )
            if multiplicity_primary_hypotheses < 1:
                raise OfficialContractValidationError(
                    "Confirmatory multiplicity policy is invalid: primary_hypotheses must be >= 1.",
                    details={"primary_hypotheses": multiplicity_primary_hypotheses},
                )
            multiplicity_primary_alpha = float(
                confirmatory_lock.get("multiplicity_primary_alpha", 0.05)
            )
            if not (0.0 < multiplicity_primary_alpha <= 1.0):
                raise OfficialContractValidationError(
                    "Confirmatory multiplicity policy is invalid: primary_alpha must be in (0, 1].",
                    details={"primary_alpha": multiplicity_primary_alpha},
                )
            multiplicity_secondary_policy = str(
                confirmatory_lock.get("multiplicity_secondary_policy", "")
            ).strip()
            if not multiplicity_secondary_policy:
                raise OfficialContractValidationError(
                    "Confirmatory multiplicity policy is invalid: secondary_policy must be non-empty.",
                    details={"secondary_policy": multiplicity_secondary_policy},
                )
            if bool(confirmatory_lock.get("multiplicity_exploratory_claims_allowed", False)):
                raise OfficialContractValidationError(
                    "Confirmatory multiplicity policy forbids exploratory_claims_allowed=true.",
                    details={"exploratory_claims_allowed": True},
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

    data_assessment = evaluate_official_data_policy(
        framework_mode=framework_mode,
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        full_index_df=frame,
        selected_index_df=selected,
        target_column=target_column,
        cv_mode=cv_mode,
        subject=subject,
        train_subject=train_subject,
        test_subject=test_subject,
        filter_task=filter_task,
        filter_modality=filter_modality,
        official_context=official_context,
        target_derivation_audit_df=target_derivation_audit_df,
        derived_label_inconsistency_audit_df=derived_label_inconsistency_audit_df,
        selection_exclusion_manifest_df=selection_result.exclusion_manifest_df,
    )
    blocking_issues = list(data_assessment.get("blocking_issues", []))
    if blocking_issues:
        first_issue = blocking_issues[0] if isinstance(blocking_issues[0], dict) else {}
        raise OfficialContractValidationError(
            "Official data-policy validation failed with blocking issues. "
            "See data_quality_report details.",
            details={
                "first_blocking_issue": first_issue,
                "n_blocking_issues": int(len(blocking_issues)),
            },
        )

    return OfficialPreflightResult(
        selected_index_df=selected,
        index_row_count=index_row_count,
        required_run_artifacts=artifact_requirements,
        required_run_metadata_fields=metadata_fields,
        data_policy_effective=dict(data_assessment.get("data_policy_effective", {})),
        data_assessment=data_assessment,
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
            details={
                "expected": bool(canonical_run),
                "actual": config_payload.get("canonical_run"),
            },
        )
    if bool(metrics_payload.get("canonical_run")) is not bool(canonical_run):
        raise OfficialArtifactContractError(
            "Official run metrics canonical_run drift detected.",
            details={
                "expected": bool(canonical_run),
                "actual": metrics_payload.get("canonical_run"),
            },
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

    config_data_policy = config_payload.get("data_policy_effective")
    metrics_data_policy = metrics_payload.get("data_policy_effective")
    if not isinstance(config_data_policy, dict) or not isinstance(metrics_data_policy, dict):
        raise OfficialArtifactContractError(
            "Official run artifacts must include data_policy_effective in both config and metrics.",
            details={
                "config_data_policy_type": type(config_data_policy).__name__,
                "metrics_data_policy_type": type(metrics_data_policy).__name__,
            },
        )
    config_data_artifacts = config_payload.get("data_artifacts")
    metrics_data_artifacts = metrics_payload.get("data_artifacts")
    if not isinstance(config_data_artifacts, dict) or not isinstance(metrics_data_artifacts, dict):
        raise OfficialArtifactContractError(
            "Official run artifacts must include data_artifacts in both config and metrics.",
            details={
                "config_data_artifacts_type": type(config_data_artifacts).__name__,
                "metrics_data_artifacts_type": type(metrics_data_artifacts).__name__,
            },
        )

    if framework_mode == FrameworkMode.CONFIRMATORY:
        config_fingerprint = config_payload.get("dataset_fingerprint")
        metrics_fingerprint = metrics_payload.get("dataset_fingerprint")
        if not isinstance(config_fingerprint, dict) or not isinstance(metrics_fingerprint, dict):
            raise OfficialArtifactContractError(
                "Confirmatory runs require dataset_fingerprint in both config and metrics artifacts.",
                details={
                    "config_dataset_fingerprint_type": type(config_fingerprint).__name__,
                    "metrics_dataset_fingerprint_type": type(metrics_fingerprint).__name__,
                },
            )

    config_primary_metric_aggregation = str(
        config_payload.get("primary_metric_aggregation", "")
    ).strip()
    metrics_primary_metric_aggregation = str(
        metrics_payload.get("primary_metric_aggregation", "")
    ).strip()
    if config_primary_metric_aggregation not in _PRIMARY_METRIC_AGGREGATIONS:
        raise OfficialArtifactContractError(
            "Official run config primary_metric_aggregation is missing or unsupported.",
            details={
                "allowed_primary_metric_aggregations": sorted(_PRIMARY_METRIC_AGGREGATIONS),
                "config_primary_metric_aggregation": config_payload.get("primary_metric_aggregation"),
            },
        )
    if metrics_primary_metric_aggregation not in _PRIMARY_METRIC_AGGREGATIONS:
        raise OfficialArtifactContractError(
            "Official run metrics primary_metric_aggregation is missing or unsupported.",
            details={
                "allowed_primary_metric_aggregations": sorted(_PRIMARY_METRIC_AGGREGATIONS),
                "metrics_primary_metric_aggregation": metrics_payload.get("primary_metric_aggregation"),
            },
        )
    if config_primary_metric_aggregation != metrics_primary_metric_aggregation:
        raise OfficialArtifactContractError(
            "Official run artifacts disagree on primary_metric_aggregation.",
            details={
                "config_primary_metric_aggregation": config_primary_metric_aggregation,
                "metrics_primary_metric_aggregation": metrics_primary_metric_aggregation,
            },
        )

    primary_metric = config_metric_policy.get("primary_metric")
    permutation_payload = metrics_payload.get("permutation_test")
    if isinstance(permutation_payload, dict) and primary_metric is not None:
        metric_name = permutation_payload.get("metric_name")
        if isinstance(metric_name, str) and validate_metric_name(
            metric_name
        ) != validate_metric_name(str(primary_metric)):
            raise OfficialArtifactContractError(
                "Permutation metric in metrics artifact does not match effective primary metric.",
                details={
                    "primary_metric": primary_metric,
                    "permutation_metric": metric_name,
                },
            )

    try:
        primary_metric_name = validate_metric_name(
            str(metrics_payload.get("primary_metric_name", "")).strip()
        )
    except ValueError as exc:
        raise OfficialArtifactContractError(
            "Official run metrics artifact must include a valid primary_metric_name.",
            details={"primary_metric_name": metrics_payload.get("primary_metric_name")},
        ) from exc
    try:
        primary_metric_value = float(metrics_payload["primary_metric_value"])
    except (TypeError, ValueError, KeyError) as exc:
        raise OfficialArtifactContractError(
            "Official run metrics artifact must include a numeric primary_metric_value.",
            details={"primary_metric_value": metrics_payload.get("primary_metric_value")},
        ) from exc
    if metrics_primary_metric_aggregation == "mean_fold_scores":
        fold_metrics_path = report_dir / "fold_metrics.csv"
        if not fold_metrics_path.exists():
            raise OfficialArtifactContractError(
                "mean_fold_scores aggregation requires fold_metrics.csv.",
                details={"fold_metrics_path": str(fold_metrics_path)},
            )
        fold_metrics_frame = pd.read_csv(fold_metrics_path)
        if primary_metric_name not in fold_metrics_frame.columns:
            raise OfficialArtifactContractError(
                "mean_fold_scores aggregation requires primary metric column in fold_metrics.csv.",
                details={
                    "primary_metric_name": primary_metric_name,
                    "fold_metrics_columns": fold_metrics_frame.columns.tolist(),
                },
            )
        fold_metric_series = pd.to_numeric(
            fold_metrics_frame[primary_metric_name],
            errors="coerce",
        ).dropna()
        if fold_metric_series.empty:
            raise OfficialArtifactContractError(
                "mean_fold_scores aggregation requires non-empty numeric fold metrics.",
                details={"primary_metric_name": primary_metric_name},
            )
        recomputed_primary_metric_value = float(fold_metric_series.mean())
    else:
        predictions_path = report_dir / "predictions.csv"
        if not predictions_path.exists():
            raise OfficialArtifactContractError(
                "pooled_held_out_predictions aggregation requires predictions.csv.",
                details={"predictions_path": str(predictions_path)},
            )
        predictions_frame = pd.read_csv(predictions_path)
        required_prediction_columns = {"y_true", "y_pred"}
        missing_prediction_columns = sorted(required_prediction_columns - set(predictions_frame.columns))
        if missing_prediction_columns:
            raise OfficialArtifactContractError(
                "predictions.csv is missing columns required for pooled_held_out_predictions.",
                details={"missing_prediction_columns": missing_prediction_columns},
            )
        recomputed_primary_metric_value = float(
            policy_metric_score(
                y_true=predictions_frame["y_true"].astype(str).tolist(),
                y_pred=predictions_frame["y_pred"].astype(str).tolist(),
                metric_name=primary_metric_name,
            )
        )
    if abs(float(primary_metric_value) - float(recomputed_primary_metric_value)) > float(
        _PRIMARY_METRIC_MATCH_ATOL
    ):
        raise OfficialArtifactContractError(
            "primary_metric_value does not match the declared primary_metric_aggregation rule.",
            details={
                "primary_metric_aggregation": metrics_primary_metric_aggregation,
                "primary_metric_name": primary_metric_name,
                "declared_primary_metric_value": float(primary_metric_value),
                "recomputed_primary_metric_value": float(recomputed_primary_metric_value),
                "tolerance": float(_PRIMARY_METRIC_MATCH_ATOL),
            },
        )
    if (
        framework_mode == FrameworkMode.CONFIRMATORY
        and bool(metrics_payload.get("tuning_enabled"))
        and str(metrics_payload.get("methodology_policy_name", "")).strip()
        == "grouped_nested_tuning"
        and str(metrics_payload.get("evidence_run_role", "")).strip() != "untuned_baseline"
        and isinstance(permutation_payload, dict)
    ):
        required_true_flags = {
            "tuning_reapplied_under_null": permutation_payload.get("tuning_reapplied_under_null"),
            "null_matches_confirmatory_setup": permutation_payload.get(
                "null_matches_confirmatory_setup"
            ),
        }
        missing_true_flags = sorted(
            key for key, value in required_true_flags.items() if value is not True
        )
        required_nonempty_fields = {
            "execution_mode": permutation_payload.get("execution_mode"),
            "null_tuning_search_space_id": permutation_payload.get("null_tuning_search_space_id"),
            "null_tuning_search_space_version": permutation_payload.get(
                "null_tuning_search_space_version"
            ),
            "null_inner_cv_scheme": permutation_payload.get("null_inner_cv_scheme"),
            "null_inner_group_field": permutation_payload.get("null_inner_group_field"),
        }
        missing_fields = sorted(
            key
            for key, value in required_nonempty_fields.items()
            if not isinstance(value, str) or not value.strip()
        )
        if missing_true_flags or missing_fields:
            raise OfficialArtifactContractError(
                "Tuned confirmatory permutation artifact must prove tuning replay under the null.",
                details={
                    "missing_true_flags": missing_true_flags,
                    "missing_fields": missing_fields,
                    "present_execution_mode": permutation_payload.get("execution_mode"),
                },
            )
        if str(permutation_payload.get("execution_mode")) != "grouped_nested_tuning_reference":
            raise OfficialArtifactContractError(
                "Tuned confirmatory permutation artifact execution_mode is invalid.",
                details={
                    "expected_execution_mode": "grouped_nested_tuning_reference",
                    "actual_execution_mode": permutation_payload.get("execution_mode"),
                },
            )


__all__ = [
    "OfficialPreflightResult",
    "validate_official_preflight",
    "validate_run_artifact_contract",
]
