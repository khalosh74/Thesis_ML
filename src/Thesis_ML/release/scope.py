from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.config.paths import PROJECT_ROOT
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
from Thesis_ML.release.hashing import canonical_target_mapping_hash
from Thesis_ML.release.loader import LoadedDatasetManifest, LoadedReleaseBundle
from Thesis_ML.release.manifests import write_json
from Thesis_ML.release.models import utc_now_iso
from Thesis_ML.release.scope_manifest import (
    build_scope_counts,
    selected_sample_ids_sha256,
)
from Thesis_ML.release.scope_models import (
    CompiledScopeManifest,
    CompiledScopeResult,
    ScopeExclusionsSummary,
)

_MIN_SCOPE_COLUMNS = {
    "sample_id",
    "subject",
    "session",
    "task",
    "modality",
    "emotion",
}
_UNKNOWN_REGRESSOR_COLUMN = "glm_has_unknown_regressors"


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at '{path}'.")
    return payload


def _resolve_scope_value_counts(frame: pd.DataFrame, column_name: str) -> dict[str, int]:
    if column_name not in frame.columns:
        return {}
    counts = frame[column_name].astype(str).value_counts(dropna=False).sort_index().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _record_exclusions(
    *,
    rows: list[dict[str, Any]],
    excluded_df: pd.DataFrame,
    stage: str,
    reason: str,
) -> None:
    if excluded_df.empty:
        return
    sample_ids_head = (
        excluded_df["sample_id"].astype(str).head(20).tolist()
        if "sample_id" in excluded_df.columns
        else []
    )
    rows.append(
        {
            "stage": str(stage),
            "reason": str(reason),
            "n_rows": int(len(excluded_df)),
            "sample_ids_head": sample_ids_head,
        }
    )


def _compile_exclusions_summary(
    *,
    exclusion_rows: list[dict[str, Any]],
    input_rows: int,
    selected_rows: int,
) -> ScopeExclusionsSummary:
    by_stage: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    excluded_rows = 0
    for row in exclusion_rows:
        n_rows = int(row.get("n_rows", 0))
        excluded_rows += n_rows
        stage = str(row.get("stage", "unknown"))
        reason = str(row.get("reason", "unknown"))
        by_stage[stage] = int(by_stage.get(stage, 0)) + n_rows
        by_reason[reason] = int(by_reason.get(reason, 0)) + n_rows
    return ScopeExclusionsSummary(
        input_rows=int(input_rows),
        selected_rows=int(selected_rows),
        excluded_rows=int(excluded_rows),
        by_stage={str(key): int(value) for key, value in sorted(by_stage.items())},
        by_reason={str(key): int(value) for key, value in sorted(by_reason.items())},
        rows=list(exclusion_rows),
    )


def _require_columns(frame: pd.DataFrame, *, required_columns: set[str], context: str) -> None:
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {', '.join(missing)}")


def _validate_unknown_regressors(frame: pd.DataFrame) -> None:
    if _UNKNOWN_REGRESSOR_COLUMN not in frame.columns:
        return
    normalized_unknown = (
        frame[_UNKNOWN_REGRESSOR_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
    )
    if bool(normalized_unknown.isna().any()):
        raise ValueError("Dataset index has invalid glm_has_unknown_regressors values.")
    if bool(normalized_unknown.any()):
        affected = frame.loc[normalized_unknown, "sample_id"].astype(str).head(20).tolist()
        raise ValueError(
            "Official release scope compilation blocked: unknown GLM regressors detected. "
            f"n_rows={int(normalized_unknown.sum())}, sample_ids_head={affected}"
        )


def _assert_non_empty(frame: pd.DataFrame, *, context: str) -> None:
    if frame.empty:
        raise ValueError(f"Official release scope compilation produced an empty selection at {context}.")


def _resolve_selected_samples_csv(scope_manifest_path: Path, selected_samples_csv: str) -> Path:
    candidate = Path(str(selected_samples_csv))
    if candidate.is_absolute():
        return candidate.resolve()
    return (scope_manifest_path.parent / candidate).resolve()


def _collect_protocol_success_report_dirs(protocol_output_dir: Path) -> list[Path]:
    report_index_path = protocol_output_dir / "report_index.csv"
    if not report_index_path.exists():
        return []
    rows: list[Path] = []
    with report_index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = str(row.get("status", "")).strip().lower()
            if status != "success":
                continue
            raw_report_dir = str(row.get("report_dir", "")).strip()
            if not raw_report_dir:
                continue
            candidate = Path(raw_report_dir)
            if not candidate.is_absolute():
                candidate = (protocol_output_dir / candidate).resolve()
            rows.append(candidate)
    return rows


def compile_release_scope(
    *,
    release_bundle: LoadedReleaseBundle,
    dataset_manifest: LoadedDatasetManifest,
    run_dir: Path,
) -> CompiledScopeResult:
    science = release_bundle.science
    scope = science.scope
    target_column = str(science.target.name)

    mapping_path = (PROJECT_ROOT / science.target.mapping_path).resolve()
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Release target mapping file does not exist: '{mapping_path}'."
        )
    actual_mapping_hash = canonical_target_mapping_hash(mapping_path).lower()
    expected_mapping_hash = science.target.mapping_hash.lower()
    if actual_mapping_hash != expected_mapping_hash:
        raise ValueError(
            "Release scope compilation blocked by target mapping hash mismatch. "
            f"expected={expected_mapping_hash}, actual={actual_mapping_hash}"
        )

    frame = pd.read_csv(dataset_manifest.index_csv_path)
    if frame.empty:
        raise ValueError("Dataset manifest index_csv is empty.")

    required_columns = set(science.dataset_contract.required_columns) | _MIN_SCOPE_COLUMNS | {
        target_column
    }
    try:
        frame = validate_dataset_index_strict(
            frame,
            data_root=dataset_manifest.data_root_path,
            required_columns=required_columns,
            require_integrity_columns=True,
        )
    except DatasetIndexValidationError as exc:
        raise ValueError(f"Release scope compilation failed strict index validation: {exc}") from exc

    derived_inconsistency_audit = build_derived_label_inconsistency_audit(
        frame,
        emotion_column="emotion",
        coarse_column="coarse_affect",
        binary_column="binary_valence_like",
    )
    blocking_inconsistency = blocking_derived_label_inconsistency_rows(
        derived_inconsistency_audit
    )
    if not blocking_inconsistency.empty:
        summary = summarize_derived_label_inconsistency_audit(blocking_inconsistency)
        raise ValueError(
            "Release scope compilation blocked by inconsistent derived labels. "
            f"n_rows={summary['n_rows']}, by_category={summary['by_category']}, "
            f"sample_ids_head={summary['sample_ids_head']}"
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

    target_audit = build_target_derivation_audit(frame, target_column=target_column)
    blocking_target_audit = blocking_target_derivation_audit_rows(target_audit)
    if not blocking_target_audit.empty:
        summary = summarize_target_derivation_audit(blocking_target_audit)
        raise ValueError(
            "Release scope compilation blocked by target derivation audit failures. "
            f"target={target_column}, n_rows={summary['n_rows']}, "
            f"by_category={summary['by_category']}, sample_ids_head={summary['sample_ids_head']}"
        )

    _validate_unknown_regressors(frame)
    _require_columns(frame, required_columns=required_columns, context="Validated dataset index")

    input_rows = int(len(frame))
    exclusion_rows: list[dict[str, Any]] = []
    selected = frame.copy()

    scope_subjects = sorted(scope.effective_subjects())
    scope_tasks = sorted(scope.effective_tasks())
    scope_modality = str(scope.effective_modality())
    cv_scope_mode = str(science.split_policy.primary_analysis.split)

    subject_keep = selected["subject"].astype(str).isin(set(scope_subjects))
    _record_exclusions(
        rows=exclusion_rows,
        excluded_df=selected.loc[~subject_keep].copy(),
        stage="scope_subjects",
        reason="subject_not_in_release_scope",
    )
    selected = selected.loc[subject_keep].copy()
    _assert_non_empty(selected, context="subject_scope")

    task_keep = selected["task"].astype(str).isin(set(scope_tasks))
    _record_exclusions(
        rows=exclusion_rows,
        excluded_df=selected.loc[~task_keep].copy(),
        stage="scope_tasks",
        reason="task_not_in_release_scope",
    )
    selected = selected.loc[task_keep].copy()
    _assert_non_empty(selected, context="task_scope")

    modality_keep = selected["modality"].astype(str) == scope_modality
    _record_exclusions(
        rows=exclusion_rows,
        excluded_df=selected.loc[~modality_keep].copy(),
        stage="scope_modality",
        reason="modality_not_in_release_scope",
    )
    selected = selected.loc[modality_keep].copy()
    _assert_non_empty(selected, context="modality_scope")

    missing_target = selected[target_column].isna()
    _record_exclusions(
        rows=exclusion_rows,
        excluded_df=selected.loc[missing_target].copy(),
        stage="target_cleanup",
        reason="target_missing_after_derivation",
    )
    selected = selected.loc[~missing_target].copy()
    _assert_non_empty(selected, context="target_cleanup")

    selected["sample_id"] = selected["sample_id"].astype(str)
    if bool(selected["sample_id"].duplicated().any()):
        duplicates = (
            selected.loc[selected["sample_id"].duplicated(keep=False), "sample_id"]
            .astype(str)
            .head(20)
            .tolist()
        )
        raise ValueError(
            "Release scope compilation requires unique sample_id values. "
            f"duplicate_sample_ids_head={duplicates}"
        )

    selected_tasks = sorted(selected["task"].astype(str).unique().tolist())
    missing_tasks = sorted(set(scope_tasks) - set(selected_tasks))
    unexpected_tasks = sorted(set(selected_tasks) - set(scope_tasks))
    if missing_tasks or unexpected_tasks:
        raise ValueError(
            "Release scope tasks mismatch after compilation. "
            f"missing_tasks={missing_tasks}, unexpected_tasks={unexpected_tasks}"
        )

    selected_modalities = sorted(selected["modality"].astype(str).unique().tolist())
    if selected_modalities != [scope_modality]:
        raise ValueError(
            "Release scope modality mismatch after compilation. "
            f"expected={[scope_modality]}, actual={selected_modalities}"
        )

    selected_subjects = sorted(selected["subject"].astype(str).unique().tolist())
    missing_subjects = sorted(set(scope_subjects) - set(selected_subjects))
    unexpected_subjects = sorted(set(selected_subjects) - set(scope_subjects))
    if missing_subjects or unexpected_subjects:
        raise ValueError(
            "Release scope subjects mismatch after compilation. "
            f"missing_subjects={missing_subjects}, unexpected_subjects={unexpected_subjects}"
        )

    if cv_scope_mode == "within_subject_loso_session":
        sessions_by_subject = (
            selected.groupby("subject", sort=True)["session"].nunique(dropna=False).to_dict()
        )
        missing_cv_subjects = sorted(
            subject_id
            for subject_id in scope_subjects
            if int(sessions_by_subject.get(subject_id, 0)) < 2
        )
        if missing_cv_subjects:
            raise ValueError(
                "Release scope missing required within-subject LOSO session coverage. "
                f"subjects_with_<2_sessions={missing_cv_subjects}"
            )

    for pair in science.scope.transfer_pairs:
        pair_subjects = {str(pair.train_subject), str(pair.test_subject)}
        pair_selected = selected[selected["subject"].astype(str).isin(pair_subjects)]
        if pair_selected.empty:
            raise ValueError(
                "Release scope transfer-pair compilation failed. "
                f"pair={pair.train_subject}->{pair.test_subject} has no selected rows."
            )
        pair_present = set(pair_selected["subject"].astype(str).unique().tolist())
        if pair_subjects - pair_present:
            raise ValueError(
                "Release scope transfer-pair compilation failed: missing subject in pair selection. "
                f"pair={pair.train_subject}->{pair.test_subject}, "
                f"missing={sorted(pair_subjects - pair_present)}"
            )

    selected = selected.sort_values(
        by=["subject", "session", "task", "modality", "sample_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    selected[target_column] = selected[target_column].astype(str)

    selected_sample_ids = selected["sample_id"].astype(str).tolist()
    selected_ids_sha256 = selected_sample_ids_sha256(selected_sample_ids)
    exclusions_summary = _compile_exclusions_summary(
        exclusion_rows=exclusion_rows,
        input_rows=input_rows,
        selected_rows=int(len(selected)),
    )

    scope_dir = (run_dir / "artifacts" / "scope").resolve()
    scope_dir.mkdir(parents=True, exist_ok=True)
    selected_samples_path = scope_dir / "selected_samples.csv"
    selected.to_csv(selected_samples_path, index=False)

    manifest = CompiledScopeManifest(
        release_id=release_bundle.release.release_id,
        release_version=release_bundle.release.release_version,
        science_hash=release_bundle.hashes.science_hash,
        dataset_manifest_path=str(dataset_manifest.manifest_path.resolve()),
        dataset_fingerprint=str(dataset_manifest.manifest.dataset_fingerprint),
        target_column=target_column,
        target_mapping_path=str(mapping_path),
        target_mapping_hash=expected_mapping_hash,
        scope_subjects=scope_subjects,
        scope_tasks=scope_tasks,
        scope_modality=scope_modality,
        cv_scope_mode=cv_scope_mode,
        selected_row_count=int(len(selected)),
        selected_sample_ids_sha256=selected_ids_sha256,
        selected_samples_csv=str(selected_samples_path.resolve()),
        counts=build_scope_counts(selected, target_column=target_column),
        exclusions_summary=exclusions_summary,
        generated_at_utc=utc_now_iso(),
    )
    scope_manifest_path = scope_dir / "scope_manifest.json"
    write_json(scope_manifest_path, manifest.model_dump(mode="json"))

    return CompiledScopeResult(
        selected_index_df=selected,
        selected_samples_path=selected_samples_path,
        scope_manifest_path=scope_manifest_path,
        selected_sample_ids=selected_sample_ids,
        selection_summary={
            "selected_row_count": int(len(selected)),
            "selected_sample_ids_sha256": selected_ids_sha256,
            "counts": manifest.counts.model_dump(mode="json"),
            "cv_units": {
                "within_subject_loso_session": {
                    str(subject_id): int(
                        (
                            selected["subject"].astype(str) == str(subject_id)
                        ).sum()
                    )
                    for subject_id in scope_subjects
                },
                "frozen_cross_person_transfer": [
                    {
                        "train_subject": str(pair.train_subject),
                        "test_subject": str(pair.test_subject),
                        "row_count": int(
                            selected[
                                selected["subject"].astype(str).isin(
                                    {str(pair.train_subject), str(pair.test_subject)}
                                )
                            ].shape[0]
                        ),
                    }
                    for pair in science.scope.transfer_pairs
                ],
            },
            "exclusions_summary": exclusions_summary.model_dump(mode="json"),
        },
    )


def verify_scope_execution_alignment(
    *,
    run_dir: Path,
    compiled_scope_manifest_path: Path,
    expected_science_hash: str,
    expected_target_mapping_hash: str,
    write_output: bool = True,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    compiled_scope_manifest = CompiledScopeManifest.model_validate(
        _load_json_object(compiled_scope_manifest_path)
    )
    expected_selected_csv = _resolve_selected_samples_csv(
        compiled_scope_manifest_path,
        compiled_scope_manifest.selected_samples_csv,
    )
    if not expected_selected_csv.exists():
        issues.append(
            {
                "code": "compiled_scope_selected_samples_missing",
                "message": "Compiled scope selected_samples.csv is missing.",
                "details": {"path": str(expected_selected_csv)},
            }
        )
        expected_df = pd.DataFrame(columns=["sample_id"])
    else:
        expected_df = pd.read_csv(expected_selected_csv)
    if "sample_id" not in expected_df.columns:
        issues.append(
            {
                "code": "compiled_scope_sample_id_missing",
                "message": "Compiled scope selected_samples.csv is missing sample_id column.",
            }
        )
        expected_df = pd.DataFrame(columns=["sample_id"])

    expected_sample_ids = expected_df["sample_id"].astype(str).tolist()
    expected_sample_id_set = set(expected_sample_ids)
    expected_hash = selected_sample_ids_sha256(expected_sample_ids)
    if expected_hash != compiled_scope_manifest.selected_sample_ids_sha256:
        issues.append(
            {
                "code": "compiled_scope_selected_sample_hash_mismatch",
                "message": "Compiled scope selected_sample_ids_sha256 does not match selected_samples.csv.",
                "details": {
                    "manifest_hash": compiled_scope_manifest.selected_sample_ids_sha256,
                    "actual_hash": expected_hash,
                },
            }
        )

    protocol_root = run_dir / "artifacts" / "protocol_runs"
    protocol_dirs = sorted(path for path in protocol_root.iterdir() if path.is_dir()) if protocol_root.exists() else []
    if len(protocol_dirs) != 1:
        issues.append(
            {
                "code": "protocol_output_missing",
                "message": "Could not resolve unique protocol output directory under artifacts/protocol_runs.",
                "details": {"protocol_root": str(protocol_root), "n_candidates": len(protocol_dirs)},
            }
        )
        protocol_output_dir = None
    else:
        protocol_output_dir = protocol_dirs[0]

    actual_rows: list[pd.DataFrame] = []
    missing_feature_qc_rows: list[str] = []
    if protocol_output_dir is not None:
        for report_dir in _collect_protocol_success_report_dirs(protocol_output_dir):
            selected_path = report_dir / "feature_qc_selected_samples.csv"
            if not selected_path.exists():
                missing_feature_qc_rows.append(str(selected_path))
                continue
            actual_df = pd.read_csv(selected_path)
            if "sample_id" not in actual_df.columns:
                missing_feature_qc_rows.append(str(selected_path))
                continue
            actual_rows.append(actual_df.copy())

    if missing_feature_qc_rows:
        issues.append(
            {
                "code": "feature_qc_selected_samples_missing",
                "message": "One or more successful runs are missing feature_qc_selected_samples.csv.",
                "details": {"paths": missing_feature_qc_rows},
            }
        )

    if actual_rows:
        all_actual = pd.concat(actual_rows, ignore_index=True)
        all_actual["sample_id"] = all_actual["sample_id"].astype(str)
        actual_unique = all_actual.drop_duplicates(subset=["sample_id"], keep="first").reset_index(
            drop=True
        )
    else:
        actual_unique = pd.DataFrame(columns=["sample_id"])

    actual_sample_ids = actual_unique["sample_id"].astype(str).tolist()
    actual_sample_id_set = set(actual_sample_ids)
    missing_sample_ids = sorted(expected_sample_id_set - actual_sample_id_set)
    extra_sample_ids = sorted(actual_sample_id_set - expected_sample_id_set)
    if missing_sample_ids or extra_sample_ids:
        issues.append(
            {
                "code": "scope_sample_id_mismatch",
                "message": "Execution-selected sample_id set does not match compiled scope sample_id set.",
                "details": {
                    "missing_sample_ids": missing_sample_ids,
                    "extra_sample_ids": extra_sample_ids,
                },
            }
        )

    expected_subject_counts = _resolve_scope_value_counts(expected_df, "subject")
    expected_task_counts = _resolve_scope_value_counts(expected_df, "task")
    expected_modality_counts = _resolve_scope_value_counts(expected_df, "modality")
    expected_session_counts = _resolve_scope_value_counts(expected_df, "session")
    actual_subject_counts = _resolve_scope_value_counts(actual_unique, "subject")
    actual_task_counts = _resolve_scope_value_counts(actual_unique, "task")
    actual_modality_counts = _resolve_scope_value_counts(actual_unique, "modality")
    actual_session_counts = _resolve_scope_value_counts(actual_unique, "session")

    if actual_subject_counts != expected_subject_counts:
        issues.append(
            {
                "code": "scope_subject_count_mismatch",
                "message": "Execution-selected subject counts do not match compiled scope.",
                "details": {
                    "expected_subject_counts": expected_subject_counts,
                    "actual_subject_counts": actual_subject_counts,
                },
            }
        )
    if actual_task_counts != expected_task_counts:
        issues.append(
            {
                "code": "scope_task_count_mismatch",
                "message": "Execution-selected task counts do not match compiled scope.",
                "details": {
                    "expected_task_counts": expected_task_counts,
                    "actual_task_counts": actual_task_counts,
                },
            }
        )
    if actual_modality_counts != expected_modality_counts:
        issues.append(
            {
                "code": "scope_modality_count_mismatch",
                "message": "Execution-selected modality counts do not match compiled scope.",
                "details": {
                    "expected_modality_counts": expected_modality_counts,
                    "actual_modality_counts": actual_modality_counts,
                },
            }
        )
    if actual_session_counts != expected_session_counts:
        issues.append(
            {
                "code": "scope_session_count_mismatch",
                "message": "Execution-selected session counts do not match compiled scope.",
                "details": {
                    "expected_session_counts": expected_session_counts,
                    "actual_session_counts": actual_session_counts,
                },
            }
        )

    if compiled_scope_manifest.science_hash != str(expected_science_hash):
        issues.append(
            {
                "code": "scope_science_hash_mismatch",
                "message": "Compiled scope science_hash does not match current release science hash.",
                "details": {
                    "manifest_science_hash": compiled_scope_manifest.science_hash,
                    "expected_science_hash": str(expected_science_hash),
                },
            }
        )
    if compiled_scope_manifest.target_mapping_hash.lower() != str(expected_target_mapping_hash).lower():
        issues.append(
            {
                "code": "scope_target_mapping_hash_mismatch",
                "message": "Compiled scope target_mapping_hash does not match release science target mapping hash.",
                "details": {
                    "manifest_target_mapping_hash": compiled_scope_manifest.target_mapping_hash,
                    "expected_target_mapping_hash": str(expected_target_mapping_hash),
                },
            }
        )

    payload = {
        "schema_version": "release-scope-alignment-verification-v1",
        "run_dir": str(run_dir.resolve()),
        "compiled_scope_manifest_path": str(compiled_scope_manifest_path.resolve()),
        "compiled_selected_samples_path": str(expected_selected_csv.resolve()),
        "passed": not issues,
        "issues": issues,
        "expected_selected_row_count": int(len(expected_df)),
        "actual_selected_row_count": int(len(actual_unique)),
        "selected_sample_ids_sha256": str(compiled_scope_manifest.selected_sample_ids_sha256),
        "missing_sample_ids": missing_sample_ids,
        "extra_sample_ids": extra_sample_ids,
        "expected_counts": {
            "by_subject": expected_subject_counts,
            "by_task": expected_task_counts,
            "by_modality": expected_modality_counts,
            "by_session": expected_session_counts,
        },
        "actual_counts": {
            "by_subject": actual_subject_counts,
            "by_task": actual_task_counts,
            "by_modality": actual_modality_counts,
            "by_session": actual_session_counts,
        },
    }
    output_path = run_dir / "verification" / "scope_alignment_verification.json"
    if write_output:
        write_json(output_path, payload)
    return payload


__all__ = [
    "compile_release_scope",
    "verify_scope_execution_alignment",
]

