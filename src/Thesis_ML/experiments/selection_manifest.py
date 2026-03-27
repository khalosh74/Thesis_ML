from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

_SELECTION_EXCLUSION_COLUMNS = [
    "sample_id",
    "subject",
    "session",
    "task",
    "modality",
    "target_column",
    "target_value",
    "exclusion_stage",
    "exclusion_reason",
    "cv_mode",
    "requested_subject",
    "requested_train_subject",
    "requested_test_subject",
    "requested_filter_task",
    "requested_filter_modality",
]


@dataclass(frozen=True)
class SelectionManifestResult:
    selected_index_df: pd.DataFrame
    exclusion_manifest_df: pd.DataFrame
    selection_summary: dict[str, Any]


def _empty_exclusion_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=_SELECTION_EXCLUSION_COLUMNS)


def _build_exclusion_rows(
    frame: pd.DataFrame,
    *,
    target_column: str,
    exclusion_stage: str,
    exclusion_reason: str,
    cv_mode: str,
    subject: str | None,
    train_subject: str | None,
    test_subject: str | None,
    filter_task: str | None,
    filter_modality: str | None,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_exclusion_manifest()

    working = frame.copy()
    for column_name in _SELECTION_EXCLUSION_COLUMNS:
        if column_name not in working.columns:
            working[column_name] = pd.NA

    working["target_column"] = str(target_column)
    if target_column in working.columns:
        working["target_value"] = working[target_column]
    else:
        working["target_value"] = pd.NA

    working["exclusion_stage"] = str(exclusion_stage)
    working["exclusion_reason"] = str(exclusion_reason)
    working["cv_mode"] = str(cv_mode)
    working["requested_subject"] = subject
    working["requested_train_subject"] = train_subject
    working["requested_test_subject"] = test_subject
    working["requested_filter_task"] = filter_task
    working["requested_filter_modality"] = filter_modality

    return working[_SELECTION_EXCLUSION_COLUMNS].reset_index(drop=True)


def _selection_summary(
    exclusion_manifest_df: pd.DataFrame,
    *,
    input_rows: int,
    selected_rows: int,
) -> dict[str, Any]:
    if exclusion_manifest_df.empty:
        return {
            "input_rows": int(input_rows),
            "selected_rows": int(selected_rows),
            "excluded_rows": 0,
            "by_stage": {},
            "by_reason": {},
        }

    by_stage = (
        exclusion_manifest_df["exclusion_stage"]
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    by_reason = (
        exclusion_manifest_df["exclusion_reason"]
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    return {
        "input_rows": int(input_rows),
        "selected_rows": int(selected_rows),
        "excluded_rows": int(len(exclusion_manifest_df)),
        "by_stage": {str(key): int(value) for key, value in by_stage.items()},
        "by_reason": {str(key): int(value) for key, value in by_reason.items()},
    }


def apply_dataset_selection_filters(
    frame: pd.DataFrame,
    *,
    target_column: str,
    cv_mode: str,
    subject: str | None,
    train_subject: str | None,
    test_subject: str | None,
    filter_task: str | None,
    filter_modality: str | None,
) -> SelectionManifestResult:
    current = frame.copy()
    input_rows = int(len(current))
    exclusion_frames: list[pd.DataFrame] = []

    if filter_task is not None:
        keep_mask = current["task"].astype(str) == str(filter_task)
        exclusion_frames.append(
            _build_exclusion_rows(
                current.loc[~keep_mask].copy(),
                target_column=target_column,
                exclusion_stage="filter_task",
                exclusion_reason="task_mismatch",
                cv_mode=cv_mode,
                subject=subject,
                train_subject=train_subject,
                test_subject=test_subject,
                filter_task=filter_task,
                filter_modality=filter_modality,
            )
        )
        current = current.loc[keep_mask].copy()

    if filter_modality is not None:
        keep_mask = current["modality"].astype(str) == str(filter_modality)
        exclusion_frames.append(
            _build_exclusion_rows(
                current.loc[~keep_mask].copy(),
                target_column=target_column,
                exclusion_stage="filter_modality",
                exclusion_reason="modality_mismatch",
                cv_mode=cv_mode,
                subject=subject,
                train_subject=train_subject,
                test_subject=test_subject,
                filter_task=filter_task,
                filter_modality=filter_modality,
            )
        )
        current = current.loc[keep_mask].copy()

    if cv_mode == "within_subject_loso_session":
        keep_mask = current["subject"].astype(str) == str(subject)
        exclusion_frames.append(
            _build_exclusion_rows(
                current.loc[~keep_mask].copy(),
                target_column=target_column,
                exclusion_stage="cv_scope",
                exclusion_reason="subject_mismatch_for_within_subject",
                cv_mode=cv_mode,
                subject=subject,
                train_subject=train_subject,
                test_subject=test_subject,
                filter_task=filter_task,
                filter_modality=filter_modality,
            )
        )
        current = current.loc[keep_mask].copy()

    elif cv_mode == "frozen_cross_person_transfer":
        if train_subject is None or test_subject is None:
            raise ValueError(
                "frozen_cross_person_transfer requires non-empty train_subject and test_subject."
            )
        allowed_subjects = {str(train_subject), str(test_subject)}
        keep_mask = current["subject"].astype(str).isin(allowed_subjects)
        exclusion_frames.append(
            _build_exclusion_rows(
                current.loc[~keep_mask].copy(),
                target_column=target_column,
                exclusion_stage="cv_scope",
                exclusion_reason="subject_not_in_transfer_pair",
                cv_mode=cv_mode,
                subject=subject,
                train_subject=train_subject,
                test_subject=test_subject,
                filter_task=filter_task,
                filter_modality=filter_modality,
            )
        )
        current = current.loc[keep_mask].copy()

    target_missing_mask = current[target_column].isna()
    exclusion_frames.append(
        _build_exclusion_rows(
            current.loc[target_missing_mask].copy(),
            target_column=target_column,
            exclusion_stage="target_cleanup",
            exclusion_reason="target_missing_after_derivation",
            cv_mode=cv_mode,
            subject=subject,
            train_subject=train_subject,
            test_subject=test_subject,
            filter_task=filter_task,
            filter_modality=filter_modality,
        )
    )
    current = current.loc[~target_missing_mask].copy()

    if target_column in current.columns:
        current[target_column] = current[target_column].astype(str)

    exclusion_manifest_df = (
        pd.concat(exclusion_frames, ignore_index=True)
        if any(not item.empty for item in exclusion_frames)
        else _empty_exclusion_manifest()
    )

    summary = _selection_summary(
        exclusion_manifest_df,
        input_rows=input_rows,
        selected_rows=int(len(current)),
    )

    return SelectionManifestResult(
        selected_index_df=current.reset_index(drop=True),
        exclusion_manifest_df=exclusion_manifest_df,
        selection_summary=summary,
    )
