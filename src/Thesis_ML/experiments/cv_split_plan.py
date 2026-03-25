from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold


@dataclass(frozen=True)
class PlannedFold:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    train_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


@dataclass(frozen=True)
class CVSplitPlan:
    cv_mode: str
    groups: np.ndarray
    folds: tuple[PlannedFold, ...]


def _sorted_unique_strings(series: pd.Series) -> tuple[str, ...]:
    if series.empty:
        return tuple()
    return tuple(sorted(series.astype(str).drop_duplicates().tolist()))


def _build_fold(
    *,
    metadata_df: pd.DataFrame,
    groups: np.ndarray,
    fold_index: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> PlannedFold:
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    train_meta = metadata_df.iloc[train_idx]
    test_meta = metadata_df.iloc[test_idx]

    train_groups = pd.Series(groups[train_idx]).astype(str)
    test_groups = pd.Series(groups[test_idx]).astype(str)

    return PlannedFold(
        fold=fold_index,
        train_idx=train_idx,
        test_idx=test_idx,
        train_subjects=_sorted_unique_strings(train_meta["subject"]),
        test_subjects=_sorted_unique_strings(test_meta["subject"]),
        train_sessions=_sorted_unique_strings(train_meta["session"]),
        test_sessions=_sorted_unique_strings(test_meta["session"]),
        train_groups=_sorted_unique_strings(train_groups),
        test_groups=_sorted_unique_strings(test_groups),
    )


def build_cv_split_plan(
    *,
    metadata_df: pd.DataFrame,
    target_column: str,
    cv_mode: str,
    subject: str | None,
    train_subject: str | None,
    test_subject: str | None,
    seed: int | None = None,
) -> CVSplitPlan:
    if target_column not in metadata_df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from metadata_df.")

    y = metadata_df[target_column].astype(str).to_numpy()
    dummy_x = np.zeros((len(metadata_df), 1), dtype=np.uint8)

    if cv_mode == "frozen_cross_person_transfer":
        subjects = metadata_df["subject"].astype(str)
        train_mask = subjects == str(train_subject)
        test_mask = subjects == str(test_subject)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())

        if len(train_idx) == 0:
            raise ValueError(
                f"No cache-aligned samples found for train_subject '{train_subject}'."
            )
        if len(test_idx) == 0:
            raise ValueError(
                f"No cache-aligned samples found for test_subject '{test_subject}'."
            )

        unique_labels_train = np.unique(y[train_idx])
        if len(unique_labels_train) < 2:
            raise ValueError("Training data requires at least 2 target classes.")

        groups = metadata_df["session"].astype(str).to_numpy()
        raw_splits: list[tuple[np.ndarray, np.ndarray]] = [(train_idx, test_idx)]

    elif cv_mode == "within_subject_loso_session":
        subjects = metadata_df["subject"].astype(str)
        unique_subjects = sorted(subjects.unique().tolist())
        if len(unique_subjects) != 1 or unique_subjects[0] != subject:
            raise ValueError(
                "within_subject_loso_session requires exactly one subject in the evaluated data."
            )

        groups = metadata_df["session"].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        unique_labels = np.unique(y)
        if len(unique_groups) < 2:
            raise ValueError("Grouped CV requires at least 2 unique subject-session groups.")
        if len(unique_labels) < 2:
            raise ValueError("Classification requires at least 2 target classes.")

        splitter = LeaveOneGroupOut()
        raw_splits = list(splitter.split(dummy_x, y, groups))
        if len(raw_splits) < 2:
            raise ValueError("Grouped CV produced fewer than 2 folds.")

    elif cv_mode == "loso_session":
        groups = (
            metadata_df["subject"].astype(str) + "_" + metadata_df["session"].astype(str)
        ).to_numpy()
        unique_groups = np.unique(groups)
        unique_labels = np.unique(y)
        if len(unique_groups) < 2:
            raise ValueError("Grouped CV requires at least 2 unique subject-session groups.")
        if len(unique_labels) < 2:
            raise ValueError("Classification requires at least 2 target classes.")

        splitter = LeaveOneGroupOut()
        raw_splits = list(splitter.split(dummy_x, y, groups))
        if len(raw_splits) < 2:
            raise ValueError("Grouped CV produced fewer than 2 folds.")

    elif cv_mode == "record_random_split":
        groups = (
            metadata_df["subject"].astype(str) + "_" + metadata_df["session"].astype(str)
        ).to_numpy()
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError("Classification requires at least 2 target classes.")
        min_class_count = int(pd.Series(y).value_counts().min())
        if min_class_count < 2:
            raise ValueError(
                "record_random_split requires at least 2 samples per class for stratified folds."
            )
        n_splits = int(min(5, min_class_count))
        if n_splits < 2:
            raise ValueError(
                "record_random_split produced fewer than 2 stratified folds from selected data."
            )
        if seed is None:
            raise ValueError("record_random_split split planning requires a non-null seed.")

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(seed),
        )
        raw_splits = list(splitter.split(dummy_x, y))
        if len(raw_splits) < 2:
            raise ValueError("record_random_split produced fewer than 2 folds.")

    else:
        raise ValueError(f"Unsupported cv_mode '{cv_mode}'.")

    folds = tuple(
        _build_fold(
            metadata_df=metadata_df,
            groups=groups,
            fold_index=fold_index,
            train_idx=train_idx,
            test_idx=test_idx,
        )
        for fold_index, (train_idx, test_idx) in enumerate(raw_splits)
    )
    
    return CVSplitPlan(
        cv_mode=str(cv_mode),
        groups=np.asarray(groups).astype(str, copy=False),
        folds=folds,
    )


__all__ = ["PlannedFold", "CVSplitPlan", "build_cv_split_plan"]