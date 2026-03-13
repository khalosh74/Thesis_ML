from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut

if TYPE_CHECKING:
    from Thesis_ML.experiments.sections import EvaluationInput, InterpretabilityInput, ModelFitInput


def execute_model_fit(section_input: ModelFitInput) -> dict[str, Any]:
    y = section_input.metadata_df[section_input.target_column].astype(str).to_numpy()

    if section_input.cv_mode == "frozen_cross_person_transfer":
        subjects = section_input.metadata_df["subject"].astype(str)
        train_mask = subjects == str(section_input.train_subject)
        test_mask = subjects == str(section_input.test_subject)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())

        if len(train_idx) == 0:
            raise ValueError(
                f"No cache-aligned samples found for train_subject '{section_input.train_subject}'."
            )
        if len(test_idx) == 0:
            raise ValueError(
                f"No cache-aligned samples found for test_subject '{section_input.test_subject}'."
            )

        unique_labels_train = np.unique(y[train_idx])
        if len(unique_labels_train) < 2:
            raise ValueError("Training data requires at least 2 target classes.")

        groups = section_input.metadata_df["session"].astype(str).to_numpy()
        splits: list[tuple[np.ndarray, np.ndarray]] = [(train_idx, test_idx)]
    elif section_input.cv_mode == "within_subject_loso_session":
        subjects = section_input.metadata_df["subject"].astype(str)
        unique_subjects = sorted(subjects.unique().tolist())
        if len(unique_subjects) != 1 or unique_subjects[0] != section_input.subject:
            raise ValueError(
                "within_subject_loso_session requires exactly one subject in the evaluated data."
            )

        groups = section_input.metadata_df["session"].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        unique_labels = np.unique(y)
        if len(unique_groups) < 2:
            raise ValueError("Grouped CV requires at least 2 unique subject-session groups.")
        if len(unique_labels) < 2:
            raise ValueError("Classification requires at least 2 target classes.")

        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(section_input.x_matrix, y, groups))
        if len(splits) < 2:
            raise ValueError("Grouped CV produced fewer than 2 folds.")
    else:
        groups = (
            section_input.metadata_df["subject"].astype(str)
            + "_"
            + section_input.metadata_df["session"].astype(str)
        ).to_numpy()
        unique_groups = np.unique(groups)
        unique_labels = np.unique(y)
        if len(unique_groups) < 2:
            raise ValueError("Grouped CV requires at least 2 unique subject-session groups.")
        if len(unique_labels) < 2:
            raise ValueError("Classification requires at least 2 target classes.")

        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(section_input.x_matrix, y, groups))
        if len(splits) < 2:
            raise ValueError("Grouped CV produced fewer than 2 folds.")

    pipeline_template = section_input.build_pipeline_fn(
        model_name=section_input.model,
        seed=section_input.seed,
    )

    interpretability_enabled = section_input.cv_mode == "within_subject_loso_session"
    interpretability_fold_rows: list[dict[str, Any]] = []
    interpretability_vectors: list[np.ndarray] = []
    interpretability_dir: Path | None = None
    if interpretability_enabled:
        interpretability_dir = section_input.report_dir / "interpretability"
        interpretability_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    y_true_all: list[str] = []
    y_pred_all: list[str] = []

    for fold_index, (train_idx, test_idx) in enumerate(splits):
        train_meta = section_input.metadata_df.iloc[train_idx].reset_index(drop=True)
        test_meta = section_input.metadata_df.iloc[test_idx].reset_index(drop=True)
        train_subjects = sorted(train_meta["subject"].astype(str).unique().tolist())
        test_subjects = sorted(test_meta["subject"].astype(str).unique().tolist())
        train_sessions = sorted(train_meta["session"].astype(str).unique().tolist())
        test_sessions = sorted(test_meta["session"].astype(str).unique().tolist())

        if section_input.cv_mode == "within_subject_loso_session":
            expected_subjects = [str(section_input.subject)]
            if train_subjects != expected_subjects or test_subjects != expected_subjects:
                raise ValueError(
                    "within_subject_loso_session produced fold(s) with unexpected "
                    "subject membership."
                )
            if set(train_sessions) & set(test_sessions):
                raise ValueError(
                    "within_subject_loso_session produced overlapping train/test sessions."
                )
        if section_input.cv_mode == "frozen_cross_person_transfer":
            expected_train = [str(section_input.train_subject)]
            expected_test = [str(section_input.test_subject)]
            if train_subjects != expected_train or test_subjects != expected_test:
                raise ValueError(
                    "frozen_cross_person_transfer produced unexpected train/test subject "
                    "membership."
                )

        estimator = clone(pipeline_template)
        estimator.fit(section_input.x_matrix[train_idx], y[train_idx])

        if interpretability_enabled:
            if interpretability_dir is None:
                raise ValueError("Interpretability directory was not initialized.")
            coef_array, intercept_array, class_labels = (
                section_input.extract_linear_coefficients_fn(estimator)
            )
            coef_path = interpretability_dir / f"fold_{fold_index:03d}_coefficients.npz"
            np.savez_compressed(
                coef_path,
                coefficients=coef_array.astype(np.float32, copy=False),
                intercept=intercept_array.astype(np.float32, copy=False),
                class_labels=np.asarray(class_labels, dtype=np.str_),
                feature_index=np.arange(coef_array.shape[1], dtype=np.int32),
            )
            interpretability_fold_rows.append(
                {
                    "fold": fold_index,
                    "experiment_mode": section_input.cv_mode,
                    "subject": str(section_input.subject),
                    "held_out_test_sessions": "|".join(test_sessions),
                    "model": section_input.model,
                    "target": section_input.target_column,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "coef_rows": int(coef_array.shape[0]),
                    "n_features": int(coef_array.shape[1]),
                    "coef_shape": json.dumps(list(coef_array.shape)),
                    "intercept_shape": json.dumps(list(intercept_array.shape)),
                    "class_labels": json.dumps(class_labels),
                    "coefficient_file": str(coef_path.resolve()),
                    "seed": int(section_input.seed),
                    "run_id": section_input.run_id,
                    "config_file": section_input.config_filename,
                }
            )
            interpretability_vectors.append(coef_array.reshape(-1).astype(np.float64))

        y_pred = estimator.predict(section_input.x_matrix[test_idx])
        y_true = y[test_idx]
        score_payload = section_input.scores_for_predictions_fn(
            estimator=estimator, x_test=section_input.x_matrix[test_idx]
        )

        fold_rows.append(
            {
                "fold": fold_index,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_groups": "|".join(sorted(np.unique(groups[test_idx]).tolist())),
                "experiment_mode": section_input.cv_mode,
                "subject": (
                    str(section_input.subject)
                    if section_input.cv_mode == "within_subject_loso_session"
                    else pd.NA
                ),
                "train_subject": (
                    str(section_input.train_subject)
                    if section_input.cv_mode == "frozen_cross_person_transfer"
                    else pd.NA
                ),
                "test_subject": (
                    str(section_input.test_subject)
                    if section_input.cv_mode == "frozen_cross_person_transfer"
                    else pd.NA
                ),
                "train_sessions": "|".join(train_sessions),
                "test_sessions": "|".join(test_sessions),
                "target": section_input.target_column,
                "model": section_input.model,
                "seed": int(section_input.seed),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            }
        )

        split_rows.append(
            {
                "fold": fold_index,
                "experiment_mode": section_input.cv_mode,
                "subject": (
                    str(section_input.subject)
                    if section_input.cv_mode == "within_subject_loso_session"
                    else pd.NA
                ),
                "train_subject": (
                    str(section_input.train_subject)
                    if section_input.cv_mode == "frozen_cross_person_transfer"
                    else pd.NA
                ),
                "test_subject": (
                    str(section_input.test_subject)
                    if section_input.cv_mode == "frozen_cross_person_transfer"
                    else pd.NA
                ),
                "train_subjects": "|".join(train_subjects),
                "test_subjects": "|".join(test_subjects),
                "train_sessions": "|".join(train_sessions),
                "test_sessions": "|".join(test_sessions),
                "train_sample_count": int(len(train_idx)),
                "test_sample_count": int(len(test_idx)),
                "target": section_input.target_column,
                "model": section_input.model,
                "seed": int(section_input.seed),
            }
        )

        for row_idx in range(len(test_meta)):
            fold_row = test_meta.loc[row_idx]
            prediction_rows.append(
                {
                    "fold": fold_index,
                    "sample_id": str(fold_row["sample_id"]),
                    "y_true": str(y_true[row_idx]),
                    "y_pred": str(y_pred[row_idx]),
                    "decision_value": score_payload["decision_value"][row_idx],
                    "decision_vector": score_payload["decision_vector"][row_idx],
                    "proba_value": score_payload["proba_value"][row_idx],
                    "proba_vector": score_payload["proba_vector"][row_idx],
                    "subject": fold_row["subject"],
                    "session": fold_row["session"],
                    "bas": fold_row["bas"],
                    "task": fold_row["task"],
                    "modality": fold_row["modality"],
                    "emotion": fold_row.get("emotion", pd.NA),
                    "coarse_affect": fold_row.get("coarse_affect", pd.NA),
                    "experiment_mode": section_input.cv_mode,
                    "train_subject": (
                        str(section_input.train_subject)
                        if section_input.cv_mode == "frozen_cross_person_transfer"
                        else pd.NA
                    ),
                    "test_subject": (
                        str(section_input.test_subject)
                        if section_input.cv_mode == "frozen_cross_person_transfer"
                        else pd.NA
                    ),
                }
            )

        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())

    return {
        "y": y,
        "splits": splits,
        "fold_rows": fold_rows,
        "split_rows": split_rows,
        "prediction_rows": prediction_rows,
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
        "interpretability_enabled": interpretability_enabled,
        "interpretability_fold_rows": interpretability_fold_rows,
        "interpretability_vectors": interpretability_vectors,
        "interpretability_fold_artifacts_path": section_input.report_dir
        / "interpretability_fold_explanations.csv",
        "interpretability_summary_path": section_input.report_dir / "interpretability_summary.json",
    }


def execute_interpretability(section_input: InterpretabilityInput) -> dict[str, Any]:
    caution_text = (
        "Linear coefficients are reported as model-behavior evidence only and must not be "
        "interpreted as direct neural localization."
    )
    if section_input.interpretability_enabled:
        pd.DataFrame(section_input.interpretability_fold_rows).to_csv(
            section_input.fold_artifacts_path, index=False
        )
        interpretability_summary: dict[str, Any] = {
            "enabled": True,
            "performed": True,
            "status": "performed",
            "reason": None,
            "caution": caution_text,
            "experiment_mode": section_input.cv_mode,
            "subject": str(section_input.subject),
            "model": section_input.model,
            "target": section_input.target_column,
            "n_fold_artifacts": int(len(section_input.interpretability_fold_rows)),
            "fold_artifacts_path": str(section_input.fold_artifacts_path.resolve()),
            "stability": section_input.compute_interpretability_stability_fn(
                section_input.interpretability_vectors
            ),
        }
    else:
        interpretability_summary = {
            "enabled": False,
            "performed": False,
            "status": "not_applicable",
            "reason": "Interpretability export is enabled only for within_subject_loso_session.",
            "caution": caution_text,
            "experiment_mode": section_input.cv_mode,
            "subject": None,
            "model": section_input.model,
            "target": section_input.target_column,
            "n_fold_artifacts": 0,
            "fold_artifacts_path": None,
            "stability": None,
        }
    section_input.summary_path.write_text(
        f"{json.dumps(interpretability_summary, indent=2)}\n",
        encoding="utf-8",
    )
    return interpretability_summary


def execute_evaluation(section_input: EvaluationInput) -> dict[str, Any]:
    overall_accuracy = float(accuracy_score(section_input.y_true_all, section_input.y_pred_all))
    overall_balanced = float(
        balanced_accuracy_score(section_input.y_true_all, section_input.y_pred_all)
    )
    overall_macro_f1 = float(
        f1_score(
            section_input.y_true_all,
            section_input.y_pred_all,
            average="macro",
            zero_division=0,
        )
    )
    labels_sorted = sorted(
        np.unique(
            np.concatenate(
                [np.asarray(section_input.y_true_all), np.asarray(section_input.y_pred_all)]
            )
        ).tolist()
    )
    cmatrix = confusion_matrix(
        section_input.y_true_all,
        section_input.y_pred_all,
        labels=labels_sorted,
    )

    metrics: dict[str, Any] = {
        "model": section_input.model,
        "target": section_input.target_column,
        "cv": section_input.cv_mode,
        "experiment_mode": section_input.cv_mode,
        "subject": (
            str(section_input.subject)
            if section_input.cv_mode == "within_subject_loso_session"
            else None
        ),
        "train_subject": (
            str(section_input.train_subject)
            if section_input.cv_mode == "frozen_cross_person_transfer"
            else None
        ),
        "test_subject": (
            str(section_input.test_subject)
            if section_input.cv_mode == "frozen_cross_person_transfer"
            else None
        ),
        "n_samples": int(len(section_input.y_true_all)),
        "n_features": int(section_input.x_matrix.shape[1]),
        "n_folds": int(len(section_input.fold_rows)),
        "accuracy": overall_accuracy,
        "balanced_accuracy": overall_balanced,
        "macro_f1": overall_macro_f1,
        "labels": labels_sorted,
        "confusion_matrix": cmatrix.tolist(),
        "spatial_compatibility": {
            "status": str(section_input.spatial_compatibility["status"]),
            "passed": bool(section_input.spatial_compatibility["passed"]),
            "n_groups_checked": int(section_input.spatial_compatibility["n_groups_checked"]),
            "reference_group_id": section_input.spatial_compatibility["reference_group_id"],
            "affine_atol": float(section_input.spatial_compatibility["affine_atol"]),
            "report_path": str(section_input.spatial_report_path.resolve()),
        },
    }

    if section_input.n_permutations > 0:
        metrics["permutation_test"] = section_input.evaluate_permutations_fn(
            pipeline_template=section_input.build_pipeline_fn(
                model_name=section_input.model,
                seed=section_input.seed,
            ),
            x_matrix=section_input.x_matrix,
            y=section_input.y,
            splits=section_input.splits,
            seed=section_input.seed,
            n_permutations=section_input.n_permutations,
            observed_accuracy=overall_accuracy,
        )

    metrics["interpretability"] = {
        "enabled": bool(section_input.interpretability_summary["enabled"]),
        "performed": bool(section_input.interpretability_summary["performed"]),
        "status": str(section_input.interpretability_summary["status"]),
        "summary_path": str(section_input.interpretability_summary_path.resolve()),
        "fold_artifacts_path": section_input.interpretability_summary["fold_artifacts_path"],
        "stability": section_input.interpretability_summary["stability"],
    }

    fold_rows = [dict(row) for row in section_input.fold_rows]
    split_rows = [dict(row) for row in section_input.split_rows]
    prediction_rows = [dict(row) for row in section_input.prediction_rows]
    for row in fold_rows:
        row["run_id"] = section_input.run_id
        row["config_file"] = section_input.config_filename
    for row in split_rows:
        row["run_id"] = section_input.run_id
        row["config_file"] = section_input.config_filename

    pd.DataFrame(fold_rows).to_csv(section_input.fold_metrics_path, index=False)
    pd.DataFrame(split_rows).to_csv(section_input.fold_splits_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(section_input.predictions_path, index=False)
    section_input.metrics_path.write_text(f"{json.dumps(metrics, indent=2)}\n", encoding="utf-8")
    return metrics
