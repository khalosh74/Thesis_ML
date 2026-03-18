from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

from Thesis_ML.config.metric_policy import metric_bundle, metric_scorer
from Thesis_ML.experiments.evidence_statistics import build_calibration_outputs
from Thesis_ML.experiments.metrics import classification_metric_score
from Thesis_ML.experiments.tuning_search_spaces import get_search_space

if TYPE_CHECKING:
    from Thesis_ML.experiments.section_models import (
        EvaluationInput,
        InterpretabilityInput,
        ModelFitInput,
    )


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
    tuning_summary_path = (
        section_input.tuning_summary_path
        if section_input.tuning_summary_path is not None
        else section_input.report_dir / "tuning_summary.json"
    )
    tuning_best_params_path = (
        section_input.tuning_best_params_path
        if section_input.tuning_best_params_path is not None
        else section_input.report_dir / "best_params_per_fold.csv"
    )
    methodology_policy_name = str(section_input.methodology_policy_name).strip()
    tuning_enabled = bool(section_input.tuning_enabled)
    if methodology_policy_name == "fixed_baselines_only" and tuning_enabled:
        raise ValueError(
            "methodology_policy_name='fixed_baselines_only' forbids tuning_enabled=true."
        )
    if methodology_policy_name not in {"fixed_baselines_only", "grouped_nested_tuning"}:
        raise ValueError(
            "Unsupported methodology_policy_name. "
            "Allowed values: fixed_baselines_only, grouped_nested_tuning."
        )

    if section_input.interpretability_enabled is None:
        interpretability_enabled = section_input.cv_mode == "within_subject_loso_session"
    else:
        interpretability_enabled = bool(section_input.interpretability_enabled)
    interpretability_fold_rows: list[dict[str, Any]] = []
    interpretability_vectors: list[np.ndarray] = []
    interpretability_dir: Path | None = None
    if interpretability_enabled:
        interpretability_dir = section_input.report_dir / "interpretability"
        interpretability_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
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
        if tuning_enabled and methodology_policy_name == "grouped_nested_tuning":
            if section_input.model == "dummy":
                estimator.fit(section_input.x_matrix[train_idx], y[train_idx])
                tuning_rows.append(
                    {
                        "fold": int(fold_index),
                        "status": "skipped_control_model",
                        "model": section_input.model,
                        "best_score": pd.NA,
                        "best_params_json": "{}",
                        "n_candidates": 0,
                        "search_space_id": section_input.tuning_search_space_id,
                        "search_space_version": section_input.tuning_search_space_version,
                        "primary_metric_name": section_input.primary_metric_name,
                        "inner_group_field": section_input.tuning_inner_group_field,
                    }
                )
            else:
                if section_input.tuning_search_space_id is None:
                    raise ValueError("grouped_nested_tuning requires tuning_search_space_id.")
                resolved_space_version, param_grid = get_search_space(
                    section_input.tuning_search_space_id,
                    section_input.model,
                )
                declared_space_version = section_input.tuning_search_space_version
                if declared_space_version and declared_space_version != resolved_space_version:
                    raise ValueError(
                        "Declared tuning_search_space_version does not match search-space registry version."
                    )
                inner_groups = groups[train_idx]
                if len(np.unique(inner_groups)) < 2:
                    raise ValueError(
                        "Grouped nested tuning requires at least two inner groups in training data."
                    )
                search = GridSearchCV(
                    estimator=clone(pipeline_template),
                    param_grid=param_grid,
                    scoring=metric_scorer(section_input.primary_metric_name),
                    cv=LeaveOneGroupOut(),
                    refit=True,
                    n_jobs=1,
                )
                search.fit(
                    section_input.x_matrix[train_idx],
                    y[train_idx],
                    groups=inner_groups,
                )
                estimator = search.best_estimator_
                tuning_rows.append(
                    {
                        "fold": int(fold_index),
                        "status": "tuned",
                        "model": section_input.model,
                        "best_score": float(search.best_score_),
                        "best_params_json": json.dumps(search.best_params_, sort_keys=True),
                        "n_candidates": int(len(search.cv_results_["params"])),
                        "search_space_id": section_input.tuning_search_space_id,
                        "search_space_version": resolved_space_version,
                        "primary_metric_name": section_input.primary_metric_name,
                        "inner_group_field": section_input.tuning_inner_group_field,
                    }
                )
        else:
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
                "accuracy": classification_metric_score(
                    y_true=y_true,
                    y_pred=y_pred,
                    metric_name="accuracy",
                ),
                "balanced_accuracy": classification_metric_score(
                    y_true=y_true,
                    y_pred=y_pred,
                    metric_name="balanced_accuracy",
                ),
                "macro_f1": classification_metric_score(
                    y_true=y_true,
                    y_pred=y_pred,
                    metric_name="macro_f1",
                ),
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

    tuning_summary = {
        "methodology_policy_name": methodology_policy_name,
        "tuning_enabled": bool(tuning_enabled),
        "status": (
            "performed"
            if tuning_enabled and any(row["status"] == "tuned" for row in tuning_rows)
            else "disabled"
        ),
        "primary_metric_name": section_input.primary_metric_name,
        "search_space_id": section_input.tuning_search_space_id,
        "search_space_version": section_input.tuning_search_space_version,
        "inner_cv_scheme": section_input.tuning_inner_cv_scheme,
        "inner_group_field": section_input.tuning_inner_group_field,
        "total_outer_folds": int(len(splits)),
        "n_tuned_folds": int(sum(row["status"] == "tuned" for row in tuning_rows)),
        "n_skipped_folds": int(sum(row["status"] != "tuned" for row in tuning_rows)),
        "best_params_path": str(tuning_best_params_path.resolve()),
    }
    tuning_summary_path.write_text(
        f"{json.dumps(tuning_summary, indent=2)}\n",
        encoding="utf-8",
    )
    tuning_frame = pd.DataFrame(tuning_rows)
    if tuning_frame.empty:
        tuning_frame = pd.DataFrame(
            columns=[
                "fold",
                "status",
                "model",
                "best_score",
                "best_params_json",
                "n_candidates",
                "search_space_id",
                "search_space_version",
                "primary_metric_name",
                "inner_group_field",
            ]
        )
    tuning_frame.to_csv(tuning_best_params_path, index=False)

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
        "tuning_summary": tuning_summary,
        "tuning_records": tuning_rows,
        "tuning_summary_path": tuning_summary_path,
        "tuning_best_params_path": tuning_best_params_path,
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


def _compute_subgroup_metrics(
    prediction_rows: list[dict[str, Any]],
    *,
    subgroup_dimensions: list[str],
    min_samples_per_group: int,
    min_classes_per_group: int,
    report_small_groups: bool,
    strict_insufficient_data_rows: bool,
) -> list[dict[str, Any]]:
    if not prediction_rows:
        return []
    frame = pd.DataFrame(prediction_rows)
    subgroup_rows: list[dict[str, Any]] = []
    for requested_dimension in subgroup_dimensions:
        if requested_dimension == "label":
            group_column = "y_true"
        else:
            group_column = requested_dimension
        if group_column not in frame.columns:
            continue
        for group_value, subset in frame.groupby(group_column, dropna=False):
            n_samples = int(len(subset))
            y_true = subset["y_true"].astype(str).tolist()
            y_pred = subset["y_pred"].astype(str).tolist()
            n_classes = int(pd.Series(y_true).nunique(dropna=False))
            class_distribution = (
                pd.Series(y_true)
                .value_counts(dropna=False)
                .sort_index()
                .astype(int)
                .to_dict()
            )
            interpretable = (
                n_samples >= int(min_samples_per_group)
                and n_classes >= int(min_classes_per_group)
            )
            insufficient_reasons: list[str] = []
            if n_samples < int(min_samples_per_group):
                insufficient_reasons.append("min_samples")
            if n_classes < int(min_classes_per_group):
                insufficient_reasons.append("min_classes")
            if not interpretable and not (
                bool(report_small_groups) or bool(strict_insufficient_data_rows)
            ):
                continue
            # For single-class or undersized slices, classification scores are not
            # scientifically stable. Emit descriptive rows with null metrics.
            can_compute_metrics = bool(interpretable)
            subgroup_rows.append(
                {
                    "subgroup_key": requested_dimension,
                    "subgroup_value": str(group_value),
                    "n_samples": n_samples,
                    "n_classes": n_classes,
                    "class_distribution_json": json.dumps(class_distribution, sort_keys=True),
                    "insufficient_data_reasons": "|".join(insufficient_reasons),
                    "status": "ok" if interpretable else "insufficient_data",
                    "interpretable": bool(interpretable),
                    "balanced_accuracy": (
                        classification_metric_score(
                            y_true=y_true,
                            y_pred=y_pred,
                            metric_name="balanced_accuracy",
                        )
                        if can_compute_metrics
                        else None
                    ),
                    "macro_f1": (
                        classification_metric_score(
                            y_true=y_true,
                            y_pred=y_pred,
                            metric_name="macro_f1",
                        )
                        if can_compute_metrics
                        else None
                    ),
                    "accuracy": (
                        classification_metric_score(
                            y_true=y_true,
                            y_pred=y_pred,
                            metric_name="accuracy",
                        )
                        if can_compute_metrics
                        else None
                    ),
                }
            )
    return subgroup_rows


def execute_evaluation(section_input: EvaluationInput) -> dict[str, Any]:
    report_dir = section_input.metrics_path.parent
    subgroup_metrics_json_path = (
        section_input.subgroup_metrics_json_path
        if section_input.subgroup_metrics_json_path is not None
        else report_dir / "subgroup_metrics.json"
    )
    subgroup_metrics_csv_path = (
        section_input.subgroup_metrics_csv_path
        if section_input.subgroup_metrics_csv_path is not None
        else report_dir / "subgroup_metrics.csv"
    )
    tuning_summary_path = (
        section_input.tuning_summary_path
        if section_input.tuning_summary_path is not None
        else report_dir / "tuning_summary.json"
    )
    tuning_best_params_path = (
        section_input.tuning_best_params_path
        if section_input.tuning_best_params_path is not None
        else report_dir / "best_params_per_fold.csv"
    )
    calibration_summary_path = (
        section_input.calibration_summary_path
        if section_input.calibration_summary_path is not None
        else report_dir / "calibration_summary.json"
    )
    calibration_table_path = (
        section_input.calibration_table_path
        if section_input.calibration_table_path is not None
        else report_dir / "calibration_table.csv"
    )
    metric_values = metric_bundle(
        section_input.y_true_all,
        section_input.y_pred_all,
        metric_names=("accuracy", "balanced_accuracy", "macro_f1"),
    )
    overall_accuracy = float(metric_values["accuracy"])
    overall_balanced = float(metric_values["balanced_accuracy"])
    overall_macro_f1 = float(metric_values["macro_f1"])
    primary_metric_name = str(section_input.primary_metric_name)
    primary_metric_value = classification_metric_score(
        section_input.y_true_all,
        section_input.y_pred_all,
        metric_name=primary_metric_name,
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
        "methodology_policy_name": section_input.methodology_policy_name,
        "evidence_run_role": str(section_input.evidence_run_role),
        "repeat_id": int(section_input.repeat_id),
        "repeat_count": int(section_input.repeat_count),
        "base_run_id": (
            str(section_input.base_run_id)
            if section_input.base_run_id is not None
            else str(section_input.run_id)
        ),
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": primary_metric_value,
        "tuning_enabled": bool(section_input.methodology_policy_name == "grouped_nested_tuning"),
        "tuning_summary_path": str(tuning_summary_path.resolve()),
        "tuning_best_params_path": str(tuning_best_params_path.resolve()),
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
        permutation_metric_name = str(
            section_input.permutation_metric_name or section_input.primary_metric_name
        )
        permutation_payload = section_input.evaluate_permutations_fn(
            pipeline_template=section_input.build_pipeline_fn(
                model_name=section_input.model,
                seed=section_input.seed,
            ),
            x_matrix=section_input.x_matrix,
            y=section_input.y,
            splits=section_input.splits,
            seed=section_input.seed,
            n_permutations=section_input.n_permutations,
            metric_name=permutation_metric_name,
            observed_metric=classification_metric_score(
                section_input.y_true_all,
                section_input.y_pred_all,
                metric_name=permutation_metric_name,
            ),
        )
        permutation_payload["alpha"] = float(section_input.permutation_alpha)
        permutation_payload["minimum_required"] = int(section_input.permutation_minimum_required)
        permutation_payload["meets_minimum"] = bool(
            int(permutation_payload.get("n_permutations", 0))
            >= int(section_input.permutation_minimum_required)
        )
        p_value = permutation_payload.get("p_value")
        passes_threshold = (
            isinstance(p_value, (int, float))
            and float(p_value) <= float(section_input.permutation_alpha)
        )
        permutation_payload["passes_threshold"] = bool(passes_threshold)
        permutation_payload["require_pass_for_validity"] = bool(
            section_input.permutation_require_pass_for_validity
        )
        permutation_payload["interpretation_status"] = (
            "passes_threshold"
            if bool(passes_threshold)
            else "fails_threshold"
        )
        metrics["permutation_test"] = permutation_payload

    metrics["interpretability"] = {
        "enabled": bool(section_input.interpretability_summary["enabled"]),
        "performed": bool(section_input.interpretability_summary["performed"]),
        "status": str(section_input.interpretability_summary["status"]),
        "summary_path": str(section_input.interpretability_summary_path.resolve()),
        "fold_artifacts_path": section_input.interpretability_summary["fold_artifacts_path"],
        "stability": section_input.interpretability_summary["stability"],
    }

    subgroup_rows = (
        _compute_subgroup_metrics(
            prediction_rows=section_input.prediction_rows,
            subgroup_dimensions=list(section_input.subgroup_dimensions),
            min_samples_per_group=int(section_input.subgroup_min_samples_per_group),
            min_classes_per_group=int(section_input.subgroup_min_classes_per_group),
            report_small_groups=bool(section_input.subgroup_report_small_groups),
            strict_insufficient_data_rows=bool(section_input.confirmatory_guardrails_enabled),
        )
        if section_input.subgroup_reporting_enabled
        else []
    )
    subgroup_payload = {
        "enabled": bool(section_input.subgroup_reporting_enabled),
        "confirmatory_guardrails_enabled": bool(section_input.confirmatory_guardrails_enabled),
        "evidence_role": str(section_input.subgroup_evidence_role),
        "primary_evidence_substitution_allowed": bool(
            section_input.subgroup_primary_evidence_allowed
        ),
        "dimensions_requested": list(section_input.subgroup_dimensions),
        "min_samples_per_group": int(section_input.subgroup_min_samples_per_group),
        "min_classes_per_group": int(section_input.subgroup_min_classes_per_group),
        "report_small_groups": bool(section_input.subgroup_report_small_groups),
        "generated": bool(subgroup_rows),
        "n_rows": int(len(subgroup_rows)),
        "json_path": str(subgroup_metrics_json_path.resolve()),
        "csv_path": str(subgroup_metrics_csv_path.resolve()),
        "rows": subgroup_rows,
    }
    subgroup_metrics_json_path.write_text(
        f"{json.dumps(subgroup_payload, indent=2)}\n",
        encoding="utf-8",
    )
    subgroup_frame = pd.DataFrame(subgroup_rows)
    if subgroup_frame.empty:
        subgroup_frame = pd.DataFrame(
            columns=[
                "subgroup_key",
                "subgroup_value",
                "n_samples",
                "n_classes",
                "class_distribution_json",
                "insufficient_data_reasons",
                "status",
                "interpretable",
                "balanced_accuracy",
                "macro_f1",
                "accuracy",
            ]
        )
    subgroup_frame.to_csv(subgroup_metrics_csv_path, index=False)
    metrics["subgroup_reporting"] = {
        key: value for key, value in subgroup_payload.items() if key != "rows"
    }

    if section_input.calibration_enabled:
        calibration_summary, calibration_table = build_calibration_outputs(
            section_input.prediction_rows,
            n_bins=int(section_input.calibration_n_bins),
        )
        if (
            str(calibration_summary.get("status")) == "not_applicable"
            and bool(section_input.calibration_require_probabilities_for_validity)
        ):
            calibration_summary["status"] = "failed"
            calibration_summary["reason"] = "probabilities_required_but_missing"
            calibration_summary["performed"] = False
    else:
        calibration_summary = {
            "status": "not_applicable",
            "reason": "calibration_disabled_by_policy",
            "performed": False,
            "n_bins": int(section_input.calibration_n_bins),
            "n_samples": int(len(section_input.prediction_rows)),
            "ece": None,
            "brier_score": None,
        }
        calibration_table = pd.DataFrame(
            columns=[
                "bin_index",
                "bin_lower",
                "bin_upper",
                "n_samples",
                "mean_confidence",
                "empirical_accuracy",
            ]
        )
    calibration_summary["enabled"] = bool(section_input.calibration_enabled)
    calibration_summary["require_probabilities_for_validity"] = bool(
        section_input.calibration_require_probabilities_for_validity
    )
    calibration_summary["policy_status"] = (
        "probabilities_required_for_validity"
        if bool(section_input.calibration_require_probabilities_for_validity)
        else "required_if_probabilities_available"
    )
    calibration_summary["probability_support_detected"] = bool(
        str(calibration_summary.get("status")) == "performed"
    )
    calibration_summary["summary_path"] = str(calibration_summary_path.resolve())
    calibration_summary["table_path"] = str(calibration_table_path.resolve())
    calibration_summary_path.write_text(
        f"{json.dumps(calibration_summary, indent=2)}\n",
        encoding="utf-8",
    )
    if calibration_table.empty:
        calibration_table = pd.DataFrame(
            columns=[
                "bin_index",
                "bin_lower",
                "bin_upper",
                "n_samples",
                "mean_confidence",
                "empirical_accuracy",
            ]
        )
    calibration_table.to_csv(calibration_table_path, index=False)
    metrics["calibration"] = calibration_summary

    if isinstance(section_input.evidence_policy_effective, dict):
        metrics["evidence_policy_effective"] = dict(section_input.evidence_policy_effective)

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
