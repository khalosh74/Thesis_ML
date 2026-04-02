from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Thesis_ML.config.metric_policy import classification_metric_score
from Thesis_ML.experiments.cv_split_plan import build_cv_split_plan
from Thesis_ML.experiments.section_models import DatasetSelectionInput
from Thesis_ML.experiments.sections import dataset_selection
from Thesis_ML.orchestration.stage_lock_rules import (
    evaluate_stage_lock_decision,
    get_preflight_stage_rule,
    preflight_experiment_ids,
)

_DEFAULT_REVISED_REGISTRY_PATH = Path("configs") / "decision_support_registry_revised_execution.json"
_DEFAULT_CONFIRMATORY_SCOPE_PATH = Path("configs") / "confirmatory" / "confirmatory_scope_v1.json"
_REQUIRED_HARD_LOCK_STAGES: tuple[str, ...] = (
    "E01",
    "E04",
    "E06",
    "E07",
    "E08",
    "E09",
    "E10",
    "E11",
)
_ADVISORY_STAGES: tuple[str, ...] = ("E02", "E03")


class PreflightReviewError(RuntimeError):
    """Raised when required preflight review inputs are missing or malformed."""


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise PreflightReviewError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PreflightReviewError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_column(frame: pd.DataFrame, logical_name: str, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    available = ", ".join(str(column) for column in frame.columns)
    raise PreflightReviewError(
        f"Missing required column '{logical_name}'. Expected one of {candidates}. "
        f"Available columns: {available}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review preflight decision-support stage evidence for E01-E11."
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="Campaign root containing run_log_export.csv and decision_support_summary.csv.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--experiment-id",
        type=str,
        help="Specific preflight experiment to review (E01-E11).",
    )
    target.add_argument(
        "--all-preflight",
        action="store_true",
        help="Review all preflight experiments (E01-E11).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=_DEFAULT_REVISED_REGISTRY_PATH,
        help=(
            "Decision-support registry used for context and model defaults. "
            "Defaults to configs/decision_support_registry_revised_execution.json"
        ),
    )
    parser.add_argument(
        "--emit-confirmatory-bundle",
        action="store_true",
        help=(
            "Emit preflight_reviews/confirmatory_selection_bundle.json from reviewed "
            "preflight stage outputs."
        ),
    )
    return parser.parse_args(argv)


def _candidate_from_key(option_key: str | None) -> dict[str, Any] | None:
    if option_key is None:
        return None
    try:
        payload = json.loads(option_key)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _key_from_fields(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    payload = {field: row.get(field) for field in fields}
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _majority_label(train_labels: np.ndarray) -> str:
    series = pd.Series(train_labels).astype(str)
    counts = series.value_counts()
    if counts.empty:
        raise PreflightReviewError("Cannot compute majority baseline from empty train labels.")
    max_count = int(counts.max())
    tied = sorted(str(label) for label, count in counts.items() if int(count) == max_count)
    return str(tied[0])


def _target_column(raw_target: str) -> str:
    target = str(raw_target).strip()
    if not target:
        raise PreflightReviewError("Encountered empty target while reconstructing baseline.")
    return target


def _compute_dummy_baseline(
    *,
    index_csv: Path,
    target: str,
    cv: str,
    subject: str | None,
    train_subject: str | None,
    test_subject: str | None,
    filter_task: str | None,
    filter_modality: str | None,
    seed: int,
) -> dict[str, Any]:
    selection_output = dataset_selection(
        DatasetSelectionInput(
            index_csv=index_csv,
            target_column=_target_column(target),
            cv_mode=str(cv),
            subject=subject,
            train_subject=train_subject,
            test_subject=test_subject,
            filter_task=filter_task,
            filter_modality=filter_modality,
        )
    )
    metadata_df = selection_output.selected_index_df.reset_index(drop=True)

    split_plan = build_cv_split_plan(
        metadata_df=metadata_df,
        target_column=_target_column(target),
        cv_mode=str(cv),
        subject=subject,
        train_subject=train_subject,
        test_subject=test_subject,
        seed=int(seed),
    )
    labels = metadata_df[_target_column(target)].astype(str).to_numpy()

    fold_rows: list[dict[str, Any]] = []
    for fold in split_plan.folds:
        y_train = labels[np.asarray(fold.train_idx, dtype=int)]
        y_test = labels[np.asarray(fold.test_idx, dtype=int)]
        majority = _majority_label(y_train)
        y_pred = np.full(shape=len(y_test), fill_value=majority, dtype=object)

        fold_rows.append(
            {
                "fold": int(fold.fold),
                "majority_class": str(majority),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "balanced_accuracy": float(
                    classification_metric_score(
                        y_true=y_test,
                        y_pred=y_pred,
                        metric_name="balanced_accuracy",
                    )
                ),
                "macro_f1": float(
                    classification_metric_score(
                        y_true=y_test,
                        y_pred=y_pred,
                        metric_name="macro_f1",
                    )
                ),
                "accuracy": float(
                    classification_metric_score(
                        y_true=y_test,
                        y_pred=y_pred,
                        metric_name="accuracy",
                    )
                ),
            }
        )

    if not fold_rows:
        raise PreflightReviewError("No folds were produced for baseline reconstruction.")

    frame = pd.DataFrame(fold_rows)
    return {
        "fold_rows": fold_rows,
        "mean_balanced_accuracy": float(frame["balanced_accuracy"].mean()),
        "mean_macro_f1": float(frame["macro_f1"].mean()),
        "mean_accuracy": float(frame["accuracy"].mean()),
    }


def _resolve_report_dir(
    artifact_path_text: str,
    *,
    campaign_root: Path,
) -> Path | None:
    text = str(artifact_path_text or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = (campaign_root / candidate).resolve()
    if candidate.is_dir():
        return candidate
    if candidate.is_file() and candidate.suffix.lower() == ".json":
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(payload, dict):
            artifacts = payload.get("artifacts")
            if isinstance(artifacts, dict):
                report_dir = artifacts.get("report_dir")
                if isinstance(report_dir, str) and report_dir.strip():
                    resolved = Path(report_dir)
                    if not resolved.is_absolute():
                        resolved = (campaign_root / resolved).resolve()
                    if resolved.exists():
                        return resolved
    return None


def _read_run_config(config_path_text: str, *, campaign_root: Path) -> dict[str, Any] | None:
    text = str(config_path_text or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (campaign_root / path).resolve()
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_param(config_payload: dict[str, Any] | None, field: str) -> Any:
    if not isinstance(config_payload, dict):
        return None
    if field in config_payload:
        return config_payload.get(field)
    config_used = config_payload.get("config_used")
    if isinstance(config_used, dict):
        params = config_used.get("params")
        if isinstance(params, dict) and field in params:
            return params.get(field)
    params_top = config_payload.get("params")
    if isinstance(params_top, dict) and field in params_top:
        return params_top.get(field)
    return None


def _summarize_fold_metrics(
    *,
    fold_metrics_path: Path,
    primary_metric_name: str,
    cv_mode: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if not fold_metrics_path.exists() or not fold_metrics_path.is_file():
        return None, f"missing_fold_metrics_csv:{fold_metrics_path}"

    try:
        frame = pd.read_csv(fold_metrics_path)
    except Exception as exc:
        return None, f"invalid_fold_metrics_csv:{fold_metrics_path} ({exc})"

    metric_column = (
        primary_metric_name
        if primary_metric_name in frame.columns
        else "balanced_accuracy"
        if "balanced_accuracy" in frame.columns
        else None
    )
    if metric_column is None:
        return (
            None,
            f"fold_metric_column_missing:{fold_metrics_path}"
            f"(needed={primary_metric_name},columns={frame.columns.tolist()})",
        )

    metric_values = pd.to_numeric(frame[metric_column], errors="coerce").dropna()
    if metric_values.empty:
        return None, f"empty_numeric_fold_metric:{fold_metrics_path}"

    values = metric_values.astype(float).tolist()
    fold_count = int(len(values))
    summary = {
        "primary_metric_column": str(metric_column),
        "fold_count": fold_count,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "values": values,
        "uncertainty_status": (
            "single_split_no_fold_variance"
            if str(cv_mode) == "frozen_cross_person_transfer"
            else "fold_spread_available"
        ),
    }
    return summary, None

def _parse_completed_runs(
    *,
    campaign_root: Path,
    run_rows: pd.DataFrame,
    experiment_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    resolved_rows: list[dict[str, Any]] = []

    config_col = _resolve_column(
        run_rows,
        "config_path",
        ("Config_File_or_Path", "config_path", "config_path_or_file"),
    )
    artifact_col = _resolve_column(run_rows, "artifact_path", ("Artifact_Path", "artifact_path"))
    status_col = _resolve_column(run_rows, "status", ("Result_Summary", "status"))
    run_id_col = _resolve_column(run_rows, "run_id", ("Run_ID", "run_id"))
    experiment_col = _resolve_column(run_rows, "experiment_id", ("Experiment_ID", "experiment_id"))

    for _, row in run_rows.iterrows():
        if str(row.get(experiment_col, "")).strip() != experiment_id:
            continue

        status = str(row.get(status_col, "")).strip()
        if status != "completed":
            continue

        run_id = str(row.get(run_id_col, "")).strip()
        config_path_text = str(row.get(config_col, ""))
        config_payload = _read_run_config(config_path_text, campaign_root=campaign_root)
        report_dir = _resolve_report_dir(str(row.get(artifact_col, "")), campaign_root=campaign_root)

        metrics_payload: dict[str, Any] = {}
        metrics_path = None
        if report_dir is not None:
            metrics_path = report_dir / "metrics.json"
            if metrics_path.exists() and metrics_path.is_file():
                try:
                    loaded_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_metrics, dict):
                        metrics_payload = loaded_metrics
                except Exception as exc:
                    warnings.append(f"invalid_metrics_json:{metrics_path} ({exc})")

        if not metrics_payload:
            warnings.append(f"metrics_unavailable_for_run:{run_id}")

        primary_metric_name = str(metrics_payload.get("primary_metric_name") or "balanced_accuracy")
        balanced_accuracy = _safe_float(metrics_payload.get("balanced_accuracy"))
        if balanced_accuracy is None:
            balanced_accuracy = _safe_float(metrics_payload.get("primary_metric_value"))
        macro_f1 = _safe_float(metrics_payload.get("macro_f1"))
        accuracy = _safe_float(metrics_payload.get("accuracy"))

        target_value = _extract_param(config_payload, "target")
        subject_value = _extract_param(config_payload, "subject")
        train_subject_value = _extract_param(config_payload, "train_subject")
        test_subject_value = _extract_param(config_payload, "test_subject")
        filter_task_value = _extract_param(config_payload, "filter_task")
        filter_modality_value = _extract_param(config_payload, "filter_modality")
        model_value = _extract_param(config_payload, "model")
        cv_value = _extract_param(config_payload, "cv")
        class_weight_policy_value = _extract_param(config_payload, "class_weight_policy")
        methodology_policy_value = _extract_param(config_payload, "methodology_policy_name")
        feature_space_value = _extract_param(config_payload, "feature_space")
        dimensionality_strategy_value = _extract_param(config_payload, "dimensionality_strategy")
        preprocessing_strategy_value = _extract_param(config_payload, "preprocessing_strategy")

        filter_task_text = _safe_text(filter_task_value)
        filter_modality_text = _safe_text(filter_modality_value)
        train_subject_text = _safe_text(train_subject_value)
        test_subject_text = _safe_text(test_subject_value)

        task_pooling_choice = "task_specific" if filter_task_text else "pooled_tasks"
        modality_pooling_choice = "modality_specific" if filter_modality_text else "pooled_modalities"
        transfer_direction = (
            f"{train_subject_text}->{test_subject_text}"
            if train_subject_text and test_subject_text
            else None
        )

        cv_mode = _safe_text(cv_value)
        fold_metrics_path = report_dir / "fold_metrics.csv" if report_dir is not None else None
        fold_summary: dict[str, Any] | None = None
        if fold_metrics_path is not None:
            fold_summary, fold_warning = _summarize_fold_metrics(
                fold_metrics_path=fold_metrics_path,
                primary_metric_name=primary_metric_name,
                cv_mode=cv_mode,
            )
            if fold_warning:
                warnings.append(fold_warning)

        baseline_summary: dict[str, Any] | None = None
        baseline_warning: str | None = None
        try:
            index_csv_raw = _extract_param(config_payload, "index_csv")
            if index_csv_raw is None:
                raise PreflightReviewError("run config is missing index_csv")
            index_csv = Path(str(index_csv_raw))
            if not index_csv.is_absolute():
                index_csv = (campaign_root / index_csv).resolve()

            baseline_summary = _compute_dummy_baseline(
                index_csv=index_csv,
                target=_safe_text(target_value),
                cv=cv_mode,
                subject=_safe_text(subject_value) or None,
                train_subject=train_subject_text or None,
                test_subject=test_subject_text or None,
                filter_task=filter_task_text or None,
                filter_modality=filter_modality_text or None,
                seed=int(_safe_float(_extract_param(config_payload, "seed")) or 42),
            )
        except Exception as exc:
            baseline_warning = f"dummy_baseline_unavailable_for_run:{run_id} ({exc})"
            warnings.append(baseline_warning)

        dummy_balanced_accuracy = (
            _safe_float(baseline_summary.get("mean_balanced_accuracy"))
            if isinstance(baseline_summary, dict)
            else None
        )
        delta_over_dummy = (
            float(balanced_accuracy) - float(dummy_balanced_accuracy)
            if balanced_accuracy is not None and dummy_balanced_accuracy is not None
            else None
        )

        resolved_rows.append(
            {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "status": status,
                "config_path": config_path_text,
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "primary_metric_name": primary_metric_name,
                "target": target_value,
                "subject": subject_value,
                "train_subject": train_subject_value,
                "test_subject": test_subject_value,
                "filter_task": filter_task_value,
                "filter_modality": filter_modality_value,
                "model": model_value,
                "cv": cv_value,
                "class_weight_policy": class_weight_policy_value,
                "methodology_policy_name": methodology_policy_value,
                "feature_space": feature_space_value,
                "dimensionality_strategy": dimensionality_strategy_value,
                "preprocessing_strategy": preprocessing_strategy_value,
                "task_pooling_choice": task_pooling_choice,
                "modality_pooling_choice": modality_pooling_choice,
                "transfer_direction": transfer_direction,
                "report_dir": str(report_dir.resolve()) if report_dir is not None else None,
                "metrics_path": str(metrics_path.resolve()) if metrics_path is not None else None,
                "fold_metrics_path": (
                    str(fold_metrics_path.resolve())
                    if fold_metrics_path is not None and fold_metrics_path.exists()
                    else None
                ),
                "uncertainty_summary": fold_summary,
                "dummy_baseline_summary": baseline_summary,
                "dummy_baseline_mean_balanced_accuracy": dummy_balanced_accuracy,
                "dummy_baseline_mean_macro_f1": (
                    _safe_float(baseline_summary.get("mean_macro_f1"))
                    if isinstance(baseline_summary, dict)
                    else None
                ),
                "dummy_baseline_mean_accuracy": (
                    _safe_float(baseline_summary.get("mean_accuracy"))
                    if isinstance(baseline_summary, dict)
                    else None
                ),
                "delta_over_dummy_balanced_accuracy": delta_over_dummy,
                "baseline_warning": baseline_warning,
            }
        )

    return resolved_rows, warnings


def _slice_winner_rows(
    *,
    completed_runs: list[dict[str, Any]],
    manipulated_fields: tuple[str, ...],
    slice_fields: tuple[str, ...],
) -> tuple[list[dict[str, Any]], str | None, dict[str, Any] | None, float | None, bool, bool]:
    by_slice: dict[str, list[dict[str, Any]]] = {}
    for row in completed_runs:
        metric_value = _safe_float(row.get("balanced_accuracy"))
        if metric_value is None:
            continue
        slice_key = _key_from_fields(row, slice_fields)
        by_slice.setdefault(slice_key, []).append(row)

    slice_rows: list[dict[str, Any]] = []
    winner_keys: list[str] = []
    margins: list[float] = []

    for slice_key in sorted(by_slice):
        slice_run_rows = by_slice[slice_key]
        by_option: dict[str, list[float]] = {}
        for run_row in slice_run_rows:
            option_key = _key_from_fields(run_row, manipulated_fields)
            by_option.setdefault(option_key, []).append(float(run_row["balanced_accuracy"]))

        option_scores = [
            {
                "option_key": option_key,
                "option_payload": _candidate_from_key(option_key),
                "mean_balanced_accuracy": float(np.mean(values)),
                "run_count": int(len(values)),
            }
            for option_key, values in by_option.items()
        ]
        option_scores.sort(key=lambda item: float(item["mean_balanced_accuracy"]), reverse=True)

        winner = option_scores[0] if option_scores else None
        runner_up = option_scores[1] if len(option_scores) > 1 else None
        margin = (
            float(winner["mean_balanced_accuracy"]) - float(runner_up["mean_balanced_accuracy"])
            if winner is not None and runner_up is not None
            else None
        )

        if winner is not None:
            winner_keys.append(str(winner["option_key"]))
        if margin is not None:
            margins.append(float(margin))

        slice_rows.append(
            {
                "slice_key": slice_key,
                "slice_payload": _candidate_from_key(slice_key),
                "winner_option_key": (str(winner["option_key"]) if winner else None),
                "winner_option": (winner.get("option_payload") if winner else None),
                "winner_mean_balanced_accuracy": (
                    float(winner["mean_balanced_accuracy"]) if winner else None
                ),
                "runner_up_option_key": (
                    str(runner_up["option_key"]) if runner_up else None
                ),
                "runner_up_option": (runner_up.get("option_payload") if runner_up else None),
                "runner_up_mean_balanced_accuracy": (
                    float(runner_up["mean_balanced_accuracy"]) if runner_up else None
                ),
                "margin_balanced_accuracy": margin,
            }
        )

    consistency_pass = bool(winner_keys) and len(set(winner_keys)) == 1
    candidate_key = str(winner_keys[0]) if consistency_pass else None
    candidate_payload = _candidate_from_key(candidate_key)

    mean_margin = float(np.mean(margins)) if margins else None
    sign_reversals_across_slices = (len(set(winner_keys)) > 1) if winner_keys else False

    return (
        slice_rows,
        candidate_key,
        candidate_payload,
        mean_margin,
        consistency_pass,
        sign_reversals_across_slices,
    )

def _winner_runner_uncertainty(
    *,
    completed_runs: list[dict[str, Any]],
    manipulated_fields: tuple[str, ...],
    candidate_winner_key: str | None,
) -> dict[str, Any]:
    option_fold_values: dict[str, list[float]] = {}
    option_run_balanced_accuracy: dict[str, list[float]] = {}
    for row in completed_runs:
        option_key = _key_from_fields(row, manipulated_fields)
        metric_value = _safe_float(row.get("balanced_accuracy"))
        if metric_value is not None:
            option_run_balanced_accuracy.setdefault(option_key, []).append(float(metric_value))
        uncertainty = row.get("uncertainty_summary")
        if isinstance(uncertainty, dict):
            values = uncertainty.get("values")
            if isinstance(values, list):
                option_fold_values.setdefault(option_key, []).extend(
                    [float(value) for value in values if _safe_float(value) is not None]
                )

    option_means = [
        (option_key, float(np.mean(values)))
        for option_key, values in option_run_balanced_accuracy.items()
        if values
    ]
    option_means.sort(key=lambda item: item[1], reverse=True)

    winner_key = (
        candidate_winner_key if candidate_winner_key is not None else (option_means[0][0] if option_means else None)
    )
    runner_up_key = None
    for option_key, _ in option_means:
        if option_key != winner_key:
            runner_up_key = option_key
            break

    winner_values = option_fold_values.get(str(winner_key), []) if winner_key is not None else []
    runner_up_values = option_fold_values.get(str(runner_up_key), []) if runner_up_key is not None else []

    winner_fold_mean = float(np.mean(winner_values)) if winner_values else None
    winner_fold_std = float(np.std(winner_values, ddof=0)) if winner_values else None
    runner_up_fold_mean = float(np.mean(runner_up_values)) if runner_up_values else None
    runner_up_fold_std = float(np.std(runner_up_values, ddof=0)) if runner_up_values else None

    raw_difference = (
        float(winner_fold_mean) - float(runner_up_fold_mean)
        if winner_fold_mean is not None and runner_up_fold_mean is not None
        else None
    )
    margin_vs_fold_std_pass = (
        raw_difference is not None
        and winner_fold_std is not None
        and float(raw_difference) >= float(winner_fold_std)
    )

    return {
        "winner_option": _candidate_from_key(winner_key) if winner_key is not None else None,
        "runner_up_option": _candidate_from_key(runner_up_key) if runner_up_key is not None else None,
        "winner_fold_mean": winner_fold_mean,
        "winner_fold_std": winner_fold_std,
        "runner_up_fold_mean": runner_up_fold_mean,
        "runner_up_fold_std": runner_up_fold_std,
        "winner_minus_runner_up": raw_difference,
        "margin_vs_fold_std_pass": bool(margin_vs_fold_std_pass),
    }


def _phase_artifact_for_experiment(experiment_id: str) -> str | None:
    stage1 = {"E01", "E02", "E03"}
    stage2 = {"E04", "E05"}
    stage3 = {"E06", "E07", "E08"}
    stage4 = {"E09", "E10", "E11"}
    if experiment_id in stage1:
        return "stage1_lock.json"
    if experiment_id in stage2:
        return "stage2_lock.json"
    if experiment_id in stage3:
        return "stage3_lock.json"
    if experiment_id in stage4:
        return "stage4_lock.json"
    return None


def _phase_experiment_ids_for_artifact(artifact_name: str) -> tuple[str, ...]:
    if artifact_name == "stage1_lock.json":
        return ("E01", "E02", "E03")
    if artifact_name == "stage2_lock.json":
        return ("E04", "E05")
    if artifact_name == "stage3_lock.json":
        return ("E06", "E07", "E08")
    if artifact_name == "stage4_lock.json":
        return ("E09", "E10", "E11")
    return tuple()


def _stage3_selected_model(
    *,
    campaign_root: Path,
    computed_reviews: dict[str, dict[str, Any]],
) -> str | None:
    if "E06" in computed_reviews:
        candidate = computed_reviews["E06"].get("candidate_winner")
        if isinstance(candidate, dict):
            model = candidate.get("model")
            if isinstance(model, str) and model.strip():
                return model.strip()
    review_path = campaign_root / "preflight_reviews" / "E06_review.json"
    if review_path.exists() and review_path.is_file():
        payload = _load_json_object(review_path, label="E06 review")
        candidate = payload.get("candidate_winner")
        if isinstance(candidate, dict):
            model = candidate.get("model")
            if isinstance(model, str) and model.strip():
                return model.strip()
    return None


def _rerun_commands(
    *,
    campaign_root: Path,
    selected_model: str,
    experiments: tuple[str, ...],
    reference_run: dict[str, Any] | None,
) -> list[str]:
    generated_registry = (
        Path("configs")
        / "generated"
        / f"decision_support_registry_stage3_{selected_model.lower()}.json"
    )
    commands = [
        "python scripts/prepare_dependent_rerun_registry.py "
        f"--selected-model {selected_model} "
        f"--experiments {' '.join(experiments)} "
        f"--output-registry {generated_registry}"
    ]

    output_root = campaign_root.parent.parent
    index_csv = Path("Data") / "processed" / "dataset_index.csv"
    data_root = Path("Data")
    cache_dir = Path("Data") / "processed" / "feature_cache"
    seed = 42
    if isinstance(reference_run, dict):
        config_path = reference_run.get("config_path")
        if isinstance(config_path, str) and config_path.strip():
            config_payload = _read_run_config(config_path, campaign_root=campaign_root)
        else:
            config_payload = None
        if isinstance(config_payload, dict):
            raw_index_csv = _extract_param(config_payload, "index_csv")
            if isinstance(raw_index_csv, str) and raw_index_csv.strip():
                index_csv = Path(raw_index_csv)
            raw_data_root = _extract_param(config_payload, "data_root")
            if isinstance(raw_data_root, str) and raw_data_root.strip():
                data_root = Path(raw_data_root)
            raw_cache_dir = _extract_param(config_payload, "cache_dir")
            if isinstance(raw_cache_dir, str) and raw_cache_dir.strip():
                cache_dir = Path(raw_cache_dir)
            raw_seed = _safe_float(_extract_param(config_payload, "seed"))
            if raw_seed is not None:
                seed = int(raw_seed)

    for experiment_id in experiments:
        commands.append(
            "thesisml-run-decision-support "
            f"--registry {generated_registry} "
            f"--experiment-id {experiment_id} "
            f"--index-csv {index_csv} "
            f"--data-root {data_root} "
            f"--cache-dir {cache_dir} "
            f"--output-root {output_root} "
            f"--seed {seed}"
        )
    return commands


def _write_markdown_review(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['experiment_id']} Preflight Review",
        "",
        "## Summary",
        f"- lock_status: {payload.get('lock_status')}",
        f"- auto_lock_allowed: {payload.get('auto_lock_allowed')}",
        f"- auto_lock_passed: {payload.get('auto_lock_passed')}",
        f"- manual_review_required: {payload.get('manual_review_required')}",
        f"- expected_completed_variants: {payload.get('expected_completed_variants')}",
        f"- completed_variants: {payload.get('completed_variants')}",
        f"- failed_variants: {payload.get('failed_variants')}",
        f"- blocked_variants: {payload.get('blocked_variants')}",
        "",
        "## Candidate",
        f"- candidate_winner: {json.dumps(payload.get('candidate_winner'))}",
        f"- candidate_winner_metric: {payload.get('candidate_winner_metric')}",
        f"- consistency_pass: {payload.get('consistency_pass')}",
        f"- mean_margin_to_runner_up: {payload.get('mean_margin_to_runner_up')}",
        f"- min_margin_pass: {payload.get('min_margin_pass')}",
        "",
        "## Baseline",
        f"- dummy_baseline_mean_balanced_accuracy: {payload.get('dummy_baseline_mean_balanced_accuracy')}",
        f"- dummy_baseline_mean_macro_f1: {payload.get('dummy_baseline_mean_macro_f1')}",
        f"- dummy_baseline_mean_accuracy: {payload.get('dummy_baseline_mean_accuracy')}",
        f"- delta_over_dummy_balanced_accuracy: {payload.get('delta_over_dummy_balanced_accuracy')}",
        f"- baseline_delta_pass: {payload.get('baseline_delta_pass')}",
        "",
        "## Uncertainty",
        f"- uncertainty_status: {payload.get('uncertainty_status')}",
        f"- winner_fold_mean: {payload.get('winner_fold_mean')}",
        f"- winner_fold_std: {payload.get('winner_fold_std')}",
        f"- runner_up_fold_mean: {payload.get('runner_up_fold_mean')}",
        f"- runner_up_fold_std: {payload.get('runner_up_fold_std')}",
        f"- margin_vs_fold_std_pass: {payload.get('margin_vs_fold_std_pass')}",
        f"- sign_reversals_across_slices: {payload.get('sign_reversals_across_slices')}",
        "",
        "## Slice Winners",
    ]

    slice_rows = payload.get("slice_level_winners")
    if isinstance(slice_rows, list) and slice_rows:
        for row in slice_rows:
            lines.append(
                "- "
                f"slice={json.dumps(row.get('slice_payload'))} "
                f"winner={json.dumps(row.get('winner_option'))} "
                f"runner_up={json.dumps(row.get('runner_up_option'))} "
                f"margin={row.get('margin_balanced_accuracy')}"
            )
    else:
        lines.append("- no_slice_winner_rows")

    lines.extend(["", "## Reasons"])
    reasons = payload.get("reasons")
    if isinstance(reasons, list) and reasons:
        for reason in reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Dependency Reruns",
            f"- dependency_reruns_required: {payload.get('dependency_reruns_required')}",
            f"- rerun_required: {payload.get('rerun_required')}",
        ]
    )

    rerun_commands = payload.get("rerun_commands")
    if isinstance(rerun_commands, list) and rerun_commands:
        lines.extend(["", "```bash"])
        lines.extend([str(command) for command in rerun_commands])
        lines.append("```")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _write_lock_review_template(path: Path, payload: dict[str, Any]) -> None:
    advisory_note = (
        "This stage is advisory by default and should not be treated as an automatic lock."
        if payload["experiment_id"] in {"E02", "E03"}
        else ""
    )

    lines = [
        f"# {payload['experiment_id']} Lock Review Template",
        "",
        "## Experiment Purpose",
        f"- {payload.get('experiment_title')}",
        "",
        "## Real Dataset Slice Used",
        f"- candidate_winner_slice: {json.dumps(payload.get('candidate_winner_slice_example'))}",
        "",
        "## Completion",
        f"- expected_completed_variants: {payload.get('expected_completed_variants')}",
        f"- completed_variants: {payload.get('completed_variants')}",
        f"- failed_variants: {payload.get('failed_variants')}",
        f"- blocked_variants: {payload.get('blocked_variants')}",
        "",
        "## Candidate",
        f"- candidate_winner: {json.dumps(payload.get('candidate_winner'))}",
        f"- candidate_winner_metric: {payload.get('candidate_winner_metric')}",
        "",
        "## Scientific Checks",
        f"- consistency_pass: {payload.get('consistency_pass')}",
        f"- baseline_delta_pass: {payload.get('baseline_delta_pass')}",
        f"- margin_vs_fold_std_pass: {payload.get('margin_vs_fold_std_pass')}",
        f"- dependency_reruns_required: {payload.get('dependency_reruns_required')}",
        f"- uncertainty_status: {payload.get('uncertainty_status')}",
        "",
        "## Human Decision",
        "- approve_lock: [ ]",
        "- hold_for_rerun: [ ]",
        "- advisory_only: [ ]",
    ]

    if advisory_note:
        lines.extend(["", f"- advisory_note: {advisory_note}"])

    lines.extend(
        [
            "",
            "## Thesis Note",
            "This lock decision was reviewed with completion checks, cross-slice consistency, "
            "dummy-baseline delta, and fold-level uncertainty evidence before any thesis claim "
            "was treated as final.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def _review_to_csv_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "record_type": "summary",
            "experiment_id": payload.get("experiment_id"),
            "expected_completed_variants": payload.get("expected_completed_variants"),
            "completed_variants": payload.get("completed_variants"),
            "failed_variants": payload.get("failed_variants"),
            "blocked_variants": payload.get("blocked_variants"),
            "candidate_winner": json.dumps(payload.get("candidate_winner"), ensure_ascii=True),
            "candidate_winner_metric": payload.get("candidate_winner_metric"),
            "mean_margin_to_runner_up": payload.get("mean_margin_to_runner_up"),
            "consistency_pass": payload.get("consistency_pass"),
            "min_margin_pass": payload.get("min_margin_pass"),
            "baseline_delta_pass": payload.get("baseline_delta_pass"),
            "manual_review_required": payload.get("manual_review_required"),
            "lock_status": payload.get("lock_status"),
            "dummy_baseline_mean_balanced_accuracy": payload.get(
                "dummy_baseline_mean_balanced_accuracy"
            ),
            "delta_over_dummy_balanced_accuracy": payload.get(
                "delta_over_dummy_balanced_accuracy"
            ),
            "winner_fold_mean": payload.get("winner_fold_mean"),
            "winner_fold_std": payload.get("winner_fold_std"),
            "runner_up_fold_mean": payload.get("runner_up_fold_mean"),
            "runner_up_fold_std": payload.get("runner_up_fold_std"),
            "margin_vs_fold_std_pass": payload.get("margin_vs_fold_std_pass"),
            "dependency_reruns_required": payload.get("dependency_reruns_required"),
            "reasons": "|".join(str(reason) for reason in payload.get("reasons", [])),
        }
    )

    for row in payload.get("slice_level_winners", []):
        rows.append(
            {
                "record_type": "slice",
                "experiment_id": payload.get("experiment_id"),
                "slice": json.dumps(row.get("slice_payload"), ensure_ascii=True),
                "winner_option": json.dumps(row.get("winner_option"), ensure_ascii=True),
                "runner_up_option": json.dumps(row.get("runner_up_option"), ensure_ascii=True),
                "winner_mean_balanced_accuracy": row.get("winner_mean_balanced_accuracy"),
                "runner_up_mean_balanced_accuracy": row.get("runner_up_mean_balanced_accuracy"),
                "margin_balanced_accuracy": row.get("margin_balanced_accuracy"),
            }
        )

    for row in payload.get("completed_runs", []):
        rows.append(
            {
                "record_type": "run",
                "experiment_id": payload.get("experiment_id"),
                "run_id": row.get("run_id"),
                "status": row.get("status"),
                "balanced_accuracy": row.get("balanced_accuracy"),
                "dummy_baseline_mean_balanced_accuracy": row.get(
                    "dummy_baseline_mean_balanced_accuracy"
                ),
                "delta_over_dummy_balanced_accuracy": row.get(
                    "delta_over_dummy_balanced_accuracy"
                ),
                "uncertainty_mean": (
                    row.get("uncertainty_summary", {}).get("mean")
                    if isinstance(row.get("uncertainty_summary"), dict)
                    else None
                ),
                "uncertainty_std": (
                    row.get("uncertainty_summary", {}).get("std")
                    if isinstance(row.get("uncertainty_summary"), dict)
                    else None
                ),
                "uncertainty_status": (
                    row.get("uncertainty_summary", {}).get("uncertainty_status")
                    if isinstance(row.get("uncertainty_summary"), dict)
                    else None
                ),
                "option": json.dumps(
                    {field: row.get(field) for field in payload.get("manipulated_factor_fields", [])},
                    ensure_ascii=True,
                ),
                "slice": json.dumps(
                    {field: row.get(field) for field in payload.get("comparison_slice_fields", [])},
                    ensure_ascii=True,
                ),
            }
        )

    return rows


def _write_review_outputs(
    *,
    campaign_root: Path,
    payload: dict[str, Any],
) -> dict[str, str]:
    reviews_dir = campaign_root / "preflight_reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = str(payload["experiment_id"])
    json_path = reviews_dir / f"{experiment_id}_review.json"
    csv_path = reviews_dir / f"{experiment_id}_review.csv"
    md_path = reviews_dir / f"{experiment_id}_review.md"
    lock_review_path = reviews_dir / f"{experiment_id}_lock_review.md"

    json_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    pd.DataFrame(_review_to_csv_rows(payload)).to_csv(csv_path, index=False)
    _write_markdown_review(md_path, payload)
    _write_lock_review_template(lock_review_path, payload)

    return {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "md": str(md_path.resolve()),
        "lock_review_md": str(lock_review_path.resolve()),
    }


def _load_summary_counts(
    *,
    summary_df: pd.DataFrame,
    experiment_id: str,
) -> dict[str, int] | None:
    if summary_df.empty:
        return None
    experiment_col = _resolve_column(summary_df, "experiment_id", ("experiment_id", "Experiment_ID"))
    matches = summary_df[summary_df[experiment_col].astype(str) == experiment_id]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return {
        "completed_variants": int(_safe_float(row.get("completed_variants")) or 0),
        "failed_variants": int(_safe_float(row.get("failed_variants")) or 0),
        "blocked_variants": int(_safe_float(row.get("blocked_variants")) or 0),
    }


def _collect_status_counts(
    *,
    run_rows: pd.DataFrame,
    experiment_id: str,
) -> dict[str, int]:
    experiment_col = _resolve_column(run_rows, "experiment_id", ("Experiment_ID", "experiment_id"))
    status_col = _resolve_column(run_rows, "status", ("Result_Summary", "status"))

    subset = run_rows[run_rows[experiment_col].astype(str) == experiment_id]
    counts: dict[str, int] = {}
    for value in subset[status_col].astype(str).tolist():
        counts[value] = int(counts.get(value, 0)) + 1
    return counts


def _load_experiment_titles(registry_payload: dict[str, Any]) -> dict[str, str]:
    titles: dict[str, str] = {}
    experiments = registry_payload.get("experiments")
    if not isinstance(experiments, list):
        return titles
    for row in experiments:
        if not isinstance(row, dict):
            continue
        experiment_id = str(row.get("experiment_id") or "").strip()
        if not experiment_id:
            continue
        titles[experiment_id] = str(row.get("title") or "").strip()
    return titles


def _relative_review_json_path(*, campaign_root: Path, experiment_id: str) -> str:
    path = campaign_root / "preflight_reviews" / f"{experiment_id}_review.json"
    return str(path.relative_to(campaign_root))


def _load_stage_review_payload_for_bundle(
    *,
    campaign_root: Path,
    experiment_id: str,
    computed_reviews: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    experiment_key = str(experiment_id).strip().upper()
    relative_path = _relative_review_json_path(campaign_root=campaign_root, experiment_id=experiment_key)
    if experiment_key in computed_reviews:
        payload = computed_reviews[experiment_key]
        if isinstance(payload, dict):
            return payload, relative_path
    review_path = campaign_root / relative_path
    if review_path.exists() and review_path.is_file():
        payload = _load_json_object(review_path, label=f"{experiment_key} review")
        return payload, relative_path
    return None, relative_path


def _candidate_value(payload: dict[str, Any], key: str) -> Any:
    candidate = payload.get("candidate_winner")
    if not isinstance(candidate, dict):
        return None
    return candidate.get(key)


def emit_confirmatory_selection_bundle(
    *,
    campaign_root: Path,
    computed_reviews: dict[str, dict[str, Any]],
    scope_config_path: Path = _DEFAULT_CONFIRMATORY_SCOPE_PATH,
) -> Path:
    resolved_scope_path = scope_config_path.resolve()
    scope_payload = _load_json_object(resolved_scope_path, label="confirmatory scope")
    scope_id = str(scope_payload.get("scope_id") or "confirmatory_scope_v1").strip()
    if not scope_id:
        raise PreflightReviewError(
            f"Invalid confirmatory scope_id in scope config: {resolved_scope_path}"
        )

    review_sources: dict[str, str] = {}
    notes: list[str] = []
    selected: dict[str, Any] = {
        "cv_transfer": "frozen_cross_person_transfer",
    }
    advisory: dict[str, Any] = {
        "task_pooling": None,
        "modality_pooling": None,
    }

    required_mapping: dict[str, tuple[str, str]] = {
        "E01": ("target", "target"),
        "E04": ("cv", "cv_within_subject"),
        "E06": ("model", "model"),
        "E07": ("class_weight_policy", "class_weight_policy"),
        "E08": ("methodology_policy_name", "methodology_policy_name"),
        "E09": ("feature_space", "feature_space"),
        "E10": ("dimensionality_strategy", "dimensionality_strategy"),
        "E11": ("preprocessing_strategy", "preprocessing_strategy"),
    }
    advisory_mapping: dict[str, tuple[str, str]] = {
        "E02": ("task_pooling_choice", "task_pooling"),
        "E03": ("modality_pooling_choice", "modality_pooling"),
    }

    freeze_ready = True
    manual_review_required = False

    for experiment_id in _REQUIRED_HARD_LOCK_STAGES:
        payload, source_rel = _load_stage_review_payload_for_bundle(
            campaign_root=campaign_root,
            experiment_id=experiment_id,
            computed_reviews=computed_reviews,
        )
        review_sources[experiment_id] = source_rel
        if payload is None:
            freeze_ready = False
            manual_review_required = True
            notes.append(f"{experiment_id}:missing_review")
            continue
        if bool(payload.get("manual_review_required")):
            freeze_ready = False
            manual_review_required = True
            notes.append(f"{experiment_id}:manual_review_required")
        candidate_key, selected_key = required_mapping[experiment_id]
        candidate_value = _candidate_value(payload, candidate_key)
        if candidate_value in (None, ""):
            freeze_ready = False
            manual_review_required = True
            notes.append(f"{experiment_id}:missing_candidate_{candidate_key}")
            continue
        selected[selected_key] = candidate_value
        if experiment_id == "E08":
            candidate_payload = payload.get("candidate_winner")
            if isinstance(candidate_payload, dict):
                for key in (
                    "tuning_search_space_id",
                    "tuning_search_space_version",
                    "tuning_inner_cv_scheme",
                    "tuning_inner_group_field",
                ):
                    value = candidate_payload.get(key)
                    if value not in (None, ""):
                        selected[key] = value

    for experiment_id in _ADVISORY_STAGES:
        payload, _ = _load_stage_review_payload_for_bundle(
            campaign_root=campaign_root,
            experiment_id=experiment_id,
            computed_reviews=computed_reviews,
        )
        candidate_key, advisory_key = advisory_mapping[experiment_id]
        if payload is None:
            advisory[advisory_key] = None
            notes.append(f"{experiment_id}:advisory_review_missing")
            continue
        advisory[advisory_key] = _candidate_value(payload, candidate_key)

    bundle_payload: dict[str, Any] = {
        "bundle_id": f"confirmatory_selection_bundle_{campaign_root.name}",
        "campaign_id": campaign_root.name,
        "scope_id": scope_id,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "review_sources": review_sources,
        "selected": selected,
        "advisory": advisory,
        "freeze_ready": bool(freeze_ready),
        "manual_review_required": bool(manual_review_required),
        "notes": sorted(set(str(note) for note in notes if str(note).strip())),
    }

    reviews_dir = campaign_root / "preflight_reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = reviews_dir / "confirmatory_selection_bundle.json"
    bundle_path.write_text(f"{json.dumps(bundle_payload, indent=2)}\n", encoding="utf-8")
    return bundle_path


def _apply_phase_artifact_review_update(
    *,
    campaign_root: Path,
    review_payload: dict[str, Any],
) -> None:
    experiment_id = str(review_payload.get("experiment_id") or "").strip()
    artifact_name = _phase_artifact_for_experiment(experiment_id)
    if artifact_name is None:
        return
    artifact_path = campaign_root / artifact_name
    if not artifact_path.exists() or not artifact_path.is_file():
        return

    phase_payload = _load_json_object(artifact_path, label=f"{artifact_name}")
    review_by_experiment = phase_payload.get("preflight_review_by_experiment")
    if not isinstance(review_by_experiment, dict):
        review_by_experiment = {}

    review_by_experiment[experiment_id] = {
        "lock_status": review_payload.get("lock_status"),
        "candidate_winner": review_payload.get("candidate_winner"),
        "candidate_winner_metric": review_payload.get("candidate_winner_metric"),
        "consistency_pass": review_payload.get("consistency_pass"),
        "min_margin_pass": review_payload.get("min_margin_pass"),
        "baseline_delta_pass": review_payload.get("baseline_delta_pass"),
        "manual_review_required": review_payload.get("manual_review_required"),
        "review_artifacts": review_payload.get("review_artifacts"),
        "dependency_reruns_required": review_payload.get("dependency_reruns_required"),
    }

    phase_ids = _phase_experiment_ids_for_artifact(artifact_name)
    phase_reviews = [
        review_by_experiment.get(exp_id)
        for exp_id in phase_ids
        if isinstance(review_by_experiment.get(exp_id), dict)
    ]

    if phase_reviews:
        manual_required = any(bool(row.get("manual_review_required")) for row in phase_reviews)
        consistency_pass = all(bool(row.get("consistency_pass")) for row in phase_reviews)
        min_margin_pass = all(bool(row.get("min_margin_pass")) for row in phase_reviews)
        baseline_delta_pass = all(bool(row.get("baseline_delta_pass")) for row in phase_reviews)
        dependency_reruns_required = any(
            bool(row.get("dependency_reruns_required")) for row in phase_reviews
        )
        lock_status = "manual_review_required" if manual_required else "auto_lock_passed"
        candidate_winner = {
            exp_id: review_by_experiment[exp_id].get("candidate_winner")
            for exp_id in phase_ids
            if isinstance(review_by_experiment.get(exp_id), dict)
        }
        candidate_winner_metric = {
            exp_id: review_by_experiment[exp_id].get("candidate_winner_metric")
            for exp_id in phase_ids
            if isinstance(review_by_experiment.get(exp_id), dict)
        }
        review_artifacts = [
            value
            for exp_id in phase_ids
            for value in (
                (review_by_experiment.get(exp_id) or {}).get("review_artifacts", {}).values()
                if isinstance((review_by_experiment.get(exp_id) or {}).get("review_artifacts"), dict)
                else []
            )
        ]
    else:
        lock_status = "not_reviewed"
        candidate_winner = None
        candidate_winner_metric = None
        consistency_pass = None
        min_margin_pass = None
        baseline_delta_pass = None
        manual_required = None
        review_artifacts = []
        dependency_reruns_required = False

    phase_payload["preflight_review_by_experiment"] = review_by_experiment
    phase_payload["lock_status"] = lock_status
    phase_payload["candidate_winner"] = candidate_winner
    phase_payload["candidate_winner_metric"] = candidate_winner_metric
    phase_payload["consistency_pass"] = consistency_pass
    phase_payload["min_margin_pass"] = min_margin_pass
    phase_payload["baseline_delta_pass"] = baseline_delta_pass
    phase_payload["manual_review_required"] = manual_required
    phase_payload["review_artifacts"] = review_artifacts
    phase_payload["dependency_reruns_required"] = bool(dependency_reruns_required)

    artifact_path.write_text(f"{json.dumps(phase_payload, indent=2)}\n", encoding="utf-8")


def review_preflight_experiment(
    *,
    campaign_root: Path,
    registry_payload: dict[str, Any],
    run_log_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    experiment_id: str,
    computed_reviews: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rule = get_preflight_stage_rule(experiment_id)
    titles = _load_experiment_titles(registry_payload)

    completed_runs, parse_warnings = _parse_completed_runs(
        campaign_root=campaign_root,
        run_rows=run_log_df,
        experiment_id=experiment_id,
    )
    status_counts = _collect_status_counts(run_rows=run_log_df, experiment_id=experiment_id)
    summary_counts = _load_summary_counts(summary_df=summary_df, experiment_id=experiment_id)

    completed_variants = int(status_counts.get("completed", 0))
    failed_variants = int(status_counts.get("failed", 0))
    blocked_variants = int(status_counts.get("blocked", 0))
    if summary_counts is not None:
        completed_variants = int(summary_counts.get("completed_variants", completed_variants))
        failed_variants = int(summary_counts.get("failed_variants", failed_variants))
        blocked_variants = int(summary_counts.get("blocked_variants", blocked_variants))

    (
        slice_rows,
        candidate_winner_key,
        candidate_winner,
        mean_margin,
        consistency_pass,
        sign_reversals_across_slices,
    ) = _slice_winner_rows(
        completed_runs=completed_runs,
        manipulated_fields=rule.manipulated_factor_fields,
        slice_fields=rule.comparison_slice_fields,
    )

    candidate_winner_runs = [
        row
        for row in completed_runs
        if candidate_winner_key is not None
        and _key_from_fields(row, rule.manipulated_factor_fields) == candidate_winner_key
    ]
    candidate_winner_metric = (
        float(np.mean([float(row["balanced_accuracy"]) for row in candidate_winner_runs]))
        if candidate_winner_runs
        else None
    )

    baseline_deltas = [
        float(row["delta_over_dummy_balanced_accuracy"])
        for row in candidate_winner_runs
        if _safe_float(row.get("delta_over_dummy_balanced_accuracy")) is not None
    ]
    delta_over_dummy_balanced_accuracy = (
        float(np.mean(baseline_deltas)) if baseline_deltas else None
    )
    dummy_baseline_mean_balanced_accuracy = (
        float(
            np.mean(
                [
                    float(row["dummy_baseline_mean_balanced_accuracy"])
                    for row in candidate_winner_runs
                    if _safe_float(row.get("dummy_baseline_mean_balanced_accuracy")) is not None
                ]
            )
        )
        if any(_safe_float(row.get("dummy_baseline_mean_balanced_accuracy")) is not None for row in candidate_winner_runs)
        else None
    )
    dummy_baseline_mean_macro_f1 = (
        float(
            np.mean(
                [
                    float(row["dummy_baseline_mean_macro_f1"])
                    for row in candidate_winner_runs
                    if _safe_float(row.get("dummy_baseline_mean_macro_f1")) is not None
                ]
            )
        )
        if any(_safe_float(row.get("dummy_baseline_mean_macro_f1")) is not None for row in candidate_winner_runs)
        else None
    )
    dummy_baseline_mean_accuracy = (
        float(
            np.mean(
                [
                    float(row["dummy_baseline_mean_accuracy"])
                    for row in candidate_winner_runs
                    if _safe_float(row.get("dummy_baseline_mean_accuracy")) is not None
                ]
            )
        )
        if any(_safe_float(row.get("dummy_baseline_mean_accuracy")) is not None for row in candidate_winner_runs)
        else None
    )

    baseline_delta_pass = (
        delta_over_dummy_balanced_accuracy is not None
        and float(delta_over_dummy_balanced_accuracy)
        >= float(rule.min_baseline_delta_balanced_accuracy)
    )

    uncertainty_summary = _winner_runner_uncertainty(
        completed_runs=completed_runs,
        manipulated_fields=rule.manipulated_factor_fields,
        candidate_winner_key=candidate_winner_key,
    )

    cv_values = {
        str(row.get("cv") or "").strip() for row in completed_runs if str(row.get("cv") or "").strip()
    }
    uncertainty_status = (
        "single_split_no_fold_variance"
        if cv_values == {"frozen_cross_person_transfer"}
        else "fold_spread_available"
    )

    selected_model = _stage3_selected_model(campaign_root=campaign_root, computed_reviews=computed_reviews)
    dependency_reruns_required = False
    rerun_required = False
    rerun_commands: list[str] = []

    if experiment_id == "E06":
        if isinstance(candidate_winner, dict):
            model_name = _safe_text(candidate_winner.get("model"))
            if model_name and model_name.lower() != "ridge":
                dependency_reruns_required = True
                rerun_required = True
                rerun_commands = _rerun_commands(
                    campaign_root=campaign_root,
                    selected_model=model_name,
                    experiments=("E07", "E08"),
                    reference_run=(completed_runs[0] if completed_runs else None),
                )
    elif experiment_id in {"E07", "E08"}:
        if selected_model is not None and selected_model.lower() != "ridge":
            dependency_reruns_required = True
            rerun_required = True
            rerun_commands = _rerun_commands(
                campaign_root=campaign_root,
                selected_model=selected_model,
                experiments=(experiment_id,),
                reference_run=(completed_runs[0] if completed_runs else None),
            )

    decision = evaluate_stage_lock_decision(
        rule=rule,
        completed_variants=completed_variants,
        failed_variants=failed_variants,
        blocked_variants=blocked_variants,
        consistency_pass=consistency_pass,
        mean_margin_balanced_accuracy=mean_margin,
        baseline_delta_pass=baseline_delta_pass,
        margin_vs_fold_std_pass=bool(uncertainty_summary.get("margin_vs_fold_std_pass")),
        dependency_reruns_required=dependency_reruns_required,
    )

    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "experiment_title": titles.get(experiment_id, ""),
        "expected_completed_variants": int(rule.expected_completed_variants),
        "completed_variants": int(completed_variants),
        "failed_variants": int(failed_variants),
        "blocked_variants": int(blocked_variants),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "manipulated_factor_fields": list(rule.manipulated_factor_fields),
        "comparison_slice_fields": list(rule.comparison_slice_fields),
        "candidate_winner": candidate_winner,
        "candidate_winner_metric": candidate_winner_metric,
        "candidate_winner_metric_name": "balanced_accuracy",
        "slice_level_winners": slice_rows,
        "mean_margin_to_runner_up": mean_margin,
        "consistency_pass": bool(consistency_pass),
        "sign_reversals_across_slices": bool(sign_reversals_across_slices),
        "auto_lock_allowed": bool(rule.auto_lock_allowed),
        "lock_status": decision["lock_status"],
        "auto_lock_passed": bool(decision["auto_lock_passed"]),
        "manual_review_required": bool(decision["manual_review_required"]),
        "min_margin_balanced_accuracy": float(rule.min_margin_balanced_accuracy),
        "min_margin_pass": bool(decision["min_margin_pass"]),
        "min_baseline_delta_balanced_accuracy": float(rule.min_baseline_delta_balanced_accuracy),
        "dummy_baseline_mean_balanced_accuracy": dummy_baseline_mean_balanced_accuracy,
        "dummy_baseline_mean_macro_f1": dummy_baseline_mean_macro_f1,
        "dummy_baseline_mean_accuracy": dummy_baseline_mean_accuracy,
        "delta_over_dummy_balanced_accuracy": delta_over_dummy_balanced_accuracy,
        "baseline_delta_pass": bool(decision["baseline_delta_pass"]),
        "winner_fold_mean": uncertainty_summary.get("winner_fold_mean"),
        "winner_fold_std": uncertainty_summary.get("winner_fold_std"),
        "runner_up_fold_mean": uncertainty_summary.get("runner_up_fold_mean"),
        "runner_up_fold_std": uncertainty_summary.get("runner_up_fold_std"),
        "winner_minus_runner_up": uncertainty_summary.get("winner_minus_runner_up"),
        "margin_vs_fold_std_pass": bool(decision["margin_vs_fold_std_pass"]),
        "uncertainty_status": uncertainty_status,
        "dependency_reruns_required": bool(decision["dependency_reruns_required"]),
        "rerun_required": bool(rerun_required),
        "rerun_commands": rerun_commands,
        "reasons": list(decision["reasons"]),
        "warnings": list(sorted(set(parse_warnings))),
        "completed_runs": completed_runs,
        "candidate_winner_slice_example": (
            slice_rows[0].get("slice_payload")
            if slice_rows and isinstance(slice_rows[0], dict)
            else None
        ),
    }

    review_artifacts = _write_review_outputs(campaign_root=campaign_root, payload=payload)
    payload["review_artifacts"] = dict(review_artifacts)

    json_path = Path(review_artifacts["json"])
    json_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    _apply_phase_artifact_review_update(campaign_root=campaign_root, review_payload=payload)

    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    campaign_root = args.campaign_root.resolve()

    if not campaign_root.exists() or not campaign_root.is_dir():
        raise PreflightReviewError(f"campaign root does not exist: {campaign_root}")

    run_log_path = campaign_root / "run_log_export.csv"
    summary_path = campaign_root / "decision_support_summary.csv"
    if not run_log_path.exists():
        raise PreflightReviewError(f"run_log_export.csv not found at campaign root: {run_log_path}")
    if not summary_path.exists():
        raise PreflightReviewError(
            f"decision_support_summary.csv not found at campaign root: {summary_path}"
        )

    registry_path = args.registry.resolve()
    registry_payload = _load_json_object(registry_path, label="decision-support registry")

    run_log_df = pd.read_csv(run_log_path)
    summary_df = pd.read_csv(summary_path)

    if args.all_preflight:
        experiment_ids = list(preflight_experiment_ids())
    else:
        experiment_ids = [str(args.experiment_id).strip().upper()]

    computed_reviews: dict[str, dict[str, Any]] = {}
    for experiment_id in experiment_ids:
        payload = review_preflight_experiment(
            campaign_root=campaign_root,
            registry_payload=registry_payload,
            run_log_df=run_log_df,
            summary_df=summary_df,
            experiment_id=experiment_id,
            computed_reviews=computed_reviews,
        )
        computed_reviews[experiment_id] = payload

    review_dir = campaign_root / "preflight_reviews"
    print(f"Wrote preflight review artifacts to: {review_dir}")
    if args.emit_confirmatory_bundle:
        bundle_path = emit_confirmatory_selection_bundle(
            campaign_root=campaign_root,
            computed_reviews=computed_reviews,
        )
        print(f"Wrote confirmatory selection bundle: {bundle_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
