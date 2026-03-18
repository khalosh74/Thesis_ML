from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def aggregate_repeated_runs(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    group_keys: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not rows:
        run_frame = pd.DataFrame(
            columns=[*group_keys, "run_id", "base_run_id", "repeat_id", "metric_name", "metric_value"]
        )
        summary_frame = pd.DataFrame(
            columns=[*group_keys, "n_runs", "mean_metric", "std_metric", "min_metric", "max_metric"]
        )
        return run_frame, summary_frame

    frame = pd.DataFrame(rows)
    frame = frame.copy()
    frame["metric_value"] = pd.to_numeric(frame.get(metric_key), errors="coerce")
    frame = frame.dropna(subset=["metric_value"])
    frame["metric_value"] = frame["metric_value"].astype(float)
    frame["metric_name"] = str(metric_key)
    run_columns = [*group_keys, "run_id", "base_run_id", "repeat_id", "metric_name", "metric_value"]
    for column in run_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    run_frame = frame[run_columns].sort_values(
        [*group_keys, "base_run_id", "repeat_id", "run_id"],
        kind="stable",
    )

    if run_frame.empty:
        summary_frame = pd.DataFrame(
            columns=[*group_keys, "n_runs", "mean_metric", "std_metric", "min_metric", "max_metric"]
        )
    else:
        summary_frame = (
            run_frame.groupby(list(group_keys), dropna=False, observed=False)["metric_value"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
            .rename(
                columns={
                    "count": "n_runs",
                    "mean": "mean_metric",
                    "std": "std_metric",
                    "min": "min_metric",
                    "max": "max_metric",
                }
            )
        )
    return run_frame, summary_frame


def grouped_bootstrap_percentile_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    group_key: str,
    confidence_level: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError("confidence_level must be in (0.0, 1.0).")
    if int(n_bootstrap) <= 0:
        raise ValueError("n_bootstrap must be > 0.")

    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return {
            "status": "not_applicable",
            "reason": "no_rows",
            "group_key": group_key,
            "value_key": value_key,
            "n_groups": 0,
            "n_samples": 0,
        }
    frame[value_key] = pd.to_numeric(frame.get(value_key), errors="coerce")
    frame = frame.dropna(subset=[value_key])
    if frame.empty:
        return {
            "status": "not_applicable",
            "reason": "no_metric_values",
            "group_key": group_key,
            "value_key": value_key,
            "n_groups": 0,
            "n_samples": 0,
        }

    if group_key not in frame.columns:
        frame[group_key] = "__all__"
    groups = [group.reset_index(drop=True) for _, group in frame.groupby(group_key, dropna=False)]
    n_groups = len(groups)
    if n_groups == 0:
        return {
            "status": "not_applicable",
            "reason": "no_groups",
            "group_key": group_key,
            "value_key": value_key,
            "n_groups": 0,
            "n_samples": int(frame.shape[0]),
        }

    rng = np.random.default_rng(int(seed))
    bootstrap_means: list[float] = []
    for _ in range(int(n_bootstrap)):
        sampled_groups = rng.integers(0, n_groups, size=n_groups)
        sampled_values = np.concatenate(
            [groups[idx][value_key].to_numpy(dtype=float, copy=False) for idx in sampled_groups]
        )
        bootstrap_means.append(float(np.mean(sampled_values)))

    alpha = 1.0 - float(confidence_level)
    lower_q = float(alpha / 2.0)
    upper_q = float(1.0 - alpha / 2.0)
    observed_values = frame[value_key].to_numpy(dtype=float, copy=False)
    return {
        "status": "ok",
        "method": "grouped_bootstrap_percentile",
        "group_key": group_key,
        "value_key": value_key,
        "confidence_level": float(confidence_level),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "n_groups": int(n_groups),
        "n_samples": int(frame.shape[0]),
        "observed_mean": float(np.mean(observed_values)),
        "observed_std": float(np.std(observed_values)),
        "interval_lower": float(np.quantile(bootstrap_means, lower_q)),
        "interval_upper": float(np.quantile(bootstrap_means, upper_q)),
    }


def paired_sign_flip_permutation(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    left_key: str,
    right_key: str,
    n_permutations: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    if int(n_permutations) <= 0:
        raise ValueError("n_permutations must be > 0.")
    if float(alpha) <= 0.0 or float(alpha) > 1.0:
        raise ValueError("alpha must be in (0.0, 1.0].")

    frame = pd.DataFrame(paired_rows).copy()
    if frame.empty:
        return {
            "status": "not_applicable",
            "reason": "no_pairs",
            "method": "paired_sign_flip_permutation",
            "n_pairs": 0,
        }
    frame[left_key] = pd.to_numeric(frame.get(left_key), errors="coerce")
    frame[right_key] = pd.to_numeric(frame.get(right_key), errors="coerce")
    frame = frame.dropna(subset=[left_key, right_key])
    if frame.empty:
        return {
            "status": "not_applicable",
            "reason": "no_valid_pairs",
            "method": "paired_sign_flip_permutation",
            "n_pairs": 0,
        }

    diffs = frame[left_key].to_numpy(dtype=float, copy=False) - frame[right_key].to_numpy(
        dtype=float, copy=False
    )
    observed_mean = float(np.mean(diffs))
    rng = np.random.default_rng(int(seed))
    null_means: list[float] = []
    for _ in range(int(n_permutations)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=diffs.shape[0], replace=True)
        null_means.append(float(np.mean(diffs * signs)))
    abs_observed = abs(observed_mean)
    p_value = float(
        (1.0 + sum(abs(value) >= abs_observed for value in null_means))
        / (float(n_permutations) + 1.0)
    )
    return {
        "status": "ok",
        "method": "paired_sign_flip_permutation",
        "n_pairs": int(diffs.shape[0]),
        "n_permutations": int(n_permutations),
        "alpha": float(alpha),
        "observed_mean_difference": float(observed_mean),
        "p_value": p_value,
        "significant": bool(p_value <= float(alpha)),
        "null_summary": {
            "mean": float(np.mean(null_means)),
            "std": float(np.std(null_means)),
            "min": float(np.min(null_means)),
            "max": float(np.max(null_means)),
            "q25": float(np.quantile(null_means, 0.25)),
            "q50": float(np.quantile(null_means, 0.50)),
            "q75": float(np.quantile(null_means, 0.75)),
        },
    }


def build_calibration_outputs(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    n_bins: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if int(n_bins) <= 1:
        raise ValueError("n_bins must be > 1.")
    frame = pd.DataFrame(prediction_rows).copy()
    if frame.empty or "proba_value" not in frame.columns:
        summary = {
            "status": "not_applicable",
            "reason": "no_probability_scores",
            "performed": False,
            "n_bins": int(n_bins),
            "n_samples": int(frame.shape[0]) if not frame.empty else 0,
            "ece": None,
            "brier_score": None,
        }
        table = pd.DataFrame(
            columns=[
                "bin_index",
                "bin_lower",
                "bin_upper",
                "n_samples",
                "mean_confidence",
                "empirical_accuracy",
            ]
        )
        return summary, table

    frame["proba_value"] = pd.to_numeric(frame["proba_value"], errors="coerce")
    frame = frame.dropna(subset=["proba_value"])
    if frame.empty:
        summary = {
            "status": "not_applicable",
            "reason": "no_probability_scores",
            "performed": False,
            "n_bins": int(n_bins),
            "n_samples": 0,
            "ece": None,
            "brier_score": None,
        }
        table = pd.DataFrame(
            columns=[
                "bin_index",
                "bin_lower",
                "bin_upper",
                "n_samples",
                "mean_confidence",
                "empirical_accuracy",
            ]
        )
        return summary, table

    frame["proba_value"] = frame["proba_value"].clip(0.0, 1.0)
    frame["correct"] = (
        frame.get("y_true", pd.Series(dtype=str)).astype(str)
        == frame.get("y_pred", pd.Series(dtype=str)).astype(str)
    ).astype(float)
    frame["squared_error"] = (frame["proba_value"] - frame["correct"]) ** 2

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    frame["bin_index"] = np.digitize(frame["proba_value"], edges[1:-1], right=False)
    grouped = frame.groupby("bin_index", dropna=False, observed=False)
    table_rows: list[dict[str, Any]] = []
    ece = 0.0
    n_total = float(frame.shape[0])
    for bin_index, subset in grouped:
        idx = int(bin_index)
        bin_lower = float(edges[idx])
        bin_upper = float(edges[idx + 1])
        n_samples = int(subset.shape[0])
        mean_confidence = float(subset["proba_value"].mean())
        empirical_accuracy = float(subset["correct"].mean())
        ece += (float(n_samples) / n_total) * abs(empirical_accuracy - mean_confidence)
        table_rows.append(
            {
                "bin_index": idx,
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "n_samples": n_samples,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
            }
        )
    table = pd.DataFrame(table_rows).sort_values("bin_index", kind="stable")
    summary = {
        "status": "performed",
        "reason": None,
        "performed": True,
        "n_bins": int(n_bins),
        "n_samples": int(frame.shape[0]),
        "ece": float(ece),
        "brier_score": float(frame["squared_error"].mean()),
    }
    return summary, table

