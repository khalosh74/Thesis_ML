from __future__ import annotations

import math

from Thesis_ML.experiments.evidence_statistics import (
    aggregate_repeated_runs,
    build_calibration_outputs,
    grouped_bootstrap_percentile_interval,
    paired_sign_flip_permutation,
)


def test_aggregate_repeated_runs_groups_rows_and_summary() -> None:
    run_rows, summary = aggregate_repeated_runs(
        [
            {
                "comparison_id": "cmp1",
                "variant_id": "ridge",
                "run_id": "run_a__r001",
                "base_run_id": "run_a",
                "repeat_id": 1,
                "primary_metric_value": 0.70,
            },
            {
                "comparison_id": "cmp1",
                "variant_id": "ridge",
                "run_id": "run_a__r002",
                "base_run_id": "run_a",
                "repeat_id": 2,
                "primary_metric_value": 0.74,
            },
        ],
        metric_key="primary_metric_value",
        group_keys=["comparison_id", "variant_id"],
    )

    assert list(run_rows["metric_name"].astype(str).unique()) == ["primary_metric_value"]
    assert run_rows.shape[0] == 2
    assert summary.shape[0] == 1
    row = summary.iloc[0]
    assert row["comparison_id"] == "cmp1"
    assert row["variant_id"] == "ridge"
    assert int(row["n_runs"]) == 2
    assert math.isclose(float(row["mean_metric"]), 0.72, rel_tol=1e-12)


def test_grouped_bootstrap_interval_is_deterministic_for_seed() -> None:
    rows = [
        {"base_run_id": "run_a", "primary_metric_value": 0.50},
        {"base_run_id": "run_a", "primary_metric_value": 0.60},
        {"base_run_id": "run_b", "primary_metric_value": 0.70},
        {"base_run_id": "run_b", "primary_metric_value": 0.80},
    ]
    first = grouped_bootstrap_percentile_interval(
        rows,
        value_key="primary_metric_value",
        group_key="base_run_id",
        confidence_level=0.95,
        n_bootstrap=200,
        seed=77,
    )
    second = grouped_bootstrap_percentile_interval(
        rows,
        value_key="primary_metric_value",
        group_key="base_run_id",
        confidence_level=0.95,
        n_bootstrap=200,
        seed=77,
    )
    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert math.isclose(float(first["interval_lower"]), float(second["interval_lower"]), rel_tol=0.0)
    assert math.isclose(float(first["interval_upper"]), float(second["interval_upper"]), rel_tol=0.0)


def test_paired_sign_flip_detects_large_matched_difference() -> None:
    paired_rows = [{"left_metric": 0.85, "right_metric": 0.40} for _ in range(12)]
    result = paired_sign_flip_permutation(
        paired_rows,
        left_key="left_metric",
        right_key="right_metric",
        n_permutations=2000,
        alpha=0.05,
        seed=17,
    )

    assert result["status"] == "ok"
    assert int(result["n_pairs"]) == 12
    assert float(result["observed_mean_difference"]) > 0.0
    assert bool(result["significant"]) is True
    assert float(result["p_value"]) <= 0.05


def test_build_calibration_outputs_handles_performed_and_not_applicable() -> None:
    performed_summary, performed_table = build_calibration_outputs(
        [
            {"y_true": "a", "y_pred": "a", "proba_value": 0.95},
            {"y_true": "a", "y_pred": "b", "proba_value": 0.60},
            {"y_true": "b", "y_pred": "b", "proba_value": 0.90},
            {"y_true": "b", "y_pred": "a", "proba_value": 0.20},
        ],
        n_bins=5,
    )
    assert performed_summary["status"] == "performed"
    assert performed_summary["performed"] is True
    assert performed_table.empty is False

    na_summary, na_table = build_calibration_outputs(
        [
            {"y_true": "a", "y_pred": "a", "proba_value": None},
            {"y_true": "b", "y_pred": "b", "proba_value": None},
        ],
        n_bins=5,
    )
    assert na_summary["status"] == "not_applicable"
    assert na_summary["performed"] is False
    assert na_table.empty is True
