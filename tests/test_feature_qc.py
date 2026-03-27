from __future__ import annotations

import numpy as np

from Thesis_ML.features.feature_qc import (
    FEATURE_QC_SAMPLE_FIELDS,
    compute_sample_feature_qc,
    merge_qc_into_metadata_records,
    summarize_group_feature_qc,
)


def test_compute_sample_feature_qc_counts_nonfinite_repair() -> None:
    before = np.asarray([1.0, np.nan, np.inf, -np.inf, 0.0], dtype=np.float64)
    after = np.nan_to_num(before, nan=0.0, posinf=0.0, neginf=0.0)
    qc = compute_sample_feature_qc(before, after)

    assert qc["n_features"] == 5
    assert qc["n_nan_before_repair"] == 1
    assert qc["n_posinf_before_repair"] == 1
    assert qc["n_neginf_before_repair"] == 1
    assert qc["n_nonfinite_before_repair"] == 3
    assert qc["repair_fraction"] == 3.0 / 5.0
    assert qc["n_zero_after_repair"] >= 3
    assert qc["all_zero_vector"] is False


def test_compute_sample_feature_qc_flags_all_zero_and_constant_vectors() -> None:
    before = np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
    after = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    qc = compute_sample_feature_qc(before, after)
    assert qc["all_zero_vector"] is True
    assert qc["constant_vector"] is True
    assert qc["std_after_repair"] == 0.0
    assert qc["l2_norm_after_repair"] == 0.0


def test_summarize_group_feature_qc_aggregates_expected_fields() -> None:
    rows = [
        {
            "group_id": "g1",
            "sample_id": "s1",
            **compute_sample_feature_qc(
                np.asarray([1.0, 0.0, np.nan], dtype=np.float64),
                np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            ),
        },
        {
            "group_id": "g1",
            "sample_id": "s2",
            **compute_sample_feature_qc(
                np.asarray([2.0, 2.0, 2.0], dtype=np.float64),
                np.asarray([2.0, 2.0, 2.0], dtype=np.float64),
            ),
        },
    ]
    summary = summarize_group_feature_qc(rows)
    assert summary["group_id"] == "g1"
    assert summary["n_samples"] == 2
    assert summary["n_features"] == 3
    assert summary["n_samples_with_any_repair"] == 1
    assert 0.0 <= summary["mean_repair_fraction"] <= summary["max_repair_fraction"]
    assert summary["n_constant_vectors"] >= 1


def test_merge_qc_into_metadata_records_attaches_all_qc_fields() -> None:
    metadata = [
        {"sample_id": "s1", "subject": "sub-001"},
        {"sample_id": "s2", "subject": "sub-001"},
    ]
    qc_rows = [
        {
            "group_id": "g1",
            "sample_id": "s1",
            **compute_sample_feature_qc(
                np.asarray([1.0, np.nan], dtype=np.float64),
                np.asarray([1.0, 0.0], dtype=np.float64),
            ),
        },
        {
            "group_id": "g1",
            "sample_id": "s2",
            **compute_sample_feature_qc(
                np.asarray([2.0, 3.0], dtype=np.float64),
                np.asarray([2.0, 3.0], dtype=np.float64),
            ),
        },
    ]
    merged = merge_qc_into_metadata_records(metadata, qc_rows)
    assert len(merged) == 2
    for row in merged:
        for field_name in FEATURE_QC_SAMPLE_FIELDS:
            assert field_name in row

