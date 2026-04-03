from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.orchestration.permutation_chunk_aggregation import (
    build_e12_table_ready_rows,
    build_reporting_variant_records,
)


def _write_metrics(
    *,
    path: Path,
    observed_score: float,
    null_scores: list[float],
    minimum_required: int = 100,
    alpha: float = 0.05,
) -> Path:
    ge_count = sum(float(value) >= float(observed_score) for value in null_scores)
    p_value = (float(ge_count) + 1.0) / (float(len(null_scores)) + 1.0)
    payload = {
        "primary_metric_name": "balanced_accuracy",
        "primary_metric_value": float(observed_score),
        "balanced_accuracy": float(observed_score),
        "macro_f1": 0.50,
        "accuracy": 0.55,
        "permutation_test": {
            "n_permutations": int(len(null_scores)),
            "metric_name": "balanced_accuracy",
            "observed_score": float(observed_score),
            "observed_metric": float(observed_score),
            "p_value": float(p_value),
            "null_scores": [float(value) for value in null_scores],
            "null_summary": {
                "mean": float(sum(null_scores) / len(null_scores)),
                "std": 0.0,
                "min": float(min(null_scores)),
                "max": float(max(null_scores)),
                "q25": float(min(null_scores)),
                "q50": float(min(null_scores)),
                "q75": float(max(null_scores)),
            },
            "alpha": float(alpha),
            "minimum_required": int(minimum_required),
            "meets_minimum": bool(len(null_scores) >= int(minimum_required)),
            "passes_threshold": bool(p_value <= float(alpha)),
            "interpretation_status": "passes_threshold" if p_value <= float(alpha) else "fails_threshold",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    return path


def _chunk_record(
    *,
    run_id: str,
    variant_id: str,
    metrics_path: Path,
    chunk_index: int,
) -> dict[str, object]:
    return {
        "experiment_id": "E12",
        "template_id": "e12_template",
        "variant_id": variant_id,
        "trial_id": variant_id,
        "cell_id": variant_id,
        "run_id": run_id,
        "status": "completed",
        "target": "coarse_affect",
        "cv": "within_subject_loso_session",
        "model": "ridge",
        "subject": "sub-001",
        "feature_space": "whole_brain_masked",
        "primary_metric_name": "balanced_accuracy",
        "primary_metric_value": 0.58,
        "balanced_accuracy": 0.58,
        "macro_f1": 0.57,
        "accuracy": 0.61,
        "n_permutations": 2,
        "metrics_path": str(metrics_path.resolve()),
        "design_metadata": {
            "special_cell_kind": "permutation_chunk",
            "chunk_index": int(chunk_index),
            "permutation_group_id": "E12::group::A",
            "expected_chunk_count": 2,
            "total_permutations_requested": 4,
            "anchor_experiment_id": "E16",
            "anchor_template_id": "e16_anchor",
        },
        "factor_settings": {
            "permutation_chunk_index": int(chunk_index),
            "permutation_group_id": "E12::group::A",
            "expected_chunk_count": 2,
            "total_permutations_requested": 4,
            "anchor_experiment_id": "E16",
            "anchor_template_id": "e16_anchor",
        },
        "notes": "",
    }


def test_e12_chunk_merge_recomputes_null_distribution(tmp_path: Path) -> None:
    metrics_one = _write_metrics(
        path=tmp_path / "chunk_1" / "metrics.json",
        observed_score=0.58,
        null_scores=[0.20, 0.30],
    )
    metrics_two = _write_metrics(
        path=tmp_path / "chunk_2" / "metrics.json",
        observed_score=0.58,
        null_scores=[0.40, 0.70],
    )
    records = [
        _chunk_record(
            run_id="run_chunk_001",
            variant_id="v001_perm_chunk_001",
            metrics_path=metrics_one,
            chunk_index=1,
        ),
        _chunk_record(
            run_id="run_chunk_002",
            variant_id="v001_perm_chunk_002",
            metrics_path=metrics_two,
            chunk_index=2,
        ),
    ]

    reporting_records, merge_summary = build_reporting_variant_records(
        campaign_root=tmp_path,
        variant_records=records,
    )

    assert len(reporting_records) == 1
    merged = reporting_records[0]
    assert str(merged["variant_id"]).endswith("__perm_merged")
    assert str(merged["run_id"]).endswith("__perm_merged")
    assert int(merged["n_permutations"]) == 4

    merged_metrics = json.loads(Path(str(merged["metrics_path"])).read_text(encoding="utf-8"))
    permutation_payload = dict(merged_metrics["permutation_test"])
    assert int(permutation_payload["n_permutations"]) == 4
    assert permutation_payload["null_scores"] == [0.2, 0.3, 0.4, 0.7]
    expected_p_value = (1.0 + 1.0) / (4.0 + 1.0)
    assert float(permutation_payload["p_value"]) == expected_p_value
    assert bool(permutation_payload["meets_minimum"]) is False
    assert int(permutation_payload["minimum_required"]) == 100

    assert isinstance(merge_summary.get("groups"), list)
    assert merge_summary["groups"]
    summary_row = merge_summary["groups"][0]
    assert summary_row["permutation_group_id"] == "E12::group::A"
    assert int(summary_row["chunk_count"]) == 2
    assert int(summary_row["completed_chunk_count"]) == 2


def test_e12_chunk_merge_keeps_groups_separate_and_builds_table_ready_rows(tmp_path: Path) -> None:
    metrics_a1 = _write_metrics(
        path=tmp_path / "group_a_chunk_1" / "metrics.json",
        observed_score=0.58,
        null_scores=[0.20, 0.30],
    )
    metrics_a2 = _write_metrics(
        path=tmp_path / "group_a_chunk_2" / "metrics.json",
        observed_score=0.58,
        null_scores=[0.40, 0.70],
    )
    metrics_b1 = _write_metrics(
        path=tmp_path / "group_b_chunk_1" / "metrics.json",
        observed_score=0.62,
        null_scores=[0.10, 0.20],
    )
    metrics_b2 = _write_metrics(
        path=tmp_path / "group_b_chunk_2" / "metrics.json",
        observed_score=0.62,
        null_scores=[0.30, 0.40],
    )

    records = [
        _chunk_record(
            run_id="run_a_001",
            variant_id="within_perm_chunk_001",
            metrics_path=metrics_a1,
            chunk_index=1,
        ),
        _chunk_record(
            run_id="run_a_002",
            variant_id="within_perm_chunk_002",
            metrics_path=metrics_a2,
            chunk_index=2,
        ),
        _chunk_record(
            run_id="run_b_001",
            variant_id="transfer_perm_chunk_001",
            metrics_path=metrics_b1,
            chunk_index=1,
        ),
        _chunk_record(
            run_id="run_b_002",
            variant_id="transfer_perm_chunk_002",
            metrics_path=metrics_b2,
            chunk_index=2,
        ),
    ]
    records[2]["design_metadata"]["permutation_group_id"] = "E12::group::B"
    records[3]["design_metadata"]["permutation_group_id"] = "E12::group::B"
    records[2]["factor_settings"]["permutation_group_id"] = "E12::group::B"
    records[3]["factor_settings"]["permutation_group_id"] = "E12::group::B"
    records[2]["cv"] = "frozen_cross_person_transfer"
    records[3]["cv"] = "frozen_cross_person_transfer"
    records[2]["subject"] = None
    records[3]["subject"] = None
    records[2]["train_subject"] = "sub-001"
    records[3]["train_subject"] = "sub-001"
    records[2]["test_subject"] = "sub-002"
    records[3]["test_subject"] = "sub-002"
    records[2]["design_metadata"]["anchor_experiment_id"] = "E18"
    records[3]["design_metadata"]["anchor_experiment_id"] = "E18"
    records[2]["design_metadata"]["anchor_analysis_label"] = (
        "frozen_cross_person_transfer:sub-001->sub-002"
    )
    records[3]["design_metadata"]["anchor_analysis_label"] = (
        "frozen_cross_person_transfer:sub-001->sub-002"
    )
    records[0]["design_metadata"]["anchor_analysis_label"] = "within_subject_loso_session:sub-001"
    records[1]["design_metadata"]["anchor_analysis_label"] = "within_subject_loso_session:sub-001"

    reporting_records, _ = build_reporting_variant_records(
        campaign_root=tmp_path,
        variant_records=records,
    )
    assert len(reporting_records) == 2
    table_rows = build_e12_table_ready_rows(reporting_variant_records=reporting_records)
    assert len(table_rows) == 2
    labels = {str(row["analysis_label"]) for row in table_rows}
    assert "within_subject_loso_session:sub-001" in labels
    assert "frozen_cross_person_transfer:sub-001->sub-002" in labels
    assert all(int(row["n_permutations"]) == 4 for row in table_rows)
    assert all(bool(row["all_chunks_present"]) is True for row in table_rows)
