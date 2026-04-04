from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.orchestration.dummy_baseline_aggregation import build_e13_table_ready_rows
from Thesis_ML.orchestration.variant_expansion import materialize_experiment_cells


def _base_variant() -> dict[str, object]:
    return {
        "template_id": "e13_template",
        "variant_index": 1,
        "supported": True,
        "blocked_reason": None,
        "params": {
            "target": "coarse_affect",
            "model": "dummy_or_majority",
            "cv": "within_subject_loso_session",
            "subject": "None",
        },
        "factor_settings": {},
        "fixed_controls": {},
        "design_metadata": {},
    }


def _registry_experiments_multi_anchor() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": "E13",
            "stage": "Stage 6 - Robustness analysis",
            "executable_now": True,
            "execution_status": "unknown",
            "variant_templates": [
                {
                    "template_id": "e13_template",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "dummy_or_majority",
                        "cv": "within_subject_loso_session",
                        "subject": "None",
                    },
                }
            ],
        },
        {
            "experiment_id": "E16",
            "stage": "Stage 5 - Confirmatory analysis",
            "executable_now": True,
            "execution_status": "unknown",
            "variant_templates": [
                {
                    "template_id": "e16_anchor",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "cv": "within_subject_loso_session",
                        "model": "ridge",
                        "subject": "sub-001",
                        "framework_mode": "confirmatory",
                        "canonical_run": True,
                    },
                },
                {
                    "template_id": "e16_anchor_sub002",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "cv": "within_subject_loso_session",
                        "model": "ridge",
                        "subject": "sub-002",
                        "framework_mode": "confirmatory",
                        "canonical_run": True,
                    },
                },
            ],
        },
        {
            "experiment_id": "E17",
            "stage": "Stage 5 - Confirmatory analysis",
            "executable_now": True,
            "execution_status": "unknown",
            "variant_templates": [
                {
                    "template_id": "e17_anchor",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "cv": "frozen_cross_person_transfer",
                        "model": "ridge",
                        "train_subject": "sub-001",
                        "test_subject": "sub-002",
                        "framework_mode": "confirmatory",
                        "canonical_run": True,
                    },
                },
                {
                    "template_id": "e17_anchor_reverse",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "cv": "frozen_cross_person_transfer",
                        "model": "ridge",
                        "train_subject": "sub-002",
                        "test_subject": "sub-001",
                        "framework_mode": "confirmatory",
                        "canonical_run": True,
                    },
                },
            ],
        },
    ]


def test_e13_materialization_inherits_anchor_identity_and_normalizes_model() -> None:
    cells, warnings = materialize_experiment_cells(
        experiment={"experiment_id": "E13"},
        variants=[_base_variant()],
        dataset_scope={},
        n_permutations=0,
        registry_experiments=_registry_experiments_multi_anchor(),
    )

    assert warnings == []
    assert len(cells) == 4
    assert {str(cell["params"].get("model")) for cell in cells} == {"dummy"}
    assert all(cell["params"].get("subject") != "None" for cell in cells)

    within = [
        cell for cell in cells if str(cell["params"].get("cv")) == "within_subject_loso_session"
    ]
    transfer = [
        cell for cell in cells if str(cell["params"].get("cv")) == "frozen_cross_person_transfer"
    ]
    assert len(within) == 2
    assert len(transfer) == 2
    assert {str(cell["params"].get("subject")) for cell in within} == {"sub-001", "sub-002"}
    assert {
        (str(cell["params"].get("train_subject")), str(cell["params"].get("test_subject")))
        for cell in transfer
    } == {("sub-001", "sub-002"), ("sub-002", "sub-001")}


def test_e13_table_ready_rows_keep_anchors_distinct(tmp_path: Path) -> None:
    metrics_within = tmp_path / "metrics_within.json"
    metrics_transfer = tmp_path / "metrics_transfer.json"
    metrics_within.write_text(
        json.dumps(
            {
                "primary_metric_name": "balanced_accuracy",
                "balanced_accuracy": 0.34,
                "majority_class_metadata": {"majority_class": "low"},
            }
        ),
        encoding="utf-8",
    )
    metrics_transfer.write_text(
        json.dumps(
            {
                "primary_metric_name": "balanced_accuracy",
                "balanced_accuracy": 0.31,
                "majority_class_metadata": {"majority_class": "high"},
            }
        ),
        encoding="utf-8",
    )

    records = [
        {
            "experiment_id": "E13",
            "status": "completed",
            "run_id": "run_within",
            "variant_id": "v_within",
            "target": "coarse_affect",
            "cv": "within_subject_loso_session",
            "subject": "sub-001",
            "model": "dummy",
            "primary_metric_name": "balanced_accuracy",
            "primary_metric_value": 0.34,
            "metrics_path": str(metrics_within),
            "report_dir": str(tmp_path),
            "design_metadata": {
                "special_cell_kind": "confirmatory_dummy_baseline",
                "anchor_experiment_id": "E16",
                "anchor_variant_id": "e16_anchor",
                "anchor_analysis_type": "within_person_loso",
                "anchor_analysis_label": "within_subject_loso_session:sub-001",
                "baseline_group_id": "g1",
            },
        },
        {
            "experiment_id": "E13",
            "status": "completed",
            "run_id": "run_transfer",
            "variant_id": "v_transfer",
            "target": "coarse_affect",
            "cv": "frozen_cross_person_transfer",
            "train_subject": "sub-001",
            "test_subject": "sub-002",
            "model": "dummy",
            "primary_metric_name": "balanced_accuracy",
            "primary_metric_value": 0.31,
            "metrics_path": str(metrics_transfer),
            "report_dir": str(tmp_path),
            "design_metadata": {
                "special_cell_kind": "confirmatory_dummy_baseline",
                "anchor_experiment_id": "E17",
                "anchor_variant_id": "e17_anchor",
                "anchor_analysis_type": "cross_person_transfer",
                "anchor_analysis_label": "frozen_cross_person_transfer:sub-001->sub-002",
                "baseline_group_id": "g2",
            },
        },
    ]

    rows = build_e13_table_ready_rows(reporting_variant_records=records)
    assert len(rows) == 2
    assert {str(row["analysis_label"]) for row in rows} == {
        "within_subject_loso_session:sub-001",
        "frozen_cross_person_transfer:sub-001->sub-002",
    }
    assert {str(row["anchor_analysis_type"]) for row in rows} == {
        "within_person_loso",
        "cross_person_transfer",
    }
    assert {str(row["model"]) for row in rows} == {"dummy"}
