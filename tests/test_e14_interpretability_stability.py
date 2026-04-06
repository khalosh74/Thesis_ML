from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Thesis_ML.orchestration.interpretability_stability_aggregation import (
    build_e14_reporting_records,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _write_anchor_artifacts(
    *,
    root: Path,
    run_id: str,
    summary_status: str = "performed",
) -> tuple[str, str, list[str]]:
    report_dir = root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "interpretability_summary.json"
    fold_csv_path = report_dir / "interpretability_fold_explanations.csv"
    coef_dir = report_dir / "interpretability"
    coef_dir.mkdir(parents=True, exist_ok=True)

    coef_a = coef_dir / "fold_001_coefficients.npz"
    coef_b = coef_dir / "fold_002_coefficients.npz"
    coef_a.write_bytes(b"npz-a")
    coef_b.write_bytes(b"npz-b")

    fold_csv_path.write_text(
        f"fold,coefficient_file\n1,{coef_a.as_posix()}\n2,{coef_b.as_posix()}\n",
        encoding="utf-8",
    )

    _write_json(
        summary_path,
        {
            "enabled": True,
            "performed": summary_status == "performed",
            "status": summary_status,
            "fold_artifacts_path": str(fold_csv_path.resolve()),
            "caution": "Linear coefficients are model-behavior evidence only.",
            "stability": {
                "status": "ok",
                "n_folds": 2,
                "mean_pairwise_correlation": 0.41,
                "mean_sign_consistency": 0.73,
                "mean_top_k_overlap": 0.22,
            },
        },
    )

    metrics_path = report_dir / "metrics.json"
    _write_json(
        metrics_path,
        {
            "primary_metric_name": "balanced_accuracy",
            "balanced_accuracy": 0.58,
            "interpretability": {
                "enabled": True,
                "performed": summary_status == "performed",
                "status": summary_status,
                "summary_path": str(summary_path.resolve()),
                "fold_artifacts_path": str(fold_csv_path.resolve()),
                "stability": {
                    "status": "ok",
                    "n_folds": 2,
                    "mean_pairwise_correlation": 0.41,
                    "mean_sign_consistency": 0.73,
                    "mean_top_k_overlap": 0.22,
                },
            },
        },
    )
    return (
        str(report_dir.resolve()),
        str(metrics_path.resolve()),
        [str(coef_a.resolve()), str(coef_b.resolve())],
    )


def _e14_record(*, variant_id: str, subject: str, anchor_variant_id: str) -> dict[str, Any]:
    return {
        "experiment_id": "E14",
        "variant_id": variant_id,
        "template_id": "e14_template",
        "status": "blocked",
        "cv": "within_subject_loso_session",
        "model": "ridge",
        "subject": subject,
        "feature_space": "whole_brain_masked",
        "design_metadata": {
            "special_cell_kind": "interpretability_stability",
            "anchor_experiment_id": "E16",
            "anchor_variant_id": anchor_variant_id,
            "anchor_subject": subject,
            "anchor_analysis_type": "within_person_loso",
            "anchor_analysis_label": f"within_subject_loso_session:{subject}",
            "robustness_group_id": f"E14::{subject}",
        },
    }


def _e16_anchor_record(
    *,
    variant_id: str,
    subject: str,
    run_id: str,
    report_dir: str,
    metrics_path: str,
    preprocessing_strategy: str = "none",
    feature_space: str = "whole_brain_masked",
) -> dict[str, Any]:
    return {
        "experiment_id": "E16",
        "variant_id": variant_id,
        "status": "completed",
        "cv": "within_subject_loso_session",
        "model": "ridge",
        "subject": subject,
        "target": "coarse_affect",
        "feature_space": feature_space,
        "filter_modality": "audiovisual",
        "preprocessing_strategy": preprocessing_strategy,
        "dimensionality_strategy": "none",
        "methodology_policy_name": "fixed_baselines_only",
        "class_weight_policy": "none",
        "run_id": run_id,
        "report_dir": report_dir,
        "metrics_path": metrics_path,
        "primary_metric_name": "balanced_accuracy",
        "primary_metric_value": 0.58,
        "balanced_accuracy": 0.58,
    }


def _e17_anchor_record() -> dict[str, Any]:
    return {
        "experiment_id": "E17",
        "variant_id": "e17_anchor",
        "status": "completed",
        "cv": "frozen_cross_person_transfer",
        "model": "ridge",
        "train_subject": "sub-001",
        "test_subject": "sub-002",
        "run_id": "e17_run",
        "report_dir": "/tmp/e17",
        "metrics_path": "/tmp/e17/metrics.json",
        "primary_metric_name": "balanced_accuracy",
        "primary_metric_value": 0.53,
    }


def test_e14_reporting_builds_one_row_per_e16_anchor_and_excludes_e17(tmp_path: Path) -> None:
    report_a, metrics_a, _ = _write_anchor_artifacts(root=tmp_path, run_id="e16_sub001")
    report_b, metrics_b, _ = _write_anchor_artifacts(root=tmp_path, run_id="e16_sub002")

    records = [
        _e16_anchor_record(
            variant_id="e16_anchor",
            subject="sub-001",
            run_id="e16_run_sub001",
            report_dir=report_a,
            metrics_path=metrics_a,
        ),
        _e16_anchor_record(
            variant_id="e16_anchor_sub002",
            subject="sub-002",
            run_id="e16_run_sub002",
            report_dir=report_b,
            metrics_path=metrics_b,
        ),
        _e17_anchor_record(),
        _e14_record(variant_id="e14_sub001", subject="sub-001", anchor_variant_id="e16_anchor"),
        _e14_record(
            variant_id="e14_sub002",
            subject="sub-002",
            anchor_variant_id="e16_anchor_sub002",
        ),
    ]

    output_records, summary_rows, payload = build_e14_reporting_records(
        reporting_variant_records=records
    )

    e14_output = [row for row in output_records if str(row.get("experiment_id")) == "E14"]
    assert len(e14_output) == 2
    assert {str(row.get("status")) for row in e14_output} == {"completed"}
    assert all(str(row.get("run_id", "")).endswith("__e14_stability") for row in e14_output)

    assert len(summary_rows) == 2
    assert {str(row.get("subject")) for row in summary_rows} == {"sub-001", "sub-002"}
    assert {str(row.get("analysis_label")) for row in summary_rows} == {
        "within_subject_loso_session:sub-001",
        "within_subject_loso_session:sub-002",
    }
    assert all(row.get("mean_pairwise_coef_correlation") is not None for row in summary_rows)
    assert all(row.get("mean_sign_consistency") is not None for row in summary_rows)
    assert all(row.get("mean_topk_overlap") is not None for row in summary_rows)
    assert all(
        "model-behavior evidence" in str(row.get("scientific_caution", "")).lower()
        for row in summary_rows
    )

    errors = list(payload.get("errors") or [])
    assert errors == []


def test_e14_reporting_marks_not_applicable_when_interpretability_missing(tmp_path: Path) -> None:
    report_a, metrics_a, _ = _write_anchor_artifacts(
        root=tmp_path,
        run_id="e16_sub001",
        summary_status="not_applicable",
    )
    records = [
        _e16_anchor_record(
            variant_id="e16_anchor",
            subject="sub-001",
            run_id="e16_run_sub001",
            report_dir=report_a,
            metrics_path=metrics_a,
        ),
        _e14_record(variant_id="e14_sub001", subject="sub-001", anchor_variant_id="e16_anchor"),
    ]

    output_records, summary_rows, payload = build_e14_reporting_records(
        reporting_variant_records=records
    )

    e14_output = [row for row in output_records if str(row.get("experiment_id")) == "E14"]
    assert len(e14_output) == 1
    assert str(e14_output[0].get("status")) == "blocked"

    assert len(summary_rows) == 1
    assert str(summary_rows[0].get("status")) == "not_applicable"
    assert str(summary_rows[0].get("completion_status")) == "ineligible_or_missing_artifacts"
    assert summary_rows[0].get("mean_pairwise_coef_correlation") is None
    assert list(payload.get("errors") or [])


def test_e14_reporting_uses_runtime_anchor_artifact_fallback_when_campaign_has_no_e16(
    tmp_path: Path,
) -> None:
    report_a, metrics_a, _ = _write_anchor_artifacts(root=tmp_path, run_id="e16_sub001")
    runtime_anchor_record = _e16_anchor_record(
        variant_id="e16_anchor",
        subject="sub-001",
        run_id="e16_run_sub001",
        report_dir=report_a,
        metrics_path=metrics_a,
    )
    runtime_anchor_rows = [
        {
            "analysis_label": "within_subject_loso_session:sub-001",
            "experiment_id": "E16",
            "cv": "within_subject_loso_session",
            "subject": "sub-001",
            "target": "coarse_affect",
            "model": "ridge",
            "feature_space": "whole_brain_masked",
            "filter_modality": "audiovisual",
            "preprocessing_strategy": "none",
            "dimensionality_strategy": "none",
            "methodology_policy_name": "fixed_baselines_only",
            "class_weight_policy": "none",
        }
    ]
    records = [
        _e14_record(variant_id="e14_sub001", subject="sub-001", anchor_variant_id="e16_anchor"),
    ]

    output_records, summary_rows, payload = build_e14_reporting_records(
        reporting_variant_records=records,
        runtime_anchor_rows=runtime_anchor_rows,
        runtime_anchor_records=[runtime_anchor_record],
    )

    e14_output = [row for row in output_records if str(row.get("experiment_id")) == "E14"]
    assert len(e14_output) == 1
    assert str(e14_output[0].get("status")) == "completed"
    assert str(summary_rows[0].get("completion_status")) == "completed"
    assert list(payload.get("errors") or []) == []


def test_e14_reporting_blocks_when_runtime_locked_core_mismatches_resolved_anchor(
    tmp_path: Path,
) -> None:
    report_a, metrics_a, _ = _write_anchor_artifacts(root=tmp_path, run_id="e16_sub001")
    runtime_anchor_record = _e16_anchor_record(
        variant_id="e16_anchor",
        subject="sub-001",
        run_id="e16_run_sub001",
        report_dir=report_a,
        metrics_path=metrics_a,
        preprocessing_strategy="standardize_zscore",
    )
    runtime_anchor_rows = [
        {
            "analysis_label": "within_subject_loso_session:sub-001",
            "experiment_id": "E16",
            "cv": "within_subject_loso_session",
            "subject": "sub-001",
            "target": "coarse_affect",
            "model": "ridge",
            "feature_space": "whole_brain_masked",
            "filter_modality": "audiovisual",
            "preprocessing_strategy": "none",
            "dimensionality_strategy": "none",
            "methodology_policy_name": "fixed_baselines_only",
            "class_weight_policy": "none",
        }
    ]
    records = [
        _e14_record(variant_id="e14_sub001", subject="sub-001", anchor_variant_id="e16_anchor"),
    ]

    output_records, summary_rows, payload = build_e14_reporting_records(
        reporting_variant_records=records,
        runtime_anchor_rows=runtime_anchor_rows,
        runtime_anchor_records=[runtime_anchor_record],
    )

    e14_output = [row for row in output_records if str(row.get("experiment_id")) == "E14"]
    assert len(e14_output) == 1
    assert str(e14_output[0].get("status")) == "blocked"
    assert str(summary_rows[0].get("completion_status")) == "anchor_locked_core_mismatch"
    assert list(payload.get("errors") or [])
