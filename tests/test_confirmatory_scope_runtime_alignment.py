from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.verification.confirmatory_scope_runtime_alignment import (
    build_confirmatory_control_coverage_rows,
    verify_confirmatory_scope_runtime_alignment,
)


def _write_scope(path: Path) -> None:
    payload = {
        "scope_id": "confirmatory_scope_v1",
        "main_tasks": ["emo", "recog"],
        "main_modality": "audiovisual",
        "main_target": "coarse_affect",
        "feature_space": "whole_brain_masked",
        "within_subjects": ["sub-001", "sub-002"],
        "transfer_pairs": [
            {"train_subject": "sub-001", "test_subject": "sub-002"},
            {"train_subject": "sub-002", "test_subject": "sub-001"},
        ],
    }
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _write_runtime_registry(path: Path, *, include_sub002: bool, include_reverse: bool) -> None:
    e16_templates: list[dict[str, object]] = [
        {
            "template_id": "e16_sub001",
            "supported": True,
            "params": {
                "target": "coarse_affect",
                "cv": "within_subject_loso_session",
                "model": "ridge",
                "subject": "sub-001",
            },
        }
    ]
    if include_sub002:
        e16_templates.append(
            {
                "template_id": "e16_sub002",
                "supported": True,
                "params": {
                    "target": "coarse_affect",
                    "cv": "within_subject_loso_session",
                    "model": "ridge",
                    "subject": "sub-002",
                },
            }
        )

    e17_templates: list[dict[str, object]] = [
        {
            "template_id": "e17_001_to_002",
            "supported": True,
            "params": {
                "target": "coarse_affect",
                "cv": "frozen_cross_person_transfer",
                "model": "ridge",
                "train_subject": "sub-001",
                "test_subject": "sub-002",
            },
        }
    ]
    if include_reverse:
        e17_templates.append(
            {
                "template_id": "e17_002_to_001",
                "supported": True,
                "params": {
                    "target": "coarse_affect",
                    "cv": "frozen_cross_person_transfer",
                    "model": "ridge",
                    "train_subject": "sub-002",
                    "test_subject": "sub-001",
                },
            }
        )

    payload = {
        "schema_version": "workbook-v1",
        "experiments": [
            {
                "experiment_id": "E16",
                "stage": "Stage 5 - Confirmatory analysis",
                "executable_now": True,
                "execution_status": "unknown",
                "variant_templates": e16_templates,
            },
            {
                "experiment_id": "E17",
                "stage": "Stage 5 - Confirmatory analysis",
                "executable_now": True,
                "execution_status": "unknown",
                "variant_templates": e17_templates,
            },
        ],
    }
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def test_scope_runtime_alignment_passes_for_full_anchor_set(tmp_path: Path) -> None:
    scope_path = tmp_path / "scope.json"
    runtime_path = tmp_path / "runtime.json"
    _write_scope(scope_path)
    _write_runtime_registry(runtime_path, include_sub002=True, include_reverse=True)

    summary = verify_confirmatory_scope_runtime_alignment(
        scope_config_path=scope_path,
        runtime_registry_path=runtime_path,
    )
    assert summary["passed"] is True
    assert len(summary["runtime_anchor_set"]) == 4


def test_scope_runtime_alignment_fails_when_runtime_is_narrower(tmp_path: Path) -> None:
    scope_path = tmp_path / "scope.json"
    runtime_path = tmp_path / "runtime.json"
    _write_scope(scope_path)
    _write_runtime_registry(runtime_path, include_sub002=False, include_reverse=False)

    summary = verify_confirmatory_scope_runtime_alignment(
        scope_config_path=scope_path,
        runtime_registry_path=runtime_path,
    )
    assert summary["passed"] is False
    issue_codes = {
        str(row.get("code")) for row in summary.get("issues", []) if isinstance(row, dict)
    }
    assert "scope_within_missing_in_runtime" in issue_codes
    assert "scope_transfer_missing_in_runtime" in issue_codes
    missing_labels = set(summary.get("missing_analysis_labels", []))
    assert "within_subject_loso_session:sub-002" in missing_labels
    assert "frozen_cross_person_transfer:sub-002->sub-001" in missing_labels


def test_scope_runtime_alignment_allows_explicit_deferred_exceptions(tmp_path: Path) -> None:
    scope_path = tmp_path / "scope.json"
    runtime_path = tmp_path / "runtime.json"
    exceptions_path = tmp_path / "exceptions.json"
    _write_scope(scope_path)
    _write_runtime_registry(runtime_path, include_sub002=False, include_reverse=False)
    exceptions_payload = {
        "scope_id": "confirmatory_scope_v1",
        "deferred_within_subjects": ["sub-002"],
        "deferred_transfer_pairs": [{"train_subject": "sub-002", "test_subject": "sub-001"}],
    }
    exceptions_path.write_text(f"{json.dumps(exceptions_payload, indent=2)}\n", encoding="utf-8")

    summary = verify_confirmatory_scope_runtime_alignment(
        scope_config_path=scope_path,
        runtime_registry_path=runtime_path,
        exceptions_config_path=exceptions_path,
    )
    assert summary["passed"] is True


def test_scope_runtime_alignment_rejects_deferred_exceptions_outside_scope(tmp_path: Path) -> None:
    scope_path = tmp_path / "scope.json"
    runtime_path = tmp_path / "runtime.json"
    exceptions_path = tmp_path / "exceptions.json"
    _write_scope(scope_path)
    _write_runtime_registry(runtime_path, include_sub002=True, include_reverse=True)
    exceptions_payload = {
        "scope_id": "confirmatory_scope_v1",
        "deferred_within_subjects": ["sub-099"],
        "deferred_transfer_pairs": [{"train_subject": "sub-001", "test_subject": "sub-099"}],
    }
    exceptions_path.write_text(f"{json.dumps(exceptions_payload, indent=2)}\n", encoding="utf-8")

    summary = verify_confirmatory_scope_runtime_alignment(
        scope_config_path=scope_path,
        runtime_registry_path=runtime_path,
        exceptions_config_path=exceptions_path,
    )
    assert summary["passed"] is False
    issue_codes = {
        str(row.get("code")) for row in summary.get("issues", []) if isinstance(row, dict)
    }
    assert "scope_exceptions_within_outside_scope" in issue_codes
    assert "scope_exceptions_transfer_outside_scope" in issue_codes


def test_scope_runtime_alignment_fails_when_within_family_not_e16(tmp_path: Path) -> None:
    scope_path = tmp_path / "scope.json"
    runtime_path = tmp_path / "runtime.json"
    _write_scope(scope_path)
    payload = {
        "schema_version": "workbook-v1",
        "experiments": [
            {
                "experiment_id": "E18",
                "stage": "Stage 5 - Confirmatory analysis",
                "executable_now": True,
                "execution_status": "unknown",
                "variant_templates": [
                    {
                        "template_id": "bad_within_anchor",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "cv": "within_subject_loso_session",
                            "model": "ridge",
                            "subject": "sub-001",
                        },
                    },
                    {
                        "template_id": "bad_within_anchor_sub002",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "cv": "within_subject_loso_session",
                            "model": "ridge",
                            "subject": "sub-002",
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
                        "template_id": "e17_001_to_002",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "cv": "frozen_cross_person_transfer",
                            "model": "ridge",
                            "train_subject": "sub-001",
                            "test_subject": "sub-002",
                        },
                    },
                    {
                        "template_id": "e17_002_to_001",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "cv": "frozen_cross_person_transfer",
                            "model": "ridge",
                            "train_subject": "sub-002",
                            "test_subject": "sub-001",
                        },
                    },
                ],
            },
        ],
    }
    runtime_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    summary = verify_confirmatory_scope_runtime_alignment(
        scope_config_path=scope_path,
        runtime_registry_path=runtime_path,
    )
    assert summary["passed"] is False
    issue_codes = {
        str(row.get("code")) for row in summary.get("issues", []) if isinstance(row, dict)
    }
    assert "runtime_within_family_experiment_mismatch" in issue_codes


def test_scope_runtime_alignment_fails_when_locked_core_mismatch_detected(tmp_path: Path) -> None:
    scope_path = tmp_path / "scope.json"
    runtime_path = tmp_path / "runtime.json"
    _write_scope(scope_path)
    payload = {
        "schema_version": "workbook-v1",
        "experiments": [
            {
                "experiment_id": "E16",
                "stage": "Stage 5 - Confirmatory analysis",
                "executable_now": True,
                "execution_status": "unknown",
                "variant_templates": [
                    {
                        "template_id": "e16_sub001",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "cv": "within_subject_loso_session",
                            "model": "ridge",
                            "subject": "sub-001",
                            "feature_space": "whole_brain_masked",
                        },
                    },
                    {
                        "template_id": "e16_sub002",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "cv": "within_subject_loso_session",
                            "model": "ridge",
                            "subject": "sub-002",
                            "feature_space": "roi_masked_predefined",
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
                        "template_id": "e17_001_to_002",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "cv": "frozen_cross_person_transfer",
                            "model": "ridge",
                            "train_subject": "sub-001",
                            "test_subject": "sub-002",
                            "feature_space": "whole_brain_masked",
                        },
                    },
                    {
                        "template_id": "e17_002_to_001",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "cv": "frozen_cross_person_transfer",
                            "model": "ridge",
                            "train_subject": "sub-002",
                            "test_subject": "sub-001",
                            "feature_space": "whole_brain_masked",
                        },
                    },
                ],
            },
        ],
    }
    runtime_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    summary = verify_confirmatory_scope_runtime_alignment(
        scope_config_path=scope_path,
        runtime_registry_path=runtime_path,
    )
    assert summary["passed"] is False
    issue_codes = {
        str(row.get("code")) for row in summary.get("issues", []) if isinstance(row, dict)
    }
    assert "runtime_confirmatory_locked_core_mismatch" in issue_codes


def test_control_coverage_rows_follow_runtime_anchor_set() -> None:
    runtime_anchors = [
        {
            "analysis_label": "within_subject_loso_session:sub-001",
            "analysis_type": "within_subject",
            "cv": "within_subject_loso_session",
            "subject": "sub-001",
            "experiment_id": "E16",
            "template_id": "e16_sub001",
        },
        {
            "analysis_label": "frozen_cross_person_transfer:sub-001->sub-002",
            "analysis_type": "cross_person_transfer",
            "cv": "frozen_cross_person_transfer",
            "train_subject": "sub-001",
            "test_subject": "sub-002",
            "experiment_id": "E17",
            "template_id": "e17_001_to_002",
        },
    ]
    e12_rows = [
        {
            "analysis_label": "within_subject_loso_session:sub-001",
            "run_id": "e12_run_sub001",
            "metrics_path": "/tmp/e12_sub001_metrics.json",
            "report_dir": "/tmp/e12_report",
        }
    ]
    e13_rows = [
        {
            "analysis_label": "frozen_cross_person_transfer:sub-001->sub-002",
            "run_id": "e13_run_001_to_002",
            "metrics_path": "/tmp/e13_001_to_002_metrics.json",
            "report_dir": "/tmp/e13_report",
        }
    ]

    rows = build_confirmatory_control_coverage_rows(
        runtime_anchors=runtime_anchors,
        e12_table_rows=e12_rows,
        e13_table_rows=e13_rows,
        e12_summary_json_path="/tmp/e12.json",
        e13_summary_json_path="/tmp/e13.json",
    )
    assert len(rows) == 2
    within_row = [row for row in rows if str(row["analysis_type"]) == "within_subject"][0]
    transfer_row = [row for row in rows if str(row["analysis_type"]) == "cross_person_transfer"][0]
    assert within_row["e12_covered"] is True
    assert within_row["e13_covered"] is False
    assert within_row["e12_run_id"] == "e12_run_sub001"
    assert within_row["e12_metrics_path"] == "/tmp/e12_sub001_metrics.json"
    assert within_row["e13_run_id"] is None
    assert transfer_row["e12_covered"] is False
    assert transfer_row["e13_covered"] is True
    assert transfer_row["e12_metrics_path"] is None
    assert transfer_row["e13_run_id"] == "e13_run_001_to_002"
