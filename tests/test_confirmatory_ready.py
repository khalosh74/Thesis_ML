from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.verification.confirmatory_ready import verify_confirmatory_ready


def _write_minimal_confirmatory_summary(
    output_dir: Path,
    *,
    confirmatory_status: str = "confirmatory",
    science_critical_deviation_detected: bool = False,
    controls_valid_for_confirmatory: bool = True,
    required_evidence_valid: bool = True,
    dataset_fingerprint_complete: bool = True,
    all_runs_completed: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    run_status = "completed" if all_runs_completed else "failed"
    execution_status = {
        "framework_mode": "confirmatory",
        "confirmatory_status": confirmatory_status,
        "science_critical_deviation_detected": bool(science_critical_deviation_detected),
        "runs": [{"run_id": "run_001", "status": run_status}],
    }
    (output_dir / "execution_status.json").write_text(
        f"{json.dumps(execution_status, indent=2)}\n",
        encoding="utf-8",
    )

    n_missing = 0 if dataset_fingerprint_complete else 1
    n_with = 1 if dataset_fingerprint_complete else 0
    suite_summary = {
        "required_evidence_status": {"valid": bool(required_evidence_valid)},
        "confirmatory_reporting_contract": {
            "controls_status": {
                "controls_valid_for_confirmatory": bool(controls_valid_for_confirmatory)
            },
            "dataset_fingerprint": {
                "n_completed_runs": 1,
                "n_with_fingerprint": n_with,
                "n_missing_fingerprint": n_missing,
                "missing_run_ids": [] if dataset_fingerprint_complete else ["run_001"],
                "unique_fingerprint_count": 1 if dataset_fingerprint_complete else 0,
                "consistent_across_runs": bool(dataset_fingerprint_complete),
                "sources": ["config"],
            },
        },
    }
    (output_dir / "suite_summary.json").write_text(
        f"{json.dumps(suite_summary, indent=2)}\n",
        encoding="utf-8",
    )

    deviation_log = {
        "science_critical_deviation_detected": bool(science_critical_deviation_detected),
        "confirmatory_status": confirmatory_status,
        "deviations": [
            {
                "run_id": None,
                "suite_id": None,
                "status": "no_deviation" if not science_critical_deviation_detected else "error",
                "science_critical": bool(science_critical_deviation_detected),
                "reason": "Synthetic test payload.",
            }
        ],
    }
    (output_dir / "deviation_log.json").write_text(
        f"{json.dumps(deviation_log, indent=2)}\n",
        encoding="utf-8",
    )


def _write_control_coverage(
    output_dir: Path,
    *,
    rows: list[dict[str, object]],
) -> Path:
    coverage_root = output_dir / "special_aggregations" / "confirmatory"
    coverage_root.mkdir(parents=True, exist_ok=True)
    path = coverage_root / "confirmatory_anchor_control_coverage.json"
    payload = {
        "generated_at_utc": "2026-04-04T00:00:00+00:00",
        "rows": rows,
    }
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    return path


def test_confirmatory_ready_passes_with_clean_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "protocol_runs" / "thesis-confirmatory__1.0.0"
    _write_minimal_confirmatory_summary(output_dir)

    monkeypatch.setattr(
        "Thesis_ML.verification.confirmatory_ready.verify_official_artifacts",
        lambda **_: {"passed": True, "framework_mode": "confirmatory", "issues": []},
    )
    summary = verify_confirmatory_ready(output_dir=output_dir)
    assert summary["passed"] is True
    assert all(bool(entry["passed"]) for entry in summary["criteria"])


def test_confirmatory_ready_fails_on_downgraded_status(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "protocol_runs" / "thesis-confirmatory__1.0.0"
    _write_minimal_confirmatory_summary(output_dir, confirmatory_status="downgraded")
    monkeypatch.setattr(
        "Thesis_ML.verification.confirmatory_ready.verify_official_artifacts",
        lambda **_: {"passed": True, "framework_mode": "confirmatory", "issues": []},
    )
    summary = verify_confirmatory_ready(output_dir=output_dir)
    assert summary["passed"] is False
    assert any(issue["code"] == "confirmatory_status_invalid" for issue in summary["issues"])


def test_confirmatory_ready_fails_when_controls_invalid(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "protocol_runs" / "thesis-confirmatory__1.0.0"
    _write_minimal_confirmatory_summary(output_dir, controls_valid_for_confirmatory=False)
    monkeypatch.setattr(
        "Thesis_ML.verification.confirmatory_ready.verify_official_artifacts",
        lambda **_: {"passed": True, "framework_mode": "confirmatory", "issues": []},
    )
    summary = verify_confirmatory_ready(output_dir=output_dir)
    assert summary["passed"] is False
    assert any(
        issue["code"] == "controls_not_valid_for_confirmatory" for issue in summary["issues"]
    )


def test_confirmatory_ready_fails_when_dataset_fingerprint_incomplete(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "protocol_runs" / "thesis-confirmatory__1.0.0"
    _write_minimal_confirmatory_summary(output_dir, dataset_fingerprint_complete=False)
    monkeypatch.setattr(
        "Thesis_ML.verification.confirmatory_ready.verify_official_artifacts",
        lambda **_: {"passed": True, "framework_mode": "confirmatory", "issues": []},
    )
    summary = verify_confirmatory_ready(output_dir=output_dir)
    assert summary["passed"] is False
    assert any(
        issue["code"] == "dataset_fingerprint_missing_or_incomplete" for issue in summary["issues"]
    )


def test_confirmatory_ready_fails_when_repro_summary_failed(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "protocol_runs" / "thesis-confirmatory__1.0.0"
    _write_minimal_confirmatory_summary(output_dir)
    repro_summary_path = tmp_path / "repro_summary.json"
    repro_summary_path.write_text(
        f"{json.dumps({'passed': False}, indent=2)}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "Thesis_ML.verification.confirmatory_ready.verify_official_artifacts",
        lambda **_: {"passed": True, "framework_mode": "confirmatory", "issues": []},
    )
    summary = verify_confirmatory_ready(
        output_dir=output_dir,
        reproducibility_summary=repro_summary_path,
    )
    assert summary["passed"] is False
    assert any(issue["code"] == "reproducibility_summary_failed" for issue in summary["issues"])


def test_confirmatory_ready_surfaces_official_verification_failure(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "protocol_runs" / "thesis-confirmatory__1.0.0"
    _write_minimal_confirmatory_summary(output_dir)
    monkeypatch.setattr(
        "Thesis_ML.verification.confirmatory_ready.verify_official_artifacts",
        lambda **_: {
            "passed": False,
            "framework_mode": "confirmatory",
            "issues": [{"code": "synthetic_issue"}],
        },
    )
    summary = verify_confirmatory_ready(output_dir=output_dir)
    assert summary["passed"] is False
    assert any(
        issue["code"] == "official_artifact_verification_failed" for issue in summary["issues"]
    )


def test_confirmatory_ready_fails_when_scope_runtime_alignment_fails(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "protocol_runs" / "thesis-confirmatory__1.0.0"
    _write_minimal_confirmatory_summary(output_dir)
    monkeypatch.setattr(
        "Thesis_ML.verification.confirmatory_ready.verify_official_artifacts",
        lambda **_: {"passed": True, "framework_mode": "confirmatory", "issues": []},
    )
    monkeypatch.setattr(
        "Thesis_ML.verification.confirmatory_ready.verify_confirmatory_scope_runtime_alignment",
        lambda **_: {
            "passed": False,
            "issues": [{"code": "scope_within_missing_in_runtime"}],
        },
    )
    summary = verify_confirmatory_ready(output_dir=output_dir)
    assert summary["passed"] is False
    assert any(
        issue["code"] == "confirmatory_scope_runtime_mismatch" for issue in summary["issues"]
    )


def test_confirmatory_ready_enforces_control_coverage_when_requested(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "protocol_runs" / "thesis-confirmatory__1.0.0"
    _write_minimal_confirmatory_summary(output_dir)
    monkeypatch.setattr(
        "Thesis_ML.verification.confirmatory_ready.verify_official_artifacts",
        lambda **_: {"passed": True, "framework_mode": "confirmatory", "issues": []},
    )
    monkeypatch.setattr(
        "Thesis_ML.verification.confirmatory_ready.verify_confirmatory_scope_runtime_alignment",
        lambda **_: {"passed": True, "issues": []},
    )
    summary = verify_confirmatory_ready(output_dir=output_dir, require_control_coverage=True)
    assert summary["passed"] is False
    assert any(
        issue["code"] == "confirmatory_control_coverage_artifact_missing"
        for issue in summary["issues"]
    )


def test_confirmatory_ready_requires_full_e12_e13_coverage_when_requested(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "protocol_runs" / "thesis-confirmatory__1.0.0"
    _write_minimal_confirmatory_summary(output_dir)
    _write_control_coverage(
        output_dir,
        rows=[
            {
                "analysis_label": "within_subject_loso_session:sub-001",
                "e12_covered": True,
                "e13_covered": True,
            },
            {
                "analysis_label": "frozen_cross_person_transfer:sub-001->sub-002",
                "e12_covered": False,
                "e13_covered": False,
            },
        ],
    )
    monkeypatch.setattr(
        "Thesis_ML.verification.confirmatory_ready.verify_official_artifacts",
        lambda **_: {"passed": True, "framework_mode": "confirmatory", "issues": []},
    )
    monkeypatch.setattr(
        "Thesis_ML.verification.confirmatory_ready.verify_confirmatory_scope_runtime_alignment",
        lambda **_: {
            "passed": True,
            "issues": [],
            "runtime_anchor_set": [
                {"analysis_label": "within_subject_loso_session:sub-001"},
                {"analysis_label": "frozen_cross_person_transfer:sub-001->sub-002"},
            ],
        },
    )
    summary = verify_confirmatory_ready(output_dir=output_dir, require_control_coverage=True)
    assert summary["passed"] is False
    issue_codes = {str(issue["code"]) for issue in summary["issues"]}
    assert "confirmatory_control_coverage_e12_missing" in issue_codes
    assert "confirmatory_control_coverage_e13_missing" in issue_codes


def test_confirmatory_ready_passes_when_full_control_coverage_present(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "protocol_runs" / "thesis-confirmatory__1.0.0"
    _write_minimal_confirmatory_summary(output_dir)
    _write_control_coverage(
        output_dir,
        rows=[
            {
                "analysis_label": "within_subject_loso_session:sub-001",
                "e12_covered": True,
                "e13_covered": True,
            },
            {
                "analysis_label": "frozen_cross_person_transfer:sub-001->sub-002",
                "e12_covered": True,
                "e13_covered": True,
            },
        ],
    )
    monkeypatch.setattr(
        "Thesis_ML.verification.confirmatory_ready.verify_official_artifacts",
        lambda **_: {"passed": True, "framework_mode": "confirmatory", "issues": []},
    )
    monkeypatch.setattr(
        "Thesis_ML.verification.confirmatory_ready.verify_confirmatory_scope_runtime_alignment",
        lambda **_: {
            "passed": True,
            "issues": [],
            "runtime_anchor_set": [
                {"analysis_label": "within_subject_loso_session:sub-001"},
                {"analysis_label": "frozen_cross_person_transfer:sub-001->sub-002"},
            ],
        },
    )
    summary = verify_confirmatory_ready(output_dir=output_dir, require_control_coverage=True)
    assert summary["passed"] is True
