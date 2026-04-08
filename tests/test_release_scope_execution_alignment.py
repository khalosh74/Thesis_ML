from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from tests._release_test_utils import dataset_manifest_path, make_temp_release_bundle
from Thesis_ML.release.loader import load_dataset_manifest, load_release_bundle
from Thesis_ML.release.scope import (
    compile_release_scope,
    verify_scope_execution_alignment,
)


def _write_protocol_success_with_mismatched_selected_samples(
    *,
    run_dir: Path,
    expected_selected_samples_path: Path,
) -> str:
    protocol_output_dir = run_dir / "artifacts" / "protocol_runs" / "thesis_confirmatory_v1__v1.1"
    report_dir = protocol_output_dir / "run_primary_sub001"
    report_dir.mkdir(parents=True, exist_ok=True)

    expected = pd.read_csv(expected_selected_samples_path)
    if "sample_id" not in expected.columns:
        raise ValueError("expected selected samples must contain sample_id")
    expected = expected.copy()
    # Simulate a drift: one expected sample missing and one extra sample leaked in.
    missing_sample_id = str(expected.iloc[0]["sample_id"])
    mismatched = expected.iloc[1:].copy()
    leaked = mismatched.iloc[[0]].copy()
    leaked["sample_id"] = "leaked_sample_id_from_runtime"
    mismatched = pd.concat([mismatched, leaked], ignore_index=True)
    mismatched.to_csv(report_dir / "feature_qc_selected_samples.csv", index=False)

    with (protocol_output_dir / "report_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "suite_id", "status", "report_dir"])
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "run_primary_sub001",
                "suite_id": "confirmatory_primary_within_subject",
                "status": "success",
                "report_dir": str(report_dir.resolve()),
            }
        )
    return missing_sample_id


def test_release_scope_alignment_verification_reports_missing_and_extra_sample_ids(
    tmp_path: Path,
) -> None:
    release_path = make_temp_release_bundle(tmp_path)
    release = load_release_bundle(release_path)
    dataset = load_dataset_manifest(dataset_manifest_path())
    run_dir = tmp_path / "run"

    compiled_scope = compile_release_scope(
        release_bundle=release,
        dataset_manifest=dataset,
        run_dir=run_dir,
    )
    missing_sample_id = _write_protocol_success_with_mismatched_selected_samples(
        run_dir=run_dir,
        expected_selected_samples_path=compiled_scope.selected_samples_path,
    )

    verification = verify_scope_execution_alignment(
        run_dir=run_dir,
        compiled_scope_manifest_path=compiled_scope.scope_manifest_path,
        expected_science_hash=release.hashes.science_hash,
        expected_target_mapping_hash=release.science.target.mapping_hash,
        write_output=True,
    )
    assert verification["passed"] is False
    mismatch_issues = [
        issue
        for issue in verification["issues"]
        if str(issue.get("code")) == "scope_sample_id_mismatch"
    ]
    assert mismatch_issues
    mismatch_details = mismatch_issues[0]["details"]
    assert missing_sample_id in mismatch_details["missing_sample_ids"]
    assert "leaked_sample_id_from_runtime" in mismatch_details["extra_sample_ids"]

    verification_path = run_dir / "verification" / "scope_alignment_verification.json"
    persisted = json.loads(verification_path.read_text(encoding="utf-8"))
    assert persisted["passed"] is False
