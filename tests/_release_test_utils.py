from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def release_bundle_path() -> Path:
    return repo_root() / "releases" / "thesis_final_v1" / "release.json"


def dataset_manifest_path() -> Path:
    return repo_root() / "demo_data" / "synthetic_v1" / "dataset_manifest.json"


def make_temp_release_bundle(tmp_path: Path) -> Path:
    source_dir = release_bundle_path().parent
    target_dir = tmp_path / "release_bundle"
    shutil.copytree(source_dir, target_dir)

    execution_path = target_dir / "execution.json"
    execution_payload = json.loads(execution_path.read_text(encoding="utf-8"))
    run_root = (tmp_path / "runs").resolve()
    execution_payload["run_root"] = str(run_root)
    execution_payload["candidate_root"] = str((run_root / "candidate").resolve())
    execution_payload["official_root"] = str((run_root / "official").resolve())
    execution_payload["exploratory_root"] = str((run_root / "exploratory").resolve())
    execution_payload["scratch_root"] = str((run_root / "scratch").resolve())
    execution_path.write_text(f"{json.dumps(execution_payload, indent=2)}\n", encoding="utf-8")

    return target_dir / "release.json"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, f"{json.dumps(payload, indent=2)}\n")


def _write_required_run_artifacts(
    report_dir: Path,
    *,
    selected_samples_index_csv: Path,
    config_overrides: dict[str, Any] | None = None,
    missing_run_artifact: str | None = None,
) -> None:
    config_payload: dict[str, Any] = {
        "model": "ridge",
        "feature_space": "whole_brain_masked",
        "preprocessing_strategy": "none",
        "dimensionality_strategy": "none",
        "tuning_enabled": False,
        "model_governance": {
            "logical_model_name": "ridge",
        },
    }
    if config_overrides:
        config_payload.update(config_overrides)

    _write_json(report_dir / "config.json", config_payload)
    _write_json(report_dir / "metrics.json", {"primary_metric_name": "balanced_accuracy"})
    _write_text(report_dir / "fold_metrics.csv", "fold,metric\n0,0.50\n")
    _write_text(report_dir / "fold_splits.csv", "fold,sample_id\n0,s1\n")
    _write_text(report_dir / "predictions.csv", "sample_id,y_true,y_pred\ns1,1,1\n")
    _write_text(report_dir / "subgroup_report.csv", "group,metric\nall,0.50\n")
    _write_json(report_dir / "spatial_report.json", {"compatible": True})
    _write_json(report_dir / "run_status.json", {"status": "success"})
    selected_df = pd.read_csv(selected_samples_index_csv)
    selected_columns = [
        column_name
        for column_name in ("sample_id", "subject", "session", "task", "modality")
        if column_name in selected_df.columns
    ]
    selected_df.loc[:, selected_columns].to_csv(
        report_dir / "feature_qc_selected_samples.csv",
        index=False,
    )

    if missing_run_artifact:
        missing_path = report_dir / missing_run_artifact
        if missing_path.exists():
            missing_path.unlink()


def write_fake_protocol_outputs(
    *,
    artifacts_dir: Path,
    selected_samples_index_csv: Path,
    config_overrides: dict[str, Any] | None = None,
    missing_run_artifact: str | None = None,
    controls_valid: bool = True,
) -> Path:
    protocol_output_dir = artifacts_dir / "protocol_runs" / "thesis_confirmatory_v1__v1.1"
    protocol_output_dir.mkdir(parents=True, exist_ok=True)

    suite_rows = [
        {
            "run_id": "run_primary_sub001",
            "suite_id": "confirmatory_primary_within_subject",
            "report_dir": str((protocol_output_dir / "run_primary_sub001").resolve()),
            "status": "success",
        },
        {
            "run_id": "run_secondary_transfer",
            "suite_id": "confirmatory_secondary_cross_person_transfer",
            "report_dir": str((protocol_output_dir / "run_secondary_transfer").resolve()),
            "status": "success",
        },
    ]

    for row in suite_rows:
        report_dir = Path(str(row["report_dir"]))
        report_dir.mkdir(parents=True, exist_ok=True)
        _write_required_run_artifacts(
            report_dir,
            selected_samples_index_csv=selected_samples_index_csv,
            config_overrides=config_overrides,
            missing_run_artifact=missing_run_artifact,
        )

    with (protocol_output_dir / "report_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "suite_id", "status", "report_dir"])
        writer.writeheader()
        writer.writerows(suite_rows)

    _write_json(
        protocol_output_dir / "compiled_protocol_manifest.json",
        {
            "suite_ids": [
                "confirmatory_primary_within_subject",
                "confirmatory_secondary_cross_person_transfer",
            ],
            "runs": [
                {
                    "run_id": "compiled_primary_sub001",
                    "cv_mode": "within_subject_loso_session",
                    "subject": "sub-001",
                    "model": "ridge",
                    "target": "coarse_affect",
                    "tuning_enabled": False,
                },
                {
                    "run_id": "compiled_primary_sub002",
                    "cv_mode": "within_subject_loso_session",
                    "subject": "sub-002",
                    "model": "ridge",
                    "target": "coarse_affect",
                    "tuning_enabled": False,
                },
                {
                    "run_id": "compiled_transfer_001_to_002",
                    "cv_mode": "frozen_cross_person_transfer",
                    "train_subject": "sub-001",
                    "test_subject": "sub-002",
                    "model": "ridge",
                    "target": "coarse_affect",
                    "tuning_enabled": False,
                },
                {
                    "run_id": "compiled_transfer_002_to_001",
                    "cv_mode": "frozen_cross_person_transfer",
                    "train_subject": "sub-002",
                    "test_subject": "sub-001",
                    "model": "ridge",
                    "target": "coarse_affect",
                    "tuning_enabled": False,
                },
            ],
        },
    )
    _write_json(protocol_output_dir / "execution_status.json", {"warnings": []})
    _write_json(
        protocol_output_dir / "suite_summary.json",
        {
            "confirmatory_reporting_contract": {
                "controls_status": {
                    "controls_valid_for_confirmatory": bool(controls_valid),
                }
            }
        },
    )
    _write_json(
        protocol_output_dir / "deviation_log.json",
        {"deviations": [], "n_total_deviations": 0, "science_critical_deviation_detected": False},
    )
    _write_json(protocol_output_dir / "claim_outcomes.json", {"claims": []})
    return protocol_output_dir


def fake_compile_and_run_protocol_factory(
    *,
    config_overrides: dict[str, Any] | None = None,
    missing_run_artifact: str | None = None,
    controls_valid: bool = True,
):
    def _fake_compile_and_run_protocol(**kwargs: Any) -> dict[str, Any]:
        reports_root = Path(str(kwargs["reports_root"])).resolve()
        selected_samples_index_csv = Path(str(kwargs["index_csv"])).resolve()
        protocol_output_dir = write_fake_protocol_outputs(
            artifacts_dir=reports_root,
            selected_samples_index_csv=selected_samples_index_csv,
            config_overrides=config_overrides,
            missing_run_artifact=missing_run_artifact,
            controls_valid=controls_valid,
        )
        return {
            "protocol_id": "thesis_confirmatory_v1",
            "protocol_version": "v1.1",
            "protocol_output_dir": str(protocol_output_dir),
            "n_success": 2,
            "n_failed": 0,
            "n_timed_out": 0,
            "n_skipped_due_to_policy": 0,
            "n_planned": 2,
            "max_parallel_runs_effective": 1,
            "artifact_paths": {
                "report_index": str((protocol_output_dir / "report_index.csv").resolve()),
                "suite_summary": str((protocol_output_dir / "suite_summary.json").resolve()),
                "deviation_log": str((protocol_output_dir / "deviation_log.json").resolve()),
            },
            "artifact_verification": {"passed": True, "issues": []},
            "run_results": [],
        }

    return _fake_compile_and_run_protocol
