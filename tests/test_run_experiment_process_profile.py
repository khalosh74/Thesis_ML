from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.experiments.run_experiment import run_experiment


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_direct_run_process_profile_artifacts_are_written_and_referenced(tmp_path: Path) -> None:
    demo_root = _repo_root() / "demo_data" / "synthetic_v1"
    result = run_experiment(
        index_csv=demo_root / "dataset_index.csv",
        data_root=demo_root / "data_root",
        cache_dir=tmp_path / "cache",
        target="coarse_affect",
        model="ridge",
        cv="within_subject_loso_session",
        subject="sub-001",
        seed=42,
        n_permutations=0,
        run_id="direct_process_profile",
        reports_root=tmp_path / "reports",
        process_profile_enabled=True,
        process_sample_interval_seconds=0.05,
        process_include_io_counters=True,
    )

    report_dir = Path(str(result["report_dir"]))
    process_samples_path = report_dir / "process_samples.jsonl"
    process_profile_summary_path = report_dir / "process_profile_summary.json"
    assert process_samples_path.exists()
    assert process_profile_summary_path.exists()

    run_status_path = Path(str(result["run_status_path"]))
    run_status_payload = json.loads(run_status_path.read_text(encoding="utf-8"))
    assert isinstance(run_status_payload.get("process_profile_summary"), dict)
    assert isinstance(run_status_payload.get("process_profile_artifacts"), dict)
    assert (
        run_status_payload["process_profile_artifacts"]["process_samples_path"]
        == str(process_samples_path.resolve())
    )
    assert (
        run_status_payload["process_profile_artifacts"]["process_profile_summary_path"]
        == str(process_profile_summary_path.resolve())
    )

    assert isinstance(result.get("process_profile_summary"), dict)
    assert isinstance(result.get("process_profile_artifacts"), dict)
    assert (
        result["process_profile_artifacts"]["process_samples_path"]
        == str(process_samples_path.resolve())
    )
    assert (
        result["process_profile_artifacts"]["process_profile_summary_path"]
        == str(process_profile_summary_path.resolve())
    )
