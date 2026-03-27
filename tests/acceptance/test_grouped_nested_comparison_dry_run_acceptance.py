from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Thesis_ML.cli.comparison_runner import main as comparison_main


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_grouped_nested_comparison_cli_dry_run_acceptance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    index_csv = tmp_path / "dataset_index.csv"
    pd.DataFrame([{"subject": "sub-001"}, {"subject": "sub-002"}]).to_csv(index_csv, index=False)

    data_root = tmp_path / "Data"
    cache_dir = tmp_path / "cache"
    reports_root = tmp_path / "reports" / "comparisons"
    comparison_path = (
        _repo_root() / "configs" / "comparisons" / "model_family_grouped_nested_comparison_v2.json"
    )

    monkeypatch.setenv("THESIS_ML_INDEX_CSV", str(index_csv))
    monkeypatch.setenv("THESIS_ML_DATA_ROOT", str(data_root))
    monkeypatch.setenv("THESIS_ML_CACHE_DIR", str(cache_dir))

    exit_code = comparison_main(
        [
            "--comparison",
            str(comparison_path),
            "--variant",
            "ridge",
            "--reports-root",
            str(reports_root),
            "--dry-run",
        ]
    )
    assert exit_code == 0

    comparison_output_dir = reports_root / "comparison_runs" / "model-family-grouped-nested__2.0.0"
    assert (comparison_output_dir / "comparison.json").exists()
    assert (comparison_output_dir / "compiled_comparison_manifest.json").exists()
    assert (comparison_output_dir / "comparison_summary.json").exists()
    assert (comparison_output_dir / "comparison_decision.json").exists()
    assert (comparison_output_dir / "execution_status.json").exists()
    assert (comparison_output_dir / "report_index.csv").exists()

    execution_payload = json.loads(
        (comparison_output_dir / "execution_status.json").read_text(encoding="utf-8")
    )
    assert execution_payload["framework_mode"] == "locked_comparison"
    assert execution_payload["dry_run"] is True
    assert all(run["status"] == "planned" for run in execution_payload["runs"])
