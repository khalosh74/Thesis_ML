from __future__ import annotations

from pathlib import Path

from Thesis_ML.cli.comparison_runner import main as comparison_main
from Thesis_ML.cli.protocol_runner import main as protocol_main


def test_protocol_cli_returns_nonzero_for_timed_out_runs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("Thesis_ML.cli.protocol_runner.load_protocol", lambda *_: object())
    monkeypatch.setattr(
        "Thesis_ML.cli.protocol_runner.compile_and_run_protocol",
        lambda **_: {
            "protocol_id": "protocol",
            "protocol_version": "1.0.0",
            "protocol_output_dir": str(tmp_path / "protocol_output"),
            "n_success": 0,
            "n_completed": 0,
            "n_failed": 0,
            "n_timed_out": 1,
            "n_skipped_due_to_policy": 0,
            "n_planned": 0,
            "max_parallel_runs_effective": 1,
            "artifact_paths": {},
        },
    )

    exit_code = protocol_main(
        [
            "--protocol",
            str(tmp_path / "protocol.json"),
            "--all-suites",
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_code == 1


def test_comparison_cli_allows_timed_out_without_forced_nonzero(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("Thesis_ML.cli.comparison_runner.load_comparison_spec", lambda *_: object())
    monkeypatch.setattr(
        "Thesis_ML.cli.comparison_runner.compile_and_run_comparison",
        lambda **_: {
            "comparison_id": "comparison",
            "comparison_version": "1.0.0",
            "comparison_output_dir": str(tmp_path / "comparison_output"),
            "n_success": 0,
            "n_completed": 0,
            "n_failed": 0,
            "n_timed_out": 2,
            "n_skipped_due_to_policy": 0,
            "n_planned": 0,
            "max_parallel_runs_effective": 1,
            "artifact_paths": {},
        },
    )

    exit_code = comparison_main(
        [
            "--comparison",
            str(tmp_path / "comparison.json"),
            "--all-variants",
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_code == 0
