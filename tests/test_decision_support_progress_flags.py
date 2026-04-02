from __future__ import annotations

from pathlib import Path

from Thesis_ML.orchestration.campaign_cli import main as campaign_cli_main


def _campaign_result_payload(tmp_path: Path) -> dict[str, object]:
    return {
        "campaign_id": "campaign-test",
        "campaign_root": str(tmp_path / "campaign"),
        "selected_experiments": ["E01"],
        "status_counts": {"dry_run": 1},
        "run_log_export_path": str(tmp_path / "run_log.csv"),
        "decision_support_summary_path": str(tmp_path / "decision_support_summary.csv"),
        "decision_recommendations_path": str(tmp_path / "decision_recommendations.md"),
        "result_aggregation_path": str(tmp_path / "result_aggregation.json"),
        "summary_outputs_export_path": str(tmp_path / "summary_outputs.csv"),
        "campaign_manifest_path": str(tmp_path / "campaign_manifest.json"),
    }


def test_campaign_cli_progress_flags_are_forwarded(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _stub_read_registry_manifest(_: Path):
        return object()

    def _stub_run_campaign(**kwargs):
        captured.update(kwargs)
        return _campaign_result_payload(tmp_path)

    exit_code = campaign_cli_main(
        [
            "--registry-alias",
            "registry.decision_support_default",
            "--all",
            "--dry-run",
            "--quiet-progress",
            "--progress-interval-seconds",
            "3.5",
            "--output-root",
            str(tmp_path / "outputs"),
        ],
        run_decision_support_campaign_fn=_stub_run_campaign,
        run_workbook_decision_support_campaign_fn=lambda **_: _campaign_result_payload(tmp_path),
        read_registry_manifest_fn=_stub_read_registry_manifest,
    )

    assert exit_code == 0
    assert captured["quiet_progress"] is True
    assert captured["progress_interval_seconds"] == 3.5


def test_campaign_cli_progress_ui_flags_are_forwarded(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _stub_read_registry_manifest(_: Path):
        return object()

    def _stub_run_campaign(**kwargs):
        captured.update(kwargs)
        return _campaign_result_payload(tmp_path)

    exit_code = campaign_cli_main(
        [
            "--registry-alias",
            "registry.decision_support_default",
            "--all",
            "--dry-run",
            "--progress-ui",
            "bar",
            "--progress-detail",
            "verbose",
            "--output-root",
            str(tmp_path / "outputs"),
        ],
        run_decision_support_campaign_fn=_stub_run_campaign,
        run_workbook_decision_support_campaign_fn=lambda **_: _campaign_result_payload(tmp_path),
        read_registry_manifest_fn=_stub_read_registry_manifest,
    )

    assert exit_code == 0
    assert captured["progress_ui"] == "bar"
    assert captured["progress_detail"] == "verbose"


def test_campaign_cli_progress_defaults_are_verbose_and_fast(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _stub_read_registry_manifest(_: Path):
        return object()

    def _stub_run_campaign(**kwargs):
        captured.update(kwargs)
        return _campaign_result_payload(tmp_path)

    exit_code = campaign_cli_main(
        [
            "--registry-alias",
            "registry.decision_support_default",
            "--all",
            "--dry-run",
            "--output-root",
            str(tmp_path / "outputs"),
        ],
        run_decision_support_campaign_fn=_stub_run_campaign,
        run_workbook_decision_support_campaign_fn=lambda **_: _campaign_result_payload(tmp_path),
        read_registry_manifest_fn=_stub_read_registry_manifest,
    )

    assert exit_code == 0
    assert captured["progress_ui"] == "auto"
    assert captured["progress_detail"] == "verbose"
    assert captured["progress_interval_seconds"] == 5.0
