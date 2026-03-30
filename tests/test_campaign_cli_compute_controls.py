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


def test_campaign_cli_forwards_compute_controls_to_campaign_runner(tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    runtime_summary = tmp_path / "runtime_profile_summary.json"
    runtime_summary.write_text("{}", encoding="utf-8")

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
            "--max-parallel-runs",
            "3",
            "--max-parallel-gpu-runs",
            "1",
            "--hardware-mode",
            "max_both",
            "--gpu-device-id",
            "0",
            "--deterministic-compute",
            "--allow-backend-fallback",
            "--phase-plan",
            "auto",
            "--runtime-profile-summary",
            str(runtime_summary),
            "--output-root",
            str(tmp_path / "outputs"),
        ],
        run_decision_support_campaign_fn=_stub_run_campaign,
        run_workbook_decision_support_campaign_fn=lambda **_: _campaign_result_payload(tmp_path),
        read_registry_manifest_fn=_stub_read_registry_manifest,
    )

    assert exit_code == 0
    assert captured["max_parallel_runs"] == 3
    assert captured["max_parallel_gpu_runs"] == 1
    assert captured["hardware_mode"] == "max_both"
    assert captured["gpu_device_id"] == 0
    assert captured["deterministic_compute"] is True
    assert captured["allow_backend_fallback"] is True
    assert captured["phase_plan"] == "auto"
    assert captured["runtime_profile_summary"] == runtime_summary
