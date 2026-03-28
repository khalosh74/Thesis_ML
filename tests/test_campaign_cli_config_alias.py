from __future__ import annotations

from pathlib import Path

from Thesis_ML.config.paths import DEFAULT_DECISION_SUPPORT_REGISTRY
from Thesis_ML.orchestration.campaign_cli import main as campaign_cli_main


def _campaign_result_payload(tmp_path: Path) -> dict[str, object]:
    return {
        "campaign_id": "campaign-test",
        "campaign_root": str(tmp_path / "campaign"),
        "selected_experiments": ["E01"],
        "status_counts": {"completed": 1},
        "run_log_export_path": str(tmp_path / "run_log.csv"),
        "decision_support_summary_path": str(tmp_path / "decision_support_summary.json"),
        "decision_recommendations_path": str(tmp_path / "decision_recommendations.json"),
        "result_aggregation_path": str(tmp_path / "result_aggregation.csv"),
        "summary_outputs_export_path": str(tmp_path / "summary_outputs.csv"),
        "campaign_manifest_path": str(tmp_path / "campaign_manifest.json"),
    }


def test_campaign_cli_registry_alias_resolves_default_registry_in_non_workbook_mode(
    tmp_path: Path,
) -> None:
    captured: dict[str, Path] = {}

    def _stub_read_registry_manifest(path: Path):
        captured["read_registry_path"] = Path(path)
        return object()

    def _stub_run_campaign(**kwargs):
        captured["run_registry_path"] = Path(kwargs["registry_path"])
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
    assert captured["read_registry_path"].resolve() == DEFAULT_DECISION_SUPPORT_REGISTRY.resolve()
    assert captured["run_registry_path"].resolve() == DEFAULT_DECISION_SUPPORT_REGISTRY.resolve()


def test_campaign_cli_explicit_registry_path_override_wins(
    tmp_path: Path,
) -> None:
    captured: dict[str, Path] = {}
    explicit_registry = tmp_path / "explicit_registry.json"
    explicit_registry.write_text("{}\n", encoding="utf-8")

    def _stub_read_registry_manifest(path: Path):
        captured["read_registry_path"] = Path(path)
        return object()

    def _stub_run_campaign(**kwargs):
        captured["run_registry_path"] = Path(kwargs["registry_path"])
        return _campaign_result_payload(tmp_path)

    exit_code = campaign_cli_main(
        [
            "--registry",
            str(explicit_registry),
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
    assert captured["read_registry_path"].resolve() == explicit_registry.resolve()
    assert captured["run_registry_path"].resolve() == explicit_registry.resolve()
