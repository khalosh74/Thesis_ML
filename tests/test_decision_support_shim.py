from __future__ import annotations

import run_decision_support_experiments as shim

from Thesis_ML.orchestration import decision_support


def test_shim_main_forwards_to_packaged_orchestrator(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        seen["argv"] = argv
        return 7

    monkeypatch.setattr(shim._decision_support, "main", fake_main)
    assert shim.main(["--all", "--dry-run"]) == 7
    assert seen["argv"] == ["--all", "--dry-run"]


def test_shim_exposes_campaign_function() -> None:
    assert shim.run_decision_support_campaign is decision_support.run_decision_support_campaign
