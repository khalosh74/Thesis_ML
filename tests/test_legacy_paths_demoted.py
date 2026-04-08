from __future__ import annotations

from pathlib import Path

import pytest

from Thesis_ML.cli.comparison_runner import _build_parser as build_comparison_parser
from Thesis_ML.cli.comparison_runner import main as comparison_main
from Thesis_ML.cli.decision_support import main as decision_support_main
from Thesis_ML.cli.protocol_runner import _build_parser as build_protocol_parser
from Thesis_ML.cli.protocol_runner import main as protocol_main
from Thesis_ML.orchestration.campaign_cli import build_parser as build_campaign_parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_docs_and_help_text_demote_legacy_paths() -> None:
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    runbook = (_repo_root() / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")

    assert "thesisml-run-release" in readme
    assert "thesisml-promote-run" in readme
    assert "non-official" in readme
    assert "release mode is the only official thesis evidence path" in runbook

    protocol_help = build_protocol_parser().format_help()
    comparison_help = build_comparison_parser().format_help()
    decision_support_help = build_campaign_parser().format_help()

    assert "non-official" in protocol_help
    assert "non-official" in comparison_help
    assert "non-official" in decision_support_help


def test_legacy_commands_still_import_and_run_help() -> None:
    with pytest.raises(SystemExit) as protocol_exit:
        protocol_main(["--help"])
    assert protocol_exit.value.code == 0

    with pytest.raises(SystemExit) as comparison_exit:
        comparison_main(["--help"])
    assert comparison_exit.value.code == 0

    with pytest.raises(SystemExit) as decision_support_exit:
        decision_support_main(["--help"])
    assert decision_support_exit.value.code == 0
