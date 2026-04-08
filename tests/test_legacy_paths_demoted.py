from __future__ import annotations

import importlib
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_docs_expose_only_release_runtime_story() -> None:
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    release_doc = (_repo_root() / "docs" / "RELEASE.md").read_text(encoding="utf-8")
    runbook = (_repo_root() / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")

    for text in (readme, release_doc, runbook):
        assert "thesisml-validate-dataset" in text
        assert "thesisml-validate-release" in text
        assert "thesisml-run-release" in text
        assert "thesisml-promote-run" in text
        assert "thesisml-run-protocol" not in text
        assert "thesisml-run-comparison" not in text
        assert "thesisml-run-decision-support" not in text
        assert "thesisml-workbook" not in text


def test_pyproject_entrypoints_remove_legacy_runners() -> None:
    pyproject = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")

    assert "thesisml-run-release" in pyproject
    assert "thesisml-promote-run" in pyproject
    assert "thesisml-validate-release" in pyproject
    assert "thesisml-validate-dataset" in pyproject

    assert "thesisml-run-protocol" not in pyproject
    assert "thesisml-run-comparison" not in pyproject
    assert "thesisml-run-decision-support" not in pyproject
    assert "thesisml-workbook" not in pyproject
    assert "thesisml-run-baseline" not in pyproject
    assert "thesisml-run-experiment" not in pyproject


def test_legacy_cli_modules_are_removed() -> None:
    for module_name in (
        "Thesis_ML.cli.protocol_runner",
        "Thesis_ML.cli.comparison_runner",
        "Thesis_ML.cli.decision_support",
        "Thesis_ML.cli.workbook",
        "Thesis_ML.cli.baseline",
        "Thesis_ML.protocols",
        "Thesis_ML.comparisons",
        "Thesis_ML.orchestration",
        "Thesis_ML.workbook",
    ):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"Legacy module must not be importable: {module_name}")
