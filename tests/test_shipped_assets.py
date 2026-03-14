from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest

from Thesis_ML.config.paths import (
    DEFAULT_DECISION_SUPPORT_REGISTRY,
    SHIPPED_WORKBOOK_TEMPLATE,
)
from Thesis_ML.orchestration.workbook_compiler import (
    NoEnabledExecutableRowsError,
    compile_workbook_file,
)
from Thesis_ML.workbook.validation import validate_workbook


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_default_registry_path_exists() -> None:
    assert DEFAULT_DECISION_SUPPORT_REGISTRY.exists()


def test_shipped_workbook_template_path_exists() -> None:
    assert SHIPPED_WORKBOOK_TEMPLATE.exists()


def test_packaged_registry_asset_matches_committed_registry() -> None:
    repo_registry = _repo_root() / "configs" / "decision_support_registry.json"
    package_registry = resources.files("Thesis_ML").joinpath(
        "assets",
        "configs",
        "decision_support_registry.json",
    )

    with resources.as_file(package_registry) as package_registry_path:
        repo_payload = json.loads(repo_registry.read_text(encoding="utf-8"))
        packaged_payload = json.loads(package_registry_path.read_text(encoding="utf-8"))

    assert packaged_payload == repo_payload


def test_packaged_workbook_asset_matches_committed_template() -> None:
    repo_template = _repo_root() / "templates" / "thesis_experiment_program.xlsx"
    package_template = resources.files("Thesis_ML").joinpath(
        "assets",
        "templates",
        "thesis_experiment_program.xlsx",
    )

    with resources.as_file(package_template) as package_template_path:
        assert package_template_path.read_bytes() == repo_template.read_bytes()


def test_committed_template_validates_with_schema_metadata() -> None:
    repo_template = _repo_root() / "templates" / "thesis_experiment_program.xlsx"
    summary = validate_workbook(repo_template)

    assert summary["sheet_order_ok"] == "True"
    assert summary["required_named_lists_present"] == "True"
    assert summary["workbook_schema_supported"] == "True"
    assert summary["schema_metadata_keys_present"] == "True"


def test_committed_template_is_non_runnable_until_enabled() -> None:
    repo_template = _repo_root() / "templates" / "thesis_experiment_program.xlsx"
    with pytest.raises(NoEnabledExecutableRowsError):
        compile_workbook_file(repo_template)
