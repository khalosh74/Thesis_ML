from __future__ import annotations

import json
from pathlib import Path

import pytest

from Thesis_ML.config import get_config_entry, resolve_config_id, validate_config_bundle
from Thesis_ML.config.paths import DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _registry_payload() -> dict[str, object]:
    return json.loads((_repo_root() / "configs" / "config_registry.json").read_text(encoding="utf-8"))


def _resolve_config_id(config_id: str) -> Path:
    repo_root = _repo_root()
    return resolve_config_id(
        config_id,
        source_repo_root=repo_root,
        project_root=repo_root,
    ).resolve()


def _require_config_entry(config_id: str) -> dict[str, object]:
    entry = get_config_entry(config_id)
    if entry is None:
        raise KeyError(config_id)
    return entry


def test_archived_protocol_compat_id_resolves_to_archive_path() -> None:
    path = _resolve_config_id("protocol.thesis_canonical_nested_v1")
    assert path.exists()
    assert path.as_posix().endswith("configs/archive/protocols/thesis_canonical_nested_v1.json")


def test_archived_comparison_compat_id_resolves_to_archive_path() -> None:
    path = _resolve_config_id("comparison.model_family_grouped_nested_comparison_v1")
    assert path.exists()
    assert path.as_posix().endswith(
        "configs/archive/comparisons/model_family_grouped_nested_comparison_v1.json"
    )


def test_retired_target_v1_is_not_a_live_registry_entry() -> None:
    with pytest.raises(KeyError):
        _require_config_entry("target.affect_mapping_v1")


def test_retired_section_contains_target_v1_with_replacement() -> None:
    payload = _registry_payload()
    retired = payload.get("retired")
    assert isinstance(retired, list)
    matching = [
        entry
        for entry in retired
        if isinstance(entry, dict) and str(entry.get("config_id", "")) == "target.affect_mapping_v1"
    ]
    assert len(matching) == 1
    assert matching[0].get("replacement") == "target.affect_mapping_v2"


def test_retired_target_mapping_file_is_removed() -> None:
    assert not (_repo_root() / "configs" / "targets" / "affect_mapping_v1.json").exists()


def test_archived_compatibility_bundle_still_validates() -> None:
    validation = validate_config_bundle(
        protocol_path=_resolve_config_id("protocol.thesis_canonical_nested_v1"),
        comparison_path=_resolve_config_id("comparison.model_family_grouped_nested_comparison_v1"),
        target_path=DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH,
    )
    assert validation["valid"] is True
    assert validation["matched_bundle_id"] == "bundle.thesis_canonical_nested_v1_grouped_nested"


def test_operational_docs_and_script_do_not_use_removed_raw_compat_paths() -> None:
    old_literals = (
        "configs/comparisons/model_family_grouped_nested_comparison_v1.json",
        "configs/protocols/thesis_canonical_nested_v1.json",
        "configs/targets/affect_mapping_v1.json",
    )
    archive_comparison_path = "configs/archive/comparisons/model_family_grouped_nested_comparison_v1.json"
    files = (
        _repo_root() / "docs" / "OPERATOR_GUIDE.md",
        _repo_root() / "docs" / "RUNBOOK.md",
        _repo_root() / "scripts" / "run_frozen_campaign.ps1",
    )
    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        for literal in old_literals:
            assert literal not in content, f"Unexpected stale path '{literal}' in {file_path}"
    assert archive_comparison_path in (_repo_root() / "docs" / "OPERATOR_GUIDE.md").read_text(
        encoding="utf-8"
    )
    assert archive_comparison_path in (_repo_root() / "docs" / "RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    assert archive_comparison_path in (_repo_root() / "scripts" / "run_frozen_campaign.ps1").read_text(
        encoding="utf-8"
    )
