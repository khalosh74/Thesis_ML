from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Thesis_ML.config.paths import (
    DEFAULT_COMPARISON_SPEC_PATH,
    DEFAULT_DECISION_SUPPORT_REGISTRY,
    DEFAULT_TARGET_CONFIGS_DIR,
    DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH,
    DEFAULT_THESIS_PROTOCOL_PATH,
)

_EXPECTED_LIFECYCLE_STATUSES = [
    "active_default",
    "active_variant",
    "frozen_confirmatory",
    "compatibility",
    "archived_deprecated",
]

_EXPECTED_ALIASES = {
    "target.coarse_affect_default": "target.affect_mapping_v2",
    "protocol.thesis_canonical_default": "protocol.thesis_canonical_nested_v2",
    "protocol.thesis_confirmatory_frozen": "protocol.thesis_confirmatory_v1",
    "comparison.grouped_nested_default": "comparison.model_family_grouped_nested_comparison_v2",
    "comparison.fixed_linear_baseline": "comparison.model_family_comparison_v1",
    "registry.decision_support_default": "registry.decision_support_registry",
}

_EXPECTED_LIFECYCLE_BY_ID = {
    "target.affect_mapping_v2": "active_default",
    "protocol.thesis_canonical_v1": "active_variant",
    "protocol.thesis_canonical_nested_v1": "compatibility",
    "protocol.thesis_canonical_nested_v2": "active_default",
    "protocol.thesis_confirmatory_v1": "frozen_confirmatory",
    "comparison.model_family_comparison_v1": "active_variant",
    "comparison.model_family_grouped_nested_comparison_v1": "compatibility",
    "comparison.model_family_grouped_nested_comparison_v2": "active_default",
    "registry.decision_support_registry": "active_default",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _registry_path() -> Path:
    return _repo_root() / "configs" / "config_registry.json"


def _load_registry() -> dict[str, Any]:
    return json.loads(_registry_path().read_text(encoding="utf-8"))


def _configs_by_id(registry_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    configs = registry_payload["configs"]
    assert isinstance(configs, list)
    return {str(entry["config_id"]): entry for entry in configs}


def test_registry_parses_and_schema_version_is_v1() -> None:
    payload = _load_registry()
    assert isinstance(payload, dict)
    assert payload.get("schema_version") == "config-registry-v1"


def test_registry_lifecycle_statuses_are_exact() -> None:
    payload = _load_registry()
    assert payload.get("lifecycle_statuses") == _EXPECTED_LIFECYCLE_STATUSES


def test_registry_lists_every_current_config_json_file() -> None:
    repo_root = _repo_root()
    payload = _load_registry()
    registry_paths = {str(entry["path"]) for entry in payload["configs"]}

    expected_paths = {
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "configs" / "targets").glob("*.json")
    }
    expected_paths.update(
        {
            path.relative_to(repo_root).as_posix()
            for path in (repo_root / "configs" / "protocols").glob("*.json")
        }
    )
    expected_paths.update(
        {
            path.relative_to(repo_root).as_posix()
            for path in (repo_root / "configs" / "comparisons").glob("*.json")
        }
    )
    expected_paths.update(
        {
            path.relative_to(repo_root).as_posix()
            for path in (repo_root / "configs" / "archive" / "protocols").glob("*.json")
        }
    )
    expected_paths.update(
        {
            path.relative_to(repo_root).as_posix()
            for path in (repo_root / "configs" / "archive" / "comparisons").glob("*.json")
        }
    )
    expected_paths.add("configs/decision_support_registry.json")

    assert registry_paths == expected_paths


def test_every_registry_path_exists() -> None:
    repo_root = _repo_root()
    payload = _load_registry()
    for entry in payload["configs"]:
        path = repo_root / str(entry["path"])
        assert path.exists(), f"Missing config path from registry: {path}"


def test_config_ids_are_unique() -> None:
    payload = _load_registry()
    config_ids = [str(entry["config_id"]) for entry in payload["configs"]]
    assert len(config_ids) == len(set(config_ids))


def test_aliases_resolve_to_known_config_ids() -> None:
    payload = _load_registry()
    aliases = payload.get("aliases", {})
    config_ids = set(_configs_by_id(payload).keys())
    assert aliases == _EXPECTED_ALIASES
    assert set(aliases.values()).issubset(config_ids)


def test_alias_paths_match_current_runtime_defaults() -> None:
    repo_root = _repo_root()
    payload = _load_registry()
    aliases = payload["aliases"]
    by_id = _configs_by_id(payload)

    def _alias_path(alias_key: str) -> Path:
        config_id = aliases[alias_key]
        return (repo_root / str(by_id[config_id]["path"])).resolve()

    assert _alias_path("protocol.thesis_canonical_default") == Path(
        DEFAULT_THESIS_PROTOCOL_PATH
    ).resolve()
    assert _alias_path("protocol.thesis_confirmatory_frozen") == Path(
        DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH
    ).resolve()
    assert _alias_path("comparison.grouped_nested_default") == Path(
        DEFAULT_COMPARISON_SPEC_PATH
    ).resolve()
    assert _alias_path("registry.decision_support_default") == Path(
        DEFAULT_DECISION_SUPPORT_REGISTRY
    ).resolve()
    assert _alias_path("target.coarse_affect_default") == (
        Path(DEFAULT_TARGET_CONFIGS_DIR) / "affect_mapping_v2.json"
    ).resolve()


def test_exact_lifecycle_classifications() -> None:
    payload = _load_registry()
    by_id = _configs_by_id(payload)
    for config_id, expected_lifecycle in _EXPECTED_LIFECYCLE_BY_ID.items():
        assert by_id[config_id]["lifecycle"] == expected_lifecycle


def test_version_alignment_with_underlying_files() -> None:
    repo_root = _repo_root()
    payload = _load_registry()

    for entry in payload["configs"]:
        kind = str(entry["kind"])
        version = str(entry["version"])
        path = repo_root / str(entry["path"])
        path_payload = json.loads(path.read_text(encoding="utf-8"))

        if kind == "protocol":
            assert version == str(path_payload["protocol_version"])
        elif kind == "comparison":
            assert version == str(path_payload["comparison_version"])
        elif kind == "registry":
            assert version == str(path_payload["schema_version"])
        elif kind == "target":
            assert version == path.stem
        else:
            raise AssertionError(f"Unsupported kind in registry: {kind}")


def test_retired_section_contains_affect_mapping_v1_replacement() -> None:
    payload = _load_registry()
    retired = payload.get("retired")
    assert isinstance(retired, list)
    retired_by_id = {
        str(entry["config_id"]): entry
        for entry in retired
        if isinstance(entry, dict) and "config_id" in entry
    }
    assert "target.affect_mapping_v1" in retired_by_id
    retired_entry = retired_by_id["target.affect_mapping_v1"]
    assert retired_entry["replacement"] == "target.affect_mapping_v2"
