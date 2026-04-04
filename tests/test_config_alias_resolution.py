from __future__ import annotations

from pathlib import Path

import pytest

from Thesis_ML.config.config_registry import resolve_config_alias
from Thesis_ML.config.paths import (
    DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH,
    DEFAULT_COMPARISON_SPEC_PATH,
    DEFAULT_CONFIG_REGISTRY_PATH,
    DEFAULT_DECISION_SUPPORT_PACKAGE_REGISTRY,
    DEFAULT_DECISION_SUPPORT_REGISTRY,
    DEFAULT_DECISION_SUPPORT_THESIS_RUNTIME_REGISTRY,
    DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH,
    DEFAULT_THESIS_NESTED_PROTOCOL_PATH,
    DEFAULT_THESIS_PROTOCOL_PATH,
    PROJECT_ROOT,
    SOURCE_REPO_ROOT,
)
from Thesis_ML.data.affect_labels import COARSE_AFFECT_MAPPING_VERSION


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_alias(alias: str) -> Path:
    return resolve_config_alias(
        alias,
        source_repo_root=SOURCE_REPO_ROOT,
        project_root=PROJECT_ROOT,
    )


def test_default_config_registry_path_points_to_repo_registry() -> None:
    expected = (_repo_root() / "configs" / "config_registry.json").resolve()
    assert DEFAULT_CONFIG_REGISTRY_PATH is not None
    assert Path(DEFAULT_CONFIG_REGISTRY_PATH).resolve() == expected
    assert expected.exists()


def test_aliases_resolve_to_expected_ground_truth_paths() -> None:
    repo_root = _repo_root()
    assert (
        _resolve_alias("target.coarse_affect_default")
        == (repo_root / "configs" / "targets" / "affect_mapping_v2.json").resolve()
    )
    assert (
        _resolve_alias("protocol.thesis_canonical_default")
        == (repo_root / "configs" / "protocols" / "thesis_canonical_nested_v2.json").resolve()
    )
    assert (
        _resolve_alias("protocol.thesis_confirmatory_frozen")
        == (repo_root / "configs" / "protocols" / "thesis_confirmatory_v1.json").resolve()
    )
    assert (
        _resolve_alias("comparison.grouped_nested_default")
        == (
            repo_root / "configs" / "comparisons" / "model_family_grouped_nested_comparison_v2.json"
        ).resolve()
    )
    assert (
        _resolve_alias("registry.decision_support_thesis_runtime")
        == (repo_root / "configs" / "decision_support_registry_revised_execution.json").resolve()
    )
    assert (
        _resolve_alias("registry.decision_support_package_default")
        == (repo_root / "configs" / "decision_support_registry.json").resolve()
    )
    assert (
        _resolve_alias("registry.decision_support_default")
        == (repo_root / "configs" / "decision_support_registry_revised_execution.json").resolve()
    )
    assert (
        _resolve_alias("registry.decision_support_registry")
        == (repo_root / "configs" / "decision_support_registry.json").resolve()
    )


def test_exported_defaults_match_alias_resolution() -> None:
    assert DEFAULT_THESIS_PROTOCOL_PATH.resolve() == _resolve_alias(
        "protocol.thesis_canonical_default"
    )
    assert DEFAULT_THESIS_NESTED_PROTOCOL_PATH.resolve() == _resolve_alias(
        "protocol.thesis_canonical_default"
    )
    assert DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH.resolve() == _resolve_alias(
        "protocol.thesis_confirmatory_frozen"
    )
    assert DEFAULT_COMPARISON_SPEC_PATH.resolve() == _resolve_alias(
        "comparison.grouped_nested_default"
    )
    assert DEFAULT_DECISION_SUPPORT_THESIS_RUNTIME_REGISTRY.resolve() == _resolve_alias(
        "registry.decision_support_thesis_runtime"
    )
    assert DEFAULT_DECISION_SUPPORT_PACKAGE_REGISTRY.resolve() == _resolve_alias(
        "registry.decision_support_package_default"
    )
    assert DEFAULT_DECISION_SUPPORT_REGISTRY.resolve() == _resolve_alias(
        "registry.decision_support_default"
    )
    assert DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH.resolve() == _resolve_alias(
        "target.coarse_affect_default"
    )


def test_coarse_affect_mapping_version_tracks_default_target_stem() -> None:
    assert COARSE_AFFECT_MAPPING_VERSION == DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH.stem
    assert COARSE_AFFECT_MAPPING_VERSION == "affect_mapping_v2"


def test_resolve_config_alias_uses_fallback_when_registry_is_missing(tmp_path: Path) -> None:
    fallback = tmp_path / "fallback.json"
    fallback.write_text("{}\n", encoding="utf-8")
    resolved = resolve_config_alias(
        "protocol.thesis_canonical_default",
        source_repo_root=tmp_path,
        project_root=tmp_path,
        fallback=fallback,
    )
    assert resolved == fallback.resolve()


def test_resolve_config_alias_raises_for_unknown_alias_when_registry_present() -> None:
    with pytest.raises(KeyError):
        _resolve_alias("unknown.alias")
