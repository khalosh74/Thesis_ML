from __future__ import annotations

from pathlib import Path

from Thesis_ML.config import aliases_for_config_id, describe_config_path
from Thesis_ML.config.paths import (
    DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH,
    DEFAULT_COMPARISON_SPEC_PATH,
    DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH,
    DEFAULT_THESIS_PROTOCOL_PATH,
)


def test_describe_canonical_default_protocol_identity() -> None:
    identity = describe_config_path(DEFAULT_THESIS_PROTOCOL_PATH)
    assert identity["registered"] is True
    assert identity["config_id"] == "protocol.thesis_canonical_nested_v2"
    assert identity["lifecycle"] == "legacy_non_official"


def test_describe_frozen_confirmatory_protocol_identity() -> None:
    identity = describe_config_path(DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH)
    assert identity["registered"] is True
    assert identity["config_id"] == "protocol.thesis_confirmatory_v1"
    assert identity["lifecycle"] == "legacy_non_official"


def test_describe_default_comparison_identity() -> None:
    identity = describe_config_path(DEFAULT_COMPARISON_SPEC_PATH)
    assert identity["registered"] is True
    assert identity["config_id"] == "comparison.model_family_grouped_nested_comparison_v2"
    assert identity["lifecycle"] == "legacy_non_official"


def test_describe_default_coarse_affect_target_identity() -> None:
    identity = describe_config_path(DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH)
    assert identity["registered"] is True
    assert identity["config_id"] == "target.affect_mapping_v2"
    assert identity["lifecycle"] == "active_default"


def test_describe_external_path_is_unregistered(tmp_path: Path) -> None:
    external_path = tmp_path / "external_config.json"
    external_path.write_text("{}\n", encoding="utf-8")
    identity = describe_config_path(external_path)
    assert identity["registered"] is False
    assert identity["config_id"] is None


def test_aliases_for_frozen_confirmatory_protocol_include_expected_alias() -> None:
    aliases = aliases_for_config_id("protocol.thesis_confirmatory_v1")
    assert "protocol.thesis_confirmatory_frozen" in aliases
