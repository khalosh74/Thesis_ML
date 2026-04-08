from __future__ import annotations

from pathlib import Path

from Thesis_ML.config import list_config_bundles, resolve_config_id, validate_config_bundle

from tests._config_refs import (
    canonical_default_protocol_path,
    canonical_v1_protocol_variant_path,
    coarse_affect_default_target_mapping_path,
    frozen_confirmatory_protocol_path,
    grouped_nested_default_comparison_path,
)


def test_registry_contains_expected_phase6_bundle_ids() -> None:
    bundle_ids = sorted(
        str(entry.get("bundle_id", ""))
        for entry in list_config_bundles()
        if isinstance(entry, dict)
    )
    assert bundle_ids == [
        "bundle.thesis_canonical_nested_v1_grouped_nested",
        "bundle.thesis_canonical_nested_v2_grouped_nested",
        "bundle.thesis_canonical_v1_fixed_linear",
        "bundle.thesis_confirmatory_v1_publishable",
        "bundle.thesis_final_v1_release_authority",
    ]


def test_canonical_default_bundle_validates_to_nested_v2_bundle() -> None:
    validation = validate_config_bundle(
        protocol_path=canonical_default_protocol_path(),
        comparison_path=grouped_nested_default_comparison_path(),
        target_path=coarse_affect_default_target_mapping_path(),
    )
    assert validation["valid"] is True
    assert validation["matched_bundle_id"] == "bundle.thesis_canonical_nested_v2_grouped_nested"


def test_frozen_confirmatory_bundle_validates_to_publishable_bundle() -> None:
    validation = validate_config_bundle(
        protocol_path=frozen_confirmatory_protocol_path(),
        comparison_path=grouped_nested_default_comparison_path(),
        target_path=coarse_affect_default_target_mapping_path(),
    )
    assert validation["valid"] is True
    assert validation["matched_bundle_id"] == "bundle.thesis_confirmatory_v1_publishable"


def test_canonical_v1_with_grouped_nested_default_is_invalid_bundle() -> None:
    validation = validate_config_bundle(
        protocol_path=canonical_v1_protocol_variant_path(),
        comparison_path=grouped_nested_default_comparison_path(),
        target_path=coarse_affect_default_target_mapping_path(),
    )
    assert validation["valid"] is False
    assert validation["matched_bundle_id"] is None


def test_bundle_validation_fails_for_unregistered_component(tmp_path: Path) -> None:
    external = tmp_path / "external_component.json"
    external.write_text("{}\n", encoding="utf-8")
    validation = validate_config_bundle(
        protocol_path=external,
        comparison_path=grouped_nested_default_comparison_path(),
        target_path=coarse_affect_default_target_mapping_path(),
    )
    assert validation["checked"] is True
    assert validation["valid"] is False
    assert any("unregistered protocol component" == str(error) for error in validation["errors"])


def test_archived_compatibility_bundle_still_validates() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    validation = validate_config_bundle(
        protocol_path=resolve_config_id(
            "protocol.thesis_canonical_nested_v1",
            source_repo_root=repo_root,
            project_root=repo_root,
        ),
        comparison_path=resolve_config_id(
            "comparison.model_family_grouped_nested_comparison_v1",
            source_repo_root=repo_root,
            project_root=repo_root,
        ),
        target_path=coarse_affect_default_target_mapping_path(),
    )
    assert validation["valid"] is True
    assert validation["matched_bundle_id"] == "bundle.thesis_canonical_nested_v1_grouped_nested"
