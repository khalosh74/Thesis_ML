from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.config.paths import (
    DEFAULT_COMPARISON_SPEC_PATH,
    DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH,
)
from Thesis_ML.verification.repro_manifest import (
    build_reproducibility_manifest,
    write_reproducibility_manifest,
)
from tests._config_refs import (
    frozen_confirmatory_protocol_path,
    grouped_nested_default_comparison_path,
)


def test_repro_manifest_includes_required_top_level_sections(tmp_path: Path) -> None:
    index_csv = tmp_path / "dataset_index.csv"
    index_csv.write_text("sample_id,subject,session,coarse_affect\ns1,sub-001,ses-01,positive\n")
    data_root = tmp_path / "Data"
    cache_dir = tmp_path / "cache"
    data_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    confirmatory_output = tmp_path / "confirmatory_output"
    confirmatory_output.mkdir(parents=True, exist_ok=True)
    (confirmatory_output / "report_index.csv").write_text(
        "run_id,status,report_dir\n", encoding="utf-8"
    )

    replay_summary = {"passed": True, "mode": "confirmatory"}
    replay_verification_summary = {
        "passed": True,
        "determinism": {"passed": True},
    }

    manifest = build_reproducibility_manifest(
        output_dirs_by_mode={"confirmatory": confirmatory_output},
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        replay_summary=replay_summary,
        replay_verification_summary=replay_verification_summary,
    )

    assert manifest["manifest_schema_version"] == "reproducibility-manifest-v1"
    assert isinstance(manifest["code_identity"], dict)
    assert isinstance(manifest["environment_identity"], dict)
    assert isinstance(manifest["dataset_identity"], dict)
    assert isinstance(manifest["spec_identity"], dict)
    assert isinstance(manifest["official_outputs"], dict)
    assert isinstance(manifest["replay_status"], dict)
    assert manifest["replay_status"]["replay_passed"] is True
    assert manifest["replay_status"]["determinism_passed"] is True

    output_path = tmp_path / "reproducibility_manifest.json"
    written_path = write_reproducibility_manifest(
        manifest=manifest,
        output_path=output_path,
    )
    assert written_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["manifest_schema_version"] == "reproducibility-manifest-v1"


def test_repro_manifest_includes_spec_registry_identity_for_selected_specs(tmp_path: Path) -> None:
    index_csv = tmp_path / "dataset_index.csv"
    index_csv.write_text("sample_id,subject,session,coarse_affect\ns1,sub-001,ses-01,positive\n")
    data_root = tmp_path / "Data"
    cache_dir = tmp_path / "cache"
    data_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    comparison_output = tmp_path / "comparison_output"
    comparison_output.mkdir(parents=True, exist_ok=True)
    (comparison_output / "report_index.csv").write_text("run_id,status,report_dir\n", encoding="utf-8")

    manifest = build_reproducibility_manifest(
        output_dirs_by_mode={"comparison": comparison_output},
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        comparison_spec_path=DEFAULT_COMPARISON_SPEC_PATH,
        protocol_path=DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH,
    )

    comparison_identity = manifest["spec_identity"]["comparison"]["registry_identity"]
    protocol_identity = manifest["spec_identity"]["confirmatory_protocol"]["registry_identity"]
    assert comparison_identity["registered"] is True
    assert comparison_identity["config_id"] == "comparison.model_family_grouped_nested_comparison_v2"
    assert comparison_identity["lifecycle"] == "active_default"
    assert protocol_identity["registered"] is True
    assert protocol_identity["config_id"] == "protocol.thesis_confirmatory_v1"
    assert protocol_identity["lifecycle"] == "frozen_confirmatory"


def test_repro_manifest_includes_valid_spec_bundle_validation_for_publishable_bundle(
    tmp_path: Path,
) -> None:
    index_csv = tmp_path / "dataset_index.csv"
    index_csv.write_text("sample_id,subject,session,coarse_affect\ns1,sub-001,ses-01,positive\n")
    data_root = tmp_path / "Data"
    cache_dir = tmp_path / "cache"
    data_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    confirmatory_output = tmp_path / "confirmatory_output"
    confirmatory_output.mkdir(parents=True, exist_ok=True)
    (confirmatory_output / "report_index.csv").write_text("run_id,status,report_dir\n", encoding="utf-8")

    manifest = build_reproducibility_manifest(
        output_dirs_by_mode={"confirmatory": confirmatory_output},
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        comparison_spec_path=grouped_nested_default_comparison_path(),
        protocol_path=frozen_confirmatory_protocol_path(),
    )

    bundle_validation = manifest["spec_bundle_validation"]
    assert isinstance(bundle_validation, dict)
    assert bundle_validation["valid"] is True
    assert bundle_validation["matched_bundle_id"] == "bundle.thesis_confirmatory_v1_publishable"
