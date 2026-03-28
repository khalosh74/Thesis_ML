from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tests._config_refs import canonical_v1_protocol_variant_path


def _load_script_module(script_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_confirmatory_output_fixture(output_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    test_path = repo_root / "tests" / "test_official_verification.py"
    spec = importlib.util.spec_from_file_location("test_official_verification_fixture", test_path)
    assert spec is not None and spec.loader is not None
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    test_module._write_confirmatory_output(output_dir)


def test_build_and_verify_publishable_bundle_roundtrip(tmp_path: Path) -> None:
    build_module = _load_script_module("build_publishable_bundle.py")
    verify_module = _load_script_module("verify_publishable_bundle.py")

    confirmatory_output = tmp_path / "protocol_runs" / "thesis-canonical__1.0.0"
    _write_confirmatory_output_fixture(confirmatory_output)

    replay_summary = tmp_path / "replay_summary.json"
    replay_summary.write_text(f"{json.dumps({'passed': True}, indent=2)}\n", encoding="utf-8")
    replay_verification = tmp_path / "replay_verification_summary.json"
    replay_verification.write_text(
        f"{json.dumps({'passed': True, 'determinism': {'passed': True}}, indent=2)}\n",
        encoding="utf-8",
    )
    repro_manifest = tmp_path / "reproducibility_manifest.json"
    repro_manifest.write_text(
        f"{json.dumps({'manifest_schema_version': 'reproducibility-manifest-v1'}, indent=2)}\n",
        encoding="utf-8",
    )
    confirmatory_ready = tmp_path / "confirmatory_ready_summary.json"
    confirmatory_ready.write_text(f"{json.dumps({'passed': True}, indent=2)}\n", encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    build_exit = build_module.main(
        [
            "--output-dir",
            str(bundle_dir),
            "--confirmatory-output",
            str(confirmatory_output),
            "--replay-summary",
            str(replay_summary),
            "--replay-verification-summary",
            str(replay_verification),
            "--repro-manifest",
            str(repro_manifest),
            "--confirmatory-ready-summary",
            str(confirmatory_ready),
        ]
    )
    assert build_exit == 0
    assert (bundle_dir / "bundle_manifest.json").exists()

    verify_exit = verify_module.main(["--bundle-dir", str(bundle_dir)])
    assert verify_exit == 0


def test_verify_publishable_bundle_fails_on_hash_mismatch(tmp_path: Path) -> None:
    build_module = _load_script_module("build_publishable_bundle.py")
    verify_module = _load_script_module("verify_publishable_bundle.py")

    confirmatory_output = tmp_path / "protocol_runs" / "thesis-canonical__1.0.0"
    _write_confirmatory_output_fixture(confirmatory_output)

    replay_summary = tmp_path / "replay_summary.json"
    replay_summary.write_text(f"{json.dumps({'passed': True}, indent=2)}\n", encoding="utf-8")
    replay_verification = tmp_path / "replay_verification_summary.json"
    replay_verification.write_text(
        f"{json.dumps({'passed': True, 'determinism': {'passed': True}}, indent=2)}\n",
        encoding="utf-8",
    )
    repro_manifest = tmp_path / "reproducibility_manifest.json"
    repro_manifest.write_text(
        f"{json.dumps({'manifest_schema_version': 'reproducibility-manifest-v1'}, indent=2)}\n",
        encoding="utf-8",
    )
    confirmatory_ready = tmp_path / "confirmatory_ready_summary.json"
    confirmatory_ready.write_text(f"{json.dumps({'passed': True}, indent=2)}\n", encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    build_module.main(
        [
            "--output-dir",
            str(bundle_dir),
            "--confirmatory-output",
            str(confirmatory_output),
            "--replay-summary",
            str(replay_summary),
            "--replay-verification-summary",
            str(replay_verification),
            "--repro-manifest",
            str(repro_manifest),
            "--confirmatory-ready-summary",
            str(confirmatory_ready),
        ]
    )

    tamper_target = bundle_dir / "verification" / "replay_summary.json"
    tamper_target.write_text(
        f"{json.dumps({'passed': False, 'tampered': True}, indent=2)}\n",
        encoding="utf-8",
    )

    verify_exit = verify_module.main(["--bundle-dir", str(bundle_dir)])
    assert verify_exit == 1


def test_build_publishable_bundle_supports_alias_spec_flags(tmp_path: Path) -> None:
    build_module = _load_script_module("build_publishable_bundle.py")

    confirmatory_output = tmp_path / "protocol_runs" / "thesis-canonical__1.0.0"
    _write_confirmatory_output_fixture(confirmatory_output)

    bundle_dir = tmp_path / "bundle_alias_flags"
    build_exit = build_module.main(
        [
            "--output-dir",
            str(bundle_dir),
            "--confirmatory-output",
            str(confirmatory_output),
            "--comparison-spec-alias",
            "comparison.grouped_nested_default",
            "--protocol-spec-alias",
            "protocol.thesis_confirmatory_frozen",
        ]
    )
    assert build_exit == 0
    assert (bundle_dir / "specs" / "model_family_grouped_nested_comparison_v2.json").exists()
    assert (bundle_dir / "specs" / "thesis_confirmatory_v1.json").exists()

    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest["spec_registry_identity"], dict)
    assert isinstance(manifest["spec_bundle_validation"], dict)
    assert manifest["spec_bundle_validation"]["valid"] is True
    assert manifest["spec_bundle_validation"]["matched_bundle_id"] == "bundle.thesis_confirmatory_v1_publishable"
    protocol_identity = manifest["spec_registry_identity"]["confirmatory_protocol"]
    assert protocol_identity["config_id"] == "protocol.thesis_confirmatory_v1"
    assert protocol_identity["lifecycle"] == "frozen_confirmatory"

    comparison_identity = manifest["spec_registry_identity"]["comparison_spec"]
    assert comparison_identity["config_id"] == "comparison.model_family_grouped_nested_comparison_v2"
    assert comparison_identity["lifecycle"] == "active_default"


def test_build_publishable_bundle_default_specs_include_valid_bundle_validation(
    tmp_path: Path,
) -> None:
    build_module = _load_script_module("build_publishable_bundle.py")

    confirmatory_output = tmp_path / "protocol_runs" / "thesis-canonical__1.0.0"
    _write_confirmatory_output_fixture(confirmatory_output)

    bundle_dir = tmp_path / "bundle_default_specs"
    build_exit = build_module.main(
        [
            "--output-dir",
            str(bundle_dir),
            "--confirmatory-output",
            str(confirmatory_output),
        ]
    )
    assert build_exit == 0
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest["spec_bundle_validation"], dict)
    assert manifest["spec_bundle_validation"]["valid"] is True
    assert manifest["spec_bundle_validation"]["matched_bundle_id"] == "bundle.thesis_confirmatory_v1_publishable"


def test_build_publishable_bundle_rejects_invalid_protocol_comparison_bundle(tmp_path: Path) -> None:
    build_module = _load_script_module("build_publishable_bundle.py")

    confirmatory_output = tmp_path / "protocol_runs" / "thesis-canonical__1.0.0"
    _write_confirmatory_output_fixture(confirmatory_output)
    protocol_path = canonical_v1_protocol_variant_path()

    with pytest.raises(ValueError):
        build_module.main(
            [
                "--output-dir",
                str(tmp_path / "bundle_invalid"),
                "--confirmatory-output",
                str(confirmatory_output),
                "--protocol-spec",
                str(protocol_path),
            ]
        )
