from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tests._config_refs import canonical_v1_protocol_variant_path


def _load_replay_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "replay_official_paths.py"
    spec = importlib.util.spec_from_file_location("replay_official_paths", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_official_paths_writes_expected_outputs(tmp_path: Path, monkeypatch) -> None:
    module = _load_replay_module()
    comparison_output = tmp_path / "comparison_output"
    comparison_output.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        module,
        "_run_comparison",
        lambda **_: {
            "comparison_id": "cmp",
            "comparison_version": "1.0.0",
            "comparison_output_dir": str(comparison_output),
            "n_completed": 1,
            "n_failed": 0,
            "n_planned": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "verify_official_artifacts",
        lambda **_: {"passed": True, "issues": []},
    )
    monkeypatch.setattr(
        module,
        "build_reproducibility_manifest",
        lambda **_: {"manifest_schema_version": "reproducibility-manifest-v1"},
    )
    monkeypatch.setattr(
        module,
        "write_reproducibility_manifest",
        lambda *, manifest, output_path: output_path.write_text(
            f"{json.dumps(manifest, indent=2)}\n",
            encoding="utf-8",
        ),
    )

    summary_out = tmp_path / "replay_summary.json"
    verification_out = tmp_path / "replay_verification_summary.json"
    manifest_out = tmp_path / "reproducibility_manifest.json"
    exit_code = module.main(
        [
            "--mode",
            "comparison",
            "--use-demo-dataset",
            "--reports-root",
            str(tmp_path / "reports"),
            "--summary-out",
            str(summary_out),
            "--verification-summary-out",
            str(verification_out),
            "--manifest-out",
            str(manifest_out),
        ]
    )
    assert exit_code == 0
    assert summary_out.exists()
    assert verification_out.exists()
    assert manifest_out.exists()


def test_replay_official_paths_fails_on_verification_error(tmp_path: Path, monkeypatch) -> None:
    module = _load_replay_module()
    comparison_output = tmp_path / "comparison_output"
    comparison_output.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        module,
        "_run_comparison",
        lambda **_: {
            "comparison_id": "cmp",
            "comparison_version": "1.0.0",
            "comparison_output_dir": str(comparison_output),
            "n_completed": 1,
            "n_failed": 0,
            "n_planned": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "verify_official_artifacts",
        lambda **_: {"passed": False, "issues": [{"code": "broken"}]},
    )
    monkeypatch.setattr(
        module,
        "build_reproducibility_manifest",
        lambda **_: {"manifest_schema_version": "reproducibility-manifest-v1"},
    )
    monkeypatch.setattr(
        module,
        "write_reproducibility_manifest",
        lambda *, manifest, output_path: output_path.write_text(
            f"{json.dumps(manifest, indent=2)}\n",
            encoding="utf-8",
        ),
    )

    exit_code = module.main(
        [
            "--mode",
            "comparison",
            "--use-demo-dataset",
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_code == 1


def test_replay_official_paths_fails_when_mode_has_timed_out_runs(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_replay_module()
    comparison_output = tmp_path / "comparison_output"
    comparison_output.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        module,
        "_run_comparison",
        lambda **_: {
            "comparison_id": "cmp",
            "comparison_version": "1.0.0",
            "comparison_output_dir": str(comparison_output),
            "n_success": 0,
            "n_completed": 0,
            "n_failed": 0,
            "n_timed_out": 1,
            "n_skipped_due_to_policy": 0,
            "n_planned": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "verify_official_artifacts",
        lambda **_: {"passed": True, "issues": []},
    )
    monkeypatch.setattr(
        module,
        "build_reproducibility_manifest",
        lambda **_: {"manifest_schema_version": "reproducibility-manifest-v1"},
    )
    monkeypatch.setattr(
        module,
        "write_reproducibility_manifest",
        lambda *, manifest, output_path: output_path.write_text(
            f"{json.dumps(manifest, indent=2)}\n",
            encoding="utf-8",
        ),
    )

    exit_code = module.main(
        [
            "--mode",
            "comparison",
            "--use-demo-dataset",
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_code == 1


def test_replay_official_paths_fails_on_split_manifest_hash_verification_issue(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_replay_module()
    comparison_output = tmp_path / "comparison_output"
    comparison_output.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        module,
        "_run_comparison",
        lambda **_: {
            "comparison_id": "cmp",
            "comparison_version": "1.0.0",
            "comparison_output_dir": str(comparison_output),
            "n_completed": 1,
            "n_failed": 0,
            "n_planned": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "verify_official_artifacts",
        lambda **_: {
            "passed": False,
            "issues": [{"code": "cv_split_manifest_hash_mismatch_recorded"}],
        },
    )
    monkeypatch.setattr(
        module,
        "build_reproducibility_manifest",
        lambda **_: {"manifest_schema_version": "reproducibility-manifest-v1"},
    )
    monkeypatch.setattr(
        module,
        "write_reproducibility_manifest",
        lambda *, manifest, output_path: output_path.write_text(
            f"{json.dumps(manifest, indent=2)}\n",
            encoding="utf-8",
        ),
    )

    exit_code = module.main(
        [
            "--mode",
            "comparison",
            "--use-demo-dataset",
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_code == 1


def test_replay_official_paths_resolves_comparison_and_protocol_aliases(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_replay_module()
    comparison_output = tmp_path / "comparison_output"
    confirmatory_output = tmp_path / "confirmatory_output"
    comparison_output.mkdir(parents=True, exist_ok=True)
    confirmatory_output.mkdir(parents=True, exist_ok=True)
    captured: dict[str, Path] = {}

    def _stub_run_comparison(**kwargs):
        captured["comparison_path"] = Path(kwargs["comparison_path"]).resolve()
        return {
            "comparison_id": "cmp",
            "comparison_version": "1.0.0",
            "comparison_output_dir": str(comparison_output),
            "n_success": 1,
            "n_completed": 1,
            "n_failed": 0,
            "n_timed_out": 0,
            "n_skipped_due_to_policy": 0,
            "n_planned": 0,
        }

    def _stub_run_confirmatory(**kwargs):
        captured["protocol_path"] = Path(kwargs["protocol_path"]).resolve()
        return {
            "protocol_id": "protocol",
            "protocol_version": "v1.1",
            "protocol_output_dir": str(confirmatory_output),
            "n_success": 1,
            "n_completed": 1,
            "n_failed": 0,
            "n_timed_out": 0,
            "n_skipped_due_to_policy": 0,
            "n_planned": 0,
        }

    monkeypatch.setattr(module, "_run_comparison", _stub_run_comparison)
    monkeypatch.setattr(module, "_run_confirmatory", _stub_run_confirmatory)
    monkeypatch.setattr(
        module,
        "verify_official_artifacts",
        lambda **_: {"passed": True, "issues": []},
    )
    monkeypatch.setattr(
        module,
        "verify_confirmatory_ready",
        lambda **_: {"passed": True, "issues": []},
    )
    monkeypatch.setattr(
        module,
        "build_reproducibility_manifest",
        lambda **_: {"manifest_schema_version": "reproducibility-manifest-v1"},
    )
    monkeypatch.setattr(
        module,
        "write_reproducibility_manifest",
        lambda *, manifest, output_path: output_path.write_text(
            f"{json.dumps(manifest, indent=2)}\n",
            encoding="utf-8",
        ),
    )

    exit_code = module.main(
        [
            "--mode",
            "both",
            "--comparison-alias",
            "comparison.grouped_nested_default",
            "--protocol-alias",
            "protocol.thesis_confirmatory_frozen",
            "--all-variants",
            "--all-suites",
            "--use-demo-dataset",
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_code == 0
    assert captured["comparison_path"] == Path(module.DEFAULT_COMPARISON_SPEC_PATH).resolve()
    assert captured["protocol_path"] == Path(module.DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH).resolve()


def test_replay_official_paths_both_mode_writes_valid_config_bundle_validation(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_replay_module()
    comparison_output = tmp_path / "comparison_output"
    confirmatory_output = tmp_path / "confirmatory_output"
    comparison_output.mkdir(parents=True, exist_ok=True)
    confirmatory_output.mkdir(parents=True, exist_ok=True)
    summary_out = tmp_path / "replay_summary.json"

    monkeypatch.setattr(
        module,
        "_run_comparison",
        lambda **_: {
            "comparison_id": "cmp",
            "comparison_version": "2.0.0",
            "comparison_output_dir": str(comparison_output),
            "n_success": 1,
            "n_completed": 1,
            "n_failed": 0,
            "n_timed_out": 0,
            "n_skipped_due_to_policy": 0,
            "n_planned": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "_run_confirmatory",
        lambda **_: {
            "protocol_id": "protocol",
            "protocol_version": "v1.1",
            "protocol_output_dir": str(confirmatory_output),
            "n_success": 1,
            "n_completed": 1,
            "n_failed": 0,
            "n_timed_out": 0,
            "n_skipped_due_to_policy": 0,
            "n_planned": 0,
        },
    )
    monkeypatch.setattr(module, "verify_official_artifacts", lambda **_: {"passed": True, "issues": []})
    monkeypatch.setattr(module, "verify_confirmatory_ready", lambda **_: {"passed": True, "issues": []})
    monkeypatch.setattr(
        module,
        "build_reproducibility_manifest",
        lambda **_: {"manifest_schema_version": "reproducibility-manifest-v1"},
    )
    monkeypatch.setattr(
        module,
        "write_reproducibility_manifest",
        lambda *, manifest, output_path: output_path.write_text(
            f"{json.dumps(manifest, indent=2)}\n",
            encoding="utf-8",
        ),
    )

    exit_code = module.main(
        [
            "--mode",
            "both",
            "--comparison-alias",
            "comparison.grouped_nested_default",
            "--protocol-alias",
            "protocol.thesis_confirmatory_frozen",
            "--all-variants",
            "--all-suites",
            "--use-demo-dataset",
            "--reports-root",
            str(tmp_path / "reports"),
            "--summary-out",
            str(summary_out),
        ]
    )
    assert exit_code == 0
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    bundle_validation = summary["config_bundle_validation"]
    assert bundle_validation["valid"] is True
    assert bundle_validation["matched_bundle_id"] == "bundle.thesis_confirmatory_v1_publishable"


def test_replay_official_paths_both_mode_rejects_invalid_bundle(monkeypatch) -> None:
    module = _load_replay_module()
    protocol_path = canonical_v1_protocol_variant_path()
    monkeypatch.setattr(module, "_run_comparison", lambda **_: None)
    monkeypatch.setattr(module, "_run_confirmatory", lambda **_: None)

    with pytest.raises(ValueError):
        module.main(
            [
                "--mode",
                "both",
                "--protocol",
                str(protocol_path),
            ]
        )


def test_replay_official_paths_verify_determinism_uses_default_confirmatory_config(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_replay_module()
    confirmatory_output = tmp_path / "confirmatory_output"
    confirmatory_output.mkdir(parents=True, exist_ok=True)
    captured: dict[str, Path] = {}

    monkeypatch.setattr(
        module,
        "_run_confirmatory",
        lambda **_: {
            "protocol_id": "protocol",
            "protocol_version": "v1.1",
            "protocol_output_dir": str(confirmatory_output),
            "n_success": 1,
            "n_completed": 1,
            "n_failed": 0,
            "n_timed_out": 0,
            "n_skipped_due_to_policy": 0,
            "n_planned": 0,
        },
    )
    monkeypatch.setattr(module, "verify_official_artifacts", lambda **_: {"passed": True, "issues": []})
    monkeypatch.setattr(module, "verify_confirmatory_ready", lambda **_: {"passed": True, "issues": []})

    def _stub_determinism_for_mode(**kwargs):
        captured["protocol_path"] = Path(kwargs["protocol_path"]).resolve()
        return {"passed": True, "mode": "confirmatory", "comparison": {"passed": True}}

    monkeypatch.setattr(module, "_determinism_for_mode", _stub_determinism_for_mode)
    monkeypatch.setattr(
        module,
        "build_reproducibility_manifest",
        lambda **_: {"manifest_schema_version": "reproducibility-manifest-v1"},
    )
    monkeypatch.setattr(
        module,
        "write_reproducibility_manifest",
        lambda *, manifest, output_path: output_path.write_text(
            f"{json.dumps(manifest, indent=2)}\n",
            encoding="utf-8",
        ),
    )

    exit_code = module.main(
        [
            "--mode",
            "confirmatory",
            "--use-demo-dataset",
            "--verify-determinism",
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_code == 0
    assert captured["protocol_path"] == Path(module.DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH).resolve()


def test_replay_official_paths_verify_determinism_uses_default_comparison_config(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_replay_module()
    comparison_output = tmp_path / "comparison_output"
    comparison_output.mkdir(parents=True, exist_ok=True)
    captured: dict[str, Path] = {}

    monkeypatch.setattr(
        module,
        "_run_comparison",
        lambda **_: {
            "comparison_id": "cmp",
            "comparison_version": "2.0.0",
            "comparison_output_dir": str(comparison_output),
            "n_success": 1,
            "n_completed": 1,
            "n_failed": 0,
            "n_timed_out": 0,
            "n_skipped_due_to_policy": 0,
            "n_planned": 0,
        },
    )
    monkeypatch.setattr(module, "verify_official_artifacts", lambda **_: {"passed": True, "issues": []})

    def _stub_determinism_for_mode(**kwargs):
        captured["comparison_path"] = Path(kwargs["comparison_path"]).resolve()
        return {"passed": True, "mode": "comparison", "comparison": {"passed": True}}

    monkeypatch.setattr(module, "_determinism_for_mode", _stub_determinism_for_mode)
    monkeypatch.setattr(
        module,
        "build_reproducibility_manifest",
        lambda **_: {"manifest_schema_version": "reproducibility-manifest-v1"},
    )
    monkeypatch.setattr(
        module,
        "write_reproducibility_manifest",
        lambda *, manifest, output_path: output_path.write_text(
            f"{json.dumps(manifest, indent=2)}\n",
            encoding="utf-8",
        ),
    )

    exit_code = module.main(
        [
            "--mode",
            "comparison",
            "--use-demo-dataset",
            "--verify-determinism",
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_code == 0
    assert captured["comparison_path"] == Path(module.DEFAULT_COMPARISON_SPEC_PATH).resolve()
