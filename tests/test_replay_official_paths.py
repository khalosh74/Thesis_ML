from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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
