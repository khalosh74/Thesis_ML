from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_verify_repro_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "verify_official_reproducibility.py"
    spec = importlib.util.spec_from_file_location("verify_official_reproducibility", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_rc1_gate_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "rc1_release_gate.py"
    spec = importlib.util.spec_from_file_location("rc1_release_gate", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_repro_uses_default_protocol_config_when_omitted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_verify_repro_module()
    captured: dict[str, Path] = {}

    def _stub_run_protocol_once(**kwargs):
        captured["config_path"] = Path(kwargs["protocol_path"])
        return {
            "n_failed": 0,
            "protocol_output_dir": str(tmp_path / "protocol_runs" / "thesis-canonical__1.0.0"),
        }

    monkeypatch.setattr(module, "_run_protocol_once", _stub_run_protocol_once)
    monkeypatch.setattr(
        module,
        "compare_official_outputs",
        lambda **_: {
            "passed": True,
            "left": {},
            "right": {},
            "mismatches": [],
        },
    )

    exit_code = module.main(
        [
            "--mode",
            "protocol",
            "--index-csv",
            "dummy_index.csv",
            "--data-root",
            "dummy_data_root",
            "--cache-dir",
            "dummy_cache",
            "--suite",
            "primary_controls",
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 0
    assert captured["config_path"] == Path(module.DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH)


def test_verify_repro_uses_default_comparison_config_when_omitted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_verify_repro_module()
    captured: dict[str, Path] = {}

    def _stub_run_comparison_once(**kwargs):
        captured["config_path"] = Path(kwargs["comparison_path"])
        return {
            "n_failed": 0,
            "comparison_output_dir": str(
                tmp_path / "comparison_runs" / "model-family-within-subject__1.0.0"
            ),
        }

    monkeypatch.setattr(module, "_run_comparison_once", _stub_run_comparison_once)
    monkeypatch.setattr(
        module,
        "compare_official_outputs",
        lambda **_: {
            "passed": True,
            "left": {},
            "right": {},
            "mismatches": [],
        },
    )

    exit_code = module.main(
        [
            "--mode",
            "comparison",
            "--index-csv",
            "dummy_index.csv",
            "--data-root",
            "dummy_data_root",
            "--cache-dir",
            "dummy_cache",
            "--variant",
            "ridge",
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 0
    assert captured["config_path"] == Path(module.DEFAULT_COMPARISON_SPEC_PATH)


def test_verify_repro_fails_when_runs_are_timed_out(tmp_path: Path, monkeypatch) -> None:
    module = _load_verify_repro_module()

    def _stub_run_protocol_once(**kwargs):
        return {
            "n_failed": 0,
            "n_timed_out": 1,
            "n_skipped_due_to_policy": 0,
            "protocol_output_dir": str(tmp_path / "protocol_runs" / "thesis-canonical__1.0.0"),
        }

    monkeypatch.setattr(module, "_run_protocol_once", _stub_run_protocol_once)
    monkeypatch.setattr(
        module,
        "compare_official_outputs",
        lambda **_: {
            "passed": True,
            "left": {},
            "right": {},
            "mismatches": [],
        },
    )

    exit_code = module.main(
        [
            "--mode",
            "protocol",
            "--index-csv",
            "dummy_index.csv",
            "--data-root",
            "dummy_data_root",
            "--cache-dir",
            "dummy_cache",
            "--suite",
            "primary_controls",
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_code == 1


def test_rc1_gate_supports_replay_and_bundle_checks(tmp_path: Path, monkeypatch) -> None:
    module = _load_rc1_gate_module()
    commands: list[list[str] | str] = []

    def _stub_run_step(command, *, cwd):
        commands.append(command)
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    def _stub_run_shell(command, *, cwd):
        commands.append(command)
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(module, "_run_step", _stub_run_step)
    monkeypatch.setattr(module, "_run_shell_step", _stub_run_shell)

    summary_out = tmp_path / "rc1_summary.json"
    exit_code = module.main(
        [
            "--summary-out",
            str(summary_out),
            "--run-official-replay",
            "--replay-use-demo-dataset",
            "--verify-bundle-dir",
            str(tmp_path / "bundle_a"),
            "--verify-bundle-dir",
            str(tmp_path / "bundle_b"),
        ]
    )

    assert exit_code == 0
    assert summary_out.exists()
    assert any(
        isinstance(command, list) and "scripts/replay_official_paths.py" in command
        for command in commands
    )
    assert (
        sum(
            1
            for command in commands
            if isinstance(command, list) and "scripts/verify_publishable_bundle.py" in command
        )
        == 2
    )
