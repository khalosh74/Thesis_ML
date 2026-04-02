from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_rc1_gate_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "rc1_release_gate.py"
    spec = importlib.util.spec_from_file_location("rc1_release_gate", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_rc1_gate_accepts_custom_repro_command_with_replay(tmp_path: Path, monkeypatch) -> None:
    module = _load_rc1_gate_module()
    shell_commands: list[str] = []

    def _stub_run_step(command, *, cwd):
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    def _stub_run_shell(command, *, cwd):
        shell_commands.append(command)
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(module, "_run_step", _stub_run_step)
    monkeypatch.setattr(module, "_run_shell_step", _stub_run_shell)

    summary_out = tmp_path / "rc1_summary.json"
    repro_command = (
        "python scripts/replay_official_paths.py --mode confirmatory "
        "--verify-determinism --skip-confirmatory-ready"
    )
    exit_code = module.main(
        [
            "--summary-out",
            str(summary_out),
            "--repro-command",
            repro_command,
        ]
    )

    assert exit_code == 0
    assert summary_out.exists()
    assert shell_commands == [repro_command]
    payload = json.loads(summary_out.read_text(encoding="utf-8"))
    assert bool(payload["passed"]) is True

