from __future__ import annotations

import importlib.util
import json
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
    captured: dict[str, list[str]] = {}

    def _stub_replay_main(argv):
        captured["argv"] = list(argv)
        summary_out = Path(argv[argv.index("--summary-out") + 1])
        verification_out = Path(argv[argv.index("--verification-summary-out") + 1])
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(
            json.dumps(
                {
                    "protocol_spec": str(module.DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        verification_out.write_text(
            json.dumps(
                {
                    "passed": True,
                    "determinism": {"by_mode": {"confirmatory": {"comparison": {"passed": True}}}},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(module, "_replay_main", _stub_replay_main)

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
    replay_argv = captured["argv"]
    assert replay_argv[0:2] == ["--mode", "confirmatory"]
    assert "--verify-determinism" in replay_argv
    assert "--skip-confirmatory-ready" in replay_argv


def test_verify_repro_protocol_mode_resolves_config_alias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_verify_repro_module()
    captured: dict[str, list[str]] = {}

    def _stub_replay_main(argv):
        captured["argv"] = list(argv)
        summary_out = Path(argv[argv.index("--summary-out") + 1])
        verification_out = Path(argv[argv.index("--verification-summary-out") + 1])
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(
            json.dumps(
                {
                    "protocol_spec": str(module.DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        verification_out.write_text(
            json.dumps(
                {
                    "passed": True,
                    "determinism": {"by_mode": {"confirmatory": {"comparison": {"passed": True}}}},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(module, "_replay_main", _stub_replay_main)

    exit_code = module.main(
        [
            "--mode",
            "protocol",
            "--config-alias",
            "protocol.thesis_confirmatory_frozen",
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
    replay_argv = captured["argv"]
    assert "--protocol-alias" in replay_argv
    alias_index = replay_argv.index("--protocol-alias")
    assert replay_argv[alias_index + 1] == "protocol.thesis_confirmatory_frozen"


def test_verify_repro_protocol_summary_includes_config_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_verify_repro_module()

    def _stub_replay_main(argv):
        summary_out = Path(argv[argv.index("--summary-out") + 1])
        verification_out = Path(argv[argv.index("--verification-summary-out") + 1])
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(
            json.dumps(
                {
                    "protocol_spec": str(module.DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        verification_out.write_text(
            json.dumps(
                {
                    "passed": True,
                    "determinism": {"by_mode": {"confirmatory": {"comparison": {"passed": True}}}},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(module, "_replay_main", _stub_replay_main)

    summary_out = tmp_path / "summary.json"
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
            "--summary-out",
            str(summary_out),
        ]
    )
    assert exit_code == 0
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert isinstance(summary["config_identity"], dict)
    assert summary["config_identity"]["config_id"] == "protocol.thesis_confirmatory_v1"
    assert summary["config_identity"]["lifecycle"] == "frozen_confirmatory"


def test_verify_repro_uses_default_comparison_config_when_omitted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_verify_repro_module()
    captured: dict[str, list[str]] = {}

    def _stub_replay_main(argv):
        captured["argv"] = list(argv)
        summary_out = Path(argv[argv.index("--summary-out") + 1])
        verification_out = Path(argv[argv.index("--verification-summary-out") + 1])
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(
            json.dumps(
                {
                    "comparison_spec": str(module.DEFAULT_COMPARISON_SPEC_PATH),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        verification_out.write_text(
            json.dumps(
                {
                    "passed": True,
                    "determinism": {"by_mode": {"comparison": {"comparison": {"passed": True}}}},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(module, "_replay_main", _stub_replay_main)

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
    replay_argv = captured["argv"]
    assert replay_argv[0:2] == ["--mode", "comparison"]
    assert "--verify-determinism" in replay_argv


def test_verify_repro_fails_when_runs_are_timed_out(tmp_path: Path, monkeypatch) -> None:
    module = _load_verify_repro_module()

    def _stub_replay_main(argv):
        summary_out = Path(argv[argv.index("--summary-out") + 1])
        verification_out = Path(argv[argv.index("--verification-summary-out") + 1])
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(
            json.dumps({"protocol_spec": str(module.DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH)}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        verification_out.write_text(
            json.dumps(
                {
                    "passed": False,
                    "determinism": {"by_mode": {"confirmatory": {"comparison": {"passed": False}}}},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 1

    monkeypatch.setattr(module, "_replay_main", _stub_replay_main)

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
