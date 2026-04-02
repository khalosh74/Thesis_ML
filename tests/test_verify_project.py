from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_script_module(script_name: str):
    script_path = _repo_root() / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verify_project_official_artifacts_subcommand_writes_summary(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module("verify_project.py")
    summary_out = tmp_path / "official_summary.json"

    monkeypatch.setattr(
        module,
        "verify_official_artifacts",
        lambda **_: {"passed": True, "issues": []},
    )

    exit_code = module.main(
        [
            "official-artifacts",
            "--output-dir",
            str(tmp_path / "official_output"),
            "--summary-out",
            str(summary_out),
        ]
    )
    assert exit_code == 0
    payload = json.loads(summary_out.read_text(encoding="utf-8"))
    assert payload["passed"] is True


def test_verify_project_publishable_bundle_subcommand_failure_exit_code(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module("verify_project.py")
    monkeypatch.setattr(
        module,
        "verify_publishable_bundle",
        lambda _: {"passed": False, "issues": [{"code": "synthetic"}]},
    )

    exit_code = module.main(
        [
            "publishable-bundle",
            "--bundle-dir",
            str(tmp_path / "bundle"),
        ]
    )
    assert exit_code == 1


@pytest.mark.parametrize(
    ("script_name", "expected_subcommand"),
    [
        ("verify_official_artifacts.py", "official-artifacts"),
        ("verify_confirmatory_ready.py", "confirmatory-ready"),
        ("verify_model_cost_policy_precheck.py", "model-cost-policy-precheck"),
        ("verify_publishable_bundle.py", "publishable-bundle"),
        ("verify_campaign_runtime_profile.py", "campaign-runtime-profile"),
    ],
)
def test_verify_wrappers_route_to_verify_project(
    script_name: str,
    expected_subcommand: str,
    monkeypatch,
) -> None:
    module = _load_script_module(script_name)
    captured: dict[str, list[str]] = {}

    def _stub_verify_project_main(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(module, "_verify_project_main", _stub_verify_project_main)
    exit_code = module.main(["--help"])

    assert exit_code == 0
    assert captured["argv"][0] == expected_subcommand
    assert captured["argv"][1:] == ["--help"]

