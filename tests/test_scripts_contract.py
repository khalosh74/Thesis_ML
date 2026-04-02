from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _scripts_dir() -> Path:
    return _repo_root() / "scripts"


def _load_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _nonempty_noncomment_line_count(content: str) -> int:
    count = 0
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        count += 1
    return count


def test_python_scripts_expose_main_and_main_guard() -> None:
    scripts_dir = _scripts_dir()
    script_paths = sorted(
        path
        for path in scripts_dir.glob("*.py")
        if path.name not in {"_common.py"}
    )
    assert script_paths

    for script_path in script_paths:
        content = script_path.read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in content, script_path.name

        module = _load_module(script_path)
        assert hasattr(module, "main"), script_path.name
        assert callable(getattr(module, "main")), script_path.name


def test_verify_wrappers_are_thin_compatibility_layers() -> None:
    wrapper_paths = [
        _scripts_dir() / "verify_official_artifacts.py",
        _scripts_dir() / "verify_confirmatory_ready.py",
        _scripts_dir() / "verify_model_cost_policy_precheck.py",
        _scripts_dir() / "verify_publishable_bundle.py",
        _scripts_dir() / "verify_campaign_runtime_profile.py",
    ]
    for wrapper_path in wrapper_paths:
        content = wrapper_path.read_text(encoding="utf-8")
        assert "Compatibility wrapper" in content
        assert "verify_project.py" in content


def test_verify_wrappers_forward_only_to_verify_project_main() -> None:
    wrapper_subcommands = {
        "verify_official_artifacts.py": "official-artifacts",
        "verify_confirmatory_ready.py": "confirmatory-ready",
        "verify_model_cost_policy_precheck.py": "model-cost-policy-precheck",
        "verify_publishable_bundle.py": "publishable-bundle",
        "verify_campaign_runtime_profile.py": "campaign-runtime-profile",
    }
    for filename, subcommand in wrapper_subcommands.items():
        wrapper_path = _scripts_dir() / filename
        content = wrapper_path.read_text(encoding="utf-8")
        assert "from verify_project import main as _verify_project_main" in content
        assert f'forwarded = ["{subcommand}"]' in content
        assert "return int(_verify_project_main(forwarded))" in content
        assert _nonempty_noncomment_line_count(content) <= 28


def test_decision_support_shim_is_explicit_deprecated_forwarder() -> None:
    shim_path = _repo_root() / "run_decision_support_experiments.py"
    content = shim_path.read_text(encoding="utf-8")
    assert "deprecated shim" in content.lower()
    assert "thesisml-run-decision-support" in content
    assert "no new logic should be added here" in content.lower()
    assert "return _decision_support.main(argv)" in content
    assert _nonempty_noncomment_line_count(content) <= 45


def test_script_role_families_are_explicit_and_stable() -> None:
    script_names = {path.name for path in _scripts_dir().glob("*.py")}
    assert "review_e01_target_lock.py" not in script_names
    assert "run_baseline.py" not in script_names
    assert "create_thesis_experiment_workbook.py" not in script_names

    verify_scripts = sorted(name for name in script_names if name.startswith("verify_"))
    build_scripts = sorted(name for name in script_names if name.startswith("build_"))
    replay_scripts = sorted(name for name in script_names if name.startswith("replay_"))

    assert verify_scripts == [
        "verify_campaign_runtime_profile.py",
        "verify_confirmatory_ready.py",
        "verify_model_cost_policy_precheck.py",
        "verify_official_artifacts.py",
        "verify_project.py",
        "verify_publishable_bundle.py",
    ]
    assert build_scripts == [
        "build_frozen_confirmatory_registry.py",
        "build_publishable_bundle.py",
    ]
    assert replay_scripts == ["replay_official_paths.py"]
