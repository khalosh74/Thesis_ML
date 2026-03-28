from __future__ import annotations

import importlib.util
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _scripts_dir() -> Path:
    return _repo_root() / "scripts"


def _load_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_wrapper_scripts_retain_deprecation_notice() -> None:
    wrapper_paths = [
        _scripts_dir() / "run_baseline.py",
        _scripts_dir() / "create_thesis_experiment_workbook.py",
    ]
    for wrapper_path in wrapper_paths:
        content = wrapper_path.read_text(encoding="utf-8")
        assert "deprecated and kept for compatibility" in content


def test_script_role_families_are_explicit_and_stable() -> None:
    script_names = {path.name for path in _scripts_dir().glob("*.py")}

    verify_scripts = sorted(name for name in script_names if name.startswith("verify_"))
    build_scripts = sorted(name for name in script_names if name.startswith("build_"))
    replay_scripts = sorted(name for name in script_names if name.startswith("replay_"))

    assert verify_scripts == [
        "verify_campaign_runtime_profile.py",
        "verify_confirmatory_ready.py",
        "verify_model_cost_policy_precheck.py",
        "verify_official_artifacts.py",
        "verify_official_reproducibility.py",
        "verify_publishable_bundle.py",
    ]
    assert build_scripts == ["build_publishable_bundle.py"]
    assert replay_scripts == ["replay_official_paths.py"]

