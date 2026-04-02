from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from Thesis_ML.script_support.cli import fail
from Thesis_ML.script_support.io import file_sha256, read_json, write_json
from Thesis_ML.script_support.paths import resolve_repo_root


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_module(script_path: Path) -> ModuleType:
    module_name = f"_script_test_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_repo_root_from_script_path() -> None:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "verify_publishable_bundle.py"

    assert resolve_repo_root(script_path) == repo_root


def test_common_compatibility_shim_re_exports_expected_helpers() -> None:
    common_module = _load_module(_repo_root() / "scripts" / "_common.py")

    assert common_module.file_sha256 is file_sha256
    assert common_module.read_json is read_json
    assert common_module.write_json is write_json
    assert common_module.fail is fail
    assert callable(common_module.resolve_repo_root)
    assert callable(common_module.write_summary)


def test_scripts_importing_common_still_import_cleanly() -> None:
    scripts_dir = _repo_root() / "scripts"
    for name in (
        "build_publishable_bundle.py",
        "verify_publishable_bundle.py",
        "replay_official_paths.py",
        "verify_official_reproducibility.py",
    ):
        module = _load_module(scripts_dir / name)
        assert hasattr(module, "main")
        assert callable(module.main)

