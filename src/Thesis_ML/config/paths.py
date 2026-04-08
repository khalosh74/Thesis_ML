from __future__ import annotations

import os
from pathlib import Path


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "pyproject.toml").exists() and (path / "src" / "Thesis_ML").exists()


def _resolve_source_repo_root() -> Path | None:
    env_root = os.getenv("THESIS_ML_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    candidate = Path(__file__).resolve().parents[3]
    if _looks_like_repo_root(candidate):
        return candidate
    return None


def _resolve_project_root(source_repo_root: Path | None) -> Path:
    if source_repo_root is not None:
        return source_repo_root
    return Path.cwd().resolve()


SOURCE_REPO_ROOT = _resolve_source_repo_root()
PROJECT_ROOT = _resolve_project_root(SOURCE_REPO_ROOT)
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

DEFAULT_REPORTS_ROOT = OUTPUTS_ROOT / "reports"
DEFAULT_EXPERIMENT_REPORTS_ROOT = DEFAULT_REPORTS_ROOT / "exploratory"
DEFAULT_BASELINE_REPORTS_DIR = OUTPUTS_ROOT / "reports"
DEFAULT_BASELINE_MODELS_DIR = OUTPUTS_ROOT / "models"

DEFAULT_TARGET_CONFIGS_DIR = PROJECT_ROOT / "configs" / "targets"
DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH = DEFAULT_TARGET_CONFIGS_DIR / "affect_mapping_v2.json"
DEFAULT_CONFIRMATORY_PROTOCOL_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "confirmatory_protocol.schema.json"

DEFAULT_RELEASES_DIR = PROJECT_ROOT / "releases"
DEFAULT_RELEASE_REGISTRY_PATH = DEFAULT_RELEASES_DIR / "release_registry.json"
DEFAULT_THESIS_FINAL_RELEASE_PATH = DEFAULT_RELEASES_DIR / "thesis_final_v1" / "release.json"


def ensure_default_output_dirs() -> None:
    for directory in (
        OUTPUTS_ROOT,
        DEFAULT_REPORTS_ROOT,
        DEFAULT_EXPERIMENT_REPORTS_ROOT,
        DEFAULT_BASELINE_REPORTS_DIR,
        DEFAULT_BASELINE_MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
