from __future__ import annotations

import os
from contextlib import ExitStack
from importlib import resources
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
    # Installed wheels do not include a repository root; fall back to cwd
    # for writable local outputs (for example, templates/ and outputs/).
    if source_repo_root is not None:
        return source_repo_root
    return Path.cwd().resolve()


_RESOURCE_STACK = ExitStack()


def _materialize_packaged_asset(*relative_parts: str) -> Path:
    traversable = resources.files("Thesis_ML").joinpath(*relative_parts)
    return Path(_RESOURCE_STACK.enter_context(resources.as_file(traversable)))


def _resolve_default_registry(source_repo_root: Path | None) -> Path:
    if source_repo_root is not None:
        source_path = source_repo_root / "configs" / "decision_support_registry.json"
        if source_path.exists():
            return source_path
    return _materialize_packaged_asset(
        "assets",
        "configs",
        "decision_support_registry.json",
    )


def _resolve_shipped_workbook_template(source_repo_root: Path | None) -> Path:
    if source_repo_root is not None:
        source_path = source_repo_root / "templates" / "thesis_experiment_program.xlsx"
        if source_path.exists():
            return source_path
    return _materialize_packaged_asset(
        "assets",
        "templates",
        "thesis_experiment_program.xlsx",
    )


SOURCE_REPO_ROOT = _resolve_source_repo_root()
PROJECT_ROOT = _resolve_project_root(SOURCE_REPO_ROOT)
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

DEFAULT_DECISION_SUPPORT_REGISTRY = _resolve_default_registry(SOURCE_REPO_ROOT)
SHIPPED_WORKBOOK_TEMPLATE = _resolve_shipped_workbook_template(SOURCE_REPO_ROOT)

# Writable default output location used by workbook-generation CLIs.
DEFAULT_WORKBOOK_TEMPLATE = PROJECT_ROOT / "templates" / "thesis_experiment_program.xlsx"

DEFAULT_DECISION_SUPPORT_OUTPUT_ROOT = OUTPUTS_ROOT / "artifacts" / "decision_support"
DEFAULT_EXPERIMENT_REPORTS_ROOT = OUTPUTS_ROOT / "reports" / "experiments"
DEFAULT_BASELINE_REPORTS_DIR = OUTPUTS_ROOT / "reports"
DEFAULT_BASELINE_MODELS_DIR = OUTPUTS_ROOT / "models"
DEFAULT_SPM_GLM_EXTRACTION_OUT = OUTPUTS_ROOT / "spm_glm_extraction_out"
DEFAULT_WORKBOOK_OUTPUT_DIR = OUTPUTS_ROOT / "workbooks"


def ensure_default_output_dirs() -> None:
    for directory in (
        OUTPUTS_ROOT,
        DEFAULT_DECISION_SUPPORT_OUTPUT_ROOT,
        DEFAULT_EXPERIMENT_REPORTS_ROOT,
        DEFAULT_BASELINE_REPORTS_DIR,
        DEFAULT_BASELINE_MODELS_DIR,
        DEFAULT_SPM_GLM_EXTRACTION_OUT,
        DEFAULT_WORKBOOK_OUTPUT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
