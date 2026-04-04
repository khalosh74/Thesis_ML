from __future__ import annotations

import os
from contextlib import ExitStack
from importlib import resources
from pathlib import Path

from Thesis_ML.config.config_registry import (
    resolve_config_alias,
    resolve_config_registry_path,
)


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


def _resolve_thesis_runtime_registry(source_repo_root: Path | None) -> Path:
    if source_repo_root is not None:
        source_path = (
            source_repo_root / "configs" / "decision_support_registry_revised_execution.json"
        )
        if source_path.exists():
            return source_path
    # Wheels do not ship thesis runtime registries; fall back to the package default registry.
    return _resolve_default_registry(source_repo_root)


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
DEFAULT_CONFIG_REGISTRY_PATH = resolve_config_registry_path(
    source_repo_root=SOURCE_REPO_ROOT,
    project_root=PROJECT_ROOT,
)

DEFAULT_DECISION_SUPPORT_THESIS_RUNTIME_REGISTRY = resolve_config_alias(
    "registry.decision_support_thesis_runtime",
    source_repo_root=SOURCE_REPO_ROOT,
    project_root=PROJECT_ROOT,
    fallback=_resolve_thesis_runtime_registry(SOURCE_REPO_ROOT),
)
DEFAULT_DECISION_SUPPORT_PACKAGE_REGISTRY = resolve_config_alias(
    "registry.decision_support_package_default",
    source_repo_root=SOURCE_REPO_ROOT,
    project_root=PROJECT_ROOT,
    fallback=_resolve_default_registry(SOURCE_REPO_ROOT),
)
# Backward-compatible alias used by existing call sites.
DEFAULT_DECISION_SUPPORT_REGISTRY = DEFAULT_DECISION_SUPPORT_THESIS_RUNTIME_REGISTRY
SHIPPED_WORKBOOK_TEMPLATE = _resolve_shipped_workbook_template(SOURCE_REPO_ROOT)

# Writable default output location used by workbook-generation CLIs.
DEFAULT_WORKBOOK_TEMPLATE = PROJECT_ROOT / "templates" / "thesis_experiment_program.xlsx"

DEFAULT_DECISION_SUPPORT_OUTPUT_ROOT = OUTPUTS_ROOT / "artifacts" / "decision_support"
DEFAULT_REPORTS_ROOT = OUTPUTS_ROOT / "reports"
DEFAULT_EXPLORATORY_REPORTS_ROOT = DEFAULT_REPORTS_ROOT / "exploratory"
DEFAULT_COMPARISON_REPORTS_ROOT = DEFAULT_REPORTS_ROOT / "comparisons"
DEFAULT_CONFIRMATORY_REPORTS_ROOT = DEFAULT_REPORTS_ROOT / "confirmatory"
# Backward-compatible alias for legacy call sites.
DEFAULT_EXPERIMENT_REPORTS_ROOT = DEFAULT_EXPLORATORY_REPORTS_ROOT
DEFAULT_PROTOCOL_REPORTS_ROOT = DEFAULT_CONFIRMATORY_REPORTS_ROOT
DEFAULT_COMPARISON_SPEC_DIR = PROJECT_ROOT / "configs" / "comparisons"
DEFAULT_PROTOCOLS_DIR = PROJECT_ROOT / "configs" / "protocols"
DEFAULT_TARGET_CONFIGS_DIR = PROJECT_ROOT / "configs" / "targets"

# thesis_canonical_nested_v2.json is the canonical nested-tuning workflow
# thesis_confirmatory_v1.json is the legacy frozen confirmatory path retained for hard-gate validation
DEFAULT_THESIS_PROTOCOL_PATH = resolve_config_alias(
    "protocol.thesis_canonical_default",
    source_repo_root=SOURCE_REPO_ROOT,
    project_root=PROJECT_ROOT,
    fallback=(DEFAULT_PROTOCOLS_DIR / "thesis_canonical_nested_v2.json"),
)
DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH = resolve_config_alias(
    "protocol.thesis_confirmatory_frozen",
    source_repo_root=SOURCE_REPO_ROOT,
    project_root=PROJECT_ROOT,
    fallback=(DEFAULT_PROTOCOLS_DIR / "thesis_confirmatory_v1.json"),
)
DEFAULT_THESIS_NESTED_PROTOCOL_PATH = resolve_config_alias(
    "protocol.thesis_canonical_default",
    source_repo_root=SOURCE_REPO_ROOT,
    project_root=PROJECT_ROOT,
    fallback=(DEFAULT_PROTOCOLS_DIR / "thesis_canonical_nested_v2.json"),
)
DEFAULT_COMPARISON_SPEC_PATH = resolve_config_alias(
    "comparison.grouped_nested_default",
    source_repo_root=SOURCE_REPO_ROOT,
    project_root=PROJECT_ROOT,
    fallback=(DEFAULT_COMPARISON_SPEC_DIR / "model_family_grouped_nested_comparison_v2.json"),
)
DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH = resolve_config_alias(
    "target.coarse_affect_default",
    source_repo_root=SOURCE_REPO_ROOT,
    project_root=PROJECT_ROOT,
    fallback=(DEFAULT_TARGET_CONFIGS_DIR / "affect_mapping_v2.json"),
)
DEFAULT_CONFIRMATORY_PROTOCOL_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "confirmatory_protocol.schema.json"
)
DEFAULT_BASELINE_REPORTS_DIR = OUTPUTS_ROOT / "reports"
DEFAULT_BASELINE_MODELS_DIR = OUTPUTS_ROOT / "models"
DEFAULT_SPM_GLM_EXTRACTION_OUT = OUTPUTS_ROOT / "spm_glm_extraction_out"
DEFAULT_WORKBOOK_OUTPUT_DIR = OUTPUTS_ROOT / "workbooks"


def ensure_default_output_dirs() -> None:
    for directory in (
        OUTPUTS_ROOT,
        DEFAULT_REPORTS_ROOT,
        DEFAULT_DECISION_SUPPORT_OUTPUT_ROOT,
        DEFAULT_EXPLORATORY_REPORTS_ROOT,
        DEFAULT_COMPARISON_REPORTS_ROOT,
        DEFAULT_CONFIRMATORY_REPORTS_ROOT,
        DEFAULT_EXPERIMENT_REPORTS_ROOT,
        DEFAULT_COMPARISON_SPEC_DIR,
        DEFAULT_PROTOCOLS_DIR,
        DEFAULT_BASELINE_REPORTS_DIR,
        DEFAULT_BASELINE_MODELS_DIR,
        DEFAULT_SPM_GLM_EXTRACTION_OUT,
        DEFAULT_WORKBOOK_OUTPUT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
