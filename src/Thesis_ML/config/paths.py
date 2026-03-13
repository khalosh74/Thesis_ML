from __future__ import annotations

import os
from pathlib import Path


def _resolve_project_root() -> Path:
    env_root = os.getenv("THESIS_ML_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _resolve_project_root()
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

DEFAULT_DECISION_SUPPORT_REGISTRY = PROJECT_ROOT / "configs" / "decision_support_registry.json"
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
