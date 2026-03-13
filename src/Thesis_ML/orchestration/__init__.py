"""Orchestration entry points for thesis experiment campaigns.

This module intentionally avoids eager imports of heavy runtime dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "ArtifactRef",
    "CompiledStudyManifest",
    "ExperimentSpec",
    "SectionName",
    "TrialResultSummary",
    "TrialSpec",
    "compile_registry_file",
    "compile_registry_payload",
    "decision_support",
    "run_decision_support_campaign",
]

_CONTRACT_EXPORTS = {
    "ArtifactRef",
    "CompiledStudyManifest",
    "ExperimentSpec",
    "SectionName",
    "TrialResultSummary",
    "TrialSpec",
}
_COMPILER_EXPORTS = {"compile_registry_file", "compile_registry_payload"}


if TYPE_CHECKING:
    from .compiler import compile_registry_file, compile_registry_payload
    from .contracts import (
        ArtifactRef,
        CompiledStudyManifest,
        ExperimentSpec,
        SectionName,
        TrialResultSummary,
        TrialSpec,
    )
    from .decision_support import run_decision_support_campaign


def __getattr__(name: str) -> Any:
    if name in _CONTRACT_EXPORTS:
        module = import_module("Thesis_ML.orchestration.contracts")
        return getattr(module, name)
    if name in _COMPILER_EXPORTS:
        module = import_module("Thesis_ML.orchestration.compiler")
        return getattr(module, name)
    if name == "run_decision_support_campaign":
        module = import_module("Thesis_ML.orchestration.decision_support")
        return getattr(module, name)
    if name == "decision_support":
        return import_module("Thesis_ML.orchestration.decision_support")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
