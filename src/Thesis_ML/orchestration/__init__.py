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
    "ReusePolicy",
    "SearchDimensionSpec",
    "SearchMode",
    "SearchSpaceSpec",
    "SectionName",
    "TrialResultSummary",
    "TrialSpec",
    "compile_registry_file",
    "compile_registry_payload",
    "compile_workbook_file",
    "compile_workbook_workbook",
    "write_workbook_results",
    "decision_support",
    "run_decision_support_campaign",
    "run_workbook_decision_support_campaign",
]

_CONTRACT_EXPORTS = {
    "ArtifactRef",
    "CompiledStudyManifest",
    "ExperimentSpec",
    "ReusePolicy",
    "SearchDimensionSpec",
    "SearchMode",
    "SearchSpaceSpec",
    "SectionName",
    "TrialResultSummary",
    "TrialSpec",
}
_COMPILER_EXPORTS = {
    "compile_registry_file",
    "compile_registry_payload",
    "compile_workbook_file",
    "compile_workbook_workbook",
}
_WRITEBACK_EXPORTS = {"write_workbook_results"}


if TYPE_CHECKING:
    from .compiler import compile_registry_file, compile_registry_payload
    from .contracts import (
        ArtifactRef,
        CompiledStudyManifest,
        ExperimentSpec,
        ReusePolicy,
        SearchDimensionSpec,
        SearchMode,
        SearchSpaceSpec,
        SectionName,
        TrialResultSummary,
        TrialSpec,
    )
    from .workbook_compiler import compile_workbook_file, compile_workbook_workbook
    from .workbook_writeback import write_workbook_results
    from .decision_support import (
        run_decision_support_campaign,
        run_workbook_decision_support_campaign,
    )


def __getattr__(name: str) -> Any:
    if name in _CONTRACT_EXPORTS:
        module = import_module("Thesis_ML.orchestration.contracts")
        return getattr(module, name)
    if name in _COMPILER_EXPORTS:
        if name in {"compile_workbook_file", "compile_workbook_workbook"}:
            module = import_module("Thesis_ML.orchestration.workbook_compiler")
        else:
            module = import_module("Thesis_ML.orchestration.compiler")
        return getattr(module, name)
    if name in _WRITEBACK_EXPORTS:
        module = import_module("Thesis_ML.orchestration.workbook_writeback")
        return getattr(module, name)
    if name == "run_decision_support_campaign":
        module = import_module("Thesis_ML.orchestration.decision_support")
        return getattr(module, name)
    if name == "run_workbook_decision_support_campaign":
        module = import_module("Thesis_ML.orchestration.decision_support")
        return getattr(module, name)
    if name == "decision_support":
        return import_module("Thesis_ML.orchestration.decision_support")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
