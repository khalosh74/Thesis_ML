"""Orchestration entry points for thesis experiment campaigns."""

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

__all__ = [
    "ArtifactRef",
    "CompiledStudyManifest",
    "ExperimentSpec",
    "SectionName",
    "TrialResultSummary",
    "TrialSpec",
    "compile_registry_file",
    "compile_registry_payload",
    "run_decision_support_campaign",
]
