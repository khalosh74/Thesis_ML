from __future__ import annotations

from pathlib import Path

from Thesis_ML.orchestration.compiler import compile_registry_file
from Thesis_ML.orchestration.contracts import CompiledStudyManifest
from Thesis_ML.orchestration.workbook_compiler import compile_workbook_file


def read_registry_manifest(path: Path) -> CompiledStudyManifest:
    return compile_registry_file(path)


def read_workbook_manifest(path: Path) -> CompiledStudyManifest:
    return compile_workbook_file(path)


__all__ = ["read_registry_manifest", "read_workbook_manifest"]
