"""Workbook builder facade.

The implementation is split into focused modules, with the primary workbook
construction/validation logic living in ``template_builder``.
"""

from __future__ import annotations

from typing import Any

from . import template_builder as _impl

OUT_XLSX = _impl.OUT_XLSX
build_workbook = _impl.build_workbook
validate = _impl.validate
main = _impl.main


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


__all__ = ["OUT_XLSX", "build_workbook", "validate", "main"]
