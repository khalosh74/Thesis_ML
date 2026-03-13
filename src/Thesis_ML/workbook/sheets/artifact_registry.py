from __future__ import annotations

from openpyxl.worksheet.worksheet import Worksheet

from Thesis_ML.workbook.template_builder import fill_artifact_registry_sheet


def fill_artifact_registry(ws: Worksheet) -> int:
    return fill_artifact_registry_sheet(ws)
