from __future__ import annotations

from openpyxl.worksheet.worksheet import Worksheet

from Thesis_ML.workbook.template_builder import fill_experiment_definitions_sheet


def fill_experiment_definitions(ws: Worksheet) -> int:
    return fill_experiment_definitions_sheet(ws)
