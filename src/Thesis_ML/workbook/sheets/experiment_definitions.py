from __future__ import annotations

from openpyxl.worksheet.worksheet import Worksheet

from Thesis_ML.workbook.builder import fill_experiment_definitions_sheet


def fill_experiment_definitions(ws: Worksheet) -> int:
    return fill_experiment_definitions_sheet(ws)
