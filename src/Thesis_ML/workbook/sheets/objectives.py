from __future__ import annotations

from openpyxl.worksheet.worksheet import Worksheet

from Thesis_ML.workbook.builder import fill_objectives_sheet


def fill_objectives(ws: Worksheet) -> int:
    return fill_objectives_sheet(ws)
