from __future__ import annotations

from openpyxl.worksheet.worksheet import Worksheet

from Thesis_ML.workbook.template_builder import fill_machine_status_sheet


def fill_machine_status(ws: Worksheet) -> int:
    return fill_machine_status_sheet(ws)
