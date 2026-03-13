from __future__ import annotations

from openpyxl.worksheet.worksheet import Worksheet

from Thesis_ML.workbook.template_builder import fill_summary_outputs_sheet


def fill_summary_outputs(ws: Worksheet) -> int:
    return fill_summary_outputs_sheet(ws)
