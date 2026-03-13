from __future__ import annotations

from openpyxl.worksheet.worksheet import Worksheet

from Thesis_ML.workbook.template_builder import fill_fixed_configs_sheet


def fill_fixed_configs(ws: Worksheet) -> int:
    return fill_fixed_configs_sheet(ws)
