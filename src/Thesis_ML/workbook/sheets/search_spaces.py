from __future__ import annotations

from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from Thesis_ML.workbook.template_builder import fill_search_spaces_sheet


def fill_search_spaces(ws: Worksheet, wb: Workbook) -> int:
    return fill_search_spaces_sheet(ws, wb)
