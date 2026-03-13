from __future__ import annotations

from openpyxl.worksheet.worksheet import Worksheet

from Thesis_ML.workbook.template_builder import fill_trial_results_sheet


def fill_trial_results(ws: Worksheet) -> int:
    return fill_trial_results_sheet(ws)
