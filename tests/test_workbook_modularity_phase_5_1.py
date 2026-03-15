from __future__ import annotations

from openpyxl import Workbook

from Thesis_ML.orchestration import workbook_compiler
from Thesis_ML.workbook import structured_execution_sheets, template_builder, template_constants
from Thesis_ML.workbook.governance_sheet_builders import (
    fill_data_selection_design_sheet as extracted_fill_data_selection_design_sheet,
)
from Thesis_ML.workbook.structured_sheet_design import fill_study_design
from Thesis_ML.workbook.structured_sheet_operations import fill_search_spaces


def test_template_builder_reuses_extracted_governance_sheet_builder() -> None:
    assert (
        template_builder.fill_data_selection_design_sheet
        is extracted_fill_data_selection_design_sheet
    )


def test_structured_execution_facade_reexports_split_modules() -> None:
    assert structured_execution_sheets.fill_study_design is fill_study_design
    assert structured_execution_sheets.fill_search_spaces is fill_search_spaces


def test_template_constants_exports_expected_catalog() -> None:
    assert "Master_Experiments" in template_constants.SHEET_ORDER
    experiments = template_constants.build_experiments()
    assert experiments
    assert experiments[0]["Experiment_ID"] == "E01"


def test_workbook_compiler_study_layer_uses_extracted_implementation(monkeypatch) -> None:
    called: dict[str, object] = {}

    def _fake_impl(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return ([], [], [], [], [], [], [], [])

    monkeypatch.setattr(workbook_compiler, "_build_study_design_layer_impl", _fake_impl)
    workbook = Workbook()
    workbook_compiler._build_study_design_layer(workbook)

    assert "kwargs" in called
    kwargs = called["kwargs"]
    assert isinstance(kwargs, dict)
    assert "_sheet_rows" in kwargs
    assert "_build_factorial_cells" in kwargs
