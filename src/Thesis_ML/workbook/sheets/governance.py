from __future__ import annotations

from openpyxl.workbook.workbook import Workbook

from Thesis_ML.workbook.template_builder import (
    fill_ai_usage_sheet,
    fill_claim_ledger_sheet,
    fill_confirmatory_sheet,
    fill_dashboard_sheet,
    fill_data_profile_sheet,
    fill_data_selection_design_sheet,
    fill_decision_log_sheet,
    fill_dictionary_sheet,
    fill_ethics_sheet,
    fill_grouping_strategy_map_sheet,
    fill_master_sheet,
    fill_readme_sheet,
    fill_run_log_sheet,
    fill_thesis_map_sheet,
)


def fill_governance_core(wb: Workbook) -> None:
    fill_readme_sheet(wb["README"])
    fill_master_sheet(wb["Master_Experiments"])
    fill_data_selection_design_sheet(wb["Data_Selection_Design"], wb)
    fill_grouping_strategy_map_sheet(wb["Grouping_Strategy_Map"], wb)
    fill_data_profile_sheet(wb["Data_Profile"])
    fill_run_log_sheet(wb["Run_Log"])
    fill_decision_log_sheet(wb["Decision_Log"])
    fill_confirmatory_sheet(wb["Confirmatory_Set"])
    fill_thesis_map_sheet(wb["Thesis_Map"])
    fill_dictionary_sheet(wb["Dictionary_Validation"], wb)
    fill_dashboard_sheet(wb["Dashboard"])
    fill_claim_ledger_sheet(wb["Claim_Ledger"])
    fill_ai_usage_sheet(wb["AI_Usage_Log"])
    fill_ethics_sheet(wb["Ethics_Governance_Notes"])
