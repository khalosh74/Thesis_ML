from __future__ import annotations

import argparse
from pathlib import Path

from Thesis_ML.config.paths import DEFAULT_WORKBOOK_TEMPLATE
from Thesis_ML.workbook.builder import build_workbook
from Thesis_ML.workbook.validation import validate_workbook


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and validate the thesis experiment workbook template."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_WORKBOOK_TEMPLATE),
        help="Output workbook path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = build_workbook()
    wb.save(output_path)
    summary = validate_workbook(output_path)

    print("Created workbook:", output_path.resolve())
    print("Sheet order valid:", summary["sheet_order_ok"])
    print("Missing required sheets:", summary["missing_sheets"])
    print("Legacy required sheets present:", summary["legacy_sheets_present"])
    print("New sheets present:", summary["new_sheets_present"])
    print("Sheet count:", summary["sheet_count"])
    print("Data validations found:", summary["data_validations_found"])
    print("Experiment_Definitions columns valid:", summary["experiment_definitions_columns_ok"])
    print("Run_Log new columns present:", summary["run_log_new_columns_present"])
    print("Required named lists present:", summary["required_named_lists_present"])
    print("Missing named lists:", summary["missing_named_lists"])
    print("Experiment_Ready formula present:", summary["experiment_ready_formula_present"])
    print("Confirmatory formulas present:", summary["confirmatory_formula_present"])
    print("Dashboard formulas present:", summary["dashboard_formula_present"])
    print("Stage vocabulary consistent:", summary["stage_vocab_consistent"])
    print("Stage rows detected:", summary["stage_vocab_rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
