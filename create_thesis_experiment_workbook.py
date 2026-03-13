"""Backward-compatible workbook generator shim.

Real workbook logic now lives in ``Thesis_ML.workbook``.
"""

from __future__ import annotations

from pathlib import Path

from Thesis_ML.cli.workbook import main as _main
from Thesis_ML.config.paths import DEFAULT_WORKBOOK_TEMPLATE
from Thesis_ML.workbook.builder import OUT_XLSX, build_workbook
from Thesis_ML.workbook.validation import validate_workbook


def validate(path: Path) -> dict[str, str]:
    return validate_workbook(path)


def main() -> None:
    _main(["--output", str(DEFAULT_WORKBOOK_TEMPLATE)])


if __name__ == "__main__":
    main()
