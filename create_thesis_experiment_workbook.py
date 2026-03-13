"""Backward-compatible workbook generator shim.

Real workbook logic now lives in ``Thesis_ML.workbook``.
"""

from __future__ import annotations

from pathlib import Path
import warnings

from Thesis_ML.cli.workbook import main as _main
from Thesis_ML.config.paths import DEFAULT_WORKBOOK_TEMPLATE
from Thesis_ML.workbook.builder import OUT_XLSX, build_workbook
from Thesis_ML.workbook.validation import validate_workbook

_DEPRECATION_MESSAGE = (
    "create_thesis_experiment_workbook.py is deprecated and kept for compatibility. "
    "Use the packaged CLI entry point 'thesisml-workbook' instead."
)


def _emit_deprecation_warning() -> None:
    warnings.warn(_DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)


def validate(path: Path) -> dict[str, str]:
    return validate_workbook(path)


def main() -> None:
    _emit_deprecation_warning()
    _main(["--output", str(DEFAULT_WORKBOOK_TEMPLATE)])


if __name__ == "__main__":
    main()


__all__ = ["OUT_XLSX", "build_workbook", "validate", "main"]
