"""Thin wrapper for workbook generation CLI."""

from __future__ import annotations

import warnings

from Thesis_ML.cli.workbook import main

_DEPRECATION_MESSAGE = (
    "scripts/create_thesis_experiment_workbook.py is deprecated and kept for compatibility. "
    "Use 'thesisml-workbook' instead."
)


def _emit_deprecation_warning() -> None:
    warnings.warn(_DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)


def _main() -> int:
    _emit_deprecation_warning()
    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
