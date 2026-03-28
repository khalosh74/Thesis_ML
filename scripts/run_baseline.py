"""Backward-compatible script wrapper for baseline CLI."""

from __future__ import annotations

import warnings

from Thesis_ML.cli.baseline import main as baseline_cli_main

_DEPRECATION_MESSAGE = (
    "scripts/run_baseline.py is deprecated and kept for compatibility. "
    "Use 'thesisml-run-baseline' instead."
)


def _emit_deprecation_warning() -> None:
    warnings.warn(_DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)


def _main() -> int:
    _emit_deprecation_warning()
    return baseline_cli_main()


def main(argv: list[str] | None = None) -> int:
    del argv
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
