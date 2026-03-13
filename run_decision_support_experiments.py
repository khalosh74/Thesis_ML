"""Backward-compatible shim for decision-support orchestration CLI."""

from __future__ import annotations

import warnings

from Thesis_ML.orchestration import decision_support as _decision_support

run_decision_support_campaign = _decision_support.run_decision_support_campaign

_DEPRECATION_MESSAGE = (
    "run_decision_support_experiments.py is deprecated and kept for compatibility. "
    "Use the packaged CLI entry point 'thesisml-run-decision-support' instead."
)


def _emit_deprecation_warning() -> None:
    warnings.warn(_DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)


def main(argv: list[str] | None = None) -> int:
    _emit_deprecation_warning()
    return _decision_support.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
