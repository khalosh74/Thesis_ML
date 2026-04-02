"""Deprecated compatibility shim for decision-support orchestration CLI.

Status: deprecated shim; plan removal after caller migration.
Canonical replacement: ``thesisml-run-decision-support``.
No new logic should be added here.
"""

from __future__ import annotations

import sys
import warnings

from Thesis_ML.orchestration import decision_support as _decision_support

run_decision_support_campaign = _decision_support.run_decision_support_campaign

_DEPRECATION_MESSAGE = (
    "DEPRECATED shim: run_decision_support_experiments.py is retained only for compatibility. "
    "Use the canonical entrypoint 'thesisml-run-decision-support'. "
    "No new logic should be added here."
)


def _emit_deprecation_warning() -> None:
    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
    print(_DEPRECATION_MESSAGE, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    _emit_deprecation_warning()
    return _decision_support.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
