"""Backward-compatible shim for decision-support orchestration CLI."""

from __future__ import annotations

from Thesis_ML.orchestration import decision_support as _decision_support

run_decision_support_campaign = _decision_support.run_decision_support_campaign


def main(argv: list[str] | None = None) -> int:
    return _decision_support.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
