"""Backward-compatible script wrapper for baseline CLI."""

from __future__ import annotations

from Thesis_ML.cli.baseline import main


if __name__ == "__main__":
    raise SystemExit(main())
