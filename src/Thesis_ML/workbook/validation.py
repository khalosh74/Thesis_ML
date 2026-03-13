from __future__ import annotations

from pathlib import Path

from .builder import validate


def validate_workbook(path: Path) -> dict[str, str]:
    return validate(path)
