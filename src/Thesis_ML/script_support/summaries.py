from __future__ import annotations

from pathlib import Path
from typing import Any

from Thesis_ML.script_support.io import write_json


def write_summary(path: Path, payload: Any) -> Path:
    write_json(path, payload)
    return path


__all__ = ["write_summary"]
