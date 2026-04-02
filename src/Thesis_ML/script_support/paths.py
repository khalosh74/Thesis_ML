from __future__ import annotations

from pathlib import Path


def resolve_repo_root(script_file: Path) -> Path:
    return script_file.resolve().parents[1]


__all__ = ["resolve_repo_root"]
