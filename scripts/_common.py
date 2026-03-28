from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, NoReturn


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def resolve_repo_root(script_file: Path) -> Path:
    return script_file.resolve().parents[1]


def fail(message: str, exit_code: int = 1) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(int(exit_code))

