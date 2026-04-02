from __future__ import annotations

"""Compatibility wrapper. Use scripts/verify_project.py official-artifacts."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_project import main as _verify_project_main


def main(argv: list[str] | None = None) -> int:
    forwarded = ["official-artifacts"]
    if argv:
        forwarded.extend(list(argv))
    return int(_verify_project_main(forwarded))


if __name__ == "__main__":
    raise SystemExit(main())
