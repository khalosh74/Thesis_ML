from __future__ import annotations

"""Compatibility wrapper. Use scripts/verify_project.py confirmatory-ready."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_project import main as _verify_project_main


def main(argv: list[str] | None = None) -> int:
    forwarded = ["confirmatory-ready"]
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    if effective_argv:
        forwarded.extend(effective_argv)
    return int(_verify_project_main(forwarded))


if __name__ == "__main__":
    raise SystemExit(main())
