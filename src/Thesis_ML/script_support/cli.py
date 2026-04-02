from __future__ import annotations

import sys
from typing import NoReturn


def fail(message: str, exit_code: int = 1) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(int(exit_code))


__all__ = ["fail"]
