from __future__ import annotations

from Thesis_ML.orchestration.decision_support import main as _main


def main(argv: list[str] | None = None) -> int:
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
