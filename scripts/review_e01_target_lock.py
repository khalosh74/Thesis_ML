from __future__ import annotations

import argparse
import sys
from pathlib import Path

from review_preflight_stage import main as review_preflight_main


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for E01 review. "
            "Delegates to scripts/review_preflight_stage.py --experiment-id E01."
        )
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="Campaign root containing run_log_export.csv and decision_support_summary.csv.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Optional registry override passed through to review_preflight_stage.py.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Deprecated compatibility flag retained for existing invocations.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Deprecated compatibility flag retained for existing invocations.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    delegated_args = [
        "--campaign-root",
        str(args.campaign_root),
        "--experiment-id",
        "E01",
    ]
    if args.registry is not None:
        delegated_args.extend(["--registry", str(args.registry)])

    return int(review_preflight_main(delegated_args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
