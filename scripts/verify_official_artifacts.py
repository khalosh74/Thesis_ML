from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from Thesis_ML.verification.official_artifacts import verify_official_artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify invariant artifact completeness/metadata for official comparison "
            "or confirmatory outputs."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to a comparison/protocol output directory containing execution_status.json.",
    )
    parser.add_argument(
        "--mode",
        choices=["comparison", "confirmatory", "protocol", "locked_comparison"],
        default=None,
        help="Optional mode hint; omitted means auto-detect from execution_status.json.",
    )
    parser.add_argument(
        "--summary-out",
        default="",
        help="Optional JSON path for verification summary output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    summary = verify_official_artifacts(
        output_dir=Path(args.output_dir),
        mode=args.mode,
    )

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(f"{json.dumps(summary, indent=2)}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not bool(summary.get("passed", False)):
        print("Official artifact verification failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
