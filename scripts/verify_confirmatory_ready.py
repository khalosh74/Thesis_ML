from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from Thesis_ML.verification.confirmatory_ready import verify_confirmatory_ready


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a confirmatory output directory satisfies governance-level "
            "confirmatory-ready criteria."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to a confirmatory protocol output directory containing execution_status.json.",
    )
    parser.add_argument(
        "--repro-summary",
        default="",
        help=(
            "Optional reproducibility summary JSON path from "
            "scripts/verify_official_reproducibility.py."
        ),
    )
    parser.add_argument(
        "--summary-out",
        default="",
        help="Optional JSON path for confirmatory-ready summary output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    summary = verify_confirmatory_ready(
        output_dir=Path(args.output_dir),
        reproducibility_summary=Path(args.repro_summary) if args.repro_summary else None,
    )

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(f"{json.dumps(summary, indent=2)}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not bool(summary.get("passed", False)):
        print("Confirmatory-ready verification failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
