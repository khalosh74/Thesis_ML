from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from Thesis_ML.verification.model_cost_policy import verify_model_cost_policy_precheck


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile official confirmatory/comparison specs and validate model-cost policy "
            "constraints before frozen campaign execution."
        )
    )
    parser.add_argument(
        "--index-csv",
        required=True,
        help="Dataset index CSV used for protocol/comparison compilation.",
    )
    parser.add_argument(
        "--confirmatory-protocol",
        required=True,
        help="Confirmatory protocol JSON path.",
    )
    parser.add_argument(
        "--comparison-spec",
        action="append",
        required=True,
        help="Comparison spec JSON path. Repeat for multiple specs.",
    )
    parser.add_argument(
        "--summary-out",
        default="",
        help="Optional JSON summary output path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    summary = verify_model_cost_policy_precheck(
        index_csv=Path(args.index_csv),
        confirmatory_protocol=Path(args.confirmatory_protocol),
        comparison_specs=[Path(path) for path in list(args.comparison_spec)],
    )

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(f"{json.dumps(summary, indent=2)}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not bool(summary.get("passed", False)):
        print("Model-cost policy precheck failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
