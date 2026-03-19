from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from Thesis_ML.verification.campaign_runtime_profile import verify_campaign_runtime_profile


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run precheck-only runtime profiling for confirmatory/comparison campaign plans "
            "and estimate campaign ETA by phase/model/cohort."
        )
    )
    parser.add_argument("--index-csv", required=True, help="Dataset index CSV path.")
    parser.add_argument("--data-root", required=True, help="Dataset data root path.")
    parser.add_argument("--cache-dir", required=True, help="Feature cache directory path.")
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
        "--profile-root",
        required=True,
        help="Output root for runtime profiling precheck runs.",
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

    summary = verify_campaign_runtime_profile(
        index_csv=Path(args.index_csv),
        data_root=Path(args.data_root),
        cache_dir=Path(args.cache_dir),
        confirmatory_protocol=Path(args.confirmatory_protocol),
        comparison_specs=[Path(path) for path in list(args.comparison_spec)],
        profile_root=Path(args.profile_root),
    )

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(f"{json.dumps(summary, indent=2)}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not bool(summary.get("passed", False)):
        print("Campaign runtime profile precheck failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
