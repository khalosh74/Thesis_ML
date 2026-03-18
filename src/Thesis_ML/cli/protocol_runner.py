from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from Thesis_ML.config.paths import (
    DEFAULT_PROTOCOL_REPORTS_ROOT,
    DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH,
    PROJECT_ROOT,
)
from Thesis_ML.protocols.loader import load_protocol
from Thesis_ML.protocols.runner import compile_and_run_protocol


def _default_index_csv() -> Path:
    override = os.getenv("THESIS_ML_INDEX_CSV")
    if override:
        return Path(override).resolve()
    return PROJECT_ROOT / "Data" / "processed" / "dataset_index.csv"


def _default_data_root() -> Path:
    override = os.getenv("THESIS_ML_DATA_ROOT")
    if override:
        return Path(override).resolve()
    return PROJECT_ROOT / "Data"


def _default_cache_dir() -> Path:
    override = os.getenv("THESIS_ML_CACHE_DIR")
    if override:
        return Path(override).resolve()
    return PROJECT_ROOT / "Data" / "processed" / "feature_cache"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run official thesis suites from a canonical protocol. "
            "Science-affecting parameters are loaded from protocol JSON only."
        )
    )
    parser.add_argument(
        "--protocol",
        default=str(DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH),
        help=(
            "Path to confirmatory protocol JSON. Defaults to the frozen "
            "confirmatory protocol for final science-freeze runs."
        ),
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Suite ID to execute. Repeat to run multiple suites.",
    )
    parser.add_argument(
        "--all-suites",
        action="store_true",
        help="Execute all enabled suites from the protocol.",
    )
    parser.add_argument(
        "--reports-root",
        default=str(DEFAULT_PROTOCOL_REPORTS_ROOT),
        help="Root directory where run reports and protocol artifacts are written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun of each run_id if output exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each run_id from existing output if available.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile protocol and emit artifacts without executing runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.all_suites and args.suite:
        parser.error("Use either --suite (one or more) or --all-suites, not both.")

    protocol = load_protocol(Path(args.protocol))
    requested_suites = list(args.suite) if args.suite else None

    result = compile_and_run_protocol(
        protocol=protocol,
        index_csv=_default_index_csv(),
        data_root=_default_data_root(),
        cache_dir=_default_cache_dir(),
        reports_root=Path(args.reports_root),
        suite_ids=(None if args.all_suites else requested_suites),
        force=bool(args.force),
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
    )

    print(
        json.dumps(
            {
                "protocol_id": result["protocol_id"],
                "protocol_version": result["protocol_version"],
                "protocol_output_dir": result["protocol_output_dir"],
                "n_success": result["n_success"],
                "n_completed": result["n_completed"],
                "n_failed": result["n_failed"],
                "n_timed_out": result["n_timed_out"],
                "n_skipped_due_to_policy": result["n_skipped_due_to_policy"],
                "n_planned": result["n_planned"],
                "artifact_paths": result["artifact_paths"],
            },
            indent=2,
        )
    )
    return (
        1
        if (
            int(result["n_failed"]) > 0
            or int(result["n_timed_out"]) > 0
            or int(result["n_skipped_due_to_policy"]) > 0
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
