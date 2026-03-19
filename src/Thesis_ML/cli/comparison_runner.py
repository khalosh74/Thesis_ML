from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from Thesis_ML.comparisons.loader import load_comparison_spec
from Thesis_ML.comparisons.runner import compile_and_run_comparison
from Thesis_ML.config.paths import (
    DEFAULT_COMPARISON_REPORTS_ROOT,
    DEFAULT_COMPARISON_SPEC_PATH,
    PROJECT_ROOT,
)


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
            "Run locked comparison experiments from registered comparison specs. "
            "Science-affecting parameters are loaded from comparison JSON only."
        )
    )
    parser.add_argument(
        "--comparison",
        default=str(DEFAULT_COMPARISON_SPEC_PATH),
        help="Path to comparison spec JSON.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant ID to execute. Repeat to run multiple variants.",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Execute all registered variants from the comparison spec.",
    )
    parser.add_argument(
        "--reports-root",
        default=str(DEFAULT_COMPARISON_REPORTS_ROOT),
        help="Root directory where run reports and comparison artifacts are written.",
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
        help="Compile comparison and emit artifacts without executing runs.",
    )
    parser.add_argument(
        "--max-parallel-runs",
        type=int,
        default=1,
        help="Operational scheduling control for independent run fan-out. Default: 1 (serial).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.all_variants and args.variant:
        parser.error("Use either --variant (one or more) or --all-variants, not both.")

    comparison = load_comparison_spec(Path(args.comparison))
    requested_variants = list(args.variant) if args.variant else None

    result = compile_and_run_comparison(
        comparison=comparison,
        index_csv=_default_index_csv(),
        data_root=_default_data_root(),
        cache_dir=_default_cache_dir(),
        reports_root=Path(args.reports_root),
        variant_ids=(None if args.all_variants else requested_variants),
        force=bool(args.force),
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        max_parallel_runs=int(args.max_parallel_runs),
    )

    print(
        json.dumps(
            {
                "comparison_id": result["comparison_id"],
                "comparison_version": result["comparison_version"],
                "comparison_output_dir": result["comparison_output_dir"],
                "n_success": result["n_success"],
                "n_completed": result["n_completed"],
                "n_failed": result["n_failed"],
                "n_timed_out": result["n_timed_out"],
                "n_skipped_due_to_policy": result["n_skipped_due_to_policy"],
                "n_planned": result["n_planned"],
                "max_parallel_runs_effective": result["max_parallel_runs_effective"],
                "artifact_paths": result["artifact_paths"],
            },
            indent=2,
        )
    )
    return 1 if int(result["n_failed"]) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
