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
from Thesis_ML.config.runtime_selection import resolve_runtime_config_path
from Thesis_ML.experiments.compute_policy import HARDWARE_MODE_CHOICES


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
        default=None,
        # Canonical modeling-layer default is grouped-nested v2.
        help="Path to comparison spec JSON.",
    )
    parser.add_argument(
        "--comparison-alias",
        default=None,
        help="Registry alias for comparison selection when --comparison is not provided.",
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
    parser.add_argument(
        "--hardware-mode",
        default="cpu_only",
        choices=list(HARDWARE_MODE_CHOICES),
        help=(
            "Operational compute control only. "
            "Official comparison runs remain conservative and admit cpu_only only."
        ),
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=None,
        help="Optional GPU device ID for future GPU-capable operational modes.",
    )
    parser.add_argument(
        "--deterministic-compute",
        action="store_true",
        help="Record deterministic compute intent in additive compute metadata.",
    )
    parser.add_argument(
        "--allow-backend-fallback",
        action="store_true",
        help="Exploratory-only compute fallback flag. Official comparison paths reject it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.all_variants and args.variant:
        parser.error("Use either --variant (one or more) or --all-variants, not both.")

    comparison_path = resolve_runtime_config_path(
        args.comparison,
        args.comparison_alias,
        default_alias="comparison.grouped_nested_default",
        fallback_path=DEFAULT_COMPARISON_SPEC_PATH,
    )
    comparison = load_comparison_spec(comparison_path)
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
        hardware_mode=args.hardware_mode,
        gpu_device_id=args.gpu_device_id,
        deterministic_compute=bool(args.deterministic_compute),
        allow_backend_fallback=bool(args.allow_backend_fallback),
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
                "source_comparison": result.get("source_comparison"),
            },
            indent=2,
        )
    )
    return 1 if int(result["n_failed"]) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
