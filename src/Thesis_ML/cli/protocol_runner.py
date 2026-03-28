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
from Thesis_ML.config.runtime_selection import resolve_runtime_config_path
from Thesis_ML.experiments.compute_policy import HARDWARE_MODE_CHOICES
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
        default=None,
        help=(
            "Path to confirmatory protocol JSON. Defaults to the frozen "
            "confirmatory protocol for final science-freeze runs."
        ),
    )
    parser.add_argument(
        "--protocol-alias",
        default=None,
        help="Registry alias for protocol selection when --protocol is not provided.",
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
            "Official protocol runs remain conservative and admit cpu_only only."
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
        help="Exploratory-only compute fallback flag. Official protocol paths reject it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.all_suites and args.suite:
        parser.error("Use either --suite (one or more) or --all-suites, not both.")

    protocol_path = resolve_runtime_config_path(
        args.protocol,
        args.protocol_alias,
        default_alias="protocol.thesis_confirmatory_frozen",
        fallback_path=DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH,
    )
    protocol = load_protocol(protocol_path)
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
        max_parallel_runs=int(args.max_parallel_runs),
        hardware_mode=args.hardware_mode,
        gpu_device_id=args.gpu_device_id,
        deterministic_compute=bool(args.deterministic_compute),
        allow_backend_fallback=bool(args.allow_backend_fallback),
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
                "max_parallel_runs_effective": result["max_parallel_runs_effective"],
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
