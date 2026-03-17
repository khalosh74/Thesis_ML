from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from Thesis_ML.comparisons.loader import load_comparison_spec
from Thesis_ML.comparisons.runner import compile_and_run_comparison
from Thesis_ML.config.paths import (
    DEFAULT_COMPARISON_SPEC_PATH,
    DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH,
)
from Thesis_ML.protocols.loader import load_protocol
from Thesis_ML.protocols.runner import compile_and_run_protocol
from Thesis_ML.verification.reproducibility import compare_official_outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small official path twice under the same config/seed and compare deterministic invariants."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["protocol", "comparison"],
        help="Official mode to rerun twice for reproducibility verification.",
    )
    parser.add_argument(
        "--config",
        default="",
        help=(
            "Protocol/comparison spec path. Defaults: protocol->"
            f"{DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH}, comparison->{DEFAULT_COMPARISON_SPEC_PATH}."
        ),
    )
    parser.add_argument("--index-csv", required=True, help="Dataset index CSV path.")
    parser.add_argument("--data-root", required=True, help="Data root path.")
    parser.add_argument("--cache-dir", required=True, help="Feature cache root path.")
    parser.add_argument(
        "--reports-root",
        default="outputs/rc_reproducibility",
        help="Base directory where the two rerun outputs are created.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Protocol suite ID to execute (repeatable).",
    )
    parser.add_argument(
        "--all-suites",
        action="store_true",
        help="Run all enabled protocol suites.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Comparison variant ID to execute (repeatable).",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Run all comparison variants.",
    )
    parser.add_argument(
        "--summary-out",
        default="",
        help="Optional JSON path for reproducibility summary output.",
    )
    return parser


def _run_protocol_once(
    *,
    protocol_path: Path,
    reports_root: Path,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    suite_ids: list[str] | None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    return compile_and_run_protocol(
        protocol=protocol,
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        reports_root=reports_root,
        suite_ids=suite_ids,
        dry_run=False,
        force=True,
        resume=False,
    )


def _run_comparison_once(
    *,
    comparison_path: Path,
    reports_root: Path,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    variant_ids: list[str] | None,
) -> dict[str, Any]:
    comparison = load_comparison_spec(comparison_path)
    return compile_and_run_comparison(
        comparison=comparison,
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        reports_root=reports_root,
        variant_ids=variant_ids,
        dry_run=False,
        force=True,
        resume=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.mode == "protocol" and args.all_suites and args.suite:
        parser.error("Use either --suite or --all-suites for protocol mode.")
    if args.mode == "comparison" and args.all_variants and args.variant:
        parser.error("Use either --variant or --all-variants for comparison mode.")

    config_text = str(args.config).strip()
    config_path = Path(config_text) if config_text else Path()
    if not config_text:
        config_path = (
            Path(DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH)
            if args.mode == "protocol"
            else Path(DEFAULT_COMPARISON_SPEC_PATH)
        )
    if not config_path.exists():
        parser.error(f"Config path does not exist: {config_path}")

    base_reports_root = Path(args.reports_root)
    run_a_root = base_reports_root / "run_a"
    run_b_root = base_reports_root / "run_b"

    for run_root in (run_a_root, run_b_root):
        if run_root.exists():
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "protocol":
        suite_ids = None if args.all_suites else list(args.suite) or None
        result_a = _run_protocol_once(
            protocol_path=config_path,
            reports_root=run_a_root,
            index_csv=Path(args.index_csv),
            data_root=Path(args.data_root),
            cache_dir=Path(args.cache_dir),
            suite_ids=suite_ids,
        )
        result_b = _run_protocol_once(
            protocol_path=config_path,
            reports_root=run_b_root,
            index_csv=Path(args.index_csv),
            data_root=Path(args.data_root),
            cache_dir=Path(args.cache_dir),
            suite_ids=suite_ids,
        )
        left_dir = Path(str(result_a["protocol_output_dir"]))
        right_dir = Path(str(result_b["protocol_output_dir"]))
    else:
        variant_ids = None if args.all_variants else list(args.variant) or None
        result_a = _run_comparison_once(
            comparison_path=config_path,
            reports_root=run_a_root,
            index_csv=Path(args.index_csv),
            data_root=Path(args.data_root),
            cache_dir=Path(args.cache_dir),
            variant_ids=variant_ids,
        )
        result_b = _run_comparison_once(
            comparison_path=config_path,
            reports_root=run_b_root,
            index_csv=Path(args.index_csv),
            data_root=Path(args.data_root),
            cache_dir=Path(args.cache_dir),
            variant_ids=variant_ids,
        )
        left_dir = Path(str(result_a["comparison_output_dir"]))
        right_dir = Path(str(result_b["comparison_output_dir"]))

    if int(result_a.get("n_failed", 0)) > 0 or int(result_b.get("n_failed", 0)) > 0:
        summary = {
            "passed": False,
            "mode": args.mode,
            "config": str(config_path.resolve()),
            "reason": "one_or_more_runs_failed",
            "run_a": result_a,
            "run_b": result_b,
        }
    else:
        comparison_summary = compare_official_outputs(left_dir=left_dir, right_dir=right_dir)
        summary = {
            "passed": bool(comparison_summary.get("passed", False)),
            "mode": args.mode,
            "config": str(config_path.resolve()),
            "left_output_dir": str(left_dir.resolve()),
            "right_output_dir": str(right_dir.resolve()),
            "comparison": comparison_summary,
        }

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(f"{json.dumps(summary, indent=2)}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not bool(summary.get("passed", False)):
        print("Official reproducibility verification failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
