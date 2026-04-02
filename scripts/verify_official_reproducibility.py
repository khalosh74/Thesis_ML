from __future__ import annotations

"""Compatibility wrapper. Use scripts/replay_official_paths.py --verify-determinism."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from Thesis_ML.config import describe_config_path
from Thesis_ML.config.paths import (
    DEFAULT_COMPARISON_SPEC_PATH,
    DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from replay_official_paths import main as _replay_main


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for deterministic official reproducibility checks. "
            "Delegates to scripts/replay_official_paths.py with --verify-determinism."
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
    parser.add_argument(
        "--config-alias",
        default="",
        help="Optional registry alias for protocol/comparison config selection.",
    )
    parser.add_argument("--index-csv", required=True, help="Dataset index CSV path.")
    parser.add_argument("--data-root", required=True, help="Data root path.")
    parser.add_argument("--cache-dir", required=True, help="Feature cache root path.")
    parser.add_argument(
        "--reports-root",
        default="outputs/rc_reproducibility",
        help="Base directory where replay outputs are created.",
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
        help="Optional JSON path for compatibility reproducibility summary output.",
    )
    return parser


def _resolve_mode(old_mode: str) -> str:
    return "confirmatory" if str(old_mode) == "protocol" else "comparison"


def _build_replay_args(args: argparse.Namespace) -> tuple[list[str], str]:
    replay_mode = _resolve_mode(str(args.mode))
    replay_args: list[str] = [
        "--mode",
        replay_mode,
        "--index-csv",
        str(args.index_csv),
        "--data-root",
        str(args.data_root),
        "--cache-dir",
        str(args.cache_dir),
        "--reports-root",
        str(args.reports_root),
        "--verify-determinism",
        "--skip-confirmatory-ready",
    ]
    if replay_mode == "confirmatory":
        if args.config:
            replay_args.extend(["--protocol", str(args.config)])
        if args.config_alias:
            replay_args.extend(["--protocol-alias", str(args.config_alias)])
        if bool(args.all_suites):
            replay_args.append("--all-suites")
        else:
            for suite in list(args.suite):
                replay_args.extend(["--suite", str(suite)])
    else:
        if args.config:
            replay_args.extend(["--comparison", str(args.config)])
        if args.config_alias:
            replay_args.extend(["--comparison-alias", str(args.config_alias)])
        if bool(args.all_variants):
            replay_args.append("--all-variants")
        else:
            for variant in list(args.variant):
                replay_args.extend(["--variant", str(variant)])

    reports_root = Path(args.reports_root).resolve()
    reports_root.mkdir(parents=True, exist_ok=True)
    replay_summary_path = reports_root / "_verify_official_reproducibility_replay_summary.json"
    replay_verification_path = (
        reports_root / "_verify_official_reproducibility_replay_verification_summary.json"
    )
    replay_args.extend(
        [
            "--summary-out",
            str(replay_summary_path),
            "--verification-summary-out",
            str(replay_verification_path),
        ]
    )
    return replay_args, replay_mode


def _compatibility_summary(
    *,
    replay_mode: str,
    replay_reports_root: Path,
) -> dict[str, Any]:
    replay_summary_path = replay_reports_root / "_verify_official_reproducibility_replay_summary.json"
    replay_verification_path = (
        replay_reports_root / "_verify_official_reproducibility_replay_verification_summary.json"
    )
    replay_summary = (
        json.loads(replay_summary_path.read_text(encoding="utf-8"))
        if replay_summary_path.exists()
        else {}
    )
    replay_verification = (
        json.loads(replay_verification_path.read_text(encoding="utf-8"))
        if replay_verification_path.exists()
        else {}
    )

    config_path_text = (
        replay_summary.get("protocol_spec")
        if replay_mode == "confirmatory"
        else replay_summary.get("comparison_spec")
    )
    config_path = Path(str(config_path_text)).resolve() if config_path_text else None

    mode_key = "confirmatory" if replay_mode == "confirmatory" else "comparison"
    determinism_by_mode = (
        replay_verification.get("determinism", {}).get("by_mode", {})
        if isinstance(replay_verification, dict)
        else {}
    )
    determinism_payload = (
        determinism_by_mode.get(mode_key) if isinstance(determinism_by_mode, dict) else None
    )
    if not isinstance(determinism_payload, dict):
        determinism_payload = {}

    return {
        "passed": bool(replay_verification.get("passed", False)),
        "mode": "protocol" if replay_mode == "confirmatory" else "comparison",
        "config": str(config_path) if config_path is not None else None,
        "config_identity": (
            describe_config_path(config_path) if config_path is not None else None
        ),
        "comparison": determinism_payload.get("comparison"),
        "replay_summary": str(replay_summary_path),
        "replay_verification_summary": str(replay_verification_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.mode == "protocol" and bool(args.all_suites) and bool(args.suite):
        parser.error("Use either --suite or --all-suites for protocol mode.")
    if args.mode == "comparison" and bool(args.all_variants) and bool(args.variant):
        parser.error("Use either --variant or --all-variants for comparison mode.")

    replay_args, replay_mode = _build_replay_args(args)
    exit_code = int(_replay_main(replay_args))

    compatibility_summary = _compatibility_summary(
        replay_mode=replay_mode,
        replay_reports_root=Path(args.reports_root).resolve(),
    )
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(f"{json.dumps(compatibility_summary, indent=2)}\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
