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
    PROJECT_ROOT,
)
from Thesis_ML.protocols.loader import load_protocol
from Thesis_ML.protocols.runner import compile_and_run_protocol
from Thesis_ML.verification.confirmatory_ready import verify_confirmatory_ready
from Thesis_ML.verification.official_artifacts import verify_official_artifacts
from Thesis_ML.verification.repro_manifest import (
    build_reproducibility_manifest,
    write_reproducibility_manifest,
)
from Thesis_ML.verification.reproducibility import compare_official_outputs


def _demo_dataset_paths() -> dict[str, Path]:
    base = PROJECT_ROOT / "demo_data" / "synthetic_v1"
    return {
        "base": base,
        "index_csv": base / "dataset_index.csv",
        "data_root": base / "data_root",
        "cache_dir": base / "cache",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-command official replay orchestrator for comparison/confirmatory paths "
            "with artifact verification and optional deterministic rerun checks."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["comparison", "confirmatory", "both"],
        default="both",
        help="Official mode(s) to execute.",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path(DEFAULT_COMPARISON_SPEC_PATH),
        help="Comparison spec path.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH),
        help="Confirmatory protocol path.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Comparison variant ID (repeatable).",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Run all variants from the comparison spec.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Protocol suite ID (repeatable).",
    )
    parser.add_argument(
        "--all-suites",
        action="store_true",
        help="Run all enabled suites from the protocol.",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=None,
        help="Dataset index CSV path. Ignored when --use-demo-dataset is set.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root path. Ignored when --use-demo-dataset is set.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache dir path. Ignored when --use-demo-dataset is set.",
    )
    parser.add_argument(
        "--use-demo-dataset",
        action="store_true",
        help="Use checked-in demo dataset under demo_data/synthetic_v1.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("outputs") / "reproducibility" / "official_replay",
        help="Root directory for replay outputs and summaries.",
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Run each selected mode twice and compare deterministic invariants.",
    )
    parser.add_argument(
        "--skip-confirmatory-ready",
        action="store_true",
        help="Skip confirmatory-ready verification for confirmatory mode.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Replay summary JSON path (default: <reports-root>/replay_summary.json).",
    )
    parser.add_argument(
        "--verification-summary-out",
        type=Path,
        default=None,
        help=(
            "Replay verification summary JSON path "
            "(default: <reports-root>/replay_verification_summary.json)."
        ),
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help=(
            "Reproducibility manifest JSON path "
            "(default: <reports-root>/reproducibility_manifest.json)."
        ),
    )
    return parser


def _resolve_dataset_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if bool(args.use_demo_dataset):
        demo_paths = _demo_dataset_paths()
        missing = [
            path
            for key, path in demo_paths.items()
            if key in {"base", "index_csv", "data_root"} and not path.exists()
        ]
        if missing:
            formatted = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                "Demo dataset is missing required files/directories: "
                f"{formatted}. Run scripts/generate_demo_dataset.py first."
            )
        demo_paths["cache_dir"].mkdir(parents=True, exist_ok=True)
        return (
            demo_paths["index_csv"],
            demo_paths["data_root"],
            demo_paths["cache_dir"],
        )
    if args.index_csv is None or args.data_root is None or args.cache_dir is None:
        raise ValueError(
            "Provide --index-csv, --data-root, and --cache-dir unless --use-demo-dataset is set."
        )
    index_csv = Path(args.index_csv)
    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (index_csv, data_root, cache_dir)


def _resolve_comparison_variants(args: argparse.Namespace) -> list[str] | None:
    if bool(args.all_variants):
        return None
    if args.variant:
        return [str(value) for value in list(args.variant)]
    comparison = load_comparison_spec(Path(args.comparison))
    return [str(comparison.allowed_variants[0].variant_id)]


def _resolve_protocol_suites(args: argparse.Namespace) -> list[str] | None:
    if bool(args.all_suites):
        return None
    if args.suite:
        return [str(value) for value in list(args.suite)]
    protocol = load_protocol(Path(args.protocol))
    enabled = [suite.suite_id for suite in protocol.official_run_suites if suite.enabled]
    if not enabled:
        raise ValueError("Selected protocol has no enabled suites.")
    return [str(enabled[0])]


def _run_comparison(
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
        force=True,
        resume=False,
        dry_run=False,
    )


def _run_confirmatory(
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
        force=True,
        resume=False,
        dry_run=False,
    )


def _determinism_for_mode(
    *,
    mode: str,
    args: argparse.Namespace,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    variant_ids: list[str] | None,
    suite_ids: list[str] | None,
    reports_root: Path,
) -> dict[str, Any]:
    mode_root = reports_root / "determinism" / mode
    run_a_root = mode_root / "run_a"
    run_b_root = mode_root / "run_b"
    for directory in (run_a_root, run_b_root):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    if mode == "comparison":
        result_a = _run_comparison(
            comparison_path=Path(args.comparison),
            reports_root=run_a_root,
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=cache_dir,
            variant_ids=variant_ids,
        )
        result_b = _run_comparison(
            comparison_path=Path(args.comparison),
            reports_root=run_b_root,
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=cache_dir,
            variant_ids=variant_ids,
        )
        output_a = Path(str(result_a["comparison_output_dir"]))
        output_b = Path(str(result_b["comparison_output_dir"]))
    else:
        result_a = _run_confirmatory(
            protocol_path=Path(args.protocol),
            reports_root=run_a_root,
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=cache_dir,
            suite_ids=suite_ids,
        )
        result_b = _run_confirmatory(
            protocol_path=Path(args.protocol),
            reports_root=run_b_root,
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=cache_dir,
            suite_ids=suite_ids,
        )
        output_a = Path(str(result_a["protocol_output_dir"]))
        output_b = Path(str(result_b["protocol_output_dir"]))

    if int(result_a.get("n_failed", 0)) > 0 or int(result_b.get("n_failed", 0)) > 0:
        return {
            "passed": False,
            "mode": mode,
            "reason": "one_or_more_runs_failed",
            "run_a": result_a,
            "run_b": result_b,
        }

    comparison_summary = compare_official_outputs(
        left_dir=output_a,
        right_dir=output_b,
    )
    return {
        "passed": bool(comparison_summary.get("passed", False)),
        "mode": mode,
        "run_a_output_dir": str(output_a.resolve()),
        "run_b_output_dir": str(output_b.resolve()),
        "comparison": comparison_summary,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if bool(args.all_variants) and bool(args.variant):
        parser.error("Use either --variant or --all-variants for comparison mode.")
    if bool(args.all_suites) and bool(args.suite):
        parser.error("Use either --suite or --all-suites for confirmatory mode.")

    mode = str(args.mode)
    run_comparison_mode = mode in {"comparison", "both"}
    run_confirmatory_mode = mode in {"confirmatory", "both"}

    index_csv, data_root, cache_dir = _resolve_dataset_paths(args)
    reports_root = Path(args.reports_root).resolve()
    reports_root.mkdir(parents=True, exist_ok=True)

    summary_out = (
        Path(args.summary_out).resolve()
        if args.summary_out is not None
        else reports_root / "replay_summary.json"
    )
    verification_summary_out = (
        Path(args.verification_summary_out).resolve()
        if args.verification_summary_out is not None
        else reports_root / "replay_verification_summary.json"
    )
    manifest_out = (
        Path(args.manifest_out).resolve()
        if args.manifest_out is not None
        else reports_root / "reproducibility_manifest.json"
    )

    variant_ids = _resolve_comparison_variants(args) if run_comparison_mode else None
    suite_ids = _resolve_protocol_suites(args) if run_confirmatory_mode else None

    mode_results: dict[str, Any] = {}
    output_dirs_by_mode: dict[str, Path] = {}

    if run_comparison_mode:
        comparison_reports_root = reports_root / "comparison"
        comparison_result = _run_comparison(
            comparison_path=Path(args.comparison),
            reports_root=comparison_reports_root,
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=cache_dir,
            variant_ids=variant_ids,
        )
        comparison_output_dir = Path(str(comparison_result["comparison_output_dir"]))
        output_dirs_by_mode["locked_comparison"] = comparison_output_dir
        artifact_check = verify_official_artifacts(
            output_dir=comparison_output_dir,
            mode="comparison",
        )
        mode_results["comparison"] = {
            **comparison_result,
            "variant_ids": variant_ids,
            "artifact_verification": artifact_check,
        }

    if run_confirmatory_mode:
        confirmatory_reports_root = reports_root / "confirmatory"
        confirmatory_result = _run_confirmatory(
            protocol_path=Path(args.protocol),
            reports_root=confirmatory_reports_root,
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=cache_dir,
            suite_ids=suite_ids,
        )
        confirmatory_output_dir = Path(str(confirmatory_result["protocol_output_dir"]))
        output_dirs_by_mode["confirmatory"] = confirmatory_output_dir
        artifact_check = verify_official_artifacts(
            output_dir=confirmatory_output_dir,
            mode="confirmatory",
        )
        confirmatory_ready_summary = None
        if not bool(args.skip_confirmatory_ready):
            confirmatory_ready_summary = verify_confirmatory_ready(
                output_dir=confirmatory_output_dir
            )
        mode_results["confirmatory"] = {
            **confirmatory_result,
            "suite_ids": suite_ids,
            "artifact_verification": artifact_check,
            "confirmatory_ready": confirmatory_ready_summary,
        }

    determinism_summary: dict[str, Any] = {
        "enabled": bool(args.verify_determinism),
        "passed": True,
        "by_mode": {},
    }
    if bool(args.verify_determinism):
        if run_comparison_mode:
            result = _determinism_for_mode(
                mode="comparison",
                args=args,
                index_csv=index_csv,
                data_root=data_root,
                cache_dir=cache_dir,
                variant_ids=variant_ids,
                suite_ids=suite_ids,
                reports_root=reports_root,
            )
            determinism_summary["by_mode"]["comparison"] = result
        if run_confirmatory_mode:
            result = _determinism_for_mode(
                mode="confirmatory",
                args=args,
                index_csv=index_csv,
                data_root=data_root,
                cache_dir=cache_dir,
                variant_ids=variant_ids,
                suite_ids=suite_ids,
                reports_root=reports_root,
            )
            determinism_summary["by_mode"]["confirmatory"] = result
        determinism_summary["passed"] = all(
            bool(value.get("passed", False))
            for value in determinism_summary["by_mode"].values()
            if isinstance(value, dict)
        )

    replay_summary = {
        "schema_version": "official-replay-summary-v1",
        "mode": mode,
        "index_csv": str(index_csv.resolve()),
        "data_root": str(data_root.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "comparison_spec": (
            str(Path(args.comparison).resolve()) if run_comparison_mode else None
        ),
        "protocol_spec": (
            str(Path(args.protocol).resolve()) if run_confirmatory_mode else None
        ),
        "results": mode_results,
        "determinism": determinism_summary,
    }

    verification_checks: dict[str, Any] = {}
    for mode_name, mode_payload in mode_results.items():
        mode_ok = int(mode_payload.get("n_failed", 0)) == 0
        artifact_ok = bool(
            mode_payload.get("artifact_verification", {}).get("passed", False)
        )
        confirmatory_ready_ok = True
        if mode_name == "confirmatory" and not bool(args.skip_confirmatory_ready):
            confirmatory_ready_ok = bool(
                mode_payload.get("confirmatory_ready", {}).get("passed", False)
            )
        verification_checks[mode_name] = {
            "execution_passed": mode_ok,
            "artifact_verification_passed": artifact_ok,
            "confirmatory_ready_passed": confirmatory_ready_ok,
        }

    overall_passed = all(
        bool(check["execution_passed"])
        and bool(check["artifact_verification_passed"])
        and bool(check["confirmatory_ready_passed"])
        for check in verification_checks.values()
    )
    if bool(args.verify_determinism):
        overall_passed = bool(overall_passed and determinism_summary["passed"])

    replay_verification_summary = {
        "schema_version": "official-replay-verification-summary-v1",
        "passed": bool(overall_passed),
        "checks": verification_checks,
        "determinism": determinism_summary,
    }

    manifest_payload = build_reproducibility_manifest(
        output_dirs_by_mode=output_dirs_by_mode,
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        comparison_spec_path=(Path(args.comparison) if run_comparison_mode else None),
        protocol_path=(Path(args.protocol) if run_confirmatory_mode else None),
        replay_summary=replay_summary,
        replay_verification_summary=replay_verification_summary,
        bundle_dir=None,
        repo_root=PROJECT_ROOT,
    )

    _write_json(summary_out, replay_summary)
    _write_json(verification_summary_out, replay_verification_summary)
    write_reproducibility_manifest(
        manifest=manifest_payload,
        output_path=manifest_out,
    )

    console_summary = {
        "passed": bool(overall_passed),
        "replay_summary": str(summary_out.resolve()),
        "replay_verification_summary": str(verification_summary_out.resolve()),
        "reproducibility_manifest": str(manifest_out.resolve()),
    }
    print(json.dumps(console_summary, indent=2))
    if not bool(overall_passed):
        print("Official replay verification failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
