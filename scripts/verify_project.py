from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from Thesis_ML.script_support.io import file_sha256
from Thesis_ML.verification.campaign_runtime_profile import verify_campaign_runtime_profile
from Thesis_ML.verification.confirmatory_ready import verify_confirmatory_ready
from Thesis_ML.verification.model_cost_policy import verify_model_cost_policy_precheck
from Thesis_ML.verification.official_artifacts import verify_official_artifacts

EXPECTED_BUNDLE_SCHEMA_VERSION = "publishable-bundle-v1"


def _write_summary(path_text: str | None, payload: dict[str, Any]) -> None:
    if not path_text:
        return
    summary_path = Path(path_text)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _run_official_artifacts(args: argparse.Namespace) -> int:
    summary = verify_official_artifacts(
        output_dir=Path(args.output_dir),
        mode=args.mode,
    )
    _write_summary(args.summary_out, summary)
    print(json.dumps(summary, indent=2))
    if not bool(summary.get("passed", False)):
        print("Official artifact verification failed.", file=sys.stderr)
        return 1
    return 0


def _run_confirmatory_ready(args: argparse.Namespace) -> int:
    summary = verify_confirmatory_ready(
        output_dir=Path(args.output_dir),
        reproducibility_summary=Path(args.repro_summary) if args.repro_summary else None,
    )
    _write_summary(args.summary_out, summary)
    print(json.dumps(summary, indent=2))
    if not bool(summary.get("passed", False)):
        print("Confirmatory-ready verification failed.", file=sys.stderr)
        return 1
    return 0


def _run_model_cost_policy_precheck(args: argparse.Namespace) -> int:
    summary = verify_model_cost_policy_precheck(
        index_csv=Path(args.index_csv),
        confirmatory_protocol=Path(args.confirmatory_protocol),
        comparison_specs=[Path(path) for path in list(args.comparison_spec)],
    )
    _write_summary(args.summary_out, summary)
    print(json.dumps(summary, indent=2))
    if not bool(summary.get("passed", False)):
        print("Model-cost policy precheck failed.", file=sys.stderr)
        return 1
    return 0


def verify_publishable_bundle(bundle_dir: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    root = bundle_dir.resolve()
    manifest_path = root / "bundle_manifest.json"
    if not root.exists() or not root.is_dir():
        return {
            "passed": False,
            "bundle_dir": str(root),
            "issues": [{"code": "bundle_dir_missing", "message": "Bundle directory is missing."}],
        }
    if not manifest_path.exists() or not manifest_path.is_file():
        return {
            "passed": False,
            "bundle_dir": str(root),
            "issues": [
                {
                    "code": "bundle_manifest_missing",
                    "message": "bundle_manifest.json is missing.",
                }
            ],
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "bundle_dir": str(root),
            "issues": [
                {
                    "code": "bundle_manifest_invalid_json",
                    "message": f"Invalid JSON in bundle_manifest.json: {exc}",
                }
            ],
        }
    if not isinstance(manifest, dict):
        return {
            "passed": False,
            "bundle_dir": str(root),
            "issues": [
                {
                    "code": "bundle_manifest_invalid_shape",
                    "message": "bundle_manifest.json must contain a JSON object.",
                }
            ],
        }

    schema_version = str(manifest.get("manifest_schema_version", ""))
    if schema_version != EXPECTED_BUNDLE_SCHEMA_VERSION:
        issues.append(
            {
                "code": "bundle_manifest_schema_version_invalid",
                "message": (
                    f"Expected manifest_schema_version='{EXPECTED_BUNDLE_SCHEMA_VERSION}', "
                    f"got '{schema_version}'."
                ),
            }
        )

    files = manifest.get("files")
    if not isinstance(files, list):
        issues.append(
            {
                "code": "bundle_files_missing",
                "message": "bundle_manifest.json must include list field 'files'.",
            }
        )
        files = []

    n_files_checked = 0
    for entry in files:
        if not isinstance(entry, dict):
            issues.append(
                {
                    "code": "bundle_file_entry_invalid",
                    "message": "Each files entry must be an object.",
                }
            )
            continue
        relative_path = str(entry.get("path", "")).strip()
        expected_hash = str(entry.get("sha256", "")).strip().lower()
        if not relative_path:
            issues.append(
                {
                    "code": "bundle_file_entry_path_missing",
                    "message": "files entry is missing 'path'.",
                }
            )
            continue
        candidate = root / relative_path
        if not candidate.exists() or not candidate.is_file():
            issues.append(
                {
                    "code": "bundle_file_missing",
                    "message": f"Listed bundle file is missing: {relative_path}",
                }
            )
            continue
        actual_hash = file_sha256(candidate)
        n_files_checked += 1
        if expected_hash != actual_hash:
            issues.append(
                {
                    "code": "bundle_file_hash_mismatch",
                    "message": f"Hash mismatch for {relative_path}.",
                    "details": {"expected": expected_hash, "actual": actual_hash},
                }
            )

    official_outputs = manifest.get("official_outputs")
    if not isinstance(official_outputs, dict):
        issues.append(
            {
                "code": "bundle_official_outputs_missing",
                "message": "bundle_manifest.json must include object field 'official_outputs'.",
            }
        )
        official_outputs = {}

    spec_files = manifest.get("spec_files")
    if not isinstance(spec_files, dict):
        issues.append(
            {
                "code": "bundle_spec_files_missing",
                "message": "bundle_manifest.json must include object field 'spec_files'.",
            }
        )
        spec_files = {}

    required_spec_keys = ("comparison_spec", "confirmatory_protocol")
    if any(spec_files.get(key) for key in required_spec_keys):
        spec_registry_identity = manifest.get("spec_registry_identity")
        if not isinstance(spec_registry_identity, dict):
            issues.append(
                {
                    "code": "bundle_spec_registry_identity_missing",
                    "message": (
                        "bundle_manifest.json must include object field "
                        "'spec_registry_identity' when spec_files are present."
                    ),
                }
            )
        else:
            for key in required_spec_keys:
                if not spec_files.get(key):
                    continue
                if key not in spec_registry_identity:
                    issues.append(
                        {
                            "code": "bundle_spec_registry_identity_entry_missing",
                            "message": f"spec_registry_identity.{key} is required when spec_files.{key} is present.",
                        }
                    )
                    continue
                identity_entry = spec_registry_identity.get(key)
                if identity_entry is None:
                    continue
                if not isinstance(identity_entry, dict):
                    issues.append(
                        {
                            "code": "bundle_spec_registry_identity_entry_invalid",
                            "message": f"spec_registry_identity.{key} must be null or an object.",
                        }
                    )
                    continue
                for required_field in ("registered", "path", "aliases"):
                    if required_field not in identity_entry:
                        issues.append(
                            {
                                "code": "bundle_spec_registry_identity_entry_field_missing",
                                "message": (
                                    f"spec_registry_identity.{key} is missing required field "
                                    f"'{required_field}'."
                                ),
                            }
                        )
    if all(spec_files.get(key) for key in required_spec_keys):
        spec_bundle_validation = manifest.get("spec_bundle_validation")
        if not isinstance(spec_bundle_validation, dict):
            issues.append(
                {
                    "code": "bundle_spec_bundle_validation_missing",
                    "message": (
                        "bundle_manifest.json must include object field "
                        "'spec_bundle_validation' when both spec_files are present."
                    ),
                }
            )
        else:
            if not bool(spec_bundle_validation.get("valid", False)):
                issues.append(
                    {
                        "code": "bundle_spec_bundle_validation_invalid",
                        "message": "spec_bundle_validation.valid must be true when both specs are present.",
                    }
                )
            matched_bundle_ids = spec_bundle_validation.get("matched_bundle_ids")
            if not isinstance(matched_bundle_ids, list) or len(matched_bundle_ids) == 0:
                issues.append(
                    {
                        "code": "bundle_spec_bundle_validation_matches_missing",
                        "message": (
                            "spec_bundle_validation.matched_bundle_ids must be a non-empty list "
                            "when both specs are present."
                        ),
                    }
                )

    official_verifications: dict[str, Any] = {}
    for key, mode in (("comparison", "comparison"), ("confirmatory", "confirmatory")):
        section = official_outputs.get(key)
        if not isinstance(section, dict):
            continue
        output_dir_text = str(section.get("bundle_output_dir", "")).strip()
        if not output_dir_text:
            issues.append(
                {
                    "code": "bundle_official_output_dir_missing",
                    "message": f"official_outputs.{key}.bundle_output_dir is missing.",
                }
            )
            continue
        output_dir = Path(output_dir_text)
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        summary = verify_official_artifacts(output_dir=output_dir, mode=mode)
        official_verifications[key] = summary
        if not bool(summary.get("passed", False)):
            issues.append(
                {
                    "code": f"bundle_{key}_official_verification_failed",
                    "message": f"Official artifact verification failed for bundled {key} output.",
                    "details": {"issues": summary.get("issues", [])},
                }
            )

    verification_files = manifest.get("verification_files")
    if not isinstance(verification_files, dict):
        issues.append(
            {
                "code": "bundle_verification_files_missing",
                "message": "bundle_manifest.json must include object field 'verification_files'.",
            }
        )
        verification_files = {}

    replay_verification_rel = str(verification_files.get("replay_verification_summary", "")).strip()
    if not replay_verification_rel:
        issues.append(
            {
                "code": "bundle_replay_verification_missing",
                "message": "verification_files.replay_verification_summary is required.",
            }
        )
    else:
        replay_verification_path = root / replay_verification_rel
        payload = None
        if replay_verification_path.exists():
            try:
                payload = json.loads(replay_verification_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = None
        if not isinstance(payload, dict):
            issues.append(
                {
                    "code": "bundle_replay_verification_invalid",
                    "message": "replay_verification_summary file must contain a JSON object.",
                }
            )
        elif not bool(payload.get("passed", False)):
            issues.append(
                {
                    "code": "bundle_replay_verification_failed",
                    "message": "replay_verification_summary reports passed=false.",
                }
            )

    confirmatory_ready_rel = str(verification_files.get("confirmatory_ready_summary", "")).strip()
    if "confirmatory" in official_outputs and confirmatory_ready_rel:
        confirmatory_ready_path = root / confirmatory_ready_rel
        payload = None
        if confirmatory_ready_path.exists():
            try:
                payload = json.loads(confirmatory_ready_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = None
        if not isinstance(payload, dict):
            issues.append(
                {
                    "code": "bundle_confirmatory_ready_invalid",
                    "message": "confirmatory_ready_summary must contain a JSON object.",
                }
            )
        elif not bool(payload.get("passed", False)):
            issues.append(
                {
                    "code": "bundle_confirmatory_ready_failed",
                    "message": "confirmatory_ready_summary reports passed=false.",
                }
            )

        output_dir_text = str(
            official_outputs.get("confirmatory", {}).get("bundle_output_dir", "")
        ).strip()
        if output_dir_text:
            ready_output_dir = Path(output_dir_text)
            if not ready_output_dir.is_absolute():
                ready_output_dir = root / ready_output_dir
            ready_summary = verify_confirmatory_ready(output_dir=ready_output_dir)
            if not bool(ready_summary.get("passed", False)):
                issues.append(
                    {
                        "code": "bundle_confirmatory_ready_recheck_failed",
                        "message": "Bundled confirmatory output failed confirmatory-ready recheck.",
                        "details": {"issues": ready_summary.get("issues", [])},
                    }
                )

    return {
        "passed": not issues,
        "bundle_dir": str(root),
        "n_files_checked": int(n_files_checked),
        "official_verifications": official_verifications,
        "issues": issues,
    }


def _run_publishable_bundle(args: argparse.Namespace) -> int:
    summary = verify_publishable_bundle(Path(args.bundle_dir))
    _write_summary(args.summary_out, summary)
    print(json.dumps(summary, indent=2))
    if not bool(summary.get("passed", False)):
        print("Publishable bundle verification failed.", file=sys.stderr)
        return 1
    return 0


class _ConsoleProgressReporter:
    def __init__(self) -> None:
        self._started = perf_counter()

    @staticmethod
    def _humanize_seconds(total_seconds: float | None) -> str:
        if total_seconds is None:
            return "unknown"
        seconds = max(0, int(round(float(total_seconds))))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def __call__(self, event: Any) -> None:
        now = perf_counter()
        elapsed = now - self._started
        fraction = None
        eta_seconds = None
        remaining_units = None

        if (
            event.completed_units is not None
            and event.total_units is not None
            and event.total_units > 0
        ):
            completed = float(event.completed_units)
            total = float(event.total_units)
            remaining_units = max(0.0, total - completed)
            if completed > 0.0:
                fraction = min(max(completed / total, 0.0), 1.0)
                eta_seconds = elapsed * (1.0 - fraction) / fraction

        percent_text = "--.-%"
        done_text = "?"
        left_text = "?"
        if (
            fraction is not None
            and event.total_units is not None
            and event.completed_units is not None
        ):
            percent_text = f"{100.0 * fraction:5.1f}%"
            done_text = f"{int(event.completed_units)}/{int(event.total_units)}"
            left_text = str(int(remaining_units or 0.0))

        phase = event.metadata.get("phase")
        model = event.metadata.get("model")
        section = event.metadata.get("section")
        fold_index = event.metadata.get("fold_index")
        total_folds = event.metadata.get("total_folds")
        lane = event.metadata.get("assigned_compute_lane")
        backend = (
            event.metadata.get("actual_estimator_backend_family")
            or event.metadata.get("assigned_backend_family")
            or event.metadata.get("effective_backend_family")
        )
        backend_id = event.metadata.get("actual_estimator_backend_id") or event.metadata.get(
            "backend_id"
        )
        hardware = event.metadata.get("hardware_mode_effective")
        hardware_requested = event.metadata.get("hardware_mode_requested")

        suffix_parts: list[str] = []
        if phase is not None:
            suffix_parts.append(f"phase={phase}")
        if model is not None:
            suffix_parts.append(f"model={model}")
        if hardware is not None:
            suffix_parts.append(f"hw={hardware}")
        if hardware_requested is not None:
            suffix_parts.append(f"hw_req={hardware_requested}")
        if lane is not None:
            suffix_parts.append(f"lane={lane}")
        if backend is not None:
            suffix_parts.append(f"backend={backend}")
        if backend_id is not None:
            suffix_parts.append(f"estimator_backend={backend_id}")
        if section is not None:
            suffix_parts.append(f"section={section}")
        if fold_index is not None and total_folds is not None:
            suffix_parts.append(f"fold={fold_index}/{total_folds}")

        suffix = " | ".join(suffix_parts)

        line = (
            f"[{event.stage}] {event.message} | done {done_text} | left {left_text} "
            f"| elapsed {self._humanize_seconds(elapsed)} "
            f"| eta {self._humanize_seconds(eta_seconds)} "
            f"| {percent_text}"
        )
        if suffix:
            line = f"{line} | {suffix}"

        print(line, file=sys.stderr, flush=True)


def _run_campaign_runtime_profile(args: argparse.Namespace) -> int:
    progress_callback = None if args.quiet_progress else _ConsoleProgressReporter()
    summary = verify_campaign_runtime_profile(
        index_csv=Path(args.index_csv),
        data_root=Path(args.data_root),
        cache_dir=Path(args.cache_dir),
        confirmatory_protocol=Path(args.confirmatory_protocol),
        comparison_specs=[Path(path) for path in list(args.comparison_spec)],
        profile_root=Path(args.profile_root),
        hardware_mode=args.hardware_mode,
        gpu_device_id=args.gpu_device_id,
        deterministic_compute=bool(args.deterministic_compute),
        allow_backend_fallback=bool(args.allow_backend_fallback),
        profile_permutations=args.profile_permutations,
        profile_inner_folds=args.profile_inner_folds,
        profile_tuning_candidates=args.profile_tuning_candidates,
        progress_callback=progress_callback,
    )

    _write_summary(args.summary_out, summary)
    print(json.dumps(summary, indent=2))
    if not bool(summary.get("passed", False)):
        print("Campaign runtime profile precheck failed.", file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonical project verification entrypoint with focused subcommands."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    official = subparsers.add_parser(
        "official-artifacts",
        help="Verify official artifact completeness/metadata.",
    )
    official.add_argument(
        "--output-dir",
        required=True,
        help="Path to comparison/protocol output directory containing execution_status.json.",
    )
    official.add_argument(
        "--mode",
        choices=["comparison", "confirmatory", "protocol", "locked_comparison"],
        default=None,
        help="Optional mode hint; omitted means auto-detect from execution_status.json.",
    )
    official.add_argument(
        "--summary-out",
        default="",
        help="Optional JSON path for verification summary output.",
    )

    confirm = subparsers.add_parser(
        "confirmatory-ready",
        help="Verify governance-level confirmatory-ready criteria.",
    )
    confirm.add_argument(
        "--output-dir",
        required=True,
        help="Path to confirmatory output directory containing execution_status.json.",
    )
    confirm.add_argument(
        "--repro-summary",
        default="",
        help="Optional reproducibility summary JSON path.",
    )
    confirm.add_argument(
        "--summary-out",
        default="",
        help="Optional JSON path for confirmatory-ready summary output.",
    )

    cost = subparsers.add_parser(
        "model-cost-policy-precheck",
        help="Validate model-cost policy constraints for official specs.",
    )
    cost.add_argument("--index-csv", required=True, help="Dataset index CSV path.")
    cost.add_argument(
        "--confirmatory-protocol",
        required=True,
        help="Confirmatory protocol JSON path.",
    )
    cost.add_argument(
        "--comparison-spec",
        action="append",
        required=True,
        help="Comparison spec JSON path. Repeat for multiple specs.",
    )
    cost.add_argument(
        "--summary-out",
        default="",
        help="Optional JSON summary output path.",
    )

    bundle = subparsers.add_parser(
        "publishable-bundle",
        help="Verify publishable bundle structure, hashes, and prerequisite summaries.",
    )
    bundle.add_argument("--bundle-dir", type=Path, required=True, help="Bundle root directory.")
    bundle.add_argument(
        "--summary-out",
        default="",
        help="Optional JSON path for verification summary output.",
    )

    runtime = subparsers.add_parser(
        "campaign-runtime-profile",
        help="Run precheck-only runtime profiling and ETA verification.",
    )
    runtime.add_argument("--index-csv", required=True, help="Dataset index CSV path.")
    runtime.add_argument("--data-root", required=True, help="Dataset data root path.")
    runtime.add_argument("--cache-dir", required=True, help="Feature cache directory path.")
    runtime.add_argument(
        "--confirmatory-protocol",
        required=True,
        help="Confirmatory protocol JSON path.",
    )
    runtime.add_argument(
        "--comparison-spec",
        action="append",
        required=True,
        help="Comparison spec JSON path. Repeat for multiple specs.",
    )
    runtime.add_argument(
        "--profile-root",
        required=True,
        help="Output root for runtime profiling precheck runs.",
    )
    runtime.add_argument(
        "--summary-out",
        default="",
        help="Optional JSON summary output path.",
    )
    runtime.add_argument(
        "--hardware-mode",
        choices=["cpu_only", "gpu_only", "max_both"],
        default="gpu_only",
        help="Compute hardware mode for profiling runs.",
    )
    runtime.add_argument(
        "--gpu-device-id",
        type=int,
        default=None,
        help="CUDA device id to use for profiling runs.",
    )
    runtime.add_argument(
        "--deterministic-compute",
        action="store_true",
        help="Enable deterministic GPU execution for profiling runs.",
    )
    runtime.add_argument(
        "--allow-backend-fallback",
        action="store_true",
        help="Allow fallback to CPU if GPU is unavailable in exploratory profiling.",
    )
    runtime.add_argument(
        "--profile-permutations",
        type=int,
        default=None,
        help="Optional profiling-only permutation override.",
    )
    runtime.add_argument(
        "--profile-inner-folds",
        type=int,
        default=None,
        help="Optional profiling-only grouped-nested inner-fold cap.",
    )
    runtime.add_argument(
        "--profile-tuning-candidates",
        type=int,
        default=None,
        help="Optional profiling-only grouped-nested candidate cap.",
    )
    runtime.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Disable live progress messages on stderr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers: dict[str, Any] = {
        "official-artifacts": _run_official_artifacts,
        "confirmatory-ready": _run_confirmatory_ready,
        "model-cost-policy-precheck": _run_model_cost_policy_precheck,
        "publishable-bundle": _run_publishable_bundle,
        "campaign-runtime-profile": _run_campaign_runtime_profile,
    }
    handler = handlers[args.subcommand]
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
