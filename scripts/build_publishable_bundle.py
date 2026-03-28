from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from Thesis_ML.config import describe_config_path, validate_config_bundle
from Thesis_ML.config.paths import (
    DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH,
    DEFAULT_COMPARISON_SPEC_PATH,
    DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH,
    PROJECT_ROOT,
)
from Thesis_ML.config.runtime_selection import resolve_runtime_config_path

BUNDLE_SCHEMA_VERSION = "publishable-bundle-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _rewrite_report_index_with_local_runs(copied_output_dir: Path) -> list[str]:
    report_index_path = copied_output_dir / "report_index.csv"
    if not report_index_path.exists():
        return []

    with report_index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{str(k): str(v) for k, v in row.items()} for row in reader]
        fieldnames = list(reader.fieldnames or [])

    copied_run_ids: list[str] = []
    run_reports_dir = copied_output_dir / "run_reports"
    run_reports_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        run_id = str(row.get("run_id", "")).strip()
        report_dir_raw = str(row.get("report_dir", "")).strip()
        if not run_id or not report_dir_raw:
            continue
        source_report_dir = Path(report_dir_raw)
        if not source_report_dir.is_absolute():
            source_report_dir = copied_output_dir / source_report_dir
        if not source_report_dir.exists() or not source_report_dir.is_dir():
            continue
        target_report_dir = run_reports_dir / run_id
        if target_report_dir.exists():
            shutil.rmtree(target_report_dir)
        shutil.copytree(source_report_dir, target_report_dir)
        relative_report_dir = str(target_report_dir.relative_to(copied_output_dir).as_posix())
        row["report_dir"] = relative_report_dir
        row["report_dir_relative"] = relative_report_dir
        copied_run_ids.append(run_id)

    if not fieldnames:
        fieldnames = ["run_id", "report_dir", "report_dir_relative"]
    for required in ("report_dir", "report_dir_relative"):
        if required not in fieldnames:
            fieldnames.append(required)

    with report_index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return sorted(set(copied_run_ids))


def _copy_official_output(
    *,
    source_dir: Path,
    destination_root: Path,
    bundle_root: Path,
    label: str,
) -> dict[str, Any]:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Official output directory does not exist: {source_dir}")
    target_dir = destination_root / label / "output"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    copied_run_ids = _rewrite_report_index_with_local_runs(target_dir)
    return {
        "label": label,
        "source_dir": str(source_dir.resolve()),
        "bundle_output_dir": str(target_dir.relative_to(bundle_root).as_posix()),
        "copied_run_ids": copied_run_ids,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build canonical publishable artifact bundle (directory + manifest)."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Bundle output directory.")
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=None,
        help="Comparison mode output directory to include.",
    )
    parser.add_argument(
        "--confirmatory-output",
        type=Path,
        default=None,
        help="Confirmatory mode output directory to include.",
    )
    parser.add_argument(
        "--replay-summary",
        type=Path,
        default=None,
        help="Optional replay_summary.json to include.",
    )
    parser.add_argument(
        "--replay-verification-summary",
        type=Path,
        default=None,
        help="Optional replay_verification_summary.json to include.",
    )
    parser.add_argument(
        "--repro-manifest",
        type=Path,
        default=None,
        help="Optional reproducibility_manifest.json to include.",
    )
    parser.add_argument(
        "--confirmatory-ready-summary",
        type=Path,
        default=None,
        help="Optional confirmatory-ready summary JSON to include.",
    )
    parser.add_argument(
        "--comparison-spec",
        type=Path,
        default=None,
        help="Comparison spec snapshot path.",
    )
    parser.add_argument(
        "--comparison-spec-alias",
        default=None,
        help="Registry alias for comparison spec snapshot selection when --comparison-spec is not provided.",
    )
    parser.add_argument(
        "--protocol-spec",
        type=Path,
        default=None,
        help="Confirmatory protocol snapshot path.",
    )
    parser.add_argument(
        "--protocol-spec-alias",
        default=None,
        help="Registry alias for protocol spec snapshot selection when --protocol-spec is not provided.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.comparison_output is None and args.confirmatory_output is None:
        parser.error("Provide at least one of --comparison-output or --confirmatory-output.")

    bundle_root = Path(args.output_dir).resolve()
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)

    official_root = bundle_root / "official_outputs"
    verification_root = bundle_root / "verification"
    specs_root = bundle_root / "specs"
    governance_root = bundle_root / "governance"
    official_root.mkdir(parents=True, exist_ok=True)
    verification_root.mkdir(parents=True, exist_ok=True)
    specs_root.mkdir(parents=True, exist_ok=True)
    governance_root.mkdir(parents=True, exist_ok=True)
    comparison_spec_path = resolve_runtime_config_path(
        args.comparison_spec,
        args.comparison_spec_alias,
        default_alias="comparison.grouped_nested_default",
        fallback_path=DEFAULT_COMPARISON_SPEC_PATH,
    )
    protocol_spec_path = resolve_runtime_config_path(
        args.protocol_spec,
        args.protocol_spec_alias,
        default_alias="protocol.thesis_confirmatory_frozen",
        fallback_path=DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH,
    )
    if comparison_spec_path is not None and protocol_spec_path is not None:
        spec_bundle_validation: dict[str, Any] | None = validate_config_bundle(
            protocol_path=protocol_spec_path,
            comparison_path=comparison_spec_path,
            target_path=DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH,
        )
        if not bool(spec_bundle_validation.get("valid", False)):
            errors = spec_bundle_validation.get("errors", [])
            raise ValueError(f"Invalid publishable bundle config combination: {errors}")
    else:
        spec_bundle_validation = None

    official_outputs: dict[str, Any] = {}
    if args.comparison_output is not None:
        official_outputs["comparison"] = _copy_official_output(
            source_dir=Path(args.comparison_output).resolve(),
            destination_root=official_root,
            bundle_root=bundle_root,
            label="comparison",
        )
    if args.confirmatory_output is not None:
        official_outputs["confirmatory"] = _copy_official_output(
            source_dir=Path(args.confirmatory_output).resolve(),
            destination_root=official_root,
            bundle_root=bundle_root,
            label="confirmatory",
        )

    verification_files: dict[str, str] = {}
    for key, candidate in (
        ("replay_summary", args.replay_summary),
        ("replay_verification_summary", args.replay_verification_summary),
        ("reproducibility_manifest", args.repro_manifest),
        ("confirmatory_ready_summary", args.confirmatory_ready_summary),
    ):
        if candidate is None:
            continue
        source = Path(candidate).resolve()
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"{key} file does not exist: {source}")
        destination = _copy_file(source, verification_root / source.name)
        verification_files[key] = str(destination.relative_to(bundle_root).as_posix())

    spec_files: dict[str, str] = {}
    for key, candidate in (
        ("comparison_spec", comparison_spec_path),
        ("confirmatory_protocol", protocol_spec_path),
    ):
        source = Path(candidate).resolve()
        if not source.exists() or not source.is_file():
            continue
        destination = _copy_file(source, specs_root / source.name)
        spec_files[key] = str(destination.relative_to(bundle_root).as_posix())

    spec_registry_identity = {
        "comparison_spec": (
            describe_config_path(comparison_spec_path) if comparison_spec_path is not None else None
        ),
        "confirmatory_protocol": (
            describe_config_path(protocol_spec_path) if protocol_spec_path is not None else None
        ),
    }

    governance_candidates = [
        PROJECT_ROOT / "LICENSE",
        PROJECT_ROOT / "CITATION.cff",
        PROJECT_ROOT / "docs" / "PRIVACY_AND_DATA_HANDLING.md",
        PROJECT_ROOT / "docs" / "USE_AND_MISUSE_BOUNDARIES.md",
        PROJECT_ROOT / "docs" / "CONFIRMATORY_READY.md",
        PROJECT_ROOT / "docs" / "RELEASE.md",
        PROJECT_ROOT / "docs" / "REPRODUCIBILITY.md",
    ]
    governance_files: list[str] = []
    for source in governance_candidates:
        if not source.exists() or not source.is_file():
            continue
        destination = _copy_file(source, governance_root / source.name)
        governance_files.append(str(destination.relative_to(bundle_root).as_posix()))

    manifest_path = bundle_root / "bundle_manifest.json"
    files: list[dict[str, Any]] = []
    for candidate in sorted(bundle_root.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate == manifest_path:
            continue
        relative = candidate.relative_to(bundle_root).as_posix()
        files.append(
            {
                "path": relative,
                "sha256": _file_sha256(candidate),
                "size_bytes": int(candidate.stat().st_size),
            }
        )

    manifest = {
        "manifest_schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "bundle_root": str(bundle_root),
        "official_outputs": official_outputs,
        "verification_files": verification_files,
        "spec_files": spec_files,
        "spec_registry_identity": spec_registry_identity,
        "spec_bundle_validation": spec_bundle_validation,
        "governance_files": governance_files,
        "files": files,
    }
    manifest_path.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")

    summary = {
        "bundle_dir": str(bundle_root),
        "bundle_manifest": str(manifest_path),
        "n_files_hashed": int(len(files)),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
