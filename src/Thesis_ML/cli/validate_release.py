from __future__ import annotations

import argparse
import json
from pathlib import Path

from Thesis_ML.release.evidence import verify_release_evidence
from Thesis_ML.release.manifests import read_run_manifest
from Thesis_ML.release.validator import validate_release


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate release bundle + dataset manifest contracts, with optional run evidence check."
        )
    )
    parser.add_argument(
        "--release",
        required=True,
        help="Release bundle path or release alias.",
    )
    parser.add_argument(
        "--dataset-manifest",
        required=True,
        help="Dataset instance manifest path.",
    )
    parser.add_argument(
        "--strict-environment",
        action="store_true",
        help="Enable strict environment-policy checks (python/uv.lock/scripts/os).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory to verify evidence artifacts against release contract.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    context = validate_release(
        release_ref=args.release,
        dataset_manifest_path=args.dataset_manifest,
        strict_environment=bool(args.strict_environment),
    )
    summary: dict[str, object] = {
        "passed": bool(context.passed),
        "release_id": context.release.release.release_id,
        "release_version": context.release.release.release_version,
        "release_path": str(context.release.release_path.resolve()),
        "dataset_manifest_path": str(context.dataset.manifest_path.resolve()),
        "issues": context.issues,
    }

    if args.run_dir is not None:
        run_manifest = read_run_manifest(Path(args.run_dir) / "run_manifest.json")
        evidence_summary = verify_release_evidence(
            run_dir=Path(args.run_dir),
            release=context.release,
            dataset=context.dataset,
            run_manifest=run_manifest,
            allow_missing_evidence_verification=False,
            write_output=False,
        )
        summary["run_evidence_verification"] = evidence_summary
        summary["passed"] = bool(summary["passed"]) and bool(evidence_summary.get("passed", False))

    print(json.dumps(summary, indent=2))
    return 0 if bool(summary.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())

