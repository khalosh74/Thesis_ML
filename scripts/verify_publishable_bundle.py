from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from Thesis_ML.verification.confirmatory_ready import verify_confirmatory_ready
from Thesis_ML.verification.official_artifacts import verify_official_artifacts

EXPECTED_SCHEMA_VERSION = "publishable-bundle-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify canonical publishable artifact bundle structure and hashes."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Bundle root directory.")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional JSON path for verification summary output.",
    )
    return parser


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
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
    if schema_version != EXPECTED_SCHEMA_VERSION:
        issues.append(
            {
                "code": "bundle_manifest_schema_version_invalid",
                "message": (
                    f"Expected manifest_schema_version='{EXPECTED_SCHEMA_VERSION}', "
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
        actual_hash = _file_sha256(candidate)
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

    replay_verification_rel = str(
        verification_files.get("replay_verification_summary", "")
    ).strip()
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = verify_bundle(Path(args.bundle_dir))
    if args.summary_out is not None:
        output_path = Path(args.summary_out).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{json.dumps(summary, indent=2)}\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not bool(summary.get("passed", False)):
        print("Publishable bundle verification failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
