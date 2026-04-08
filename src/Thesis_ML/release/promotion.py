from __future__ import annotations

import json
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from Thesis_ML.release.evidence import load_release_verification
from Thesis_ML.release.loader import load_release_bundle
from Thesis_ML.release.manifests import (
    build_promotion_manifest,
    read_run_manifest,
    write_json,
)
from Thesis_ML.release.models import RunClass, RunStatus
from Thesis_ML.release.paths import (
    find_candidate_run_by_id,
    list_official_runs_for_release,
    resolve_run_paths,
)


def _make_read_only(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        # Remove write bits for owner/group/other where supported.
        path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    except Exception:
        return


def _copy_approved_artifacts(candidate_run_dir: Path, official_run_dir: Path) -> None:
    approved_files = [
        "release_manifest.json",
        "release_summary.json",
        "dataset_snapshot.json",
        "warnings.json",
        "deviations.json",
        "evidence_verification.json",
    ]
    approved_dirs = ["artifacts", "verification", "logs"]

    for filename in approved_files:
        src = candidate_run_dir / filename
        if src.exists():
            shutil.copy2(src, official_run_dir / filename)
    for dirname in approved_dirs:
        src_dir = candidate_run_dir / dirname
        dst_dir = official_run_dir / dirname
        if src_dir.exists() and src_dir.is_dir():
            shutil.copytree(src_dir, dst_dir)


def _official_release_manifest_payload(
    *,
    release_manifest_payload: dict[str, Any],
    candidate_run_id: str,
    official_run_id: str,
) -> dict[str, Any]:
    payload = dict(release_manifest_payload)
    payload["schema_version"] = "official-release-manifest-v1"
    payload["candidate_run_id"] = candidate_run_id
    payload["official_run_id"] = official_run_id
    payload["promoted_at_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    return payload


def promote_candidate_run(candidate_run: Path | str) -> dict[str, Any]:
    candidate_run_dir = find_candidate_run_by_id(str(candidate_run))
    candidate_manifest_path = candidate_run_dir / "run_manifest.json"
    if not candidate_manifest_path.exists():
        raise FileNotFoundError(f"Missing candidate run manifest: {candidate_manifest_path}")
    candidate_manifest = read_run_manifest(candidate_manifest_path)

    if candidate_manifest.run_class != RunClass.CANDIDATE:
        raise ValueError(
            "Promotion requires a candidate run. "
            f"Received run_class='{candidate_manifest.run_class.value}'."
        )
    if not bool(candidate_manifest.promotable):
        raise ValueError("Candidate run is not promotable (promotable=false).")
    if not bool(candidate_manifest.evidence_verified):
        raise ValueError("Candidate run evidence_verified is false.")

    evidence_verification = load_release_verification(candidate_run_dir)
    if not isinstance(evidence_verification, dict) or not bool(
        evidence_verification.get("passed", False)
    ):
        raise ValueError("Candidate run evidence verification is missing or failed.")

    release_manifest_path = candidate_run_dir / "release_manifest.json"
    if not release_manifest_path.exists():
        raise FileNotFoundError(f"Missing candidate release manifest: {release_manifest_path}")
    release_manifest_payload = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(release_manifest_payload, dict):
        raise ValueError("Candidate release_manifest.json must be a JSON object.")

    release_json_path = Path(str(release_manifest_payload.get("release_json_path", ""))).resolve()
    if not release_json_path.exists():
        raise FileNotFoundError(
            "release_manifest.json does not contain a valid release_json_path for promotion."
        )
    loaded_release = load_release_bundle(release_json_path)

    existing_official = list_official_runs_for_release(
        execution=loaded_release.execution,
        release_id=candidate_manifest.release_id,
    )
    if existing_official:
        raise ValueError(
            "Official singleton violation: an official run already exists for this release_id. "
            f"release_id='{candidate_manifest.release_id}'"
        )

    official_run_id = datetime.now(UTC).strftime("official_%Y%m%dT%H%M%SZ")
    official_paths = resolve_run_paths(
        execution=loaded_release.execution,
        run_class=RunClass.OFFICIAL,
        release_id=candidate_manifest.release_id,
        run_id=official_run_id,
    )
    official_paths.root.mkdir(parents=True, exist_ok=False)

    _copy_approved_artifacts(candidate_run_dir, official_paths.root)

    promotion_manifest = build_promotion_manifest(
        official_run_id=official_run_id,
        candidate_manifest=candidate_manifest,
        candidate_run_path=candidate_run_dir,
        official_run_path=official_paths.root,
    )
    write_json(
        official_paths.root / "promotion_manifest.json",
        promotion_manifest.model_dump(mode="json"),
    )
    write_json(
        official_paths.root / "official_release_manifest.json",
        _official_release_manifest_payload(
            release_manifest_payload=release_manifest_payload,
            candidate_run_id=candidate_manifest.run_id,
            official_run_id=official_run_id,
        ),
    )
    write_json(
        official_paths.root / "official_lock.json",
        {
            "schema_version": "official-lock-v1",
            "release_id": candidate_manifest.release_id,
            "official_run_id": official_run_id,
            "immutable": True,
            "enforced_by": "sentinel_and_code_guard",
        },
    )

    official_manifest = candidate_manifest.model_copy(
        update={
            "run_id": official_run_id,
            "run_class": RunClass.OFFICIAL,
            "status": RunStatus.PROMOTED,
            "parent_run_id": candidate_manifest.run_id,
            "promotable": False,
            "official": True,
            "evidence_verified": True,
            "command_line": "thesisml-promote-run --candidate-run " + str(candidate_run_dir),
        }
    )
    write_json(
        official_paths.root / "run_manifest.json",
        official_manifest.model_dump(mode="json"),
    )

    # Best-effort read-only hardening without breaking cross-platform behavior.
    for path in official_paths.root.rglob("*"):
        if path.is_file():
            _make_read_only(path)
    _make_read_only(official_paths.root)

    return {
        "passed": True,
        "release_id": candidate_manifest.release_id,
        "candidate_run_id": candidate_manifest.run_id,
        "candidate_run_path": str(candidate_run_dir.resolve()),
        "official_run_id": official_run_id,
        "official_run_path": str(official_paths.root.resolve()),
    }


__all__ = ["promote_candidate_run"]
