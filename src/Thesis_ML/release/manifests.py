from __future__ import annotations

import json
import platform
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Thesis_ML.experiments.provenance import collect_git_provenance
from Thesis_ML.release.hashing import combined_release_hash
from Thesis_ML.release.models import (
    PromotionManifest,
    ReleaseBundle,
    ReleaseManifest,
    RunClass,
    RunManifest,
    RunStatus,
    utc_now_iso,
)


@dataclass(frozen=True)
class AuthorityHashes:
    release_hash: str
    science_hash: str
    execution_hash: str
    environment_hash: str
    evidence_hash: str
    claims_hash: str
    combined_hash: str

    @classmethod
    def from_components(
        cls,
        *,
        release_hash: str,
        science_hash: str,
        execution_hash: str,
        environment_hash: str,
        evidence_hash: str,
        claims_hash: str,
    ) -> AuthorityHashes:
        combined = combined_release_hash(
            {
                "release_hash": release_hash,
                "science_hash": science_hash,
                "execution_hash": execution_hash,
                "environment_hash": environment_hash,
                "evidence_hash": evidence_hash,
                "claims_hash": claims_hash,
            }
        )
        return cls(
            release_hash=release_hash,
            science_hash=science_hash,
            execution_hash=execution_hash,
            environment_hash=environment_hash,
            evidence_hash=evidence_hash,
            claims_hash=claims_hash,
            combined_hash=combined,
        )


def write_json(path: Path | str, payload: dict[str, Any]) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    return resolved


def build_release_manifest(
    *,
    release_bundle: ReleaseBundle,
    release_json_path: Path,
    science_json_path: Path,
    execution_json_path: Path,
    environment_json_path: Path,
    evidence_json_path: Path,
    claims_json_path: Path,
    hashes: AuthorityHashes,
) -> ReleaseManifest:
    return ReleaseManifest(
        release_id=release_bundle.release_id,
        release_version=release_bundle.release_version,
        generated_at_utc=utc_now_iso(),
        release_json_path=str(release_json_path.resolve()),
        science_json_path=str(science_json_path.resolve()),
        execution_json_path=str(execution_json_path.resolve()),
        environment_json_path=str(environment_json_path.resolve()),
        evidence_json_path=str(evidence_json_path.resolve()),
        claims_json_path=str(claims_json_path.resolve()),
        release_hash=hashes.combined_hash,
        science_hash=hashes.science_hash,
        execution_hash=hashes.execution_hash,
        environment_hash=hashes.environment_hash,
        evidence_hash=hashes.evidence_hash,
        claims_hash=hashes.claims_hash,
    )


def build_run_manifest(
    *,
    run_id: str,
    run_class: RunClass,
    release_bundle: ReleaseBundle,
    hashes: AuthorityHashes,
    dataset_manifest_path: Path,
    dataset_fingerprint: str,
    cache_policy: str,
    command_line: list[str] | None = None,
    parent_run_id: str | None = None,
    status: RunStatus,
    promotable: bool,
    official: bool,
    evidence_verified: bool = False,
    compiled_scope_manifest_path: str | None = None,
    selected_samples_path: str | None = None,
    selected_sample_ids_sha256: str | None = None,
    scope_alignment_passed: bool = False,
) -> RunManifest:
    git_payload = collect_git_provenance()
    if isinstance(command_line, list) and command_line:
        command_line_text = shlex.join(command_line)
    else:
        command_line_text = ""
    return RunManifest(
        run_id=run_id,
        run_class=run_class,
        release_id=release_bundle.release_id,
        release_version=release_bundle.release_version,
        release_hash=hashes.combined_hash,
        science_hash=hashes.science_hash,
        execution_hash=hashes.execution_hash,
        environment_hash=hashes.environment_hash,
        evidence_hash=hashes.evidence_hash,
        claims_hash=hashes.claims_hash,
        dataset_manifest_path=str(dataset_manifest_path.resolve()),
        dataset_fingerprint=str(dataset_fingerprint),
        git_commit=str(git_payload.get("git_commit", "")),
        git_dirty=bool(git_payload.get("git_dirty", False)),
        python_version=str(platform.python_version()),
        platform=str(sys.platform),
        timestamp_utc=utc_now_iso(),
        parent_run_id=parent_run_id,
        status=status,
        promotable=promotable,
        official=official,
        cache_policy=cache_policy,
        command_line=command_line_text,
        evidence_verified=bool(evidence_verified),
        compiled_scope_manifest_path=compiled_scope_manifest_path,
        selected_samples_path=selected_samples_path,
        selected_sample_ids_sha256=selected_sample_ids_sha256,
        scope_alignment_passed=bool(scope_alignment_passed),
    )


def build_promotion_manifest(
    *,
    official_run_id: str,
    candidate_manifest: RunManifest,
    candidate_run_path: Path,
    official_run_path: Path,
) -> PromotionManifest:
    return PromotionManifest(
        release_id=candidate_manifest.release_id,
        release_version=candidate_manifest.release_version,
        official_run_id=official_run_id,
        candidate_run_id=candidate_manifest.run_id,
        candidate_run_path=str(candidate_run_path.resolve()),
        official_run_path=str(official_run_path.resolve()),
        timestamp_utc=utc_now_iso(),
        release_hash=candidate_manifest.release_hash,
        science_hash=candidate_manifest.science_hash,
        execution_hash=candidate_manifest.execution_hash,
        environment_hash=candidate_manifest.environment_hash,
        evidence_hash=candidate_manifest.evidence_hash,
        claims_hash=candidate_manifest.claims_hash,
        dataset_fingerprint=candidate_manifest.dataset_fingerprint,
    )


def read_run_manifest(path: Path | str) -> RunManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run manifest must be a JSON object: {path}")
    return RunManifest.model_validate(payload)


__all__ = [
    "AuthorityHashes",
    "build_promotion_manifest",
    "build_release_manifest",
    "build_run_manifest",
    "read_run_manifest",
    "write_json",
]
