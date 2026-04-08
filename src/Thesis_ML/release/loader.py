from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from Thesis_ML.release.hashing import sha256_file
from Thesis_ML.release.manifests import AuthorityHashes
from Thesis_ML.release.models import (
    DatasetManifest,
    ReleaseBundle,
    ReleaseClaims,
    ReleaseEnvironment,
    ReleaseEvidence,
    ReleaseExecution,
    ReleaseScience,
)
from Thesis_ML.release.paths import resolve_release_path


@dataclass(frozen=True)
class LoadedReleaseBundle:
    release_path: Path
    release: ReleaseBundle
    science_path: Path
    science: ReleaseScience
    execution_path: Path
    execution: ReleaseExecution
    environment_path: Path
    environment: ReleaseEnvironment
    evidence_path: Path
    evidence: ReleaseEvidence
    claims_path: Path
    claims: ReleaseClaims
    hashes: AuthorityHashes


@dataclass(frozen=True)
class LoadedDatasetManifest:
    manifest_path: Path
    manifest: DatasetManifest
    index_csv_path: Path
    data_root_path: Path
    cache_dir_path: Path


def _read_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in '{path}'.")
    return payload


def load_release_bundle(release_ref: Path | str) -> LoadedReleaseBundle:
    release_path = resolve_release_path(release_ref)
    release = ReleaseBundle.model_validate(_read_json_object(release_path))
    release_root = release_path.parent

    science_path = (release_root / release.science_path).resolve()
    execution_path = (release_root / release.execution_path).resolve()
    environment_path = (release_root / release.environment_path).resolve()
    evidence_path = (release_root / release.evidence_path).resolve()
    claims_path = (release_root / release.claims_path).resolve()

    science = ReleaseScience.model_validate(_read_json_object(science_path))
    execution = ReleaseExecution.model_validate(_read_json_object(execution_path))
    environment = ReleaseEnvironment.model_validate(_read_json_object(environment_path))
    evidence = ReleaseEvidence.model_validate(_read_json_object(evidence_path))
    claims = ReleaseClaims.model_validate(_read_json_object(claims_path))

    hashes = AuthorityHashes.from_components(
        release_hash=sha256_file(release_path),
        science_hash=sha256_file(science_path),
        execution_hash=sha256_file(execution_path),
        environment_hash=sha256_file(environment_path),
        evidence_hash=sha256_file(evidence_path),
        claims_hash=sha256_file(claims_path),
    )
    return LoadedReleaseBundle(
        release_path=release_path,
        release=release,
        science_path=science_path,
        science=science,
        execution_path=execution_path,
        execution=execution,
        environment_path=environment_path,
        environment=environment,
        evidence_path=evidence_path,
        evidence=evidence,
        claims_path=claims_path,
        claims=claims,
        hashes=hashes,
    )


def load_dataset_manifest(path: Path | str) -> LoadedDatasetManifest:
    manifest_path = Path(path).resolve()
    manifest = DatasetManifest.model_validate(_read_json_object(manifest_path))
    base = manifest_path.parent

    index_csv_path = (base / manifest.index_csv).resolve()
    data_root_path = (base / manifest.data_root).resolve()
    cache_rel = manifest.cache_dir if manifest.cache_dir else ".cache"
    cache_dir_path = (base / cache_rel).resolve()

    return LoadedDatasetManifest(
        manifest_path=manifest_path,
        manifest=manifest,
        index_csv_path=index_csv_path,
        data_root_path=data_root_path,
        cache_dir_path=cache_dir_path,
    )


__all__ = [
    "LoadedDatasetManifest",
    "LoadedReleaseBundle",
    "load_dataset_manifest",
    "load_release_bundle",
]

