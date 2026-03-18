from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from Thesis_ML.experiments.provenance import collect_git_provenance
from Thesis_ML.experiments.run_states import is_run_success_status

MANIFEST_SCHEMA_VERSION = "reproducibility-manifest-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(payload: Any) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _resolve_report_dir(output_dir: Path, report_dir_raw: str) -> Path:
    candidate = Path(report_dir_raw)
    if candidate.is_absolute():
        return candidate
    return output_dir / candidate


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def _spec_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        return {
            "path": str(resolved),
            "exists": False,
        }
    payload = _load_json(resolved)
    identity: dict[str, Any] = {
        "path": str(resolved.resolve()),
        "exists": True,
        "sha256": _file_sha256(resolved),
    }
    if isinstance(payload, dict):
        for key in (
            "protocol_id",
            "protocol_version",
            "comparison_id",
            "comparison_version",
            "protocol_schema_version",
            "comparison_schema_version",
        ):
            if key in payload:
                identity[key] = payload.get(key)
    return identity


def _dataset_fingerprint_summary(output_dir: Path) -> dict[str, Any]:
    report_index_path = output_dir / "report_index.csv"
    if not report_index_path.exists():
        return {
            "available": False,
            "reason": "report_index_missing",
        }

    signatures: set[str] = set()
    missing_run_ids: list[str] = []
    n_completed = 0
    with report_index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not is_run_success_status(str(row.get("status", "")).strip().lower()):
                continue
            n_completed += 1
            run_id = str(row.get("run_id", "")).strip()
            report_dir_raw = str(row.get("report_dir", "")).strip()
            if not run_id or not report_dir_raw:
                continue
            report_dir = _resolve_report_dir(output_dir, report_dir_raw)
            config_payload = _load_json(report_dir / "config.json")
            metrics_payload = _load_json(report_dir / "metrics.json")
            fingerprint = None
            if isinstance(config_payload, dict) and isinstance(
                config_payload.get("dataset_fingerprint"), dict
            ):
                fingerprint = dict(config_payload["dataset_fingerprint"])
            elif isinstance(metrics_payload, dict) and isinstance(
                metrics_payload.get("dataset_fingerprint"), dict
            ):
                fingerprint = dict(metrics_payload["dataset_fingerprint"])
            if fingerprint is None:
                missing_run_ids.append(run_id)
                continue
            marker = "|".join(
                [
                    str(fingerprint.get("index_csv_sha256", "")),
                    str(fingerprint.get("selected_sample_id_sha256", "")),
                    str(fingerprint.get("target_column", "")),
                    str(fingerprint.get("cv_mode", "")),
                ]
            )
            signatures.add(marker)

    n_present = int(n_completed - len(missing_run_ids))
    return {
        "available": True,
        "n_success_runs": int(n_completed),
        "n_completed_runs": int(n_completed),
        "n_with_fingerprint": int(n_present),
        "n_missing_fingerprint": int(len(missing_run_ids)),
        "missing_run_ids": sorted(missing_run_ids),
        "unique_fingerprint_count": int(len(signatures)),
        "consistent_across_runs": bool(len(signatures) <= 1 if n_present > 0 else True),
    }


def _seed_summary(output_dir: Path) -> dict[str, Any]:
    report_index_path = output_dir / "report_index.csv"
    if not report_index_path.exists():
        return {"available": False, "reason": "report_index_missing"}
    seeds: set[int] = set()
    with report_index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_seed = row.get("seed")
            if raw_seed is None or str(raw_seed).strip() == "":
                continue
            try:
                seeds.add(int(raw_seed))
            except ValueError:
                continue
    return {
        "available": True,
        "n_unique_seeds": int(len(seeds)),
        "seed_values": sorted(seeds),
    }


def _environment_identity(repo_root: Path) -> dict[str, Any]:
    uv_lock = repo_root / "uv.lock"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "uv_lock_path": str(uv_lock.resolve()) if uv_lock.exists() else None,
        "uv_lock_sha256": _file_sha256(uv_lock) if uv_lock.exists() else None,
        "package_versions": {
            "numpy": _module_version("numpy"),
            "pandas": _module_version("pandas"),
            "sklearn": _module_version("sklearn"),
            "nibabel": _module_version("nibabel"),
            "pydantic": _module_version("pydantic"),
        },
    }


def build_reproducibility_manifest(
    *,
    output_dirs_by_mode: dict[str, Path],
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    comparison_spec_path: Path | None = None,
    protocol_path: Path | None = None,
    replay_summary: dict[str, Any] | None = None,
    replay_verification_summary: dict[str, Any] | None = None,
    bundle_dir: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[3]
    )
    index_csv_resolved = Path(index_csv).resolve()
    data_root_resolved = Path(data_root).resolve()
    cache_dir_resolved = Path(cache_dir).resolve()

    demo_manifest_path = index_csv_resolved.parent / "demo_dataset_manifest.json"
    bundle_manifest_path = (
        Path(bundle_dir).resolve() / "bundle_manifest.json" if bundle_dir is not None else None
    )
    mode_outputs: dict[str, Any] = {}
    for mode, output_dir in sorted(output_dirs_by_mode.items()):
        resolved_output_dir = Path(output_dir).resolve()
        mode_outputs[str(mode)] = {
            "output_dir": str(resolved_output_dir),
            "exists": bool(resolved_output_dir.exists()),
            "dataset_fingerprint_summary": _dataset_fingerprint_summary(resolved_output_dir),
            "seed_summary": _seed_summary(resolved_output_dir),
        }

    replay_summary_hash = _stable_sha256(replay_summary) if replay_summary is not None else None
    replay_verification_hash = (
        _stable_sha256(replay_verification_summary)
        if replay_verification_summary is not None
        else None
    )

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "code_identity": collect_git_provenance(),
        "environment_identity": _environment_identity(resolved_repo_root),
        "dataset_identity": {
            "index_csv_path": str(index_csv_resolved),
            "index_csv_sha256": (
                _file_sha256(index_csv_resolved) if index_csv_resolved.exists() else None
            ),
            "data_root_path": str(data_root_resolved),
            "cache_dir_path": str(cache_dir_resolved),
            "demo_dataset_manifest_path": (
                str(demo_manifest_path.resolve()) if demo_manifest_path.exists() else None
            ),
            "demo_dataset_manifest_sha256": (
                _file_sha256(demo_manifest_path) if demo_manifest_path.exists() else None
            ),
        },
        "spec_identity": {
            "comparison": _spec_identity(comparison_spec_path),
            "confirmatory_protocol": _spec_identity(protocol_path),
        },
        "official_outputs": mode_outputs,
        "replay_status": {
            "replay_summary_hash": replay_summary_hash,
            "replay_verification_summary_hash": replay_verification_hash,
            "replay_passed": (
                bool(replay_verification_summary.get("passed", False))
                if isinstance(replay_verification_summary, dict)
                else None
            ),
            "determinism_passed": (
                bool(replay_verification_summary.get("determinism", {}).get("passed", False))
                if isinstance(replay_verification_summary, dict)
                and isinstance(replay_verification_summary.get("determinism"), dict)
                else None
            ),
        },
        "bundle_identity": {
            "bundle_dir": str(Path(bundle_dir).resolve()) if bundle_dir is not None else None,
            "bundle_manifest_path": (
                str(bundle_manifest_path.resolve())
                if bundle_manifest_path is not None and bundle_manifest_path.exists()
                else None
            ),
            "bundle_manifest_sha256": (
                _file_sha256(bundle_manifest_path)
                if bundle_manifest_path is not None and bundle_manifest_path.exists()
                else None
            ),
        },
    }


def write_reproducibility_manifest(
    *,
    manifest: dict[str, Any],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")
    return output_path


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "build_reproducibility_manifest",
    "write_reproducibility_manifest",
]
