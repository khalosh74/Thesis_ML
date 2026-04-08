from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.config.paths import (
    DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH,
    PROJECT_ROOT,
)
from Thesis_ML.protocols.compiler import compile_protocol
from Thesis_ML.protocols.loader import load_protocol
from Thesis_ML.protocols.models import CompiledProtocolManifest, ThesisProtocol
from Thesis_ML.release.hashing import canonical_target_mapping_hash
from Thesis_ML.release.loader import LoadedDatasetManifest, LoadedReleaseBundle
from Thesis_ML.release.scope_models import CompiledScopeManifest, CompiledScopeResult


@dataclass(frozen=True)
class ReleaseAdapterPlan:
    protocol_path: Path
    protocol: ThesisProtocol
    scoped_index_csv: Path
    compiled_manifest: CompiledProtocolManifest
    compiled_scope_manifest_path: Path
    compiled_scope_selected_samples_path: Path
    compiled_scope_manifest: CompiledScopeManifest
    protocol_context_overrides_by_run_id: dict[str, dict[str, Any]]
    alignment_report: dict[str, Any]


def _load_compiled_scope_manifest(path: Path) -> CompiledScopeManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Compiled scope manifest must be a JSON object: '{path}'.")
    return CompiledScopeManifest.model_validate(payload)


def _adapt_protocol_for_release(
    *,
    release: LoadedReleaseBundle,
    protocol: ThesisProtocol,
) -> tuple[ThesisProtocol, dict[str, Any]]:
    lock_payload = protocol.confirmatory_lock if isinstance(protocol.confirmatory_lock, dict) else {}
    if not lock_payload:
        return protocol, {"confirmatory_lock_split_override": None}

    lock = dict(lock_payload)
    original_split = str(lock.get("split", "")).strip()
    expected_primary_split = str(release.science.split_policy.primary_analysis.split).strip()
    if original_split and expected_primary_split and original_split != expected_primary_split:
        raise ValueError(
            "Release science primary split does not match protocol confirmatory lock split. "
            f"science={expected_primary_split}, protocol_lock={original_split}"
        )

    # Release runs execute both primary and frozen secondary suites. The legacy lock only
    # carries the primary split and rejects secondary transfer runs; release-level alignment
    # enforcement handles split/scope fidelity explicitly.
    if "split" in lock:
        lock.pop("split", None)
        protocol = protocol.model_copy(update={"confirmatory_lock": lock})

    return protocol, {
        "confirmatory_lock_split_override": {
            "removed": bool(original_split),
            "original_split": original_split or None,
            "release_primary_split": expected_primary_split or None,
        }
    }


def _validate_target_mapping(release: LoadedReleaseBundle, protocol: ThesisProtocol) -> dict[str, Any]:
    mapping_path = (PROJECT_ROOT / release.science.target.mapping_path).resolve()
    if not mapping_path.exists():
        raise FileNotFoundError(f"target mapping path does not exist: {mapping_path}")

    mapping_hash_actual = canonical_target_mapping_hash(mapping_path)
    mapping_hash_expected = release.science.target.mapping_hash.lower()
    if mapping_hash_actual.lower() != mapping_hash_expected:
        raise ValueError(
            "Target mapping hash mismatch against release science. "
            f"expected={mapping_hash_expected}, actual={mapping_hash_actual.lower()}"
        )

    confirmatory_lock = protocol.confirmatory_lock if isinstance(protocol.confirmatory_lock, dict) else {}
    lock_hash = str(confirmatory_lock.get("target_mapping_hash", "")).strip().lower()
    if lock_hash and lock_hash != mapping_hash_expected:
        raise ValueError(
            "Protocol confirmatory lock mapping hash does not match release science hash. "
            f"lock={lock_hash}, science={mapping_hash_expected}"
        )
    return {
        "target_mapping_path": str(mapping_path),
        "target_mapping_hash_expected": mapping_hash_expected,
        "target_mapping_hash_actual": mapping_hash_actual.lower(),
        "protocol_lock_mapping_hash": lock_hash or None,
    }


def _validate_compiled_alignment(
    *,
    release: LoadedReleaseBundle,
    compiled_manifest: CompiledProtocolManifest,
    compiled_scope_manifest: CompiledScopeManifest,
    compiled_scope_df: pd.DataFrame,
) -> dict[str, Any]:
    issues: list[str] = []
    science = release.science

    compiled_targets = sorted({spec.target for spec in compiled_manifest.runs})
    if compiled_targets != [science.target.name]:
        issues.append(
            f"compiled target mismatch: expected={[science.target.name]}, actual={compiled_targets}"
        )

    non_dummy_models = sorted(
        {
            spec.model
            for spec in compiled_manifest.runs
            if str(spec.model).strip().lower() != "dummy"
        }
    )
    if non_dummy_models != [science.model_policy.model_family]:
        issues.append(
            "compiled model mismatch: "
            f"expected={[science.model_policy.model_family]}, actual={non_dummy_models}"
        )

    if science.model_policy.tuning_enabled:
        issues.append("release science tuning_enabled must be false")
    tuned_run_ids = [spec.run_id for spec in compiled_manifest.runs if bool(spec.tuning_enabled)]
    if tuned_run_ids:
        issues.append("compiled manifest contains tuning-enabled runs")

    within_subjects_compiled = sorted(
        {
            str(spec.subject)
            for spec in compiled_manifest.runs
            if spec.cv_mode == "within_subject_loso_session" and spec.subject is not None
        }
    )
    expected_within_subjects = sorted(science.scope.effective_subjects())
    if within_subjects_compiled != expected_within_subjects:
        issues.append(
            "compiled within-subject scope mismatch: "
            f"expected={expected_within_subjects}, actual={within_subjects_compiled}"
        )

    transfer_pairs_compiled = sorted(
        {
            (str(spec.train_subject), str(spec.test_subject))
            for spec in compiled_manifest.runs
            if spec.cv_mode == "frozen_cross_person_transfer"
            and spec.train_subject is not None
            and spec.test_subject is not None
        }
    )
    expected_transfer_pairs = sorted(
        {(pair.train_subject, pair.test_subject) for pair in science.scope.transfer_pairs}
    )
    if transfer_pairs_compiled != expected_transfer_pairs:
        issues.append(
            "compiled transfer-pair scope mismatch: "
            f"expected={expected_transfer_pairs}, actual={transfer_pairs_compiled}"
        )

    configured_filter_tasks = sorted(
        {
            str(spec.filter_task)
            for spec in compiled_manifest.runs
            if spec.filter_task is not None and str(spec.filter_task).strip()
        }
    )
    configured_filter_modalities = sorted(
        {
            str(spec.filter_modality)
            for spec in compiled_manifest.runs
            if spec.filter_modality is not None and str(spec.filter_modality).strip()
        }
    )
    for spec in compiled_manifest.runs:
        if spec.filter_task is not None:
            issues.append(
                f"compiled run '{spec.run_id}' sets filter_task='{spec.filter_task}', "
                "but release scope enforcement requires filter_task=None."
            )
        if spec.filter_modality is not None:
            issues.append(
                f"compiled run '{spec.run_id}' sets filter_modality='{spec.filter_modality}', "
                "but release scope enforcement requires filter_modality=None."
            )

    scope_tasks = sorted(compiled_scope_df["task"].astype(str).unique().tolist())
    scope_modalities = sorted(compiled_scope_df["modality"].astype(str).unique().tolist())
    if scope_tasks != sorted(science.scope.effective_tasks()):
        issues.append(
            "compiled scope task mismatch: "
            f"expected={sorted(science.scope.effective_tasks())}, actual={scope_tasks}"
        )
    if scope_modalities != [science.scope.effective_modality()]:
        issues.append(
            "compiled scope modality mismatch: "
            f"expected={[science.scope.effective_modality()]}, actual={scope_modalities}"
        )

    if compiled_scope_manifest.science_hash != release.hashes.science_hash:
        issues.append(
            "compiled scope science hash mismatch: "
            f"expected={release.hashes.science_hash}, actual={compiled_scope_manifest.science_hash}"
        )
    if compiled_scope_manifest.target_mapping_hash.lower() != release.science.target.mapping_hash.lower():
        issues.append(
            "compiled scope target mapping hash mismatch: "
            f"expected={release.science.target.mapping_hash.lower()}, "
            f"actual={compiled_scope_manifest.target_mapping_hash.lower()}"
        )

    if issues:
        raise ValueError(
            "Release scope alignment failed before execution: " + "; ".join(sorted(set(issues)))
        )

    return {
        "passed": True,
        "compiled_targets": compiled_targets,
        "compiled_non_dummy_models": non_dummy_models,
        "compiled_within_subjects": within_subjects_compiled,
        "compiled_transfer_pairs": [
            {"train_subject": pair[0], "test_subject": pair[1]} for pair in transfer_pairs_compiled
        ],
        "compiled_scope_tasks": scope_tasks,
        "compiled_scope_modalities": scope_modalities,
        "compiled_scope_selected_row_count": int(compiled_scope_manifest.selected_row_count),
        "compiled_scope_selected_sample_ids_sha256": compiled_scope_manifest.selected_sample_ids_sha256,
        "compiled_filter_task_values": configured_filter_tasks,
        "compiled_filter_modality_values": configured_filter_modalities,
        "tuning_run_ids": tuned_run_ids,
    }


def build_release_adapter_plan(
    *,
    release: LoadedReleaseBundle,
    dataset: LoadedDatasetManifest,
    compiled_scope: CompiledScopeResult,
) -> ReleaseAdapterPlan:
    _ = dataset  # dataset is intentionally kept in the signature for compatibility and traceability.
    compiled_scope_manifest = _load_compiled_scope_manifest(compiled_scope.scope_manifest_path)

    protocol_path = Path(DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH).resolve()
    protocol = load_protocol(protocol_path)
    protocol, protocol_override_summary = _adapt_protocol_for_release(
        release=release,
        protocol=protocol,
    )
    target_mapping_summary = _validate_target_mapping(release, protocol)

    compiled_manifest = compile_protocol(protocol, index_csv=compiled_scope.selected_samples_path, suite_ids=None)
    compiled_alignment = _validate_compiled_alignment(
        release=release,
        compiled_manifest=compiled_manifest,
        compiled_scope_manifest=compiled_scope_manifest,
        compiled_scope_df=compiled_scope.selected_index_df,
    )

    run_context_overrides: dict[str, dict[str, Any]] = {}
    for run_spec in compiled_manifest.runs:
        run_context_overrides[str(run_spec.run_id)] = {
            "release_scope_enforcement": True,
            "compiled_scope_manifest_path": str(compiled_scope.scope_manifest_path.resolve()),
            "compiled_scope_selected_samples_path": str(compiled_scope.selected_samples_path.resolve()),
            "compiled_scope_selected_sample_ids_sha256": str(
                compiled_scope_manifest.selected_sample_ids_sha256
            ),
            "compiled_scope_science_hash": str(release.hashes.science_hash),
            "compiled_scope_target_mapping_hash": str(release.science.target.mapping_hash).lower(),
        }

    return ReleaseAdapterPlan(
        protocol_path=protocol_path,
        protocol=protocol,
        scoped_index_csv=compiled_scope.selected_samples_path,
        compiled_manifest=compiled_manifest,
        compiled_scope_manifest_path=compiled_scope.scope_manifest_path,
        compiled_scope_selected_samples_path=compiled_scope.selected_samples_path,
        compiled_scope_manifest=compiled_scope_manifest,
        protocol_context_overrides_by_run_id=run_context_overrides,
        alignment_report={
            "target_mapping": target_mapping_summary,
            "protocol_overrides": protocol_override_summary,
            "compiled_scope_alignment": compiled_alignment,
        },
    )


__all__ = ["ReleaseAdapterPlan", "build_release_adapter_plan"]
