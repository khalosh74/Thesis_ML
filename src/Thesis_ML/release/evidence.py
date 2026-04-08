from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.config.paths import PROJECT_ROOT
from Thesis_ML.release.hashing import canonical_target_mapping_hash
from Thesis_ML.release.loader import LoadedDatasetManifest, LoadedReleaseBundle
from Thesis_ML.release.manifests import write_json
from Thesis_ML.release.models import RunManifest
from Thesis_ML.release.scope_manifest import selected_sample_ids_sha256

_FORBIDDEN_SCIENCE_OVERRIDE_FLAGS = {
    "--protocol",
    "--suite",
    "--index-csv",
    "--data-root",
    "--cache-dir",
    "--model",
    "--feature-space",
    "--target",
    "--cv",
}


def _add_issue(
    issues: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {"code": code, "message": message}
    if details:
        payload["details"] = details
    issues.append(payload)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _find_protocol_output_dir(artifacts_dir: Path) -> Path | None:
    protocol_root = artifacts_dir / "protocol_runs"
    if not protocol_root.exists() or not protocol_root.is_dir():
        return None
    candidates = sorted(path for path in protocol_root.iterdir() if path.is_dir())
    if len(candidates) != 1:
        return None
    return candidates[0]


def _iter_success_rows(report_index_path: Path) -> list[dict[str, str]]:
    if not report_index_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with report_index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = str(row.get("status", "")).strip().lower()
            if status == "success":
                rows.append({str(key): str(value) for key, value in row.items()})
    return rows


def _resolve_report_dir(protocol_output_dir: Path, raw_report_dir: str) -> Path:
    candidate = Path(raw_report_dir)
    if candidate.is_absolute():
        return candidate
    return (protocol_output_dir / candidate).resolve()


def _normalize_strategy_text(value: Any, *, none_fallback: str = "none") -> str:
    if value is None:
        return none_fallback
    normalized = str(value).strip().lower()
    if not normalized:
        return none_fallback
    return normalized


def _verify_release_artifacts(
    *,
    release: LoadedReleaseBundle,
    run_dir: Path,
    allow_missing_evidence_verification: bool,
    issues: list[dict[str, Any]],
) -> None:
    for relative_name in release.evidence.required_release_artifacts:
        candidate = run_dir / relative_name
        if candidate.exists():
            continue
        if allow_missing_evidence_verification and relative_name == "evidence_verification.json":
            continue
        _add_issue(
            issues,
            code="required_release_artifact_missing",
            message=f"Missing required release artifact '{relative_name}'.",
            details={"path": str(candidate)},
        )


def _verify_manifest_hash_fields(
    run_manifest: RunManifest,
    *,
    release: LoadedReleaseBundle,
    issues: list[dict[str, Any]],
) -> None:
    expected = {
        "release_hash": release.hashes.combined_hash,
        "science_hash": release.hashes.science_hash,
        "execution_hash": release.hashes.execution_hash,
        "environment_hash": release.hashes.environment_hash,
        "evidence_hash": release.hashes.evidence_hash,
        "claims_hash": release.hashes.claims_hash,
    }
    verify_flag_by_key = {
        "science_hash": bool(release.evidence.verify_science_hash),
        "execution_hash": bool(release.evidence.verify_execution_hash),
        "environment_hash": bool(release.evidence.verify_environment_hash),
        "claims_hash": bool(release.evidence.verify_claims_hash),
    }
    for key, value in expected.items():
        actual = str(getattr(run_manifest, key))
        if not actual:
            _add_issue(
                issues,
                code="manifest_hash_missing",
                message=f"run_manifest missing required hash field '{key}'.",
            )
            continue

        should_verify = verify_flag_by_key.get(key, True)
        if should_verify and actual != value:
            _add_issue(
                issues,
                code="manifest_hash_mismatch",
                message=f"run_manifest {key} does not match release authority hash.",
                details={"expected": value, "actual": actual},
            )


def _verify_scope_alignment_runtime(
    *,
    release: LoadedReleaseBundle,
    run_dir: Path,
    run_manifest: RunManifest,
    protocol_output_dir: Path,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    compiled_scope_manifest_path_raw = run_manifest.compiled_scope_manifest_path
    selected_samples_path_raw = run_manifest.selected_samples_path
    if not compiled_scope_manifest_path_raw:
        _add_issue(
            issues,
            code="compiled_scope_manifest_missing",
            message="run_manifest is missing compiled_scope_manifest_path.",
        )
        return {}
    if not selected_samples_path_raw:
        _add_issue(
            issues,
            code="compiled_scope_selected_samples_missing",
            message="run_manifest is missing selected_samples_path.",
        )
        return {}

    compiled_scope_manifest_path = Path(str(compiled_scope_manifest_path_raw)).resolve()
    selected_samples_path = Path(str(selected_samples_path_raw)).resolve()
    if not compiled_scope_manifest_path.exists():
        _add_issue(
            issues,
            code="compiled_scope_manifest_missing",
            message="compiled scope manifest path from run_manifest does not exist.",
            details={"path": str(compiled_scope_manifest_path)},
        )
        return {}
    if not selected_samples_path.exists():
        _add_issue(
            issues,
            code="compiled_scope_selected_samples_missing",
            message="selected_samples_path from run_manifest does not exist.",
            details={"path": str(selected_samples_path)},
        )
        return {}

    compiled_scope_manifest = _read_json(compiled_scope_manifest_path)
    if not isinstance(compiled_scope_manifest, dict):
        _add_issue(
            issues,
            code="compiled_scope_manifest_invalid",
            message="compiled scope manifest is not valid JSON object.",
            details={"path": str(compiled_scope_manifest_path)},
        )
        return {}

    selected_samples_df = pd.read_csv(selected_samples_path)
    if "sample_id" not in selected_samples_df.columns:
        _add_issue(
            issues,
            code="compiled_scope_sample_id_missing",
            message="selected_samples.csv is missing sample_id column.",
            details={"path": str(selected_samples_path)},
        )
    else:
        selected_sample_ids = selected_samples_df["sample_id"].astype(str).tolist()
        selected_ids_hash = selected_sample_ids_sha256(selected_sample_ids)
        if str(compiled_scope_manifest.get("selected_sample_ids_sha256", "")).strip() != selected_ids_hash:
            _add_issue(
                issues,
                code="compiled_scope_selected_sample_hash_mismatch",
                message="compiled scope selected_sample_ids_sha256 does not match selected_samples.csv.",
                details={
                    "manifest_hash": compiled_scope_manifest.get("selected_sample_ids_sha256"),
                    "actual_hash": selected_ids_hash,
                },
            )
        if (
            run_manifest.selected_sample_ids_sha256 is not None
            and str(run_manifest.selected_sample_ids_sha256).strip() != selected_ids_hash
        ):
            _add_issue(
                issues,
                code="run_manifest_scope_hash_mismatch",
                message="run_manifest selected_sample_ids_sha256 does not match selected_samples.csv.",
                details={
                    "run_manifest_selected_sample_ids_sha256": run_manifest.selected_sample_ids_sha256,
                    "actual_hash": selected_ids_hash,
                },
            )

    scope_tasks = sorted(selected_samples_df["task"].astype(str).unique().tolist())
    scope_modalities = sorted(selected_samples_df["modality"].astype(str).unique().tolist())
    scope_subjects = sorted(selected_samples_df["subject"].astype(str).unique().tolist())
    if scope_tasks != sorted(release.science.scope.effective_tasks()):
        _add_issue(
            issues,
            code="scope_task_alignment_failed",
            message="Compiled scope tasks do not match release science tasks.",
            details={
                "expected_tasks": sorted(release.science.scope.effective_tasks()),
                "actual_tasks": scope_tasks,
            },
        )
    if scope_modalities != [release.science.scope.effective_modality()]:
        _add_issue(
            issues,
            code="scope_modality_alignment_failed",
            message="Compiled scope modality does not match release science modality.",
            details={
                "expected_modality": release.science.scope.effective_modality(),
                "actual_modalities": scope_modalities,
            },
        )
    if scope_subjects != sorted(release.science.scope.effective_subjects()):
        _add_issue(
            issues,
            code="scope_subject_alignment_failed",
            message="Compiled scope subjects do not match release science subjects.",
            details={
                "expected_subjects": sorted(release.science.scope.effective_subjects()),
                "actual_subjects": scope_subjects,
            },
        )
    if str(compiled_scope_manifest.get("science_hash", "")).strip() != str(release.hashes.science_hash):
        _add_issue(
            issues,
            code="scope_science_hash_mismatch",
            message="Compiled scope science_hash does not match release science hash.",
            details={
                "expected_science_hash": release.hashes.science_hash,
                "actual_science_hash": compiled_scope_manifest.get("science_hash"),
            },
        )
    if str(compiled_scope_manifest.get("target_mapping_hash", "")).strip().lower() != str(
        release.science.target.mapping_hash
    ).strip().lower():
        _add_issue(
            issues,
            code="scope_target_mapping_hash_mismatch",
            message="Compiled scope target_mapping_hash does not match release science target mapping hash.",
            details={
                "expected_target_mapping_hash": release.science.target.mapping_hash,
                "actual_target_mapping_hash": compiled_scope_manifest.get("target_mapping_hash"),
            },
        )

    mapping_path = (PROJECT_ROOT / release.science.target.mapping_path).resolve()
    if not mapping_path.exists():
        _add_issue(
            issues,
            code="target_mapping_missing",
            message="release science target_mapping_path does not exist.",
            details={"path": str(mapping_path)},
        )
    else:
        actual_mapping_hash = canonical_target_mapping_hash(mapping_path).lower()
        expected_mapping_hash = release.science.target.mapping_hash.lower()
        if actual_mapping_hash != expected_mapping_hash:
            _add_issue(
                issues,
                code="target_mapping_hash_mismatch",
                message="target mapping hash does not match release science hash.",
                details={"expected": expected_mapping_hash, "actual": actual_mapping_hash},
            )

    compiled_manifest = _read_json(protocol_output_dir / "compiled_protocol_manifest.json")
    run_specs = list(compiled_manifest.get("runs", [])) if isinstance(compiled_manifest, dict) else []
    within_subjects = sorted(
        {
            str(spec.get("subject"))
            for spec in run_specs
            if str(spec.get("cv_mode")) == "within_subject_loso_session" and spec.get("subject")
        }
    )
    transfer_pairs = sorted(
        {
            (str(spec.get("train_subject")), str(spec.get("test_subject")))
            for spec in run_specs
            if str(spec.get("cv_mode")) == "frozen_cross_person_transfer"
            and spec.get("train_subject")
            and spec.get("test_subject")
        }
    )
    expected_within = sorted(release.science.scope.effective_subjects())
    expected_pairs = sorted(
        {(pair.train_subject, pair.test_subject) for pair in release.science.scope.transfer_pairs}
    )
    if within_subjects != expected_within:
        _add_issue(
            issues,
            code="within_subject_alignment_failed",
            message="Compiled protocol within-subject set does not match release scope.",
            details={"expected": expected_within, "actual": within_subjects},
        )
    if transfer_pairs != expected_pairs:
        _add_issue(
            issues,
            code="transfer_pair_alignment_failed",
            message="Compiled protocol transfer pairs do not match release scope.",
            details={
                "expected": [
                    {"train_subject": pair[0], "test_subject": pair[1]} for pair in expected_pairs
                ],
                "actual": [
                    {"train_subject": pair[0], "test_subject": pair[1]} for pair in transfer_pairs
                ],
            },
        )
    scope_alignment_verification_path = run_dir / "verification" / "scope_alignment_verification.json"
    scope_alignment_verification = _read_json(scope_alignment_verification_path)
    if not isinstance(scope_alignment_verification, dict):
        _add_issue(
            issues,
            code="scope_alignment_verification_missing",
            message="Missing verification/scope_alignment_verification.json.",
            details={"path": str(scope_alignment_verification_path)},
        )
    elif not bool(scope_alignment_verification.get("passed", False)):
        _add_issue(
            issues,
            code="scope_alignment_failed",
            message="Scope alignment verification failed.",
            details={"path": str(scope_alignment_verification_path)},
        )
    return {
        "scope_tasks": scope_tasks,
        "scope_modalities": scope_modalities,
        "scope_subjects": scope_subjects,
        "compiled_within_subjects": within_subjects,
        "compiled_transfer_pairs": [
            {"train_subject": pair[0], "test_subject": pair[1]} for pair in transfer_pairs
        ],
        "compiled_scope_manifest_path": str(compiled_scope_manifest_path),
        "selected_samples_path": str(selected_samples_path),
        "scope_alignment_verification_path": str(scope_alignment_verification_path),
        "scope_alignment_passed": bool(
            isinstance(scope_alignment_verification, dict)
            and scope_alignment_verification.get("passed", False)
        ),
    }


def _verify_run_level_contract(
    *,
    release: LoadedReleaseBundle,
    protocol_output_dir: Path,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    report_index_path = protocol_output_dir / "report_index.csv"
    success_rows = _iter_success_rows(report_index_path)
    all_rows: list[dict[str, str]] = []
    if report_index_path.exists():
        with report_index_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            all_rows = [{str(key): str(value) for key, value in row.items()} for row in reader]
    if not success_rows:
        _add_issue(
            issues,
            code="success_rows_missing",
            message="No successful runs found in report_index.csv.",
        )
        return {"n_success_runs": 0}

    for row in success_rows:
        run_id = str(row.get("run_id", "")).strip()
        report_dir = _resolve_report_dir(protocol_output_dir, str(row.get("report_dir", "")))
        if not report_dir.exists():
            _add_issue(
                issues,
                code="run_report_dir_missing",
                message="report_index successful row points to missing report_dir.",
                details={"run_id": run_id, "report_dir": str(report_dir)},
            )
            continue
        for artifact_name in release.evidence.required_run_artifacts:
            candidate = report_dir / artifact_name
            if not candidate.exists():
                _add_issue(
                    issues,
                    code="required_run_artifact_missing",
                    message=f"Missing required run artifact '{artifact_name}'.",
                    details={"run_id": run_id, "path": str(candidate)},
                )

        config_payload = _read_json(report_dir / "config.json")
        if not isinstance(config_payload, dict):
            _add_issue(
                issues,
                code="run_config_missing_or_invalid",
                message="Missing or invalid config.json in successful run directory.",
                details={"run_id": run_id},
            )
            continue

        resolved_model = str(
            config_payload.get("model_governance", {}).get(
                "logical_model_name",
                config_payload.get("model", ""),
            )
        ).strip()
        if resolved_model and resolved_model not in {
            release.science.model_policy.model_family,
            "dummy",
        }:
            _add_issue(
                issues,
                code="model_family_alignment_failed",
                message="Executed model differs from release science model family.",
                details={
                    "run_id": run_id,
                    "expected": release.science.model_policy.model_family,
                    "actual": resolved_model,
                },
            )

        expected_feature_space = release.science.feature_policy.feature_space
        actual_feature_space = str(config_payload.get("feature_space", "")).strip()
        if actual_feature_space and actual_feature_space != expected_feature_space:
            _add_issue(
                issues,
                code="feature_space_alignment_failed",
                message="Executed feature_space differs from release science feature_space.",
                details={
                    "run_id": run_id,
                    "expected": expected_feature_space,
                    "actual": actual_feature_space,
                },
            )

        expected_pre = _normalize_strategy_text(
            release.science.feature_policy.preprocessing_strategy,
            none_fallback="none",
        )
        actual_pre = _normalize_strategy_text(
            config_payload.get("preprocessing_strategy"),
            none_fallback="none",
        )
        if actual_pre != expected_pre:
            _add_issue(
                issues,
                code="preprocessing_alignment_failed",
                message="Executed preprocessing strategy differs from release science.",
                details={"run_id": run_id, "expected": expected_pre, "actual": actual_pre},
            )

        expected_dim = _normalize_strategy_text(
            release.science.feature_policy.dimensionality_strategy,
            none_fallback="none",
        )
        actual_dim = _normalize_strategy_text(
            config_payload.get("dimensionality_strategy"),
            none_fallback="none",
        )
        if actual_dim != expected_dim:
            _add_issue(
                issues,
                code="dimensionality_alignment_failed",
                message="Executed dimensionality strategy differs from release science.",
                details={"run_id": run_id, "expected": expected_dim, "actual": actual_dim},
            )

        if bool(config_payload.get("tuning_enabled", False)) is not False:
            _add_issue(
                issues,
                code="tuning_alignment_failed",
                message="Executed run has tuning_enabled=true but release science requires false.",
                details={"run_id": run_id},
            )
    suites_all = sorted(
        {
            str(row.get("suite_id", "")).strip()
            for row in all_rows
            if str(row.get("suite_id", "")).strip()
        }
    )
    suites_success = sorted(
        {
            str(row.get("suite_id", "")).strip()
            for row in success_rows
            if str(row.get("suite_id", "")).strip()
        }
    )
    missing_suite_success = sorted(set(suites_all) - set(suites_success))
    if missing_suite_success:
        _add_issue(
            issues,
            code="suite_output_missing",
            message="One or more required suites have no successful run outputs.",
            details={"missing_suite_ids": missing_suite_success},
        )
    return {
        "n_success_runs": len(success_rows),
        "suite_ids_all": suites_all,
        "suite_ids_with_success": suites_success,
        "missing_suite_success": missing_suite_success,
    }


def verify_release_evidence(
    *,
    run_dir: Path | str,
    release: LoadedReleaseBundle,
    dataset: LoadedDatasetManifest,
    run_manifest: RunManifest,
    allow_missing_evidence_verification: bool = False,
    write_output: bool = True,
) -> dict[str, Any]:
    resolved_run_dir = Path(run_dir).resolve()
    issues: list[dict[str, Any]] = []
    run_manifest_payload = run_manifest.model_dump(mode="json")

    for required_field in release.evidence.required_manifest_fields:
        if required_field not in run_manifest_payload:
            _add_issue(
                issues,
                code="run_manifest_field_missing",
                message=f"run_manifest missing required field '{required_field}'.",
            )
    if not bool(run_manifest.scope_alignment_passed):
        _add_issue(
            issues,
            code="scope_alignment_manifest_flag_false",
            message="run_manifest scope_alignment_passed is false.",
        )

    _verify_manifest_hash_fields(run_manifest, release=release, issues=issues)
    if (
        bool(release.evidence.verify_dataset_fingerprint)
        and run_manifest.dataset_fingerprint != dataset.manifest.dataset_fingerprint
    ):
        _add_issue(
            issues,
            code="dataset_fingerprint_mismatch",
            message="run_manifest dataset_fingerprint does not match dataset manifest fingerprint.",
            details={
                "run_manifest_dataset_fingerprint": run_manifest.dataset_fingerprint,
                "dataset_manifest_dataset_fingerprint": dataset.manifest.dataset_fingerprint,
            },
        )

    command_line_text = run_manifest.command_line or ""
    used_forbidden = sorted(
        flag for flag in _FORBIDDEN_SCIENCE_OVERRIDE_FLAGS if flag in command_line_text
    )
    if used_forbidden:
        _add_issue(
            issues,
            code="science_override_detected",
            message="Detected forbidden science-affecting CLI override flags in command line.",
            details={"flags": used_forbidden, "command_line": command_line_text},
        )

    _verify_release_artifacts(
        release=release,
        run_dir=resolved_run_dir,
        allow_missing_evidence_verification=allow_missing_evidence_verification,
        issues=issues,
    )
    for control_name in release.evidence.required_control_artifacts:
        if not (resolved_run_dir / "artifacts" / "protocol_runs").exists():
            continue
        protocol_output = _find_protocol_output_dir(resolved_run_dir / "artifacts")
        if protocol_output is None:
            continue
        if not (protocol_output / control_name).exists():
            _add_issue(
                issues,
                code="required_control_artifact_missing",
                message=f"Missing required control artifact '{control_name}'.",
                details={"path": str(protocol_output / control_name)},
            )

    protocol_output_dir = _find_protocol_output_dir(resolved_run_dir / "artifacts")
    if protocol_output_dir is None:
        _add_issue(
            issues,
            code="protocol_output_missing",
            message="Could not resolve unique protocol output directory under artifacts/protocol_runs.",
        )
        runtime_scope_alignment = {}
        run_level_summary = {"n_success_runs": 0}
    else:
        runtime_scope_alignment = _verify_scope_alignment_runtime(
            release=release,
            run_dir=resolved_run_dir,
            run_manifest=run_manifest,
            protocol_output_dir=protocol_output_dir,
            issues=issues,
        )
        run_level_summary = _verify_run_level_contract(
            release=release,
            protocol_output_dir=protocol_output_dir,
            issues=issues,
        )
        suite_summary = _read_json(protocol_output_dir / "suite_summary.json")
        controls_status = (
            suite_summary.get("confirmatory_reporting_contract", {}).get("controls_status", {})
            if isinstance(suite_summary, dict)
            else {}
        )
        if isinstance(controls_status, dict):
            if not bool(controls_status.get("controls_valid_for_confirmatory", False)):
                _add_issue(
                    issues,
                    code="controls_not_valid",
                    message="confirmatory controls are not valid per suite_summary controls_status.",
                    details={"controls_status": controls_status},
                )
        else:
            _add_issue(
                issues,
                code="controls_status_missing",
                message="suite_summary confirmatory controls_status is missing/invalid.",
            )

    verification_payload = {
        "schema_version": "release-evidence-verification-v1",
        "run_dir": str(resolved_run_dir),
        "release_id": release.release.release_id,
        "release_version": release.release.release_version,
        "run_id": run_manifest.run_id,
        "run_class": run_manifest.run_class.value,
        "passed": not issues,
        "issues": issues,
        "runtime_scope_alignment": runtime_scope_alignment,
        "run_level_summary": run_level_summary,
    }
    verification_path = resolved_run_dir / "verification" / "evidence_verification.json"
    if write_output:
        write_json(verification_path, verification_payload)
    return verification_payload


def load_release_verification(run_dir: Path | str) -> dict[str, Any] | None:
    candidate = Path(run_dir).resolve() / "verification" / "evidence_verification.json"
    return _read_json(candidate)


__all__ = [
    "load_release_verification",
    "verify_release_evidence",
]
