from __future__ import annotations

import json
import platform
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.config.paths import PROJECT_ROOT
from Thesis_ML.data.index_validation import (
    DatasetIndexValidationError,
    validate_dataset_index_strict,
)
from Thesis_ML.release.loader import (
    LoadedDatasetManifest,
    LoadedReleaseBundle,
    load_dataset_manifest,
    load_release_bundle,
)


@dataclass(frozen=True)
class ValidatedReleaseContext:
    release: LoadedReleaseBundle
    dataset: LoadedDatasetManifest
    issues: list[dict[str, Any]]
    passed: bool


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


def _validate_schema_node(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str,
    errors: list[str],
) -> None:
    schema_type = schema.get("type")
    allowed_types = schema_type if isinstance(schema_type, list) else [schema_type]
    if schema_type is not None:
        type_ok = False
        for candidate in allowed_types:
            if candidate == "object" and isinstance(value, dict):
                type_ok = True
            elif candidate == "array" and isinstance(value, list):
                type_ok = True
            elif candidate == "string" and isinstance(value, str):
                type_ok = True
            elif candidate == "integer" and isinstance(value, int) and not isinstance(value, bool):
                type_ok = True
            elif candidate == "number" and isinstance(value, (int, float)) and not isinstance(
                value, bool
            ):
                type_ok = True
            elif candidate == "boolean" and isinstance(value, bool):
                type_ok = True
            elif candidate == "null" and value is None:
                type_ok = True
        if not type_ok:
            errors.append(
                f"{path}: expected type {allowed_types!r}, got '{type(value).__name__}'."
            )
            return

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected const value {schema['const']!r}, got {value!r}.")

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path}: value {value!r} is not in enum {enum_values!r}.")

    pattern = schema.get("pattern")
    if isinstance(pattern, str) and isinstance(value, str):
        import re

        if re.fullmatch(pattern, value) is None:
            errors.append(f"{path}: value {value!r} does not match pattern {pattern!r}.")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and float(value) < float(minimum):
            errors.append(f"{path}: value {value!r} is below minimum {minimum!r}.")
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and float(value) > float(maximum):
            errors.append(f"{path}: value {value!r} is above maximum {maximum!r}.")
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(exclusive_minimum, (int, float)) and float(value) <= float(exclusive_minimum):
            errors.append(
                f"{path}: value {value!r} must be > exclusiveMinimum {exclusive_minimum!r}."
            )

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            errors.append(f"{path}: expected minLength {min_length}, got {len(value)}.")
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and len(value) > max_length:
            errors.append(f"{path}: expected maxLength {max_length}, got {len(value)}.")

    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    errors.append(f"{path}: missing required property '{key}'.")
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key not in value or not isinstance(child_schema, dict):
                    continue
                _validate_schema_node(value[key], child_schema, path=f"{path}.{key}", errors=errors)
            if schema.get("additionalProperties") is False:
                allowed = set(properties)
                for key in value:
                    if key not in allowed:
                        errors.append(f"{path}: unexpected property '{key}'.")
        min_properties = schema.get("minProperties")
        if isinstance(min_properties, int) and len(value) < min_properties:
            errors.append(f"{path}: expected at least {min_properties} properties.")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"{path}: expected at least {min_items} items, got {len(value)}.")
        if schema.get("uniqueItems") is True:
            seen: set[str] = set()
            for idx, item in enumerate(value):
                marker = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                if marker in seen:
                    errors.append(f"{path}[{idx}]: duplicate item not allowed.")
                seen.add(marker)
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                _validate_schema_node(item, item_schema, path=f"{path}[{idx}]", errors=errors)


def _validate_against_schema(payload: dict[str, Any], schema_path: Path) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        return [f"Schema is not a JSON object: {schema_path}"]
    errors: list[str] = []
    _validate_schema_node(payload, schema, path="$", errors=errors)
    return errors


def _schema_path(schema_filename: str) -> Path:
    return (PROJECT_ROOT / "schemas" / schema_filename).resolve()


def _validate_dataset_manifest_payload(dataset: LoadedDatasetManifest) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    manifest = dataset.manifest

    schema_errors = _validate_against_schema(
        manifest.model_dump(mode="json", exclude_none=True),
        _schema_path("dataset_instance.schema.json"),
    )
    for error in schema_errors:
        _add_issue(issues, code="dataset_schema_error", message=error)

    if not dataset.index_csv_path.exists() or not dataset.index_csv_path.is_file():
        _add_issue(
            issues,
            code="dataset_index_missing",
            message="dataset_manifest index_csv does not resolve to an existing file.",
            details={"index_csv_path": str(dataset.index_csv_path)},
        )
        return issues
    if not dataset.data_root_path.exists() or not dataset.data_root_path.is_dir():
        _add_issue(
            issues,
            code="dataset_data_root_missing",
            message="dataset_manifest data_root does not resolve to an existing directory.",
            details={"data_root_path": str(dataset.data_root_path)},
        )
        return issues

    frame = pd.read_csv(dataset.index_csv_path)
    try:
        validated = validate_dataset_index_strict(
            frame,
            data_root=dataset.data_root_path,
            required_columns=manifest.required_columns,
            require_integrity_columns=False,
        )
    except DatasetIndexValidationError as exc:
        _add_issue(
            issues,
            code="dataset_index_validation_failed",
            message=f"strict dataset index validation failed: {exc}",
        )
        return issues

    # Optional compatibility check against cached manifest summary values.
    actual_subject_count = int(validated["subject"].astype(str).nunique())
    if actual_subject_count != int(manifest.subject_count):
        _add_issue(
            issues,
            code="dataset_subject_count_mismatch",
            message="dataset_manifest subject_count does not match index content.",
            details={
                "manifest_subject_count": int(manifest.subject_count),
                "actual_subject_count": actual_subject_count,
            },
        )

    actual_sessions_by_subject = {
        str(subject): int(group["session"].astype(str).nunique())
        for subject, group in validated.groupby("subject", sort=True)
    }
    if actual_sessions_by_subject != {
        str(key): int(value) for key, value in manifest.session_counts_by_subject.items()
    }:
        _add_issue(
            issues,
            code="dataset_session_counts_mismatch",
            message="dataset_manifest session_counts_by_subject does not match index content.",
            details={
                "manifest": manifest.session_counts_by_subject,
                "actual": actual_sessions_by_subject,
            },
        )

    return issues


def validate_dataset_manifest(dataset_manifest_path: Path | str) -> dict[str, Any]:
    dataset = load_dataset_manifest(dataset_manifest_path)
    issues = _validate_dataset_manifest_payload(dataset)
    return {
        "passed": not issues,
        "dataset_manifest_path": str(dataset.manifest_path.resolve()),
        "index_csv_path": str(dataset.index_csv_path.resolve()),
        "data_root_path": str(dataset.data_root_path.resolve()),
        "cache_dir_path": str(dataset.cache_dir_path.resolve()),
        "cache_dir_defaulted": dataset.manifest.cache_dir is None,
        "issues": issues,
    }


def _validate_release_schemas(release: LoadedReleaseBundle) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    schema_checks = [
        (
            "release_bundle",
            release.release.model_dump(mode="json", exclude_none=True),
            "release_bundle.schema.json",
        ),
        (
            "release_science",
            release.science.model_dump(mode="json", exclude_none=True),
            "release_science.schema.json",
        ),
        (
            "release_execution",
            release.execution.model_dump(mode="json", exclude_none=True),
            "release_execution.schema.json",
        ),
        (
            "release_environment",
            release.environment.model_dump(mode="json", exclude_none=True),
            "release_environment.schema.json",
        ),
        (
            "release_evidence",
            release.evidence.model_dump(mode="json", exclude_none=True),
            "release_evidence.schema.json",
        ),
        (
            "release_claims",
            release.claims.model_dump(mode="json", exclude_none=True),
            "release_claims.schema.json",
        ),
    ]
    for check_name, payload, schema_name in schema_checks:
        schema_path = _schema_path(schema_name)
        if not schema_path.exists():
            _add_issue(
                issues,
                code="schema_missing",
                message=f"required schema file missing: {schema_name}",
            )
            continue
        errors = _validate_against_schema(payload, schema_path)
        for error in errors:
            _add_issue(
                issues,
                code="schema_validation_failed",
                message=f"{check_name}: {error}",
            )
    return issues


def _validate_release_internal_consistency(
    release: LoadedReleaseBundle, dataset: LoadedDatasetManifest
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if release.release.release_id != release.science.release_id:
        _add_issue(
            issues,
            code="release_id_mismatch",
            message="release_id differs between release.json and science.json.",
            details={
                "release_json_release_id": release.release.release_id,
                "science_release_id": release.science.release_id,
            },
        )
    if release.science.dataset_contract.dataset_contract_version != dataset.manifest.dataset_contract_version:
        _add_issue(
            issues,
            code="dataset_contract_version_mismatch",
            message="dataset manifest contract version does not match science.dataset_contract.",
            details={
                "science_dataset_contract_version": release.science.dataset_contract.dataset_contract_version,
                "dataset_manifest_contract_version": dataset.manifest.dataset_contract_version,
            },
        )
    if release.science.sample_unit != dataset.manifest.sample_unit:
        _add_issue(
            issues,
            code="sample_unit_mismatch",
            message="dataset manifest sample_unit does not match release science sample_unit.",
            details={
                "science_sample_unit": release.science.sample_unit,
                "dataset_manifest_sample_unit": dataset.manifest.sample_unit,
            },
        )
    scope_target = release.science.scope.target
    if scope_target is not None and str(scope_target).strip() != str(release.science.target.name).strip():
        _add_issue(
            issues,
            code="scope_target_mismatch",
            message="release science scope.target does not match release science target.name.",
            details={
                "scope_target": scope_target,
                "science_target": release.science.target.name,
            },
        )
    contract_path = (
        PROJECT_ROOT
        / "data"
        / "contracts"
        / f"{release.science.dataset_contract.dataset_contract_version}.json"
    ).resolve()
    if not contract_path.exists():
        _add_issue(
            issues,
            code="dataset_contract_file_missing",
            message="Release science dataset_contract_version has no matching contract file.",
            details={"contract_path": str(contract_path)},
        )
    else:
        contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
        required_columns = list(contract_payload.get("required_columns", []))
        if sorted(required_columns) != sorted(release.science.dataset_contract.required_columns):
            _add_issue(
                issues,
                code="dataset_contract_columns_mismatch",
                message="Release science required columns differ from data/contracts contract.",
                details={
                    "contract_required_columns": sorted(required_columns),
                    "science_required_columns": sorted(
                        release.science.dataset_contract.required_columns
                    ),
                },
            )
        contract_sample_unit = str(contract_payload.get("sample_unit", "")).strip()
        if contract_sample_unit and contract_sample_unit != release.science.sample_unit:
            _add_issue(
                issues,
                code="dataset_contract_sample_unit_mismatch",
                message="Release science sample_unit differs from data/contracts sample_unit.",
                details={
                    "contract_sample_unit": contract_sample_unit,
                    "science_sample_unit": release.science.sample_unit,
                },
            )
    if dataset.manifest.subject_count < release.science.dataset_contract.minimum_subjects:
        _add_issue(
            issues,
            code="dataset_subject_floor_violation",
            message="dataset manifest violates minimum_subjects in science contract.",
            details={
                "minimum_subjects": release.science.dataset_contract.minimum_subjects,
                "subject_count": dataset.manifest.subject_count,
            },
        )
    if release.science.model_policy.tuning_enabled:
        _add_issue(
            issues,
            code="science_tuning_policy_invalid",
            message="release science tuning_enabled must be false for thesis_final_v1.",
        )
    if release.execution.hardware_mode != "cpu_only":
        _add_issue(
            issues,
            code="execution_hardware_invalid",
            message="execution.hardware_mode must be cpu_only.",
            details={"hardware_mode": release.execution.hardware_mode},
        )
    return issues


def _validate_strict_environment(release: LoadedReleaseBundle) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    expected = release.environment.official_python
    actual = platform.python_version()
    if not actual.startswith(f"{expected}.") and actual != expected:
        _add_issue(
            issues,
            code="python_version_mismatch",
            message="Current python version does not match official_python policy.",
            details={"expected": expected, "actual": actual},
        )

    if release.environment.requires_uv_lock:
        uv_lock_path = (PROJECT_ROOT / "uv.lock").resolve()
        if not uv_lock_path.exists():
            _add_issue(
                issues,
                code="uv_lock_missing",
                message="environment.requires_uv_lock=true but uv.lock is missing.",
            )

    normalized_platform = "windows" if sys.platform.startswith("win") else "linux"
    if normalized_platform not in set(release.environment.supported_os):
        _add_issue(
            issues,
            code="unsupported_platform",
            message="Current platform is not allowed by release environment policy.",
            details={
                "platform": normalized_platform,
                "supported_os": release.environment.supported_os,
            },
        )

    pyproject_path = (PROJECT_ROOT / "pyproject.toml").resolve()
    if pyproject_path.exists():
        pyproject_payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        scripts_payload = (
            pyproject_payload.get("project", {}).get("scripts", {})
            if isinstance(pyproject_payload, dict)
            else {}
        )
        if not isinstance(scripts_payload, dict):
            scripts_payload = {}
    else:
        scripts_payload = {}

    missing_scripts = sorted(
        script_name
        for script_name in release.environment.required_project_scripts
        if script_name not in scripts_payload
    )
    if missing_scripts:
        _add_issue(
            issues,
            code="required_scripts_missing",
            message="Release environment required scripts are not registered in pyproject.",
            details={"missing_scripts": missing_scripts},
        )
    return issues


def validate_release(
    *,
    release_ref: Path | str,
    dataset_manifest_path: Path | str,
    strict_environment: bool = False,
) -> ValidatedReleaseContext:
    release = load_release_bundle(release_ref)
    dataset = load_dataset_manifest(dataset_manifest_path)

    issues: list[dict[str, Any]] = []
    issues.extend(_validate_release_schemas(release))
    issues.extend(_validate_dataset_manifest_payload(dataset))
    issues.extend(_validate_release_internal_consistency(release, dataset))
    if strict_environment:
        issues.extend(_validate_strict_environment(release))

    return ValidatedReleaseContext(
        release=release,
        dataset=dataset,
        issues=issues,
        passed=not issues,
    )


__all__ = [
    "ValidatedReleaseContext",
    "validate_dataset_manifest",
    "validate_release",
]
