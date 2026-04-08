from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_FEATURE_CACHE,
    ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE,
    ArtifactRecord,
    compute_config_hash,
    find_latest_compatible_artifact,
    get_artifact,
    list_artifacts_for_run,
    register_artifact,
)
from Thesis_ML.experiments.contracts import ReusePolicy, SectionName

EXECUTION_SECTION_ORDER: tuple[SectionName, ...] = (
    SectionName.DATASET_SELECTION,
    SectionName.FEATURE_CACHE_BUILD,
    SectionName.FEATURE_MATRIX_LOAD,
    SectionName.SPATIAL_VALIDATION,
    SectionName.MODEL_FIT,
    SectionName.INTERPRETABILITY,
    SectionName.EVALUATION,
)

_SECTION_TO_INDEX = {name: idx for idx, name in enumerate(EXECUTION_SECTION_ORDER)}
_EARLIEST_SECTION = EXECUTION_SECTION_ORDER[0]
_LATEST_SECTION = EXECUTION_SECTION_ORDER[-1]


def normalize_section_name(
    value: str | SectionName | None,
    *,
    field_name: str,
    default: SectionName,
) -> SectionName:
    if value is None:
        return default
    if isinstance(value, SectionName):
        return value
    try:
        return SectionName(str(value))
    except ValueError as exc:
        allowed = ", ".join(section.value for section in EXECUTION_SECTION_ORDER)
        raise ValueError(f"Invalid {field_name}='{value}'. Allowed values: {allowed}") from exc


def normalize_reuse_policy(value: str | ReusePolicy | None) -> ReusePolicy:
    if value is None:
        return ReusePolicy.AUTO
    if isinstance(value, ReusePolicy):
        return value
    try:
        return ReusePolicy(str(value))
    except ValueError as exc:
        allowed = ", ".join(policy.value for policy in ReusePolicy)
        raise ValueError(f"Invalid reuse_policy='{value}'. Allowed values: {allowed}") from exc


def plan_section_path(
    start_section: str | SectionName | None = None,
    end_section: str | SectionName | None = None,
) -> list[SectionName]:
    start = normalize_section_name(
        start_section,
        field_name="start_section",
        default=_EARLIEST_SECTION,
    )
    end = normalize_section_name(
        end_section,
        field_name="end_section",
        default=_LATEST_SECTION,
    )
    start_idx = _SECTION_TO_INDEX[start]
    end_idx = _SECTION_TO_INDEX[end]
    if start_idx > end_idx:
        raise ValueError(
            "Invalid section range: start_section must be before or equal to end_section."
        )
    return list(EXECUTION_SECTION_ORDER[start_idx : end_idx + 1])


def _requires_base_artifact(start: SectionName) -> bool:
    return _SECTION_TO_INDEX[start] >= _SECTION_TO_INDEX[SectionName.FEATURE_MATRIX_LOAD]


def _expected_base_artifact_type(start: SectionName) -> str:
    if start == SectionName.FEATURE_MATRIX_LOAD:
        return ARTIFACT_TYPE_FEATURE_CACHE
    return ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE


def resolve_base_artifact(
    request: Any,
    *,
    start_section: SectionName,
    reuse_policy: ReusePolicy,
) -> ArtifactRecord | None:
    if not _requires_base_artifact(start_section):
        if request.base_artifact_id:
            raise ValueError(
                "base_artifact_id is only valid when start_section is at or after feature_matrix_load."
            )
        return None

    if reuse_policy == ReusePolicy.DISALLOW:
        raise ValueError(
            "reuse_policy='disallow' is incompatible with segmented execution that starts after "
            "feature_cache_build."
        )

    expected_type = _expected_base_artifact_type(start_section)
    if request.base_artifact_id:
        record = get_artifact(
            registry_path=request.artifact_registry_path,
            artifact_id=request.base_artifact_id,
        )
        if record is None:
            fallback_paths = tuple(
                Path(path)
                for path in getattr(request, "artifact_registry_fallback_paths", ()) or ()
                if path is not None
            )
            for fallback_path in fallback_paths:
                try:
                    if fallback_path.resolve() == Path(request.artifact_registry_path).resolve():
                        continue
                except Exception:
                    continue
                fallback_record = get_artifact(
                    registry_path=fallback_path,
                    artifact_id=request.base_artifact_id,
                )
                if fallback_record is None:
                    continue
                record = register_artifact(
                    registry_path=request.artifact_registry_path,
                    artifact_type=fallback_record.artifact_type,
                    run_id=fallback_record.run_id,
                    upstream_artifact_ids=list(fallback_record.upstream_artifact_ids),
                    config_hash=fallback_record.config_hash,
                    code_ref=fallback_record.code_ref,
                    path=fallback_record.path,
                    status=fallback_record.status,
                    artifact_schema_version=fallback_record.artifact_schema_version,
                    artifact_id=fallback_record.artifact_id,
                    created_at=fallback_record.created_at,
                )
                break
        if record is None:
            raise ValueError(
                f"Base artifact '{request.base_artifact_id}' was not found in registry."
            )
        if record.artifact_type != expected_type:
            raise ValueError(
                f"Incompatible base artifact '{record.artifact_id}': expected artifact_type "
                f"'{expected_type}', got '{record.artifact_type}'."
            )
        return record

    if reuse_policy == ReusePolicy.REQUIRE_EXPLICIT_BASE:
        raise ValueError(
            "reuse_policy='require_explicit_base' requires base_artifact_id for segmented runs."
        )

    if start_section == SectionName.FEATURE_MATRIX_LOAD:
        feature_cache_config_hash = compute_config_hash(
            {
                "index_csv": str(request.index_csv.resolve()),
                "data_root": str(request.data_root.resolve()),
                "cache_dir": str(request.cache_dir.resolve()),
                "group_key": "subject_session_bas",
                "force": False,
            }
        )
        cached = find_latest_compatible_artifact(
            registry_path=request.artifact_registry_path,
            artifact_type=ARTIFACT_TYPE_FEATURE_CACHE,
            config_hash=feature_cache_config_hash,
            code_ref=request.code_ref,
        )
        if cached is not None:
            return cached

    raise ValueError(
        f"start_section='{start_section.value}' requires a compatible base artifact. "
        "Provide base_artifact_id explicitly."
    )


def expected_base_artifact_type(start: SectionName) -> str:
    return _expected_base_artifact_type(start)


def is_after_or_equal(left: SectionName, right: SectionName) -> bool:
    return _SECTION_TO_INDEX[left] >= _SECTION_TO_INDEX[right]


def require_callable(name: str, value: Callable[..., Any] | None) -> Callable[..., Any]:
    if value is None:
        raise ValueError(f"Missing required callable for segment execution: {name}")
    return value


def is_reusable_status(status: str) -> bool:
    lowered = str(status).strip().lower()
    return lowered not in {"failed", "error", "blocked", "running", "planned"}


def find_reusable_run_artifact(
    request: Any,
    *,
    artifact_type: str,
    expected_config_hash: str | None,
) -> ArtifactRecord | None:
    for record in list_artifacts_for_run(
        registry_path=request.artifact_registry_path,
        run_id=request.run_id,
    ):
        if record.artifact_type != artifact_type:
            continue
        if expected_config_hash is not None and record.config_hash != expected_config_hash:
            continue
        if not is_reusable_status(record.status):
            continue
        if not Path(record.path).exists():
            continue
        return record
    return None


__all__ = [
    "EXECUTION_SECTION_ORDER",
    "expected_base_artifact_type",
    "find_reusable_run_artifact",
    "is_after_or_equal",
    "normalize_reuse_policy",
    "plan_section_path",
    "require_callable",
    "resolve_base_artifact",
]
