from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class _ScopeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ScopeCounts(_ScopeModel):
    by_subject: dict[str, int] = Field(default_factory=dict)
    by_session: dict[str, int] = Field(default_factory=dict)
    by_task: dict[str, int] = Field(default_factory=dict)
    by_modality: dict[str, int] = Field(default_factory=dict)
    by_target: dict[str, int] = Field(default_factory=dict)


class ScopeExclusionsSummary(_ScopeModel):
    input_rows: int = Field(ge=0)
    selected_rows: int = Field(ge=0)
    excluded_rows: int = Field(ge=0)
    by_stage: dict[str, int] = Field(default_factory=dict)
    by_reason: dict[str, int] = Field(default_factory=dict)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class CompiledScopeManifest(_ScopeModel):
    schema_version: str = "release-compiled-scope-v1"
    release_id: str = Field(min_length=1)
    release_version: str = Field(min_length=1)
    science_hash: str = Field(min_length=64, max_length=64)
    dataset_manifest_path: str = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    target_mapping_path: str = Field(min_length=1)
    target_mapping_hash: str = Field(min_length=64, max_length=64)
    scope_subjects: list[str] = Field(min_length=1)
    scope_tasks: list[str] = Field(min_length=1)
    scope_modality: str = Field(min_length=1)
    cv_scope_mode: str = Field(min_length=1)
    selected_row_count: int = Field(ge=1)
    selected_sample_ids_sha256: str = Field(min_length=64, max_length=64)
    selected_samples_csv: str = Field(min_length=1)
    counts: ScopeCounts
    exclusions_summary: ScopeExclusionsSummary
    generated_at_utc: str = Field(min_length=1)


class CompiledScopeResult(_ScopeModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    selected_index_df: pd.DataFrame
    selected_samples_path: Path
    scope_manifest_path: Path
    selected_sample_ids: list[str] = Field(default_factory=list)
    selection_summary: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "CompiledScopeManifest",
    "CompiledScopeResult",
    "ScopeCounts",
    "ScopeExclusionsSummary",
]

