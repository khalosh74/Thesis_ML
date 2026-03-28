from __future__ import annotations

from pathlib import Path

from Thesis_ML.config import resolve_config_id
from Thesis_ML.config.paths import (
    DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH,
    DEFAULT_COMPARISON_SPEC_PATH,
    DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH,
    DEFAULT_THESIS_PROTOCOL_PATH,
)

PROTOCOL_DEFAULT_ALIAS = "protocol.thesis_canonical_default"
PROTOCOL_FROZEN_CONFIRMATORY_ALIAS = "protocol.thesis_confirmatory_frozen"
COMPARISON_DEFAULT_ALIAS = "comparison.grouped_nested_default"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_default_protocol_path() -> Path:
    return DEFAULT_THESIS_PROTOCOL_PATH.resolve()


def frozen_confirmatory_protocol_path() -> Path:
    return DEFAULT_THESIS_CONFIRMATORY_PROTOCOL_PATH.resolve()


def grouped_nested_default_comparison_path() -> Path:
    return DEFAULT_COMPARISON_SPEC_PATH.resolve()


def coarse_affect_default_target_mapping_path() -> Path:
    return DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH.resolve()


def canonical_v1_protocol_variant_path() -> Path:
    repo_root = _repo_root()
    return resolve_config_id(
        "protocol.thesis_canonical_v1",
        source_repo_root=repo_root,
        project_root=repo_root,
    ).resolve()


def canonical_nested_v1_protocol_compat_path() -> Path:
    repo_root = _repo_root()
    return resolve_config_id(
        "protocol.thesis_canonical_nested_v1",
        source_repo_root=repo_root,
        project_root=repo_root,
    ).resolve()


def model_family_v1_comparison_variant_path() -> Path:
    repo_root = _repo_root()
    return resolve_config_id(
        "comparison.model_family_comparison_v1",
        source_repo_root=repo_root,
        project_root=repo_root,
    ).resolve()


def grouped_nested_v1_comparison_compat_path() -> Path:
    repo_root = _repo_root()
    return resolve_config_id(
        "comparison.model_family_grouped_nested_comparison_v1",
        source_repo_root=repo_root,
        project_root=repo_root,
    ).resolve()
