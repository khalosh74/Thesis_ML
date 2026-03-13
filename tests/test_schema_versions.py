from __future__ import annotations

from Thesis_ML.config.schema_versions import (
    ARTIFACT_SCHEMA_VERSION,
    COMPILED_MANIFEST_SCHEMA_VERSION,
    PUBLIC_SCHEMA_VERSIONS,
    SUMMARY_RESULT_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    WORKBOOK_SCHEMA_METADATA_ROWS,
    WORKBOOK_SCHEMA_VERSION,
    WORKBOOK_WRITEBACK_SCHEMA_VERSION,
)


def test_public_schema_versions_map_matches_constants() -> None:
    assert PUBLIC_SCHEMA_VERSIONS == {
        "workbook_schema_version": WORKBOOK_SCHEMA_VERSION,
        "compiled_manifest_schema_version": COMPILED_MANIFEST_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "workbook_writeback_schema_version": WORKBOOK_WRITEBACK_SCHEMA_VERSION,
        "summary_result_schema_version": SUMMARY_RESULT_SCHEMA_VERSION,
    }


def test_supported_schema_versions_include_current_versions() -> None:
    for key, version in PUBLIC_SCHEMA_VERSIONS.items():
        assert key in SUPPORTED_SCHEMA_VERSIONS
        assert version in SUPPORTED_SCHEMA_VERSIONS[key]


def test_workbook_schema_metadata_rows_follow_public_schema_versions() -> None:
    metadata_map = dict(WORKBOOK_SCHEMA_METADATA_ROWS)
    assert metadata_map == PUBLIC_SCHEMA_VERSIONS
