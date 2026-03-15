from __future__ import annotations

# Public contract versions.
WORKBOOK_SCHEMA_VERSION = "workbook-v1"
COMPILED_MANIFEST_SCHEMA_VERSION = "compiled-manifest-v1"
ARTIFACT_SCHEMA_VERSION = "artifact-v1"
WORKBOOK_WRITEBACK_SCHEMA_VERSION = "workbook-writeback-v1"
SUMMARY_RESULT_SCHEMA_VERSION = "summary-result-v1"
THESIS_PROTOCOL_SCHEMA_VERSION = "thesis-protocol-v1"

SUPPORTED_WORKBOOK_SCHEMA_VERSIONS = frozenset({WORKBOOK_SCHEMA_VERSION})
SUPPORTED_COMPILED_MANIFEST_SCHEMA_VERSIONS = frozenset({COMPILED_MANIFEST_SCHEMA_VERSION})
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = frozenset({ARTIFACT_SCHEMA_VERSION})
SUPPORTED_WORKBOOK_WRITEBACK_SCHEMA_VERSIONS = frozenset({WORKBOOK_WRITEBACK_SCHEMA_VERSION})
SUPPORTED_SUMMARY_RESULT_SCHEMA_VERSIONS = frozenset({SUMMARY_RESULT_SCHEMA_VERSION})
SUPPORTED_THESIS_PROTOCOL_SCHEMA_VERSIONS = frozenset({THESIS_PROTOCOL_SCHEMA_VERSION})

# Authoritative schema/version maps for operators and maintainers.
PUBLIC_SCHEMA_VERSIONS: dict[str, str] = {
    "workbook_schema_version": WORKBOOK_SCHEMA_VERSION,
    "compiled_manifest_schema_version": COMPILED_MANIFEST_SCHEMA_VERSION,
    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
    "workbook_writeback_schema_version": WORKBOOK_WRITEBACK_SCHEMA_VERSION,
    "summary_result_schema_version": SUMMARY_RESULT_SCHEMA_VERSION,
    "thesis_protocol_schema_version": THESIS_PROTOCOL_SCHEMA_VERSION,
}
SUPPORTED_SCHEMA_VERSIONS: dict[str, frozenset[str]] = {
    "workbook_schema_version": SUPPORTED_WORKBOOK_SCHEMA_VERSIONS,
    "compiled_manifest_schema_version": SUPPORTED_COMPILED_MANIFEST_SCHEMA_VERSIONS,
    "artifact_schema_version": SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
    "workbook_writeback_schema_version": SUPPORTED_WORKBOOK_WRITEBACK_SCHEMA_VERSIONS,
    "summary_result_schema_version": SUPPORTED_SUMMARY_RESULT_SCHEMA_VERSIONS,
    "thesis_protocol_schema_version": SUPPORTED_THESIS_PROTOCOL_SCHEMA_VERSIONS,
}

# Deterministic workbook metadata placement (README sheet).
WORKBOOK_SCHEMA_METADATA_START_ROW = 45
WORKBOOK_SCHEMA_METADATA_KEY_COLUMN = 1
WORKBOOK_SCHEMA_METADATA_VALUE_COLUMN = 2
WORKBOOK_SCHEMA_METADATA_HEADER_KEY = "Schema_Metadata_Key"
WORKBOOK_SCHEMA_METADATA_HEADER_VALUE = "Value"
WORKBOOK_SCHEMA_METADATA_ROWS: tuple[tuple[str, str], ...] = (
    ("workbook_schema_version", PUBLIC_SCHEMA_VERSIONS["workbook_schema_version"]),
    (
        "compiled_manifest_schema_version",
        PUBLIC_SCHEMA_VERSIONS["compiled_manifest_schema_version"],
    ),
    ("artifact_schema_version", PUBLIC_SCHEMA_VERSIONS["artifact_schema_version"]),
    (
        "workbook_writeback_schema_version",
        PUBLIC_SCHEMA_VERSIONS["workbook_writeback_schema_version"],
    ),
    ("summary_result_schema_version", PUBLIC_SCHEMA_VERSIONS["summary_result_schema_version"]),
    ("thesis_protocol_schema_version", PUBLIC_SCHEMA_VERSIONS["thesis_protocol_schema_version"]),
)
