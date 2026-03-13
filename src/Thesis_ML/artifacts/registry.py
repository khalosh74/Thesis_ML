from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from Thesis_ML.config.schema_versions import (
    ARTIFACT_SCHEMA_VERSION,
    SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
)

ARTIFACT_TYPE_FEATURE_CACHE = "feature_cache"
ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE = "feature_matrix_bundle"
ARTIFACT_TYPE_EXPERIMENT_REPORT = "experiment_report"
ARTIFACT_TYPE_METRICS_BUNDLE = "metrics_bundle"
ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE = "interpretability_bundle"

SUPPORTED_ARTIFACT_TYPES = {
    ARTIFACT_TYPE_FEATURE_CACHE,
    ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE,
    ARTIFACT_TYPE_EXPERIMENT_REPORT,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
}


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    artifact_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    run_id: str | None = None
    upstream_artifact_ids: list[str] = Field(default_factory=list)
    config_hash: str | None = None
    code_ref: str | None = None
    path: str = Field(min_length=1)
    status: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    artifact_schema_version: str = ARTIFACT_SCHEMA_VERSION

    @field_validator("artifact_type")
    @classmethod
    def _validate_artifact_type(cls, value: str) -> str:
        if value not in SUPPORTED_ARTIFACT_TYPES:
            allowed = ", ".join(sorted(SUPPORTED_ARTIFACT_TYPES))
            raise ValueError(f"Unsupported artifact_type '{value}'. Allowed values: {allowed}")
        return value

    @field_validator("artifact_schema_version")
    @classmethod
    def _validate_artifact_schema_version(cls, value: str) -> str:
        if value not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
            allowed = ", ".join(sorted(SUPPORTED_ARTIFACT_SCHEMA_VERSIONS))
            raise ValueError(
                f"Unsupported artifact schema version '{value}'. Allowed values: {allowed}"
            )
        return value


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def compute_config_hash(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_artifact_id(
    artifact_type: str,
    run_id: str | None,
    path: str,
    config_hash: str | None,
    code_ref: str | None,
) -> str:
    basis = "|".join([artifact_type, run_id or "", path, config_hash or "", code_ref or ""])
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]
    return f"{artifact_type}_{digest}"


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            artifact_type TEXT NOT NULL,
            run_id TEXT,
            upstream_artifact_ids TEXT NOT NULL,
            config_hash TEXT,
            code_ref TEXT,
            path TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            artifact_schema_version TEXT NOT NULL
        )
        """
    )
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(artifacts)").fetchall()}
    if "artifact_schema_version" not in columns:
        escaped = ARTIFACT_SCHEMA_VERSION.replace("'", "''")
        connection.execute(
            "ALTER TABLE artifacts ADD COLUMN artifact_schema_version "
            f"TEXT NOT NULL DEFAULT '{escaped}'"
        )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON artifacts(run_id)")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_artifacts_type_hash "
        "ON artifacts(artifact_type, config_hash)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_artifacts_created_at ON artifacts(created_at)"
    )
    connection.commit()


def _connect(registry_path: Path) -> sqlite3.Connection:
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(registry_path))
    connection.row_factory = sqlite3.Row
    _ensure_schema(connection)
    return connection


def _row_to_record(row: sqlite3.Row) -> ArtifactRecord:
    upstream_raw = row["upstream_artifact_ids"]
    upstream = json.loads(upstream_raw) if upstream_raw else []
    return ArtifactRecord(
        artifact_id=str(row["artifact_id"]),
        artifact_type=str(row["artifact_type"]),
        run_id=(str(row["run_id"]) if row["run_id"] is not None else None),
        upstream_artifact_ids=[str(value) for value in upstream],
        config_hash=(str(row["config_hash"]) if row["config_hash"] is not None else None),
        code_ref=(str(row["code_ref"]) if row["code_ref"] is not None else None),
        path=str(row["path"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        artifact_schema_version=str(row["artifact_schema_version"]),
    )


def register_artifact(
    *,
    registry_path: Path,
    artifact_type: str,
    run_id: str | None,
    upstream_artifact_ids: list[str] | None,
    config_hash: str | None,
    code_ref: str | None,
    path: Path | str,
    status: str,
    artifact_schema_version: str = ARTIFACT_SCHEMA_VERSION,
    artifact_id: str | None = None,
    created_at: str | None = None,
) -> ArtifactRecord:
    normalized_path = str(Path(path).resolve())
    resolved_created_at = created_at or _utc_timestamp()
    upstream = [str(value) for value in (upstream_artifact_ids or [])]
    resolved_artifact_id = artifact_id or _build_artifact_id(
        artifact_type=artifact_type,
        run_id=run_id,
        path=normalized_path,
        config_hash=config_hash,
        code_ref=code_ref,
    )

    record = ArtifactRecord(
        artifact_id=resolved_artifact_id,
        artifact_type=artifact_type,
        run_id=run_id,
        upstream_artifact_ids=upstream,
        config_hash=config_hash,
        code_ref=code_ref,
        path=normalized_path,
        status=status,
        created_at=resolved_created_at,
        artifact_schema_version=artifact_schema_version,
    )

    with _connect(Path(registry_path)) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO artifacts (
                artifact_id,
                artifact_type,
                run_id,
                upstream_artifact_ids,
                config_hash,
                code_ref,
                path,
                status,
                created_at,
                artifact_schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.artifact_id,
                record.artifact_type,
                record.run_id,
                json.dumps(record.upstream_artifact_ids, separators=(",", ":")),
                record.config_hash,
                record.code_ref,
                record.path,
                record.status,
                record.created_at,
                record.artifact_schema_version,
            ),
        )
        connection.commit()
    return record


def get_artifact(*, registry_path: Path, artifact_id: str) -> ArtifactRecord | None:
    with _connect(Path(registry_path)) as connection:
        row = connection.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_record(row)


def list_artifacts_for_run(*, registry_path: Path, run_id: str) -> list[ArtifactRecord]:
    with _connect(Path(registry_path)) as connection:
        rows = connection.execute(
            "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at DESC, artifact_id DESC",
            (run_id,),
        ).fetchall()
    return [_row_to_record(row) for row in rows]


def find_latest_compatible_artifact(
    *,
    registry_path: Path,
    artifact_type: str,
    config_hash: str | None = None,
    code_ref: str | None = None,
    status: str | None = "created",
) -> ArtifactRecord | None:
    query = "SELECT * FROM artifacts WHERE artifact_type = ?"
    values: list[Any] = [artifact_type]
    if config_hash is not None:
        query += " AND config_hash = ?"
        values.append(config_hash)
    if code_ref is not None:
        query += " AND code_ref = ?"
        values.append(code_ref)
    if status is not None:
        query += " AND status = ?"
        values.append(status)
    query += " ORDER BY created_at DESC, artifact_id DESC LIMIT 1"

    with _connect(Path(registry_path)) as connection:
        row = connection.execute(query, values).fetchone()
    if row is None:
        return None
    return _row_to_record(row)
