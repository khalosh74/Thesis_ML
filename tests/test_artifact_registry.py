from __future__ import annotations

import sqlite3
from pathlib import Path

from Thesis_ML.artifacts.registry import (
    ARTIFACT_SCHEMA_VERSION,
    ARTIFACT_TYPE_EXPERIMENT_REPORT,
    ARTIFACT_TYPE_FEATURE_CACHE,
    compute_config_hash,
    find_latest_compatible_artifact,
    get_artifact,
    list_artifacts_for_run,
    register_artifact,
)


def test_register_and_get_artifact(tmp_path: Path) -> None:
    registry_path = tmp_path / "artifact_registry.sqlite3"
    report_path = tmp_path / "reports" / "run_001"
    report_path.mkdir(parents=True, exist_ok=True)
    config_hash = compute_config_hash({"target": "coarse_affect", "cv": "loso_session"})

    record = register_artifact(
        registry_path=registry_path,
        artifact_type=ARTIFACT_TYPE_EXPERIMENT_REPORT,
        run_id="run_001",
        upstream_artifact_ids=[],
        config_hash=config_hash,
        code_ref="abc123",
        path=report_path,
        status="created",
    )

    loaded = get_artifact(registry_path=registry_path, artifact_id=record.artifact_id)
    assert loaded is not None
    assert loaded.artifact_id == record.artifact_id
    assert loaded.run_id == "run_001"
    assert loaded.artifact_type == ARTIFACT_TYPE_EXPERIMENT_REPORT
    assert loaded.path == str(report_path.resolve())
    assert record.artifact_schema_version == ARTIFACT_SCHEMA_VERSION
    assert loaded.artifact_schema_version == ARTIFACT_SCHEMA_VERSION


def test_find_latest_compatible_and_list_for_run(tmp_path: Path) -> None:
    registry_path = tmp_path / "artifact_registry.sqlite3"
    cache_path_a = tmp_path / "cache" / "a.csv"
    cache_path_b = tmp_path / "cache" / "b.csv"
    cache_path_a.parent.mkdir(parents=True, exist_ok=True)
    cache_path_a.write_text("a\n", encoding="utf-8")
    cache_path_b.write_text("b\n", encoding="utf-8")

    hash_value = compute_config_hash({"cache_key": "same"})
    first = register_artifact(
        registry_path=registry_path,
        artifact_type=ARTIFACT_TYPE_FEATURE_CACHE,
        run_id="run_001",
        upstream_artifact_ids=[],
        config_hash=hash_value,
        code_ref="abc123",
        path=cache_path_a,
        status="created",
        created_at="2026-01-01T00:00:00+00:00",
    )
    second = register_artifact(
        registry_path=registry_path,
        artifact_type=ARTIFACT_TYPE_FEATURE_CACHE,
        run_id="run_001",
        upstream_artifact_ids=[first.artifact_id],
        config_hash=hash_value,
        code_ref="abc123",
        path=cache_path_b,
        status="created",
        created_at="2026-01-02T00:00:00+00:00",
    )

    latest = find_latest_compatible_artifact(
        registry_path=registry_path,
        artifact_type=ARTIFACT_TYPE_FEATURE_CACHE,
        config_hash=hash_value,
        code_ref="abc123",
    )
    assert latest is not None
    assert latest.artifact_id == second.artifact_id

    listed = list_artifacts_for_run(registry_path=registry_path, run_id="run_001")
    assert len(listed) == 2
    assert {item.artifact_id for item in listed} == {first.artifact_id, second.artifact_id}
    assert all(item.artifact_schema_version == ARTIFACT_SCHEMA_VERSION for item in listed)


def test_register_artifact_migrates_legacy_schema(tmp_path: Path) -> None:
    registry_path = tmp_path / "artifact_registry.sqlite3"
    with sqlite3.connect(str(registry_path)) as connection:
        connection.execute(
            """
            CREATE TABLE artifacts (
                artifact_id TEXT PRIMARY KEY,
                artifact_type TEXT NOT NULL,
                run_id TEXT,
                upstream_artifact_ids TEXT NOT NULL,
                config_hash TEXT,
                code_ref TEXT,
                path TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()

    record = register_artifact(
        registry_path=registry_path,
        artifact_type=ARTIFACT_TYPE_EXPERIMENT_REPORT,
        run_id="run_legacy",
        upstream_artifact_ids=[],
        config_hash=compute_config_hash({"legacy": True}),
        code_ref="abc123",
        path=tmp_path / "legacy_report",
        status="created",
    )
    assert record.artifact_schema_version == ARTIFACT_SCHEMA_VERSION
    loaded = get_artifact(registry_path=registry_path, artifact_id=record.artifact_id)
    assert loaded is not None
    assert loaded.artifact_schema_version == ARTIFACT_SCHEMA_VERSION
