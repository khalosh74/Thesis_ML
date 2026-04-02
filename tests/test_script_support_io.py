from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from Thesis_ML.script_support.io import file_sha256, read_json, write_json
from Thesis_ML.script_support.summaries import write_summary


def test_file_sha256_matches_hashlib_digest(tmp_path: Path) -> None:
    payload = b"script-support-io-check\n"
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    assert file_sha256(file_path) == expected


def test_read_write_json_round_trip(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "summary.json"
    payload = {"status": "ok", "count": 3}

    write_json(output_path, payload)
    assert read_json(output_path) == payload
    assert output_path.read_text(encoding="utf-8").endswith("\n")


def test_read_json_requires_object_payload(tmp_path: Path) -> None:
    path = tmp_path / "list_payload.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON payload must be an object"):
        read_json(path)


def test_write_summary_writes_json_and_returns_path(tmp_path: Path) -> None:
    summary_path = tmp_path / "out" / "summary.json"
    payload = {"name": "demo", "passed": True}

    written = write_summary(summary_path, payload)

    assert written == summary_path
    assert read_json(summary_path) == payload


def test_generate_demo_dataset_uses_shared_hash_helper() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_demo_dataset.py"
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    defined_function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "_file_sha256" not in defined_function_names
    assert "from Thesis_ML.script_support.io import file_sha256" in source

