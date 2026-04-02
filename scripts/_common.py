from __future__ import annotations

"""Compatibility shim re-exporting script helpers.

Status: supported compatibility wrapper; do not extend.
Canonical helper modules live in ``Thesis_ML.script_support``.
"""

from Thesis_ML.script_support.cli import fail
from Thesis_ML.script_support.io import file_sha256, read_json, write_json
from Thesis_ML.script_support.paths import resolve_repo_root
from Thesis_ML.script_support.summaries import write_summary

__all__ = [
    "fail",
    "file_sha256",
    "read_json",
    "resolve_repo_root",
    "write_json",
    "write_summary",
]

