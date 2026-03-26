from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _stable_json_sha256(payload: Any) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dataframe_records_sha256(frame: pd.DataFrame, *, columns: list[str]) -> str:
    subset = frame.loc[:, columns].copy()
    subset = subset.astype(str)
    subset = subset.sort_values(by=columns, kind="mergesort").reset_index(drop=True)
    return _stable_json_sha256(subset.to_dict(orient="records"))


def collect_dataset_fingerprint(
    *,
    index_csv: Path,
    selected_index_df: pd.DataFrame,
    index_row_count: int | None = None,
    target_column: str,
    cv_mode: str,
    subject: str | None,
    train_subject: str | None,
    test_subject: str | None,
    filter_task: str | None,
    filter_modality: str | None,
    selected_beta_path_sha256: str | None = None,
    cv_split_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    selected = selected_index_df.copy()
    selected_subjects = sorted({str(value) for value in selected.get("subject", [])})
    selected_sessions = sorted({str(value) for value in selected.get("session", [])})
    target_counts = (
        selected[target_column].astype(str).value_counts(dropna=False).sort_index().to_dict()
        if target_column in selected.columns
        else {}
    )

    hash_columns = ["sample_id"]
    if target_column in selected.columns:
        hash_columns.append(target_column)

    beta_path_column = "beta_path_canonical" if "beta_path_canonical" in selected.columns else "beta_path"
    if selected_beta_path_sha256 is None and beta_path_column in selected.columns:
        selected_beta_path_sha256 = dataframe_records_sha256(
            selected,
            columns=["sample_id", beta_path_column],
        )

    return {
        "index_csv": str(Path(index_csv).resolve()),
        "index_csv_sha256": file_sha256(Path(index_csv)),
        "index_file_size_bytes": int(Path(index_csv).stat().st_size),
        "index_row_count": int(index_row_count)
        if index_row_count is not None
        else int(pd.read_csv(index_csv).shape[0]),
        "selected_row_count": int(selected.shape[0]),
        "selected_sample_id_sha256": dataframe_records_sha256(
            selected,
            columns=hash_columns,
        ),
        "selected_subjects": selected_subjects,
        "selected_sessions": selected_sessions,
        "target_distribution": {str(key): int(value) for key, value in target_counts.items()},
        "target_column": str(target_column),
        "cv_mode": str(cv_mode),
        "subject": subject,
        "train_subject": train_subject,
        "test_subject": test_subject,
        "filter_task": filter_task,
        "filter_modality": filter_modality,
        "selected_beta_path_sha256": selected_beta_path_sha256,
        "cv_split_manifest_sha256": cv_split_manifest_sha256,
    }


def _run_git_command(args: list[str]) -> str | None:
    try:
        process = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = process.stdout.strip()
    return value or None


def collect_git_provenance() -> dict[str, Any]:
    commit = _run_git_command(["rev-parse", "HEAD"])
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git_command(["status", "--porcelain"])
    dirty = bool(status)
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
    }


__all__ = [
    "collect_dataset_fingerprint",
    "collect_git_provenance",
    "dataframe_records_sha256",
    "file_sha256",
]
