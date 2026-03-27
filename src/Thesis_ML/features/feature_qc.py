from __future__ import annotations

from typing import Any

import numpy as np

FEATURE_QC_SAMPLE_FIELDS: tuple[str, ...] = (
    "n_features",
    "n_nan_before_repair",
    "n_posinf_before_repair",
    "n_neginf_before_repair",
    "n_nonfinite_before_repair",
    "repair_fraction",
    "n_zero_after_repair",
    "all_zero_vector",
    "constant_vector",
    "mean_after_repair",
    "std_after_repair",
    "l2_norm_after_repair",
    "max_abs_after_repair",
)


def _as_vector_1d(value: Any, *, field_name: str) -> np.ndarray:
    vector = np.asarray(value)
    if vector.ndim != 1:
        raise ValueError(f"{field_name} must be a 1D vector.")
    return vector


def compute_sample_feature_qc(
    vector_before_repair: Any,
    vector_after_repair: Any,
) -> dict[str, Any]:
    before = _as_vector_1d(vector_before_repair, field_name="vector_before_repair")
    after = _as_vector_1d(vector_after_repair, field_name="vector_after_repair")
    if before.shape[0] != after.shape[0]:
        raise ValueError(
            "vector_before_repair and vector_after_repair must have matching feature length."
        )

    before_float = np.asarray(before, dtype=np.float64)
    after_float = np.asarray(after, dtype=np.float64)
    n_features = int(after_float.shape[0])

    n_nan = int(np.isnan(before_float).sum())
    n_posinf = int(np.isposinf(before_float).sum())
    n_neginf = int(np.isneginf(before_float).sum())
    n_nonfinite = int((~np.isfinite(before_float)).sum())
    n_zero_after = int((after_float == 0.0).sum())
    std_after = float(np.std(after_float)) if n_features > 0 else 0.0

    return {
        "n_features": n_features,
        "n_nan_before_repair": n_nan,
        "n_posinf_before_repair": n_posinf,
        "n_neginf_before_repair": n_neginf,
        "n_nonfinite_before_repair": n_nonfinite,
        "repair_fraction": (float(n_nonfinite) / float(n_features)) if n_features > 0 else 0.0,
        "n_zero_after_repair": n_zero_after,
        "all_zero_vector": bool(n_features > 0 and n_zero_after == n_features),
        "constant_vector": bool(n_features > 0 and std_after == 0.0),
        "mean_after_repair": float(np.mean(after_float)) if n_features > 0 else 0.0,
        "std_after_repair": std_after,
        "l2_norm_after_repair": float(np.linalg.norm(after_float, ord=2)),
        "max_abs_after_repair": float(np.max(np.abs(after_float))) if n_features > 0 else 0.0,
    }


def summarize_group_feature_qc(sample_qc_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not sample_qc_rows:
        raise ValueError("sample_qc_rows must not be empty.")

    group_id = str(sample_qc_rows[0].get("group_id", ""))
    n_features = int(sample_qc_rows[0]["n_features"])
    for row in sample_qc_rows:
        if int(row["n_features"]) != n_features:
            raise ValueError("All sample QC rows must share the same n_features.")

    repair_fractions = [float(row["repair_fraction"]) for row in sample_qc_rows]
    std_values = [float(row["std_after_repair"]) for row in sample_qc_rows]
    n_samples_with_any_repair = int(
        sum(int(row["n_nonfinite_before_repair"]) > 0 for row in sample_qc_rows)
    )
    n_all_zero_vectors = int(sum(bool(row["all_zero_vector"]) for row in sample_qc_rows))
    n_constant_vectors = int(sum(bool(row["constant_vector"]) for row in sample_qc_rows))

    return {
        "group_id": group_id,
        "n_samples": int(len(sample_qc_rows)),
        "n_features": int(n_features),
        "n_samples_with_any_repair": n_samples_with_any_repair,
        "max_repair_fraction": float(max(repair_fractions)),
        "mean_repair_fraction": float(np.mean(np.asarray(repair_fractions, dtype=np.float64))),
        "n_all_zero_vectors": n_all_zero_vectors,
        "n_constant_vectors": n_constant_vectors,
        "mean_vector_std_after_repair": float(np.mean(np.asarray(std_values, dtype=np.float64))),
        "min_vector_std_after_repair": float(min(std_values)),
    }


def merge_qc_into_metadata_records(
    metadata_records: list[dict[str, Any]],
    sample_qc_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(metadata_records) != len(sample_qc_rows):
        raise ValueError("metadata_records and sample_qc_rows must have matching row counts.")

    sample_qc_by_id: dict[str, dict[str, Any]] = {}
    for qc_row in sample_qc_rows:
        sample_id = str(qc_row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("sample_qc_rows must include non-empty sample_id values.")
        if sample_id in sample_qc_by_id:
            raise ValueError(f"sample_qc_rows contains duplicate sample_id '{sample_id}'.")
        sample_qc_by_id[sample_id] = qc_row

    merged_rows: list[dict[str, Any]] = []
    for metadata in metadata_records:
        sample_id = str(metadata.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("metadata_records must include non-empty sample_id values.")
        qc_row = sample_qc_by_id.get(sample_id)
        if qc_row is None:
            raise ValueError(f"Missing QC row for sample_id '{sample_id}'.")
        merged = dict(metadata)
        for field_name in FEATURE_QC_SAMPLE_FIELDS:
            merged[field_name] = qc_row[field_name]
        merged_rows.append(merged)
    return merged_rows


__all__ = [
    "FEATURE_QC_SAMPLE_FIELDS",
    "compute_sample_feature_qc",
    "summarize_group_feature_qc",
    "merge_qc_into_metadata_records",
]
