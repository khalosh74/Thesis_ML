from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Thesis_ML.experiments.spatial_validation import (
    SPATIAL_AFFINE_ATOL,
    build_spatial_compatibility_report,
    raise_spatial_compatibility_error,
)
from Thesis_ML.features.feature_qc import FEATURE_QC_SAMPLE_FIELDS

LOGGER = logging.getLogger(__name__)


def _canonical_metadata_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _resolve_metadata_beta_path(metadata: dict[str, Any]) -> str:
    for key in ("beta_path_canonical", "beta_path"):
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


_GROUP_QC_MANIFEST_COLUMNS: tuple[str, ...] = (
    "feature_qc_n_samples",
    "feature_qc_n_features",
    "feature_qc_n_samples_with_any_repair",
    "feature_qc_max_repair_fraction",
    "feature_qc_mean_repair_fraction",
    "feature_qc_n_all_zero_vectors",
    "feature_qc_n_constant_vectors",
    "feature_qc_mean_vector_std_after_repair",
    "feature_qc_min_vector_std_after_repair",
)


def _manifest_row_expects_feature_qc(row: pd.Series) -> bool:
    return any(column_name in row.index for column_name in _GROUP_QC_MANIFEST_COLUMNS)


def _validate_sample_qc_fields(
    *,
    metadata_record: dict[str, Any],
    require_qc: bool,
    cache_path: Path,
    group_id: str,
    sample_id: str,
) -> None:
    present_fields = {
        field_name
        for field_name in FEATURE_QC_SAMPLE_FIELDS
        if field_name in metadata_record
    }
    if present_fields and present_fields != set(FEATURE_QC_SAMPLE_FIELDS):
        missing_fields = sorted(set(FEATURE_QC_SAMPLE_FIELDS) - present_fields)
        raise ValueError(
            "Cache metadata row contains partial feature QC fields, which is not allowed. "
            f"group_id='{group_id}', sample_id='{sample_id}', cache_path='{cache_path}', "
            f"missing_fields={missing_fields}."
        )
    if require_qc and present_fields != set(FEATURE_QC_SAMPLE_FIELDS):
        missing_fields = sorted(set(FEATURE_QC_SAMPLE_FIELDS) - present_fields)
        raise ValueError(
            "Upgraded cache metadata row is missing required feature QC fields. "
            f"group_id='{group_id}', sample_id='{sample_id}', cache_path='{cache_path}', "
            f"missing_fields={missing_fields}."
        )


def load_features_from_cache(
    index_df: pd.DataFrame,
    cache_manifest_path: Path,
    spatial_report_path: Path | None = None,
    affine_atol: float = SPATIAL_AFFINE_ATOL,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_csv(cache_manifest_path)
    if manifest.empty:
        raise ValueError(f"Cache manifest is empty: {cache_manifest_path}")
    required_manifest_columns = {"group_id", "cache_path"}
    missing_manifest_columns = sorted(required_manifest_columns - set(manifest.columns))
    if missing_manifest_columns:
        raise ValueError(
            "Cache manifest is missing required columns: "
            + ", ".join(missing_manifest_columns)
        )

    duplicate_group_mask = manifest["group_id"].astype(str).duplicated(keep=False)
    if bool(duplicate_group_mask.any()):
        duplicates = (
            manifest.loc[duplicate_group_mask, "group_id"].astype(str).drop_duplicates().tolist()
        )
        raise ValueError(
            "Cache manifest contains duplicate group_id values, which is not allowed. "
            f"duplicates={duplicates[:10]}"
        )

    duplicate_cache_path_mask = manifest["cache_path"].astype(str).duplicated(keep=False)
    if bool(duplicate_cache_path_mask.any()):
        duplicates = (
            manifest.loc[duplicate_cache_path_mask, "cache_path"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "Cache manifest contains duplicate cache_path values, which is not allowed. "
            f"duplicates={duplicates[:10]}"
        )

    selected_ids = set(index_df["sample_id"].astype(str))
    feature_map: dict[str, np.ndarray] = {}
    metadata_map: dict[str, dict[str, Any]] = {}
    sample_source_map: dict[str, dict[str, Any]] = {}
    selected_cache_groups: list[dict[str, Any]] = []

    for _, row in manifest.iterrows():
        cache_path = Path(str(row["cache_path"]))
        if not cache_path.exists():
            LOGGER.warning("Skipping missing cache file: %s", cache_path)
            continue

        with np.load(cache_path, allow_pickle=False) as npz:
            x_block = np.asarray(npz["X"], dtype=np.float32)
            metadata_json = str(npz["metadata_json"].item())
            metadata_records = json.loads(metadata_json)
            raw_signature = None
            if "spatial_signature_json" in npz.files:
                raw_signature = json.loads(str(npz["spatial_signature_json"].item()))
            group_qc_summary = None
            if "group_qc_summary_json" in npz.files:
                group_qc_summary = json.loads(str(npz["group_qc_summary_json"].item()))
            group_id = (
                str(npz["group_id"].item())
                if "group_id" in npz.files
                else str(row.get("group_id", cache_path.name))
            )

        requires_qc_fields = _manifest_row_expects_feature_qc(row)
        if requires_qc_fields and group_qc_summary is None:
            raise ValueError(
                "Cache manifest indicates upgraded feature QC fields, but cache group is missing "
                "group_qc_summary_json. "
                f"group_id='{group_id}', cache_path='{cache_path}'."
            )

        if x_block.shape[0] != len(metadata_records):
            raise ValueError(
                f"Cache row mismatch in {cache_path}: {x_block.shape[0]} != {len(metadata_records)}"
            )

        selected_in_group = 0
        seen_in_group: set[str] = set()
        for row_idx, metadata in enumerate(metadata_records):
            sample_id = str(metadata.get("sample_id", ""))
            if sample_id:
                if sample_id in seen_in_group:
                    raise ValueError(
                        "Cache file contains duplicate sample_id rows in the same group. "
                        f"group_id='{group_id}', sample_id='{sample_id}', cache_path='{cache_path}'."
                    )
                seen_in_group.add(sample_id)
                _validate_sample_qc_fields(
                    metadata_record=metadata,
                    require_qc=requires_qc_fields,
                    cache_path=cache_path,
                    group_id=str(group_id),
                    sample_id=sample_id,
                )
            if sample_id and sample_id in selected_ids:
                current_vector = np.asarray(x_block[row_idx], dtype=np.float32)
                metadata_payload = dict(metadata)
                canonical_beta_path = _resolve_metadata_beta_path(metadata_payload)
                metadata_json = _canonical_metadata_json(metadata_payload)

                existing_entry = sample_source_map.get(sample_id)
                if existing_entry is not None:
                    previous_group_id = str(existing_entry["group_id"])
                    previous_cache_path = str(existing_entry["cache_path"])
                    previous_metadata_json = str(existing_entry["metadata_json"])
                    previous_beta_path = str(existing_entry["beta_path"])
                    previous_vector = np.asarray(existing_entry["vector"], dtype=np.float32)

                    differences: list[str] = []
                    if previous_metadata_json != metadata_json:
                        differences.append("metadata_mismatch")
                    if previous_beta_path != canonical_beta_path:
                        differences.append("canonical_beta_path_mismatch")
                    if previous_vector.shape != current_vector.shape or not np.array_equal(
                        previous_vector, current_vector
                    ):
                        differences.append("feature_vector_mismatch")

                    if previous_group_id != str(group_id):
                        raise ValueError(
                            "Duplicate sample_id detected across multiple cache groups, which is not allowed. "
                            f"sample_id='{sample_id}', first_group='{previous_group_id}', "
                            f"second_group='{group_id}', first_cache='{previous_cache_path}', "
                            f"second_cache='{cache_path}', differences={differences or ['none']}."
                        )
                    raise ValueError(
                        "Duplicate sample_id detected within cache loading with conflicting entries. "
                        f"sample_id='{sample_id}', group_id='{group_id}', cache_path='{cache_path}', "
                        f"differences={differences or ['none']}."
                    )

                feature_map[sample_id] = current_vector
                metadata_map[sample_id] = metadata_payload
                sample_source_map[sample_id] = {
                    "group_id": str(group_id),
                    "cache_path": str(cache_path.resolve()),
                    "metadata_json": metadata_json,
                    "beta_path": canonical_beta_path,
                    "vector": current_vector.copy(),
                }
                selected_in_group += 1

        if selected_in_group > 0:
            selected_cache_groups.append(
                {
                    "group_id": group_id,
                    "cache_path": str(cache_path.resolve()),
                    "n_selected_samples": int(selected_in_group),
                    "n_features": int(x_block.shape[1]),
                    "raw_signature": raw_signature,
                    "group_qc_summary": group_qc_summary,
                }
            )

    spatial_report = build_spatial_compatibility_report(
        cache_groups=selected_cache_groups,
        affine_atol=affine_atol,
    )
    if spatial_report_path is not None:
        spatial_report_path.write_text(
            f"{json.dumps(spatial_report, indent=2)}\n", encoding="utf-8"
        )
    if not spatial_report["passed"]:
        raise_spatial_compatibility_error(spatial_report)

    vectors: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    missing_samples: list[str] = []

    for _, row in index_df.iterrows():
        sample_id = str(row["sample_id"])
        if sample_id not in feature_map:
            missing_samples.append(sample_id)
            continue
        vectors.append(feature_map[sample_id])
        merged = dict(metadata_map[sample_id])
        merged.update(row.to_dict())
        metadata_rows.append(merged)

    if missing_samples:
        preview = ", ".join(missing_samples[:5])
        raise ValueError(
            f"{len(missing_samples)} samples were missing in cache. "
            f"First missing sample_id values: {preview}"
        )

    x_matrix = np.vstack(vectors).astype(np.float32, copy=False)
    metadata_df = pd.DataFrame(metadata_rows)
    return x_matrix, metadata_df, spatial_report
