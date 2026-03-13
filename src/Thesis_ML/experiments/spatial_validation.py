from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SPATIAL_AFFINE_ATOL = 1e-5


def normalize_spatial_signature(raw_signature: Any, cache_path: Path) -> dict[str, Any]:
    if not isinstance(raw_signature, dict):
        raise ValueError(f"invalid spatial signature payload in {cache_path}")

    required = ("image_shape", "affine", "mask_voxel_count", "feature_count", "mask_sha256")
    missing = [key for key in required if key not in raw_signature]
    if missing:
        raise ValueError(f"missing required spatial signature field(s) {missing} in {cache_path}")

    image_shape = [int(value) for value in list(raw_signature["image_shape"])]
    affine_array = np.asarray(raw_signature["affine"], dtype=np.float64)
    if affine_array.shape != (4, 4):
        raise ValueError(
            f"invalid affine shape in {cache_path}: expected (4, 4), got {affine_array.shape}"
        )

    voxel_size_raw = raw_signature.get("voxel_size", [])
    voxel_size = [float(value) for value in list(voxel_size_raw)]
    mask_sha256 = str(raw_signature["mask_sha256"]).strip()
    if not mask_sha256:
        raise ValueError(f"empty mask_sha256 in {cache_path}")

    return {
        "signature_version": int(raw_signature.get("signature_version", 1)),
        "image_shape": image_shape,
        "affine": affine_array.tolist(),
        "voxel_size": voxel_size,
        "mask_voxel_count": int(raw_signature["mask_voxel_count"]),
        "feature_count": int(raw_signature["feature_count"]),
        "mask_sha256": mask_sha256,
    }


def spatial_mismatch_reasons(
    reference_signature: dict[str, Any],
    candidate_signature: dict[str, Any],
    affine_atol: float,
) -> list[str]:
    reasons: list[str] = []

    ref_shape = [int(value) for value in reference_signature["image_shape"]]
    cand_shape = [int(value) for value in candidate_signature["image_shape"]]
    if cand_shape != ref_shape:
        reasons.append(f"image_shape mismatch ({cand_shape} != {ref_shape})")

    ref_affine = np.asarray(reference_signature["affine"], dtype=np.float64)
    cand_affine = np.asarray(candidate_signature["affine"], dtype=np.float64)
    if not np.allclose(cand_affine, ref_affine, rtol=0.0, atol=affine_atol):
        reasons.append("affine mismatch")

    ref_voxel_size = np.asarray(reference_signature.get("voxel_size", []), dtype=np.float64)
    cand_voxel_size = np.asarray(candidate_signature.get("voxel_size", []), dtype=np.float64)
    if ref_voxel_size.size > 0 and cand_voxel_size.size > 0:
        if not np.allclose(cand_voxel_size, ref_voxel_size, rtol=0.0, atol=1e-6):
            reasons.append("voxel_size mismatch")

    ref_mask_voxels = int(reference_signature["mask_voxel_count"])
    cand_mask_voxels = int(candidate_signature["mask_voxel_count"])
    if cand_mask_voxels != ref_mask_voxels:
        reasons.append(f"mask_voxel_count mismatch ({cand_mask_voxels} != {ref_mask_voxels})")

    ref_feature_count = int(reference_signature["feature_count"])
    cand_feature_count = int(candidate_signature["feature_count"])
    if cand_feature_count != ref_feature_count:
        reasons.append(f"feature_count mismatch ({cand_feature_count} != {ref_feature_count})")

    ref_hash = str(reference_signature["mask_sha256"])
    cand_hash = str(candidate_signature["mask_sha256"])
    if cand_hash != ref_hash:
        reasons.append("mask_sha256 mismatch")

    return reasons


def build_spatial_compatibility_report(
    cache_groups: list[dict[str, Any]],
    affine_atol: float,
) -> dict[str, Any]:
    checked_groups: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    reference_signature: dict[str, Any] | None = None
    reference_group_id: str | None = None

    for group in cache_groups:
        group_id = str(group["group_id"])
        cache_path = str(group["cache_path"])
        n_features = int(group["n_features"])
        n_selected_samples = int(group["n_selected_samples"])
        raw_signature = group.get("raw_signature")
        normalized_signature: dict[str, Any] | None = None
        reasons: list[str] = []

        if raw_signature is None:
            reasons.append("missing spatial signature metadata")
        else:
            try:
                normalized_signature = normalize_spatial_signature(
                    raw_signature=raw_signature,
                    cache_path=Path(cache_path),
                )
            except ValueError as exc:
                reasons.append(str(exc))

        if normalized_signature is not None:
            signature_feature_count = int(normalized_signature["feature_count"])
            signature_mask_voxels = int(normalized_signature["mask_voxel_count"])
            if signature_feature_count != n_features:
                reasons.append(
                    "feature_count mismatch against cached matrix width "
                    f"({signature_feature_count} != {n_features})"
                )
            if signature_mask_voxels != n_features:
                reasons.append(
                    "mask_voxel_count mismatch against cached matrix width "
                    f"({signature_mask_voxels} != {n_features})"
                )
            if reference_signature is None:
                reference_signature = normalized_signature
                reference_group_id = group_id
            else:
                reasons.extend(
                    spatial_mismatch_reasons(
                        reference_signature=reference_signature,
                        candidate_signature=normalized_signature,
                        affine_atol=affine_atol,
                    )
                )

        checked_groups.append(
            {
                "group_id": group_id,
                "cache_path": cache_path,
                "n_selected_samples": n_selected_samples,
                "n_features": n_features,
                "spatial_signature": normalized_signature,
            }
        )
        if reasons:
            mismatches.append(
                {
                    "group_id": group_id,
                    "cache_path": cache_path,
                    "reasons": reasons,
                }
            )

    if not cache_groups:
        mismatches.append(
            {
                "group_id": None,
                "cache_path": None,
                "reasons": ["no cache groups matched selected samples"],
            }
        )

    passed = bool(cache_groups) and not mismatches and reference_signature is not None
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "affine_atol": float(affine_atol),
        "n_groups_checked": int(len(cache_groups)),
        "reference_group_id": reference_group_id,
        "reference_signature": reference_signature,
        "checked_groups": checked_groups,
        "mismatches": mismatches,
    }


def raise_spatial_compatibility_error(report: dict[str, Any]) -> None:
    mismatch_summaries: list[str] = []
    for mismatch in report.get("mismatches", [])[:5]:
        group_id = mismatch.get("group_id")
        group_label = str(group_id) if group_id is not None else "<unknown-group>"
        reasons = mismatch.get("reasons", [])
        if reasons:
            reason_text = "; ".join(str(reason) for reason in reasons)
        else:
            reason_text = "unknown mismatch"
        mismatch_summaries.append(f"{group_label}: {reason_text}")

    details = " | ".join(mismatch_summaries) if mismatch_summaries else "unknown mismatch"
    raise ValueError(
        "Spatial compatibility validation failed before feature stacking. "
        f"{details}. Rebuild cache with thesisml-cache-features --force if metadata is stale."
    )
