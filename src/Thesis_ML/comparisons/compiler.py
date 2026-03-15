from __future__ import annotations

import re
from itertools import permutations
from pathlib import Path

import pandas as pd

from Thesis_ML.comparisons.models import (
    ComparisonSpec,
    ComparisonTransferPair,
    CompiledComparisonManifest,
    CompiledComparisonRunControls,
    CompiledComparisonRunSpec,
    SubjectSource,
    TransferPairSource,
)
from Thesis_ML.config.framework_mode import FrameworkMode


def _slug(value: str) -> str:
    lowered = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")


def _load_subjects(index_csv: Path | str) -> list[str]:
    index_df = pd.read_csv(index_csv)
    if "subject" not in index_df.columns:
        raise ValueError(
            f"Dataset index '{Path(index_csv)}' is missing required column 'subject' for comparison compilation."
        )
    subjects = sorted(
        {
            str(value).strip()
            for value in index_df["subject"].astype(str).tolist()
            if str(value).strip()
        }
    )
    if not subjects:
        raise ValueError(
            f"Dataset index '{Path(index_csv)}' does not contain any non-empty subject values."
        )
    return subjects


def _resolve_subjects(comparison: ComparisonSpec, all_subjects: list[str]) -> list[str]:
    policy = comparison.scientific_contract.subject_policy
    if policy.source == SubjectSource.ALL_FROM_INDEX:
        return list(all_subjects)
    resolved = sorted({str(value).strip() for value in policy.subjects if str(value).strip()})
    if not resolved:
        raise ValueError(
            f"Comparison '{comparison.comparison_id}' resolved to an empty subject list."
        )
    return resolved


def _resolve_transfer_pairs(
    comparison: ComparisonSpec,
    all_subjects: list[str],
) -> list[ComparisonTransferPair]:
    policy = comparison.scientific_contract.transfer_policy
    if policy.source == TransferPairSource.ALL_ORDERED_PAIRS_FROM_INDEX:
        pairs = [
            ComparisonTransferPair(train_subject=left, test_subject=right)
            for left, right in permutations(all_subjects, 2)
        ]
        if not pairs:
            raise ValueError(
                f"Comparison '{comparison.comparison_id}' requires at least two subjects for transfer runs."
            )
        return pairs
    if not policy.pairs:
        raise ValueError(
            f"Comparison '{comparison.comparison_id}' resolved to an empty transfer pair list."
        )
    return list(policy.pairs)


def _build_run_id(
    comparison: ComparisonSpec,
    variant_id: str,
    *,
    subject: str | None = None,
    train_subject: str | None = None,
    test_subject: str | None = None,
) -> str:
    parts = [
        "cmp",
        _slug(comparison.comparison_id),
        _slug(comparison.comparison_version),
        _slug(variant_id),
    ]
    if subject is not None:
        parts.append(_slug(subject))
    if train_subject is not None and test_subject is not None:
        parts.append(f"{_slug(train_subject)}_to_{_slug(test_subject)}")
    return "_".join(parts)


def compile_comparison(
    comparison: ComparisonSpec,
    *,
    index_csv: Path | str,
    variant_ids: list[str] | None = None,
) -> CompiledComparisonManifest:
    all_subjects = _load_subjects(index_csv)
    allowed_variants = {variant.variant_id: variant for variant in comparison.allowed_variants}
    if variant_ids is None:
        selected_variant_ids = [variant.variant_id for variant in comparison.allowed_variants]
    else:
        selected_variant_ids = []
        for variant_id in variant_ids:
            if variant_id not in allowed_variants:
                known = ", ".join(sorted(allowed_variants))
                raise ValueError(
                    f"Unknown comparison variant '{variant_id}'. Allowed variants: {known}."
                )
            selected_variant_ids.append(variant_id)
    if not selected_variant_ids:
        raise ValueError("No comparison variants selected for compilation.")

    contract = comparison.scientific_contract
    controls = CompiledComparisonRunControls(
        permutation_enabled=bool(contract.control_policy.permutation_enabled),
        permutation_metric=str(contract.control_policy.permutation_metric),
        n_permutations=int(contract.control_policy.n_permutations),
        dummy_baseline_enabled=bool(contract.control_policy.dummy_baseline_enabled),
    )

    runs: list[CompiledComparisonRunSpec] = []
    claim_to_run_map: dict[str, list[str]] = {}
    for variant_id in selected_variant_ids:
        variant = allowed_variants[variant_id]
        interpretability_enabled = bool(contract.interpretability_policy.enabled) and (
            variant.model in set(contract.interpretability_policy.allowed_models)
        )

        if contract.split_mode == "within_subject_loso_session":
            for subject in _resolve_subjects(comparison, all_subjects):
                run_id = _build_run_id(comparison, variant_id, subject=subject)
                run = CompiledComparisonRunSpec(
                    run_id=run_id,
                    framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
                    canonical_run=False,
                    comparison_id=comparison.comparison_id,
                    comparison_version=comparison.comparison_version,
                    variant_id=variant.variant_id,
                    claim_ids=list(variant.claim_ids),
                    target=contract.target,
                    model=variant.model,
                    cv_mode=contract.split_mode,
                    subject=subject,
                    filter_task=contract.filter_task,
                    filter_modality=contract.filter_modality,
                    seed=int(contract.seed_policy.global_seed),
                    primary_metric=contract.primary_metric,
                    controls=controls,
                    interpretability_enabled=interpretability_enabled,
                    artifact_requirements=list(comparison.artifact_contract.required_run_artifacts),
                )
                runs.append(run)
                for claim_id in variant.claim_ids:
                    claim_to_run_map.setdefault(claim_id, []).append(run.run_id)

        if contract.split_mode == "frozen_cross_person_transfer":
            for pair in _resolve_transfer_pairs(comparison, all_subjects):
                run_id = _build_run_id(
                    comparison,
                    variant_id,
                    train_subject=pair.train_subject,
                    test_subject=pair.test_subject,
                )
                run = CompiledComparisonRunSpec(
                    run_id=run_id,
                    framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
                    canonical_run=False,
                    comparison_id=comparison.comparison_id,
                    comparison_version=comparison.comparison_version,
                    variant_id=variant.variant_id,
                    claim_ids=list(variant.claim_ids),
                    target=contract.target,
                    model=variant.model,
                    cv_mode=contract.split_mode,
                    train_subject=pair.train_subject,
                    test_subject=pair.test_subject,
                    filter_task=contract.filter_task,
                    filter_modality=contract.filter_modality,
                    seed=int(contract.seed_policy.global_seed),
                    primary_metric=contract.primary_metric,
                    controls=controls,
                    interpretability_enabled=interpretability_enabled,
                    artifact_requirements=list(comparison.artifact_contract.required_run_artifacts),
                )
                runs.append(run)
                for claim_id in variant.claim_ids:
                    claim_to_run_map.setdefault(claim_id, []).append(run.run_id)

    if not runs:
        raise ValueError("Comparison compilation produced zero concrete runs.")

    return CompiledComparisonManifest(
        framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
        comparison_id=comparison.comparison_id,
        comparison_version=comparison.comparison_version,
        status=comparison.status,
        comparison_dimension=comparison.comparison_dimension,
        variant_ids=selected_variant_ids,
        runs=runs,
        claim_to_run_map=claim_to_run_map,
        required_comparison_artifacts=list(
            comparison.artifact_contract.required_comparison_artifacts
        ),
        required_run_artifacts=list(comparison.artifact_contract.required_run_artifacts),
    )
