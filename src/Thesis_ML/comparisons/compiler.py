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
from Thesis_ML.config.methodology import EvidenceRunRole, MethodologyPolicyName
from Thesis_ML.experiments.comparison_contract import (
    validate_locked_comparison_fairness_contract,
)
from Thesis_ML.experiments.model_catalog import (
    get_model_cost_entry,
    projected_runtime_seconds,
)


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
    validate_locked_comparison_fairness_contract(
        comparison=comparison,
        selected_variant_ids=selected_variant_ids,
    )

    contract = comparison.scientific_contract
    control_policy = comparison.control_policy
    controls = CompiledComparisonRunControls(
        permutation_enabled=bool(control_policy.permutation_enabled),
        permutation_metric=str(
            control_policy.permutation_metric or comparison.metric_policy.primary_metric
        ),
        n_permutations=int(control_policy.n_permutations),
        dummy_baseline_enabled=bool(control_policy.dummy_baseline_enabled),
    )
    repeat_count = int(comparison.evidence_policy.repeat_evaluation.repeat_count)
    seed_stride = int(comparison.evidence_policy.repeat_evaluation.seed_stride)
    require_untuned_baseline = bool(
        comparison.methodology_policy.policy_name == MethodologyPolicyName.GROUPED_NESTED_TUNING
        and comparison.evidence_policy.required_package.require_untuned_baseline_if_tuning
    )

    runs: list[CompiledComparisonRunSpec] = []
    claim_to_run_map: dict[str, list[str]] = {}

    def _append_run_variants(
        *,
        base_run_id: str,
        variant_id: str,
        claim_ids: list[str],
        model_name: str,
        interpretability_enabled: bool,
        subject: str | None = None,
        train_subject: str | None = None,
        test_subject: str | None = None,
    ) -> None:
        base_seed = int(contract.seed_policy.global_seed)
        for repeat_id in range(1, repeat_count + 1):
            repeat_seed = int(base_seed + ((repeat_id - 1) * seed_stride))
            repeated_run_id = (
                f"{base_run_id}__r{repeat_id:03d}" if repeat_count > 1 else base_run_id
            )
            model_cost_entry = get_model_cost_entry(model_name)
            run = CompiledComparisonRunSpec(
                run_id=repeated_run_id,
                base_run_id=base_run_id,
                repeat_id=repeat_id,
                repeat_count=repeat_count,
                evidence_run_role=EvidenceRunRole.PRIMARY,
                framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
                canonical_run=False,
                comparison_id=comparison.comparison_id,
                comparison_version=comparison.comparison_version,
                variant_id=variant_id,
                claim_ids=list(claim_ids),
                target=contract.target,
                model=model_name,
                model_cost_tier=model_cost_entry.cost_tier,
                projected_runtime_seconds=projected_runtime_seconds(
                    model_name=model_name,
                    framework_mode=FrameworkMode.LOCKED_COMPARISON,
                    methodology_policy=comparison.methodology_policy.policy_name,
                    tuning_enabled=bool(comparison.methodology_policy.tuning_enabled),
                ),
                cv_mode=contract.split_mode,
                subject=subject,
                train_subject=train_subject,
                test_subject=test_subject,
                filter_task=contract.filter_task,
                filter_modality=contract.filter_modality,
                seed=repeat_seed,
                primary_metric=comparison.metric_policy.primary_metric,
                controls=controls,
                interpretability_enabled=interpretability_enabled,
                methodology_policy_name=comparison.methodology_policy.policy_name,
                class_weight_policy=comparison.methodology_policy.class_weight_policy,
                tuning_enabled=bool(comparison.methodology_policy.tuning_enabled),
                tuning_search_space_id=comparison.methodology_policy.tuning_search_space_id,
                tuning_search_space_version=comparison.methodology_policy.tuning_search_space_version,
                tuning_inner_cv_scheme=comparison.methodology_policy.inner_cv_scheme,
                tuning_inner_group_field=comparison.methodology_policy.inner_group_field,
                subgroup_reporting_enabled=bool(comparison.subgroup_reporting_policy.enabled),
                subgroup_dimensions=list(comparison.subgroup_reporting_policy.subgroup_dimensions),
                subgroup_min_samples_per_group=int(
                    comparison.subgroup_reporting_policy.min_samples_per_group
                ),
                artifact_requirements=list(comparison.artifact_contract.required_run_artifacts),
            )
            runs.append(run)
            for claim_id in claim_ids:
                claim_to_run_map.setdefault(claim_id, []).append(run.run_id)

            if require_untuned_baseline and model_name != "dummy":
                untuned_run = run.model_copy(
                    update={
                        "run_id": f"{repeated_run_id}__untuned",
                        "evidence_run_role": EvidenceRunRole.UNTUNED_BASELINE,
                        "tuning_enabled": False,
                        "tuning_search_space_id": None,
                        "tuning_search_space_version": None,
                        "tuning_inner_cv_scheme": None,
                        "tuning_inner_group_field": None,
                    }
                )
                runs.append(untuned_run)
                for claim_id in claim_ids:
                    claim_to_run_map.setdefault(claim_id, []).append(untuned_run.run_id)

    for variant_id in selected_variant_ids:
        variant = allowed_variants[variant_id]
        interpretability_enabled = bool(comparison.interpretability_policy.enabled) and (
            variant.model in set(comparison.interpretability_policy.allowed_models)
        )

        if contract.split_mode == "within_subject_loso_session":
            for subject in _resolve_subjects(comparison, all_subjects):
                base_run_id = _build_run_id(comparison, variant_id, subject=subject)
                _append_run_variants(
                    base_run_id=base_run_id,
                    variant_id=variant.variant_id,
                    claim_ids=list(variant.claim_ids),
                    model_name=variant.model,
                    interpretability_enabled=interpretability_enabled,
                    subject=subject,
                )

        if contract.split_mode == "frozen_cross_person_transfer":
            for pair in _resolve_transfer_pairs(comparison, all_subjects):
                base_run_id = _build_run_id(
                    comparison,
                    variant_id,
                    train_subject=pair.train_subject,
                    test_subject=pair.test_subject,
                )
                _append_run_variants(
                    base_run_id=base_run_id,
                    variant_id=variant.variant_id,
                    claim_ids=list(variant.claim_ids),
                    model_name=variant.model,
                    interpretability_enabled=interpretability_enabled,
                    train_subject=pair.train_subject,
                    test_subject=pair.test_subject,
                )

    if not runs:
        raise ValueError("Comparison compilation produced zero concrete runs.")
    if comparison.methodology_policy.policy_name == MethodologyPolicyName.GROUPED_NESTED_TUNING:
        if all(run.model == "dummy" for run in runs):
            raise ValueError("grouped_nested_tuning requires at least one non-dummy model run.")

    return CompiledComparisonManifest(
        framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
        comparison_id=comparison.comparison_id,
        comparison_version=comparison.comparison_version,
        status=comparison.status,
        comparison_dimension=comparison.comparison_dimension,
        methodology_policy=comparison.methodology_policy,
        metric_policy=comparison.metric_policy,
        subgroup_reporting_policy=comparison.subgroup_reporting_policy,
        data_policy=comparison.data_policy,
        decision_policy=comparison.decision_policy,
        evidence_policy=comparison.evidence_policy,
        variant_ids=selected_variant_ids,
        runs=runs,
        claim_to_run_map=claim_to_run_map,
        required_comparison_artifacts=list(
            comparison.artifact_contract.required_comparison_artifacts
        ),
        required_run_artifacts=list(comparison.artifact_contract.required_run_artifacts),
        required_run_metadata_fields=list(
            comparison.artifact_contract.required_run_metadata_fields
        ),
    )
