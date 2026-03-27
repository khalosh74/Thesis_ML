from __future__ import annotations

import re
from itertools import permutations
from pathlib import Path

import pandas as pd

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.methodology import EvidenceRunRole, MethodologyPolicyName
from Thesis_ML.experiments.comparison_contract import (
    validate_confirmatory_protocol_fairness_contract,
)
from Thesis_ML.experiments.model_catalog import (
    get_model_cost_entry,
    projected_runtime_seconds,
)
from Thesis_ML.protocols.models import (
    CompiledProtocolManifest,
    CompiledRunControls,
    CompiledRunSpec,
    SubjectSource,
    SuiteSpec,
    ThesisProtocol,
    TransferPair,
    TransferPairSource,
)


def _slug(value: str) -> str:
    lowered = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")


def _load_subjects(index_csv: Path | str) -> list[str]:
    index_df = pd.read_csv(index_csv)
    if "subject" not in index_df.columns:
        raise ValueError(
            f"Dataset index '{Path(index_csv)}' is missing required column 'subject' for protocol compilation."
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


def _resolve_suite_subjects(suite: SuiteSpec, all_subjects: list[str]) -> list[str]:
    if suite.subject_source == SubjectSource.ALL_FROM_INDEX:
        return all_subjects
    resolved = sorted({str(value).strip() for value in suite.subjects if str(value).strip()})
    if not resolved:
        raise ValueError(
            f"Suite '{suite.suite_id}' resolved to an empty subject list for within-subject runs."
        )
    return resolved


def _resolve_transfer_pairs(suite: SuiteSpec, all_subjects: list[str]) -> list[TransferPair]:
    if suite.transfer_pair_source == TransferPairSource.ALL_ORDERED_PAIRS_FROM_INDEX:
        ordered_pairs = [
            TransferPair(train_subject=left, test_subject=right)
            for left, right in permutations(all_subjects, 2)
        ]
        if not ordered_pairs:
            raise ValueError(
                f"Suite '{suite.suite_id}' requires at least two subjects for transfer runs."
            )
        return ordered_pairs
    if not suite.transfer_pairs:
        raise ValueError(f"Suite '{suite.suite_id}' resolved to an empty transfer pair list.")
    return list(suite.transfer_pairs)


def _resolve_suite_models(protocol: ThesisProtocol, suite: SuiteSpec) -> list[str]:
    base_models = (
        list(suite.models) if suite.models is not None else list(protocol.model_policy.models)
    )
    if protocol.control_policy.dummy_baseline.enabled and suite.suite_id in set(
        protocol.control_policy.dummy_baseline.suites
    ):
        if "dummy" not in base_models:
            base_models.append("dummy")
    unique_models = sorted(set(base_models), key=base_models.index)
    if suite.controls_required:
        has_dummy = "dummy" in unique_models
        has_permutation = protocol.control_policy.permutation.enabled and suite.suite_id in set(
            protocol.control_policy.permutation.suites
        )
        if not has_dummy and not has_permutation:
            raise ValueError(
                f"Suite '{suite.suite_id}' is marked controls_required but no dummy/permutation control applies."
            )
    return unique_models


def _resolve_suite_seed(protocol: ThesisProtocol, suite: SuiteSpec) -> int:
    if suite.seed_override is None:
        return int(protocol.scientific_contract.seed_policy.global_seed)
    if not protocol.scientific_contract.seed_policy.per_suite_overrides_allowed:
        raise ValueError(
            f"Suite '{suite.suite_id}' sets seed_override but per-suite seed overrides are disabled by protocol."
        )
    return int(suite.seed_override)


def _interpretability_enabled(protocol: ThesisProtocol, suite: SuiteSpec, model_name: str) -> bool:
    if not suite.interpretability_requested:
        return False
    policy = protocol.interpretability_policy
    return (
        bool(policy.enabled)
        and suite.suite_id in set(policy.suites)
        and suite.split_mode in set(policy.modes)
        and model_name in set(policy.models)
    )


def _build_run_id(
    protocol: ThesisProtocol,
    suite: SuiteSpec,
    model_name: str,
    *,
    subject: str | None = None,
    train_subject: str | None = None,
    test_subject: str | None = None,
) -> str:
    parts = [
        "canonical",
        _slug(protocol.protocol_id),
        _slug(protocol.protocol_version),
        _slug(suite.suite_id),
        _slug(model_name),
    ]
    if subject is not None:
        parts.append(_slug(subject))
    if train_subject is not None and test_subject is not None:
        parts.append(f"{_slug(train_subject)}_to_{_slug(test_subject)}")
    return "_".join(part for part in parts if part)


def compile_protocol(
    protocol: ThesisProtocol,
    *,
    index_csv: Path | str,
    suite_ids: list[str] | None = None,
) -> CompiledProtocolManifest:
    all_subjects = _load_subjects(index_csv)

    available_suites = {
        suite.suite_id: suite for suite in protocol.official_run_suites if suite.enabled
    }
    if suite_ids is None:
        selected_suite_ids = sorted(available_suites.keys())
    else:
        selected_suite_ids = []
        for suite_id in suite_ids:
            if suite_id not in available_suites:
                known = ", ".join(sorted(available_suites))
                raise ValueError(
                    f"Requested suite '{suite_id}' was not found among enabled suites. Enabled suites: {known}."
                )
            selected_suite_ids.append(suite_id)

    if not selected_suite_ids:
        raise ValueError("No enabled suites were selected for protocol compilation.")
    validate_confirmatory_protocol_fairness_contract(
        protocol=protocol,
        selected_suite_ids=selected_suite_ids,
    )

    runs: list[CompiledRunSpec] = []
    claim_to_run_map: dict[str, list[str]] = {}
    permutation_enabled_suites = set(protocol.control_policy.permutation.suites)
    permutation_metric = (
        protocol.control_policy.permutation.metric or protocol.metric_policy.primary_metric
    )
    repeat_count = int(protocol.evidence_policy.repeat_evaluation.repeat_count)
    seed_stride = int(protocol.evidence_policy.repeat_evaluation.seed_stride)
    require_untuned_baseline = bool(
        protocol.methodology_policy.policy_name == MethodologyPolicyName.GROUPED_NESTED_TUNING
        and protocol.evidence_policy.required_package.require_untuned_baseline_if_tuning
    )

    def _append_run_variants(
        *,
        base_run_id: str,
        suite: SuiteSpec,
        model_name: str,
        suite_seed: int,
        controls: CompiledRunControls,
        subject: str | None = None,
        train_subject: str | None = None,
        test_subject: str | None = None,
    ) -> None:
        for repeat_id in range(1, repeat_count + 1):
            repeat_seed = int(suite_seed + ((repeat_id - 1) * seed_stride))
            repeated_run_id = (
                f"{base_run_id}__r{repeat_id:03d}" if repeat_count > 1 else base_run_id
            )
            model_cost_entry = get_model_cost_entry(model_name)
            run = CompiledRunSpec(
                run_id=repeated_run_id,
                base_run_id=base_run_id,
                repeat_id=repeat_id,
                repeat_count=repeat_count,
                evidence_run_role=EvidenceRunRole.PRIMARY,
                suite_id=suite.suite_id,
                claim_ids=list(suite.claim_ids),
                target=protocol.scientific_contract.target,
                model=model_name,
                model_cost_tier=model_cost_entry.cost_tier,
                projected_runtime_seconds=projected_runtime_seconds(
                    model_name=model_name,
                    framework_mode=FrameworkMode.CONFIRMATORY,
                    methodology_policy=protocol.methodology_policy.policy_name,
                    tuning_enabled=bool(protocol.methodology_policy.tuning_enabled),
                ),
                cv_mode=suite.split_mode,
                subject=subject,
                train_subject=train_subject,
                test_subject=test_subject,
                filter_task=suite.filter_task,
                filter_modality=suite.filter_modality,
                seed=repeat_seed,
                primary_metric=protocol.metric_policy.primary_metric,
                controls=controls,
                interpretability_enabled=_interpretability_enabled(protocol, suite, model_name),
                methodology_policy_name=protocol.methodology_policy.policy_name,
                class_weight_policy=protocol.methodology_policy.class_weight_policy,
                feature_recipe_id=protocol.feature_engineering_policy.feature_recipe_id,
                emit_feature_qc_artifacts=bool(
                    protocol.feature_engineering_policy.emit_feature_qc_artifacts
                ),
                tuning_enabled=bool(protocol.methodology_policy.tuning_enabled),
                tuning_search_space_id=protocol.methodology_policy.tuning_search_space_id,
                tuning_search_space_version=protocol.methodology_policy.tuning_search_space_version,
                tuning_inner_cv_scheme=protocol.methodology_policy.inner_cv_scheme,
                tuning_inner_group_field=protocol.methodology_policy.inner_group_field,
                subgroup_reporting_enabled=bool(protocol.subgroup_reporting_policy.enabled),
                subgroup_dimensions=list(protocol.subgroup_reporting_policy.subgroup_dimensions),
                subgroup_min_samples_per_group=int(
                    protocol.subgroup_reporting_policy.min_samples_per_group
                ),
                framework_mode=FrameworkMode.CONFIRMATORY.value,
                canonical_run=True,
                artifact_requirements=list(protocol.artifact_contract.required_run_artifacts),
                protocol_id=protocol.protocol_id,
                protocol_version=protocol.protocol_version,
                protocol_schema_version=protocol.protocol_schema_version,
            )
            runs.append(run)
            for claim_id in suite.claim_ids:
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
                for claim_id in suite.claim_ids:
                    claim_to_run_map.setdefault(claim_id, []).append(untuned_run.run_id)

    for suite_id in selected_suite_ids:
        suite = available_suites[suite_id]
        suite_models = _resolve_suite_models(protocol, suite)
        if suite.interpretability_requested:
            unsupported_models = [
                model_name
                for model_name in suite_models
                if not _interpretability_enabled(protocol, suite, model_name)
            ]
            if unsupported_models:
                raise ValueError(
                    f"Suite '{suite.suite_id}' requests interpretability for unsupported models: "
                    + ", ".join(sorted(set(unsupported_models)))
                )
        suite_seed = _resolve_suite_seed(protocol, suite)

        if suite.split_mode == "within_subject_loso_session":
            for subject in _resolve_suite_subjects(suite, all_subjects):
                for model_name in suite_models:
                    base_run_id = _build_run_id(protocol, suite, model_name, subject=subject)
                    controls = CompiledRunControls(
                        dummy_baseline_run=(model_name == "dummy"),
                        permutation_enabled=(
                            protocol.control_policy.permutation.enabled
                            and suite.suite_id in permutation_enabled_suites
                        ),
                        permutation_metric=(
                            permutation_metric
                            if protocol.control_policy.permutation.enabled
                            and suite.suite_id in permutation_enabled_suites
                            else None
                        ),
                        n_permutations=(
                            int(protocol.control_policy.permutation.n_permutations)
                            if protocol.control_policy.permutation.enabled
                            and suite.suite_id in permutation_enabled_suites
                            else 0
                        ),
                    )
                    _append_run_variants(
                        base_run_id=base_run_id,
                        suite=suite,
                        model_name=model_name,
                        suite_seed=suite_seed,
                        controls=controls,
                        subject=subject,
                    )

        if suite.split_mode == "frozen_cross_person_transfer":
            transfer_pairs = _resolve_transfer_pairs(suite, all_subjects)
            for pair in transfer_pairs:
                for model_name in suite_models:
                    base_run_id = _build_run_id(
                        protocol,
                        suite,
                        model_name,
                        train_subject=pair.train_subject,
                        test_subject=pair.test_subject,
                    )
                    controls = CompiledRunControls(
                        dummy_baseline_run=(model_name == "dummy"),
                        permutation_enabled=(
                            protocol.control_policy.permutation.enabled
                            and suite.suite_id in permutation_enabled_suites
                        ),
                        permutation_metric=(
                            permutation_metric
                            if protocol.control_policy.permutation.enabled
                            and suite.suite_id in permutation_enabled_suites
                            else None
                        ),
                        n_permutations=(
                            int(protocol.control_policy.permutation.n_permutations)
                            if protocol.control_policy.permutation.enabled
                            and suite.suite_id in permutation_enabled_suites
                            else 0
                        ),
                    )
                    _append_run_variants(
                        base_run_id=base_run_id,
                        suite=suite,
                        model_name=model_name,
                        suite_seed=suite_seed,
                        controls=controls,
                        train_subject=pair.train_subject,
                        test_subject=pair.test_subject,
                    )

    if not runs:
        raise ValueError("Protocol compilation produced zero concrete runs.")

    if protocol.methodology_policy.policy_name == MethodologyPolicyName.GROUPED_NESTED_TUNING:
        if all(run.model == "dummy" for run in runs):
            raise ValueError("grouped_nested_tuning requires at least one non-dummy model run.")

    return CompiledProtocolManifest(
        framework_mode=FrameworkMode.CONFIRMATORY.value,
        protocol_schema_version=protocol.protocol_schema_version,
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.protocol_version,
        status=protocol.status,
        methodology_policy=protocol.methodology_policy,
        metric_policy=protocol.metric_policy,
        subgroup_reporting_policy=protocol.subgroup_reporting_policy,
        feature_engineering_policy=protocol.feature_engineering_policy,
        data_policy=protocol.data_policy,
        evidence_policy=protocol.evidence_policy,
        suite_ids=selected_suite_ids,
        runs=runs,
        claim_to_run_map=claim_to_run_map,
        required_protocol_artifacts=list(protocol.artifact_contract.required_protocol_artifacts),
        required_run_artifacts=list(protocol.artifact_contract.required_run_artifacts),
        required_run_metadata_fields=list(protocol.artifact_contract.required_run_metadata_fields),
    )
