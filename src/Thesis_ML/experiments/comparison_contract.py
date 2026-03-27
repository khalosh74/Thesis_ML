from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.model_admission import (
    official_deterministic_compute_required,
    official_gpu_only_model_backend_allowed,
    official_max_both_gpu_lane_eligible,
)
from Thesis_ML.experiments.model_registry import get_model_spec
from Thesis_ML.features.preprocessing import BASELINE_STANDARD_SCALER_RECIPE_ID


@dataclass(frozen=True)
class ComparisonContractSignature:
    target: str
    data_slice_semantics: tuple[str, ...]
    split_policy: tuple[str, ...]
    feature_recipe_id: str
    metric_policy: tuple[str, ...]
    methodology_policy: tuple[str, ...]
    tuning_budget_semantics: tuple[str, ...]
    repeat_policy: tuple[int, int]
    control_policy: tuple[str, ...]
    class_weight_policy: str
    deterministic_compute_requirements: tuple[bool, bool]
    backend_parity_semantics: tuple[str, ...]


def _signature_payload(signature: ComparisonContractSignature) -> dict[str, Any]:
    return {
        "target": signature.target,
        "data_slice_semantics": signature.data_slice_semantics,
        "split_policy": signature.split_policy,
        "feature_recipe_id": signature.feature_recipe_id,
        "metric_policy": signature.metric_policy,
        "methodology_policy": signature.methodology_policy,
        "tuning_budget_semantics": signature.tuning_budget_semantics,
        "repeat_policy": signature.repeat_policy,
        "control_policy": signature.control_policy,
        "class_weight_policy": signature.class_weight_policy,
        "deterministic_compute_requirements": signature.deterministic_compute_requirements,
        "backend_parity_semantics": signature.backend_parity_semantics,
    }


def _admitted_backend_families_for_model(
    *,
    framework_mode: FrameworkMode,
    model_name: str,
) -> tuple[str, ...]:
    spec = get_model_spec(model_name)
    admitted_backend_families: set[str] = set()
    for binding in spec.backend_bindings:
        backend_family = str(binding.backend_family).strip().lower()
        compute_family = str(binding.compute_backend_family).strip().lower()

        if framework_mode == FrameworkMode.CONFIRMATORY:
            if compute_family == "sklearn_cpu":
                admitted_backend_families.add(backend_family)
            elif official_gpu_only_model_backend_allowed(
                framework_mode=framework_mode,
                model_name=spec.logical_name,
                backend_family=backend_family,
            ):
                admitted_backend_families.add(backend_family)
            continue

        if framework_mode == FrameworkMode.LOCKED_COMPARISON:
            if compute_family == "sklearn_cpu":
                admitted_backend_families.add(backend_family)
            elif official_gpu_only_model_backend_allowed(
                framework_mode=framework_mode,
                model_name=spec.logical_name,
                backend_family=backend_family,
            ) or official_max_both_gpu_lane_eligible(
                framework_mode=framework_mode,
                model_name=spec.logical_name,
                backend_family=backend_family,
            ):
                admitted_backend_families.add(backend_family)
            continue

        admitted_backend_families.add(backend_family)

    if not admitted_backend_families:
        raise ValueError(
            f"No admitted backend parity families resolved for model '{spec.logical_name}'."
        )

    return tuple(sorted(admitted_backend_families))


def _common_backend_parity_semantics(
    framework_mode: FrameworkMode,
    model_names: list[str],
) -> tuple[str, ...]:
    if not model_names:
        return ()
    admitted_sets = {}
    for name in model_names:
        admitted_sets[name] = set(_admitted_backend_families_for_model(
            framework_mode=framework_mode, model_name=name
        ))
    common = set.intersection(*admitted_sets.values())
    if not common:
        raise ValueError(
            f"No common backend parity semantics resolved for models: {model_names}. "
            f"Per-model admitted families: {admitted_sets}"
        )
    return tuple(sorted(common))

def validate_contract_signature_parity(
    *,
    signatures_by_id: dict[str, ComparisonContractSignature],
    context: str,
) -> None:
    if not signatures_by_id:
        raise ValueError(f"{context}: no signatures were provided for fairness validation.")

    ordered_items = list(signatures_by_id.items())
    reference_id, reference_signature = ordered_items[0]
    reference_payload = _signature_payload(reference_signature)

    for candidate_id, candidate_signature in ordered_items[1:]:
        candidate_payload = _signature_payload(candidate_signature)
        mismatched_fields = sorted(
            key
            for key in reference_payload.keys()
            if reference_payload.get(key) != candidate_payload.get(key)
        )
        if mismatched_fields:
            raise ValueError(
                f"{context}: fairness contract mismatch between '{reference_id}' and "
                f"'{candidate_id}'. Mismatched fields: {', '.join(mismatched_fields)}."
            )


def build_locked_comparison_signatures(
    *,
    comparison: Any,
    selected_variant_ids: list[str],
) -> dict[str, ComparisonContractSignature]:
    variants_by_id = {variant.variant_id: variant for variant in comparison.allowed_variants}
    signatures: dict[str, ComparisonContractSignature] = {}

    models = [str(variants_by_id[v].model) for v in selected_variant_ids]
    common_parity = _common_backend_parity_semantics(FrameworkMode.LOCKED_COMPARISON, models) if models else ()

    for variant_id in selected_variant_ids:
        variant = variants_by_id[variant_id]
        spec = get_model_spec(str(variant.model))
        if BASELINE_STANDARD_SCALER_RECIPE_ID not in set(spec.allowed_feature_recipe_ids):
            raise ValueError(
                "Locked comparison fairness contract requires baseline feature recipe support for "
                f"model '{spec.logical_name}'."
            )

        signatures[variant_id] = ComparisonContractSignature(
            target=str(comparison.scientific_contract.target),
            data_slice_semantics=(
                str(comparison.scientific_contract.split_mode),
                str(comparison.scientific_contract.subject_policy.source.value),
                str(comparison.scientific_contract.transfer_policy.source.value),
                str(comparison.scientific_contract.filter_task),
                str(comparison.scientific_contract.filter_modality),
            ),
            split_policy=(
                str(comparison.scientific_contract.split_mode),
                str(comparison.scientific_contract.grouping_policy),
            ),
            feature_recipe_id=BASELINE_STANDARD_SCALER_RECIPE_ID,
            metric_policy=(
                str(comparison.metric_policy.primary_metric),
                *tuple(str(value) for value in comparison.metric_policy.secondary_metrics),
            ),
            methodology_policy=(
                str(comparison.methodology_policy.policy_name.value),
                str(bool(comparison.methodology_policy.tuning_enabled)),
                str(comparison.methodology_policy.inner_cv_scheme),
                str(comparison.methodology_policy.inner_group_field),
                str(comparison.methodology_policy.tuning_search_space_id),
                str(comparison.methodology_policy.tuning_search_space_version),
            ),
            tuning_budget_semantics=(
                str(comparison.methodology_policy.tuning_search_space_id),
                str(comparison.methodology_policy.tuning_search_space_version),
                str(comparison.methodology_policy.inner_cv_scheme),
                str(comparison.methodology_policy.inner_group_field),
                str(bool(comparison.methodology_policy.tuning_enabled)),
            ),
            repeat_policy=(
                int(comparison.evidence_policy.repeat_evaluation.repeat_count),
                int(comparison.evidence_policy.repeat_evaluation.seed_stride),
            ),
            control_policy=(
                str(bool(comparison.control_policy.permutation_enabled)),
                str(comparison.control_policy.permutation_metric),
                str(int(comparison.control_policy.n_permutations)),
                str(bool(comparison.control_policy.dummy_baseline_enabled)),
            ),
            class_weight_policy=str(comparison.methodology_policy.class_weight_policy.value),
            deterministic_compute_requirements=(
                bool(
                    official_deterministic_compute_required(
                        framework_mode=FrameworkMode.LOCKED_COMPARISON,
                        hardware_mode="gpu_only",
                    )
                ),
                bool(
                    official_deterministic_compute_required(
                        framework_mode=FrameworkMode.LOCKED_COMPARISON,
                        hardware_mode="max_both",
                    )
                ),
            ),
            backend_parity_semantics=common_parity,
        )

    return signatures


def validate_locked_comparison_fairness_contract(
    *,
    comparison: Any,
    selected_variant_ids: list[str],
) -> None:
    signatures = build_locked_comparison_signatures(
        comparison=comparison,
        selected_variant_ids=selected_variant_ids,
    )
    validate_contract_signature_parity(
        signatures_by_id=signatures,
        context="locked_comparison",
    )


def _resolve_suite_models(protocol: Any, suite: Any) -> list[str]:
    base_models = list(suite.models) if suite.models is not None else list(protocol.model_policy.models)
    if protocol.control_policy.dummy_baseline.enabled and suite.suite_id in set(
        protocol.control_policy.dummy_baseline.suites
    ):
        if "dummy" not in base_models:
            base_models.append("dummy")
    return list(dict.fromkeys(base_models))


def validate_confirmatory_protocol_fairness_contract(
    *,
    protocol: Any,
    selected_suite_ids: list[str],
) -> None:
    suites_by_id = {
        suite.suite_id: suite for suite in protocol.official_run_suites if bool(suite.enabled)
    }

    for suite_id in selected_suite_ids:
        suite = suites_by_id[suite_id]
        suite_models = _resolve_suite_models(protocol, suite)
        common_parity = _common_backend_parity_semantics(FrameworkMode.CONFIRMATORY, suite_models) if suite_models else ()
        signatures: dict[str, ComparisonContractSignature] = {}

        for model_name in suite_models:
            spec = get_model_spec(str(model_name))
            if BASELINE_STANDARD_SCALER_RECIPE_ID not in set(spec.allowed_feature_recipe_ids):
                raise ValueError(
                    "Confirmatory fairness contract requires baseline feature recipe support "
                    f"for model '{spec.logical_name}'."
                )

            signatures[str(model_name)] = ComparisonContractSignature(
                target=str(protocol.scientific_contract.target),
                data_slice_semantics=(
                    str(suite.split_mode),
                    str(suite.subject_source.value),
                    str(suite.transfer_pair_source.value),
                    str(suite.filter_task),
                    str(suite.filter_modality),
                ),
                split_policy=(
                    str(suite.split_mode),
                    str(protocol.split_policy.grouping_field),
                ),
                feature_recipe_id=str(protocol.feature_engineering_policy.feature_recipe_id),
                metric_policy=(
                    str(protocol.metric_policy.primary_metric),
                    *tuple(str(value) for value in protocol.metric_policy.secondary_metrics),
                ),
                methodology_policy=(
                    str(protocol.methodology_policy.policy_name.value),
                    str(bool(protocol.methodology_policy.tuning_enabled)),
                    str(protocol.methodology_policy.inner_cv_scheme),
                    str(protocol.methodology_policy.inner_group_field),
                    str(protocol.methodology_policy.tuning_search_space_id),
                    str(protocol.methodology_policy.tuning_search_space_version),
                ),
                tuning_budget_semantics=(
                    str(protocol.methodology_policy.tuning_search_space_id),
                    str(protocol.methodology_policy.tuning_search_space_version),
                    str(protocol.methodology_policy.inner_cv_scheme),
                    str(protocol.methodology_policy.inner_group_field),
                    str(bool(protocol.methodology_policy.tuning_enabled)),
                ),
                repeat_policy=(
                    int(protocol.evidence_policy.repeat_evaluation.repeat_count),
                    int(protocol.evidence_policy.repeat_evaluation.seed_stride),
                ),
                control_policy=(
                    str(bool(protocol.control_policy.permutation.enabled)),
                    str(protocol.control_policy.permutation.metric),
                    str(int(protocol.control_policy.permutation.n_permutations)),
                    str(bool(protocol.control_policy.dummy_baseline.enabled)),
                ),
                class_weight_policy=str(protocol.methodology_policy.class_weight_policy.value),
                deterministic_compute_requirements=(
                    bool(
                        official_deterministic_compute_required(
                            framework_mode=FrameworkMode.CONFIRMATORY,
                            hardware_mode="gpu_only",
                        )
                    ),
                    bool(
                        official_deterministic_compute_required(
                            framework_mode=FrameworkMode.CONFIRMATORY,
                            hardware_mode="max_both",
                        )
                    ),
                ),
            backend_parity_semantics=common_parity,
            )

        validate_contract_signature_parity(
            signatures_by_id=signatures,
            context=f"confirmatory_suite:{suite_id}",
        )


__all__ = [
    "ComparisonContractSignature",
    "build_locked_comparison_signatures",
    "validate_confirmatory_protocol_fairness_contract",
    "validate_contract_signature_parity",
    "validate_locked_comparison_fairness_contract",
]
