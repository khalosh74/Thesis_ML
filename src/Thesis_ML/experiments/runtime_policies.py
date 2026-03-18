from __future__ import annotations

from typing import Any, Literal, cast

from Thesis_ML.config.framework_mode import FrameworkMode, coerce_framework_mode
from Thesis_ML.config.methodology import (
    ClassWeightPolicy,
    MethodologyPolicy,
    MethodologyPolicyName,
    SubgroupReportingPolicy,
)
from Thesis_ML.config.metric_policy import (
    EffectiveMetricPolicy,
    enforce_primary_metric_alignment,
    resolve_effective_metric_policy,
    validate_metric_name,
)


def resolve_framework_context(
    framework_mode: FrameworkMode | str,
    *,
    protocol_context: dict[str, Any] | None,
    comparison_context: dict[str, Any] | None,
) -> tuple[
    FrameworkMode,
    bool,
    dict[str, Any],
    dict[str, Any],
]:
    resolved_mode = coerce_framework_mode(framework_mode)
    resolved_protocol_context = dict(protocol_context or {})
    resolved_comparison_context = dict(comparison_context or {})

    if resolved_mode == FrameworkMode.EXPLORATORY:
        if protocol_context is not None:
            raise ValueError(
                "framework_mode='exploratory' cannot accept protocol_context. "
                "Use framework_mode='confirmatory' via thesisml-run-protocol."
            )
        if comparison_context is not None:
            raise ValueError(
                "framework_mode='exploratory' cannot accept comparison_context. "
                "Use framework_mode='locked_comparison' via thesisml-run-comparison."
            )
        return resolved_mode, False, {}, {}

    if resolved_mode == FrameworkMode.CONFIRMATORY:
        if not resolved_protocol_context:
            raise ValueError("framework_mode='confirmatory' requires non-empty protocol_context.")
        if comparison_context is not None:
            raise ValueError("framework_mode='confirmatory' cannot accept comparison_context.")
        required_keys = [
            "framework_mode",
            "protocol_id",
            "protocol_version",
            "protocol_schema_version",
            "suite_id",
            "claim_ids",
            "methodology_policy_name",
            "class_weight_policy",
            "tuning_enabled",
            "subgroup_reporting_enabled",
            "subgroup_dimensions",
            "subgroup_min_samples_per_group",
            "metric_policy",
            "data_policy",
            "required_run_metadata_fields",
        ]
        missing = [key for key in required_keys if key not in resolved_protocol_context]
        if missing:
            raise ValueError(
                "framework_mode='confirmatory' protocol_context is missing required keys: "
                + ", ".join(missing)
            )
        if bool(resolved_protocol_context.get("canonical_run", True)) is not True:
            raise ValueError(
                "framework_mode='confirmatory' requires protocol_context['canonical_run']=true."
            )
        if str(resolved_protocol_context.get("framework_mode")) != FrameworkMode.CONFIRMATORY.value:
            raise ValueError(
                "framework_mode='confirmatory' requires protocol_context['framework_mode']='confirmatory'."
            )
        resolved_protocol_context["canonical_run"] = True
        return resolved_mode, True, resolved_protocol_context, {}

    if resolved_mode == FrameworkMode.LOCKED_COMPARISON:
        if not resolved_comparison_context:
            raise ValueError(
                "framework_mode='locked_comparison' requires non-empty comparison_context."
            )
        if protocol_context is not None:
            raise ValueError("framework_mode='locked_comparison' cannot accept protocol_context.")
        required_keys = [
            "framework_mode",
            "comparison_id",
            "comparison_version",
            "variant_id",
            "methodology_policy_name",
            "class_weight_policy",
            "tuning_enabled",
            "subgroup_reporting_enabled",
            "subgroup_dimensions",
            "subgroup_min_samples_per_group",
            "metric_policy",
            "data_policy",
            "required_run_metadata_fields",
        ]
        missing = [key for key in required_keys if key not in resolved_comparison_context]
        if missing:
            raise ValueError(
                "framework_mode='locked_comparison' comparison_context is missing required keys: "
                + ", ".join(missing)
            )
        if (
            str(resolved_comparison_context.get("framework_mode"))
            != FrameworkMode.LOCKED_COMPARISON.value
        ):
            raise ValueError(
                "framework_mode='locked_comparison' requires comparison_context['framework_mode']='locked_comparison'."
            )
        return resolved_mode, False, {}, resolved_comparison_context

    raise ValueError(f"Unsupported framework_mode '{resolved_mode}'.")


def resolve_methodology_runtime(
    *,
    framework_mode: FrameworkMode,
    methodology_policy_name: str,
    class_weight_policy: str,
    tuning_enabled: bool,
    tuning_search_space_id: str | None,
    tuning_search_space_version: str | None,
    tuning_inner_cv_scheme: str | None,
    tuning_inner_group_field: str | None,
    subgroup_reporting_enabled: bool,
    subgroup_dimensions: list[str] | None,
    subgroup_min_samples_per_group: int,
    evidence_run_role: str | None,
    protocol_context: dict[str, Any],
    comparison_context: dict[str, Any],
) -> tuple[MethodologyPolicy, SubgroupReportingPolicy]:
    source_context: dict[str, Any] = {}
    if framework_mode == FrameworkMode.CONFIRMATORY:
        source_context = dict(protocol_context)
    if framework_mode == FrameworkMode.LOCKED_COMPARISON:
        source_context = dict(comparison_context)

    if framework_mode in {FrameworkMode.CONFIRMATORY, FrameworkMode.LOCKED_COMPARISON}:
        required_context_keys = {
            "methodology_policy_name",
            "class_weight_policy",
            "tuning_enabled",
            "subgroup_reporting_enabled",
            "subgroup_dimensions",
            "subgroup_min_samples_per_group",
        }
        missing = [key for key in sorted(required_context_keys) if key not in source_context]
        if missing:
            raise ValueError(
                "Official run context is missing methodology/subgroup keys: "
                + ", ".join(missing)
            )
        mismatch_checks = {
            "methodology_policy_name": str(methodology_policy_name),
            "class_weight_policy": str(class_weight_policy),
            "tuning_enabled": bool(tuning_enabled),
        }
        for key, local_value in mismatch_checks.items():
            context_value = source_context.get(key)
            if context_value is None:
                continue
            if key == "tuning_enabled":
                if bool(context_value) != bool(local_value):
                    raise ValueError(
                        f"Illegal override for official run key '{key}'. "
                        "Use protocol/comparison spec values only."
                    )
                continue
            if str(context_value) != str(local_value):
                raise ValueError(
                    f"Illegal override for official run key '{key}'. "
                    "Use protocol/comparison spec values only."
                )

    resolved_policy_name = str(
        source_context.get("methodology_policy_name", methodology_policy_name)
    )
    resolved_class_weight_policy = str(
        source_context.get("class_weight_policy", class_weight_policy)
    )
    resolved_tuning_enabled = bool(source_context.get("tuning_enabled", tuning_enabled))
    resolved_tuning_space_id = source_context.get("tuning_search_space_id", tuning_search_space_id)
    resolved_tuning_space_version = source_context.get(
        "tuning_search_space_version", tuning_search_space_version
    )
    resolved_tuning_inner_cv_scheme = source_context.get(
        "tuning_inner_cv_scheme", tuning_inner_cv_scheme
    )
    resolved_tuning_inner_group_field = source_context.get(
        "tuning_inner_group_field", tuning_inner_group_field
    )
    resolved_subgroup_enabled = bool(
        source_context.get("subgroup_reporting_enabled", subgroup_reporting_enabled)
    )
    resolved_subgroup_dimensions = source_context.get(
        "subgroup_dimensions",
        subgroup_dimensions
        if subgroup_dimensions is not None
        else ["label", "task", "modality", "session", "subject"],
    )
    resolved_subgroup_min_samples = int(
        source_context.get("subgroup_min_samples_per_group", subgroup_min_samples_per_group)
    )
    resolved_evidence_run_role = str(
        source_context.get("evidence_run_role", evidence_run_role or "primary")
    ).strip()
    if resolved_evidence_run_role not in {"primary", "untuned_baseline"}:
        raise ValueError(
            "Unsupported evidence_run_role. Allowed values: primary, untuned_baseline."
        )

    resolved_inner_cv_scheme_literal: Literal["grouped_leave_one_group_out"] | None
    if resolved_tuning_inner_cv_scheme is None:
        resolved_inner_cv_scheme_literal = None
    else:
        normalized_inner_cv = str(resolved_tuning_inner_cv_scheme).strip()
        if normalized_inner_cv != "grouped_leave_one_group_out":
            raise ValueError(
                "Unsupported tuning_inner_cv_scheme. "
                "Allowed value: grouped_leave_one_group_out."
            )
        resolved_inner_cv_scheme_literal = cast(
            Literal["grouped_leave_one_group_out"], normalized_inner_cv
        )

    if resolved_evidence_run_role == "untuned_baseline":
        if resolved_policy_name != MethodologyPolicyName.GROUPED_NESTED_TUNING.value:
            raise ValueError(
                "evidence_run_role='untuned_baseline' requires methodology_policy_name='grouped_nested_tuning'."
            )
        if bool(resolved_tuning_enabled):
            raise ValueError(
                "evidence_run_role='untuned_baseline' requires tuning_enabled=false."
            )
        if any(
            value is not None
            for value in (
                resolved_tuning_space_id,
                resolved_tuning_space_version,
                resolved_inner_cv_scheme_literal,
                resolved_tuning_inner_group_field,
            )
        ):
            raise ValueError(
                "evidence_run_role='untuned_baseline' forbids tuning search-space and inner-CV metadata."
            )
        methodology_policy = MethodologyPolicy.model_construct(
            policy_name=MethodologyPolicyName.GROUPED_NESTED_TUNING,
            class_weight_policy=ClassWeightPolicy(resolved_class_weight_policy),
            tuning_enabled=False,
            inner_cv_scheme=None,
            inner_group_field=None,
            tuning_search_space_id=None,
            tuning_search_space_version=None,
            notes=None,
        )
    else:
        methodology_policy = MethodologyPolicy(
            policy_name=MethodologyPolicyName(resolved_policy_name),
            class_weight_policy=ClassWeightPolicy(resolved_class_weight_policy),
            tuning_enabled=resolved_tuning_enabled,
            inner_cv_scheme=resolved_inner_cv_scheme_literal,
            inner_group_field=resolved_tuning_inner_group_field,
            tuning_search_space_id=resolved_tuning_space_id,
            tuning_search_space_version=resolved_tuning_space_version,
        )
    subgroup_policy = SubgroupReportingPolicy(
        enabled=resolved_subgroup_enabled,
        subgroup_dimensions=list(resolved_subgroup_dimensions),
        min_samples_per_group=resolved_subgroup_min_samples,
    )
    return methodology_policy, subgroup_policy


def resolve_metric_policy_runtime(
    *,
    framework_mode: FrameworkMode,
    official_context: dict[str, Any],
    primary_metric_name: str,
    permutation_metric_name: str | None,
    n_permutations: int,
    interpretability_enabled_override: bool | None,
) -> tuple[str, str, EffectiveMetricPolicy]:
    resolved_primary_metric_name = validate_metric_name(primary_metric_name)
    resolved_permutation_metric_name = (
        validate_metric_name(permutation_metric_name)
        if permutation_metric_name is not None
        else resolved_primary_metric_name
    )

    resolved_secondary_metrics: list[str] = []
    resolved_decision_metric = resolved_primary_metric_name
    resolved_tuning_metric = resolved_primary_metric_name
    if official_context:
        context_primary_metric = official_context.get("primary_metric")
        if context_primary_metric is not None and validate_metric_name(
            str(context_primary_metric)
        ) != resolved_primary_metric_name:
            raise ValueError(
                "Illegal override for official run key 'primary_metric'. "
                "Use protocol/comparison spec values only."
            )
        controls_payload = official_context.get("controls")
        if isinstance(controls_payload, dict):
            context_perm_metric = controls_payload.get("permutation_metric")
            if context_perm_metric is not None and validate_metric_name(
                str(context_perm_metric)
            ) != resolved_permutation_metric_name:
                raise ValueError(
                    "Illegal override for official run key 'permutation_metric'. "
                    "Use protocol/comparison spec values only."
                )
            context_n_permutations = controls_payload.get("n_permutations")
            if context_n_permutations is not None and int(context_n_permutations) != int(
                n_permutations
            ):
                raise ValueError(
                    "Illegal override for official run key 'n_permutations'. "
                    "Use protocol/comparison spec values only."
                )
        context_interpretability = official_context.get("interpretability_enabled")
        if (
            context_interpretability is not None
            and interpretability_enabled_override is not None
            and bool(context_interpretability) != bool(interpretability_enabled_override)
        ):
            raise ValueError(
                "Illegal override for official run key 'interpretability_enabled'. "
                "Use protocol/comparison spec values only."
            )
        metric_policy_payload = official_context.get("metric_policy")
        if not isinstance(metric_policy_payload, dict):
            raise ValueError(
                "Official run context is missing metric_policy payload. "
                "Use protocol/comparison spec values only."
            )
        payload_primary_metric = metric_policy_payload.get("primary_metric")
        if payload_primary_metric is None:
            raise ValueError("Official run context metric_policy is missing primary_metric.")
        if validate_metric_name(str(payload_primary_metric)) != resolved_primary_metric_name:
            raise ValueError(
                "Illegal override for official run key 'metric_policy.primary_metric'. "
                "Use protocol/comparison spec values only."
            )
        payload_secondary_metrics = metric_policy_payload.get("secondary_metrics", [])
        if payload_secondary_metrics is not None and not isinstance(
            payload_secondary_metrics, list
        ):
            raise ValueError("Official run context metric_policy.secondary_metrics must be a list.")
        resolved_secondary_metrics = (
            [str(value) for value in payload_secondary_metrics]
            if isinstance(payload_secondary_metrics, list)
            else []
        )
        payload_decision_metric = metric_policy_payload.get("decision_metric")
        if payload_decision_metric is not None:
            resolved_decision_metric = validate_metric_name(str(payload_decision_metric))
        payload_tuning_metric = metric_policy_payload.get("tuning_metric")
        if payload_tuning_metric is not None:
            resolved_tuning_metric = validate_metric_name(str(payload_tuning_metric))
        payload_permutation_metric = metric_policy_payload.get("permutation_metric")
        if payload_permutation_metric is not None:
            if (
                validate_metric_name(str(payload_permutation_metric))
                != resolved_permutation_metric_name
            ):
                raise ValueError(
                    "Illegal override for official run key 'metric_policy.permutation_metric'. "
                    "Use protocol/comparison spec values only."
                )
    metric_policy_effective = resolve_effective_metric_policy(
        primary_metric=resolved_primary_metric_name,
        secondary_metrics=resolved_secondary_metrics,
        decision_metric=resolved_decision_metric,
        tuning_metric=resolved_tuning_metric,
        permutation_metric=resolved_permutation_metric_name,
    )
    if framework_mode in {FrameworkMode.CONFIRMATORY, FrameworkMode.LOCKED_COMPARISON}:
        metric_policy_effective = enforce_primary_metric_alignment(
            metric_policy_effective,
            context=f"framework_mode='{framework_mode.value}'",
        )
    return resolved_primary_metric_name, resolved_permutation_metric_name, metric_policy_effective
