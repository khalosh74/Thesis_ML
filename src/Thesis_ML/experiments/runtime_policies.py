from __future__ import annotations

from copy import deepcopy
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
from Thesis_ML.experiments.model_catalog import model_timeout_overrides_seconds

CONFIRMATORY_PROTOCOL_CONTEXT_REQUIRED_KEYS: tuple[str, ...] = (
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
    "model_cost_tier",
    "projected_runtime_seconds",
    "required_run_metadata_fields",
)

LOCKED_COMPARISON_CONTEXT_REQUIRED_KEYS: tuple[str, ...] = (
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
    "model_cost_tier",
    "projected_runtime_seconds",
    "required_run_metadata_fields",
)

_DEFAULT_TIMEOUT_POLICY: dict[str, Any] = {
    "enabled": True,
    "default_timeout_seconds": 90 * 60,
    "mode_timeouts_seconds": {
        FrameworkMode.CONFIRMATORY.value: 45 * 60,
        FrameworkMode.LOCKED_COMPARISON.value: 90 * 60,
    },
    "model_timeouts_seconds": model_timeout_overrides_seconds(),
    "shutdown_grace_seconds": 30,
    "absolute_hard_ceiling_seconds": 180 * 60,
}


def default_timeout_policy_payload() -> dict[str, Any]:
    return deepcopy(_DEFAULT_TIMEOUT_POLICY)


def _coerce_timeout_seconds(
    value: Any,
    *,
    field_name: str,
    minimum: int = 1,
) -> int:
    seconds = int(value)
    if seconds < int(minimum):
        raise ValueError(f"{field_name} must be >= {minimum} seconds.")
    return seconds


def _coerce_optional_timeout_map(raw: Any, *, field_name: str) -> dict[str, int]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be an object when provided.")
    resolved: dict[str, int] = {}
    for key, value in raw.items():
        key_text = str(key).strip().lower()
        if not key_text:
            raise ValueError(f"{field_name} contains an empty key.")
        resolved[key_text] = _coerce_timeout_seconds(
            value,
            field_name=f"{field_name}.{key_text}",
        )
    return resolved


def _merge_timeout_policy_overrides(
    base: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = deepcopy(base)
    if not isinstance(overrides, dict):
        return merged
    if "enabled" in overrides:
        merged["enabled"] = bool(overrides.get("enabled"))
    if "default_timeout_seconds" in overrides:
        merged["default_timeout_seconds"] = _coerce_timeout_seconds(
            overrides.get("default_timeout_seconds"),
            field_name="default_timeout_seconds",
        )
    if "mode_timeouts_seconds" in overrides:
        merged["mode_timeouts_seconds"] = _coerce_optional_timeout_map(
            overrides.get("mode_timeouts_seconds"),
            field_name="mode_timeouts_seconds",
        )
    if "model_timeouts_seconds" in overrides:
        merged["model_timeouts_seconds"] = _coerce_optional_timeout_map(
            overrides.get("model_timeouts_seconds"),
            field_name="model_timeouts_seconds",
        )
    if "shutdown_grace_seconds" in overrides:
        merged["shutdown_grace_seconds"] = _coerce_timeout_seconds(
            overrides.get("shutdown_grace_seconds"),
            field_name="shutdown_grace_seconds",
        )
    if "absolute_hard_ceiling_seconds" in overrides:
        merged["absolute_hard_ceiling_seconds"] = _coerce_timeout_seconds(
            overrides.get("absolute_hard_ceiling_seconds"),
            field_name="absolute_hard_ceiling_seconds",
        )
    return merged


def resolve_run_timeout_policy(
    *,
    framework_mode: FrameworkMode | str,
    model_name: str,
    policy_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_mode = coerce_framework_mode(framework_mode)
    merged = _merge_timeout_policy_overrides(
        default_timeout_policy_payload(),
        policy_overrides,
    )
    enabled = bool(merged.get("enabled", True))
    default_timeout_seconds = _coerce_timeout_seconds(
        merged.get("default_timeout_seconds", _DEFAULT_TIMEOUT_POLICY["default_timeout_seconds"]),
        field_name="default_timeout_seconds",
    )
    mode_timeouts = _coerce_optional_timeout_map(
        merged.get("mode_timeouts_seconds"),
        field_name="mode_timeouts_seconds",
    )
    model_timeouts = _coerce_optional_timeout_map(
        merged.get("model_timeouts_seconds"),
        field_name="model_timeouts_seconds",
    )
    shutdown_grace_seconds = _coerce_timeout_seconds(
        merged.get("shutdown_grace_seconds", _DEFAULT_TIMEOUT_POLICY["shutdown_grace_seconds"]),
        field_name="shutdown_grace_seconds",
    )
    hard_ceiling_seconds = _coerce_timeout_seconds(
        merged.get(
            "absolute_hard_ceiling_seconds",
            _DEFAULT_TIMEOUT_POLICY["absolute_hard_ceiling_seconds"],
        ),
        field_name="absolute_hard_ceiling_seconds",
    )

    effective_timeout_seconds: int | None = None
    effective_source = "disabled"
    if enabled:
        candidate = default_timeout_seconds
        effective_source = "default"

        mode_key = resolved_mode.value
        if mode_key in mode_timeouts:
            candidate = int(mode_timeouts[mode_key])
            effective_source = "mode_default"

        model_key = str(model_name).strip().lower()
        if model_key in model_timeouts:
            candidate = int(model_timeouts[model_key])
            effective_source = "model_override"

        effective_timeout_seconds = min(int(candidate), int(hard_ceiling_seconds))

    return {
        "enabled": enabled,
        "default_timeout_seconds": int(default_timeout_seconds),
        "mode_timeouts_seconds": dict(mode_timeouts),
        "model_timeouts_seconds": dict(model_timeouts),
        "shutdown_grace_seconds": int(shutdown_grace_seconds),
        "absolute_hard_ceiling_seconds": int(hard_ceiling_seconds),
        "effective_timeout_seconds": (
            int(effective_timeout_seconds) if effective_timeout_seconds is not None else None
        ),
        "effective_timeout_source": str(effective_source),
        "framework_mode": resolved_mode.value,
        "model_name": str(model_name),
    }


def _validate_required_context_keys(
    *,
    framework_mode: FrameworkMode,
    context_name: str,
    context: dict[str, Any],
    required_keys: tuple[str, ...],
) -> None:
    missing = [key for key in required_keys if key not in context]
    if missing:
        raise ValueError(
            f"framework_mode='{framework_mode.value}' {context_name} is missing required keys: "
            + ", ".join(missing)
        )
    data_policy_payload = context.get("data_policy")
    if not isinstance(data_policy_payload, dict):
        raise ValueError(
            f"framework_mode='{framework_mode.value}' {context_name}.data_policy must be a JSON object."
        )


def validate_official_context_payload(
    *,
    framework_mode: FrameworkMode | str,
    context_name: Literal["protocol_context", "comparison_context"],
    context: dict[str, Any],
) -> dict[str, Any]:
    resolved_mode = coerce_framework_mode(framework_mode)
    resolved_context = dict(context)

    if resolved_mode == FrameworkMode.CONFIRMATORY:
        if context_name != "protocol_context":
            raise ValueError(
                "framework_mode='confirmatory' requires context_name='protocol_context'."
            )
        _validate_required_context_keys(
            framework_mode=resolved_mode,
            context_name=context_name,
            context=resolved_context,
            required_keys=CONFIRMATORY_PROTOCOL_CONTEXT_REQUIRED_KEYS,
        )
        if bool(resolved_context.get("canonical_run", True)) is not True:
            raise ValueError(
                "framework_mode='confirmatory' requires protocol_context['canonical_run']=true."
            )
        if str(resolved_context.get("framework_mode")) != FrameworkMode.CONFIRMATORY.value:
            raise ValueError(
                "framework_mode='confirmatory' requires protocol_context['framework_mode']='confirmatory'."
            )
        resolved_context["canonical_run"] = True
        return resolved_context

    if resolved_mode == FrameworkMode.LOCKED_COMPARISON:
        if context_name != "comparison_context":
            raise ValueError(
                "framework_mode='locked_comparison' requires context_name='comparison_context'."
            )
        _validate_required_context_keys(
            framework_mode=resolved_mode,
            context_name=context_name,
            context=resolved_context,
            required_keys=LOCKED_COMPARISON_CONTEXT_REQUIRED_KEYS,
        )
        if str(resolved_context.get("framework_mode")) != FrameworkMode.LOCKED_COMPARISON.value:
            raise ValueError(
                "framework_mode='locked_comparison' requires comparison_context['framework_mode']='locked_comparison'."
            )
        return resolved_context

    raise ValueError(
        "validate_official_context_payload only supports official framework modes: "
        "confirmatory and locked_comparison."
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
        resolved_protocol_context = validate_official_context_payload(
            framework_mode=resolved_mode,
            context_name="protocol_context",
            context=resolved_protocol_context,
        )
        return resolved_mode, True, resolved_protocol_context, {}

    if resolved_mode == FrameworkMode.LOCKED_COMPARISON:
        if not resolved_comparison_context:
            raise ValueError(
                "framework_mode='locked_comparison' requires non-empty comparison_context."
            )
        if protocol_context is not None:
            raise ValueError("framework_mode='locked_comparison' cannot accept protocol_context.")
        resolved_comparison_context = validate_official_context_payload(
            framework_mode=resolved_mode,
            context_name="comparison_context",
            context=resolved_comparison_context,
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
                "Official run context is missing methodology/subgroup keys: " + ", ".join(missing)
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
                "Unsupported tuning_inner_cv_scheme. Allowed value: grouped_leave_one_group_out."
            )
        resolved_inner_cv_scheme_literal = cast(
            Literal["grouped_leave_one_group_out"], normalized_inner_cv
        )

    if resolved_policy_name == MethodologyPolicyName.FIXED_BASELINES_ONLY.value:
        resolved_inner_cv_scheme_literal = None
        resolved_tuning_inner_group_field = None
        resolved_tuning_space_id = None
        resolved_tuning_space_version = None

    if resolved_evidence_run_role == "untuned_baseline":
        if resolved_policy_name != MethodologyPolicyName.GROUPED_NESTED_TUNING.value:
            raise ValueError(
                "evidence_run_role='untuned_baseline' requires methodology_policy_name='grouped_nested_tuning'."
            )
        if bool(resolved_tuning_enabled):
            raise ValueError("evidence_run_role='untuned_baseline' requires tuning_enabled=false.")
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
        if (
            context_primary_metric is not None
            and validate_metric_name(str(context_primary_metric)) != resolved_primary_metric_name
        ):
            raise ValueError(
                "Illegal override for official run key 'primary_metric'. "
                "Use protocol/comparison spec values only."
            )
        controls_payload = official_context.get("controls")
        if isinstance(controls_payload, dict):
            context_perm_metric = controls_payload.get("permutation_metric")
            if (
                context_perm_metric is not None
                and validate_metric_name(str(context_perm_metric))
                != resolved_permutation_metric_name
            ):
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
