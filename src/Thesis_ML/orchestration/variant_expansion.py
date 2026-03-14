from __future__ import annotations

from typing import Any

from Thesis_ML.orchestration.search_space import expand_variant_search_space


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except Exception:
        return None


def variant_label(params: dict[str, Any]) -> str:
    keys = [
        "target",
        "cv",
        "model",
        "subject",
        "train_subject",
        "test_subject",
        "filter_task",
        "filter_modality",
    ]
    parts: list[str] = []
    for key in keys:
        value = params.get(key)
        if value is None:
            continue
        value_text = str(value).strip()
        if not value_text:
            continue
        parts.append(f"{key}={value_text}")
    return ", ".join(parts)


def expand_template_variants(
    experiment: dict[str, Any],
    template: dict[str, Any],
    dataset_scope: dict[str, Any],
    search_space_map: dict[str, Any] | None = None,
    search_seed: int = 42,
    optuna_enabled: bool = False,
    optuna_trials: int | None = None,
) -> list[dict[str, Any]]:
    search_map = search_space_map or {}
    template_id = str(template.get("template_id", "template"))
    supported = bool(template.get("supported", False))
    base_params = dict(template.get("params", {}))
    start_section = template.get("start_section")
    end_section = template.get("end_section")
    base_artifact_id = template.get("base_artifact_id")
    reuse_policy = template.get("reuse_policy")
    search_space_id = (
        str(template.get("search_space_id")).strip() if template.get("search_space_id") else None
    )
    repeat_raw = template.get("repeat_id")
    seed_raw = template.get("seed")
    study_id = str(template.get("study_id")).strip() if template.get("study_id") else None
    trial_id = str(template.get("trial_id")).strip() if template.get("trial_id") else None
    cell_id = str(template.get("cell_id")).strip() if template.get("cell_id") else None
    repeat_id = _optional_int(repeat_raw)
    seed = _optional_int(seed_raw)
    factor_settings = (
        dict(template.get("factor_settings", {}))
        if isinstance(template.get("factor_settings"), dict)
        else {}
    )
    fixed_controls = (
        dict(template.get("fixed_controls", {}))
        if isinstance(template.get("fixed_controls"), dict)
        else {}
    )
    design_metadata = (
        dict(template.get("design_metadata", {}))
        if isinstance(template.get("design_metadata"), dict)
        else {}
    )
    if not supported:
        reason = str(template.get("unsupported_reason", "template marked unsupported"))
        return [
            {
                "template_id": template_id,
                "variant_index": 1,
                "params": base_params,
                "supported": False,
                "blocked_reason": reason,
                "start_section": start_section,
                "end_section": end_section,
                "base_artifact_id": base_artifact_id,
                "reuse_policy": reuse_policy,
                "search_space_id": search_space_id,
                "search_assignment": None,
                "study_id": study_id,
                "trial_id": trial_id,
                "cell_id": cell_id,
                "repeat_id": repeat_id,
                "seed": seed,
                "factor_settings": factor_settings,
                "fixed_controls": fixed_controls,
                "design_metadata": design_metadata,
            }
        ]

    expand_config = dict(template.get("expand", {}))
    variants: list[dict[str, Any]] = [
        {
            "params": base_params,
            "supported": True,
            "blocked_reason": None,
        }
    ]

    for param_name, scope_key in expand_config.items():
        expanded: list[dict[str, Any]] = []
        values = dataset_scope.get(str(scope_key))
        if not isinstance(values, list) or not values:
            blocked_reason = (
                f"Expansion for '{param_name}' requires non-empty scope '{scope_key}', "
                "but no values were available."
            )
            return [
                {
                    "template_id": template_id,
                    "variant_index": 1,
                    "params": base_params,
                    "supported": False,
                    "blocked_reason": blocked_reason,
                    "start_section": start_section,
                    "end_section": end_section,
                    "base_artifact_id": base_artifact_id,
                    "reuse_policy": reuse_policy,
                    "search_space_id": search_space_id,
                    "search_assignment": None,
                    "study_id": study_id,
                    "trial_id": trial_id,
                    "cell_id": cell_id,
                    "repeat_id": repeat_id,
                    "seed": seed,
                    "factor_settings": factor_settings,
                    "fixed_controls": fixed_controls,
                    "design_metadata": design_metadata,
                }
            ]

        for row in variants:
            for value in values:
                params = dict(row["params"])
                if param_name == "train_test_pair":
                    if not isinstance(value, (list, tuple)) or len(value) != 2:
                        raise ValueError(
                            f"Invalid ordered_subject_pairs value for experiment "
                            f"{experiment['experiment_id']}: {value}"
                        )
                    params["train_subject"] = str(value[0])
                    params["test_subject"] = str(value[1])
                else:
                    params[str(param_name)] = value
                expanded.append(
                    {
                        "params": params,
                        "supported": True,
                        "blocked_reason": None,
                    }
                )
        variants = expanded

    unresolved: list[dict[str, Any]] = []
    for row in variants:
        base_variant = {
            "template_id": template_id,
            "params": row["params"],
            "supported": bool(row["supported"]),
            "blocked_reason": row["blocked_reason"],
            "start_section": start_section,
            "end_section": end_section,
            "base_artifact_id": base_artifact_id,
            "reuse_policy": reuse_policy,
            "search_space_id": search_space_id,
            "search_assignment": None,
            "study_id": study_id,
            "trial_id": trial_id,
            "cell_id": cell_id,
            "repeat_id": repeat_id,
            "seed": seed,
            "factor_settings": factor_settings,
            "fixed_controls": fixed_controls,
            "design_metadata": design_metadata,
        }
        if not search_space_id:
            unresolved.append(base_variant)
            continue
        search_space = search_map.get(search_space_id)
        if search_space is None:
            unresolved.append(
                {
                    **base_variant,
                    "supported": False,
                    "blocked_reason": (
                        f"Search space '{search_space_id}' was referenced by template "
                        f"'{template_id}' but is not defined."
                    ),
                }
            )
            continue
        try:
            expanded_variants = expand_variant_search_space(
                base_variant,
                search_space=search_space,
                seed=search_seed,
                optuna_enabled=optuna_enabled,
                optuna_trials=optuna_trials,
            )
        except ValueError as exc:
            unresolved.append(
                {
                    **base_variant,
                    "supported": False,
                    "blocked_reason": str(exc),
                }
            )
            continue
        unresolved.extend(expanded_variants)

    resolved: list[dict[str, Any]] = []
    for idx, row in enumerate(unresolved, start=1):
        trial_id_value = (
            str(row.get("trial_id")).strip() if row.get("trial_id") is not None else None
        )
        resolved_trial_id = None
        if trial_id_value:
            resolved_trial_id = (
                trial_id_value if len(unresolved) == 1 else f"{trial_id_value}__v{idx:03d}"
            )
        resolved.append(
            {
                "template_id": template_id,
                "variant_index": idx,
                "params": row["params"],
                "supported": bool(row.get("supported", True)),
                "blocked_reason": row.get("blocked_reason"),
                "start_section": row.get("start_section"),
                "end_section": row.get("end_section"),
                "base_artifact_id": row.get("base_artifact_id"),
                "reuse_policy": row.get("reuse_policy"),
                "search_space_id": row.get("search_space_id"),
                "search_assignment": row.get("search_assignment"),
                "study_id": row.get("study_id"),
                "trial_id": resolved_trial_id,
                "cell_id": row.get("cell_id"),
                "repeat_id": row.get("repeat_id"),
                "seed": row.get("seed"),
                "factor_settings": row.get("factor_settings")
                if isinstance(row.get("factor_settings"), dict)
                else {},
                "fixed_controls": row.get("fixed_controls")
                if isinstance(row.get("fixed_controls"), dict)
                else {},
                "design_metadata": row.get("design_metadata")
                if isinstance(row.get("design_metadata"), dict)
                else {},
            }
        )
    return resolved


def expand_experiment_variants(
    experiment: dict[str, Any],
    dataset_scope: dict[str, Any],
    search_space_map: dict[str, Any] | None = None,
    search_seed: int = 42,
    optuna_enabled: bool = False,
    optuna_trials: int | None = None,
    max_runs_per_experiment: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    templates = list(experiment.get("variant_templates", []))
    if not templates:
        blocked_reasons = [
            str(reason) for reason in list(experiment.get("blocked_reasons", [])) if str(reason)
        ]
        reason = (
            "; ".join(blocked_reasons)
            if blocked_reasons
            else "No variant templates are defined for this experiment in the registry."
        )
        return (
            [
                {
                    "template_id": "no_template",
                    "variant_index": 1,
                    "params": {},
                    "supported": False,
                    "blocked_reason": reason,
                    "study_id": (
                        str(experiment.get("experiment_id", "")).strip()
                        if bool(experiment.get("is_study_design"))
                        else None
                    ),
                }
            ],
            [reason],
        )

    variants: list[dict[str, Any]] = []
    warnings: list[str] = []
    for template in templates:
        template_variants = expand_template_variants(
            experiment=experiment,
            template=template,
            dataset_scope=dataset_scope,
            search_space_map=search_space_map or {},
            search_seed=search_seed,
            optuna_enabled=optuna_enabled,
            optuna_trials=optuna_trials,
        )
        variants.extend(template_variants)

    if max_runs_per_experiment is not None and max_runs_per_experiment > 0:
        executable = [item for item in variants if item["supported"]]
        non_executable = [item for item in variants if not item["supported"]]
        if len(executable) > max_runs_per_experiment:
            warnings.append(
                f"Truncated executable variants from {len(executable)} to "
                f"{max_runs_per_experiment} due to --max-runs-per-experiment."
            )
            executable = executable[:max_runs_per_experiment]
        variants = executable + non_executable

    return variants, warnings


__all__ = [
    "expand_experiment_variants",
    "expand_template_variants",
    "variant_label",
]
