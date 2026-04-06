from __future__ import annotations

import math
from typing import Any

from Thesis_ML.experiments.model_factory import model_supports_linear_interpretability
from Thesis_ML.orchestration.search_space import expand_variant_search_space

_RUN_EXPERIMENT_SUPPORTED_CV_MODES: set[str] = {
    "loso_session",
    "within_subject_loso_session",
    "frozen_cross_person_transfer",
    "record_random_split",
}
_E12_ANCHOR_PREFERRED_EXPERIMENT_IDS: tuple[str, ...] = ("E16",)
_E12_CONFIRMATORY_CV_MODES: set[str] = {
    "within_subject_loso_session",
    "frozen_cross_person_transfer",
}
_E12_ANCHOR_PARAM_KEYS: tuple[str, ...] = (
    "target",
    "cv",
    "model",
    "subject",
    "train_subject",
    "test_subject",
    "filter_task",
    "filter_modality",
    "feature_space",
    "roi_spec_path",
    "preprocessing_strategy",
    "dimensionality_strategy",
    "pca_n_components",
    "pca_variance_ratio",
    "methodology_policy_name",
    "class_weight_policy",
    "tuning_search_space_id",
    "tuning_search_space_version",
    "tuning_inner_cv_scheme",
    "tuning_inner_group_field",
    "framework_mode",
    "canonical_run",
    "protocol_context",
    "scope_task_ids",
)
_E13_CONFIRMATORY_CV_MODES: set[str] = {
    "within_subject_loso_session",
    "frozen_cross_person_transfer",
}
_E13_ANCHOR_PARAM_KEYS: tuple[str, ...] = (
    "target",
    "cv",
    "subject",
    "train_subject",
    "test_subject",
    "feature_space",
    "roi_spec_path",
    "preprocessing_strategy",
    "dimensionality_strategy",
    "pca_n_components",
    "pca_variance_ratio",
    "methodology_policy_name",
    "class_weight_policy",
    "framework_mode",
    "canonical_run",
    "protocol_context",
    "scope_task_ids",
    "filter_task",
    "filter_modality",
)
_E14_ANCHOR_PREFERRED_EXPERIMENT_IDS: tuple[str, ...] = ("E16",)
_E14_ANCHOR_PARAM_KEYS: tuple[str, ...] = (
    "target",
    "cv",
    "model",
    "subject",
    "feature_space",
    "roi_spec_path",
    "preprocessing_strategy",
    "dimensionality_strategy",
    "pca_n_components",
    "pca_variance_ratio",
    "methodology_policy_name",
    "class_weight_policy",
    "framework_mode",
    "canonical_run",
    "protocol_context",
    "scope_task_ids",
    "filter_task",
    "filter_modality",
)
_E15_LOCKED_CORE_REQUIRED_KEYS: tuple[str, ...] = (
    "feature_space",
    "preprocessing_strategy",
    "dimensionality_strategy",
    "methodology_policy_name",
    "class_weight_policy",
)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except Exception:
        return None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_executable_experiment(row: dict[str, Any]) -> bool:
    if bool(row.get("executable_now", True)) is False:
        return False
    execution_status = _safe_text(row.get("execution_status")).lower()
    if execution_status == "blocked":
        return False
    return True


def _missing_locked_core_keys(
    params: dict[str, Any],
    *,
    required_keys: tuple[str, ...],
) -> list[str]:
    missing: list[str] = []
    for key in required_keys:
        if _safe_text(params.get(key)):
            continue
        missing.append(str(key))
    return missing


def _anchor_analysis_key(anchor_params: dict[str, Any]) -> str:
    payload = {
        key: anchor_params.get(key)
        for key in sorted(_E12_ANCHOR_PARAM_KEYS)
        if anchor_params.get(key) not in (None, "")
    }
    return str(payload)


def _e13_anchor_analysis_key(anchor_params: dict[str, Any]) -> str:
    payload = {
        key: anchor_params.get(key)
        for key in sorted(_E13_ANCHOR_PARAM_KEYS)
        if anchor_params.get(key) not in (None, "")
    }
    return str(payload)


def _e14_anchor_analysis_key(anchor_params: dict[str, Any]) -> str:
    payload = {
        key: anchor_params.get(key)
        for key in sorted(_E14_ANCHOR_PARAM_KEYS)
        if anchor_params.get(key) not in (None, "")
    }
    return str(payload)


def _resolve_e12_anchor_identities(
    *,
    experiment: dict[str, Any],
    variants: list[dict[str, Any]],
    registry_experiments: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], str | None]:
    experiments = list(registry_experiments or [])
    if not experiments:
        return [], "E12 anchor resolution failed: registry experiments are unavailable."

    hinted_anchor_ids: list[str] = []
    for variant in variants:
        params = variant.get("params")
        if not isinstance(params, dict):
            continue
        anchor_hint = _safe_text(params.get("anchor_experiment_id"))
        if anchor_hint and anchor_hint not in hinted_anchor_ids:
            hinted_anchor_ids.append(anchor_hint)

    experiments_by_id = {
        _safe_text(row.get("experiment_id")): dict(row)
        for row in experiments
        if isinstance(row, dict) and _safe_text(row.get("experiment_id"))
    }
    stage5_candidates = [
        row
        for row in experiments_by_id.values()
        if _safe_text(row.get("stage")) == "Stage 5 - Confirmatory analysis"
    ]
    fallback_candidates = [
        row for row in experiments_by_id.values() if row not in stage5_candidates
    ]

    ordered_experiments: list[dict[str, Any]] = []
    seen_order_ids: set[str] = set()
    for experiment_id in hinted_anchor_ids + list(_E12_ANCHOR_PREFERRED_EXPERIMENT_IDS):
        experiment_key = str(experiment_id).strip()
        candidate = experiments_by_id.get(experiment_key)
        if candidate is None or experiment_key in seen_order_ids:
            continue
        ordered_experiments.append(candidate)
        seen_order_ids.add(experiment_key)
    candidate_pools = [stage5_candidates] if stage5_candidates else [fallback_candidates]
    for pool in candidate_pools:
        for row in pool:
            experiment_id = _safe_text(row.get("experiment_id"))
            if not experiment_id or experiment_id in seen_order_ids:
                continue
            ordered_experiments.append(row)
            seen_order_ids.add(experiment_id)

    anchors: list[dict[str, Any]] = []
    seen_analysis_keys: set[str] = set()
    for experiment_row in ordered_experiments:
        if not isinstance(experiment_row, dict):
            continue
        if not _is_executable_experiment(experiment_row):
            continue
        templates = list(experiment_row.get("variant_templates", []))
        for template in templates:
            if not isinstance(template, dict):
                continue
            if bool(template.get("supported", True)) is False:
                continue
            params = template.get("params")
            if not isinstance(params, dict):
                continue
            cv_mode = _safe_text(params.get("cv"))
            if cv_mode not in _E12_CONFIRMATORY_CV_MODES:
                continue
            subject = _safe_text(params.get("subject"))
            train_subject = _safe_text(params.get("train_subject"))
            test_subject = _safe_text(params.get("test_subject"))
            if cv_mode == "within_subject_loso_session" and not subject:
                continue
            if cv_mode == "frozen_cross_person_transfer":
                if not train_subject or not test_subject:
                    continue
                if train_subject == test_subject:
                    continue
            anchor_params: dict[str, Any] = {
                key: params.get(key)
                for key in _E12_ANCHOR_PARAM_KEYS
                if params.get(key) not in (None, "")
            }
            analysis_key = _anchor_analysis_key(anchor_params)
            if analysis_key in seen_analysis_keys:
                continue
            seen_analysis_keys.add(analysis_key)
            anchor_experiment_id = _safe_text(experiment_row.get("experiment_id"))
            anchor_template_id = _safe_text(template.get("template_id"))
            anchor_subject = _safe_text(anchor_params.get("subject"))
            anchor_train_subject = _safe_text(anchor_params.get("train_subject"))
            anchor_test_subject = _safe_text(anchor_params.get("test_subject"))
            if anchor_subject:
                analysis_label = f"{cv_mode}:{anchor_subject}"
            elif anchor_train_subject and anchor_test_subject:
                analysis_label = f"{cv_mode}:{anchor_train_subject}->{anchor_test_subject}"
            else:
                analysis_label = cv_mode
            anchors.append(
                {
                    "anchor_experiment_id": anchor_experiment_id,
                    "anchor_experiment_title": _safe_text(experiment_row.get("title")),
                    "anchor_stage": _safe_text(experiment_row.get("stage")),
                    "anchor_template_id": anchor_template_id,
                    "anchor_subject": anchor_subject,
                    "anchor_train_subject": anchor_train_subject,
                    "anchor_test_subject": anchor_test_subject,
                    "anchor_cv": _safe_text(anchor_params.get("cv")),
                    "anchor_target": _safe_text(anchor_params.get("target")),
                    "anchor_model": _safe_text(anchor_params.get("model")),
                    "anchor_feature_space": _safe_text(anchor_params.get("feature_space"))
                    or "whole_brain_masked",
                    "anchor_analysis_label": analysis_label,
                    "anchor_analysis_key": analysis_key,
                    "anchor_params": anchor_params,
                }
            )

    experiment_id = _safe_text(experiment.get("experiment_id")) or "E12"
    if not anchors:
        return (
            [],
            f"{experiment_id} anchor resolution failed: no executable confirmatory anchors were found.",
        )
    return anchors, None


def _build_e12_permutation_group_id(
    *,
    params: dict[str, Any],
    template_id: str,
    anchor_experiment_id: str,
    anchor_template_id: str,
    anchor_analysis_label: str,
) -> str:
    subject = _safe_text(params.get("subject")) or "unknown_subject"
    target = _safe_text(params.get("target")) or "unknown_target"
    model = _safe_text(params.get("model")) or "unknown_model"
    cv_mode = _safe_text(params.get("cv")) or "unknown_cv"
    task = _safe_text(params.get("filter_task")) or "all_tasks"
    modality = _safe_text(params.get("filter_modality")) or "all_modalities"
    feature_space = _safe_text(params.get("feature_space")) or "whole_brain_masked"
    anchor_experiment = _safe_text(anchor_experiment_id) or "unknown_anchor"
    anchor_template = _safe_text(anchor_template_id) or "unknown_template"
    return (
        f"E12::{template_id}::anchor={anchor_experiment}:{anchor_template}::"
        f"{anchor_analysis_label}::{subject}::{target}::{model}::{cv_mode}::{task}::{modality}::{feature_space}"
    )


def _build_e13_baseline_group_id(
    *,
    template_id: str,
    anchor_experiment_id: str,
    anchor_template_id: str,
    anchor_analysis_label: str,
) -> str:
    return (
        f"E13::{template_id}::anchor={anchor_experiment_id or 'unknown_anchor'}:"
        f"{anchor_template_id or 'unknown_template'}::{anchor_analysis_label or 'unknown_analysis'}"
    )


def _resolve_e13_anchor_identities(
    *,
    experiment: dict[str, Any],
    variants: list[dict[str, Any]],
    registry_experiments: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], str | None]:
    experiments = list(registry_experiments or [])
    if not experiments:
        return [], "E13 anchor resolution failed: registry experiments are unavailable."

    hinted_anchor_ids: list[str] = []
    for variant in variants:
        params = variant.get("params")
        if not isinstance(params, dict):
            continue
        anchor_hint = _safe_text(params.get("anchor_experiment_id"))
        if anchor_hint and anchor_hint not in hinted_anchor_ids:
            hinted_anchor_ids.append(anchor_hint)

    experiments_by_id = {
        _safe_text(row.get("experiment_id")): dict(row)
        for row in experiments
        if isinstance(row, dict) and _safe_text(row.get("experiment_id"))
    }
    stage5_candidates = [
        row
        for row in experiments_by_id.values()
        if _safe_text(row.get("stage")) == "Stage 5 - Confirmatory analysis"
    ]
    if not stage5_candidates:
        return [], "E13 anchor resolution failed: no Stage 5 confirmatory experiments were found."

    ordered_experiments: list[dict[str, Any]] = []
    seen_order_ids: set[str] = set()
    for experiment_id in hinted_anchor_ids:
        candidate = experiments_by_id.get(str(experiment_id).strip())
        if candidate is None:
            continue
        experiment_key = _safe_text(candidate.get("experiment_id"))
        if not experiment_key or experiment_key in seen_order_ids:
            continue
        ordered_experiments.append(candidate)
        seen_order_ids.add(experiment_key)
    for row in stage5_candidates:
        experiment_id = _safe_text(row.get("experiment_id"))
        if not experiment_id or experiment_id in seen_order_ids:
            continue
        ordered_experiments.append(row)
        seen_order_ids.add(experiment_id)

    anchors: list[dict[str, Any]] = []
    seen_analysis_keys: set[str] = set()
    for experiment_row in ordered_experiments:
        if not isinstance(experiment_row, dict):
            continue
        if not _is_executable_experiment(experiment_row):
            continue
        templates = list(experiment_row.get("variant_templates", []))
        for template in templates:
            if not isinstance(template, dict):
                continue
            if bool(template.get("supported", True)) is False:
                continue
            params = template.get("params")
            if not isinstance(params, dict):
                continue
            cv_mode = _safe_text(params.get("cv"))
            if cv_mode not in _E13_CONFIRMATORY_CV_MODES:
                continue
            subject = _safe_text(params.get("subject"))
            train_subject = _safe_text(params.get("train_subject"))
            test_subject = _safe_text(params.get("test_subject"))
            if cv_mode == "within_subject_loso_session" and not subject:
                continue
            if cv_mode == "frozen_cross_person_transfer":
                if not train_subject or not test_subject:
                    continue
                if train_subject == test_subject:
                    continue
            anchor_params: dict[str, Any] = {
                key: params.get(key)
                for key in _E13_ANCHOR_PARAM_KEYS
                if params.get(key) not in (None, "")
            }
            analysis_key = _e13_anchor_analysis_key(anchor_params)
            if analysis_key in seen_analysis_keys:
                continue
            seen_analysis_keys.add(analysis_key)
            anchor_experiment_id = _safe_text(experiment_row.get("experiment_id"))
            anchor_template_id = _safe_text(template.get("template_id"))
            analysis_type = (
                "within_person_loso"
                if cv_mode == "within_subject_loso_session"
                else "cross_person_transfer"
            )
            if subject:
                analysis_label = f"{cv_mode}:{subject}"
            elif train_subject and test_subject:
                analysis_label = f"{cv_mode}:{train_subject}->{test_subject}"
            else:
                analysis_label = cv_mode
            anchors.append(
                {
                    "anchor_experiment_id": anchor_experiment_id,
                    "anchor_experiment_title": _safe_text(experiment_row.get("title")),
                    "anchor_stage": _safe_text(experiment_row.get("stage")),
                    "anchor_template_id": anchor_template_id,
                    "anchor_subject": subject,
                    "anchor_train_subject": train_subject,
                    "anchor_test_subject": test_subject,
                    "anchor_cv": cv_mode,
                    "anchor_target": _safe_text(params.get("target")),
                    "anchor_feature_space": _safe_text(params.get("feature_space")),
                    "anchor_analysis_type": analysis_type,
                    "anchor_analysis_label": analysis_label,
                    "anchor_analysis_key": analysis_key,
                    "anchor_params": anchor_params,
                }
            )

    experiment_id = _safe_text(experiment.get("experiment_id")) or "E13"
    if not anchors:
        return (
            [],
            f"{experiment_id} anchor resolution failed: no executable confirmatory anchors were found.",
        )
    return anchors, None


def _expand_e13_dummy_baseline_cells(
    *,
    experiment: dict[str, Any],
    variants: list[dict[str, Any]],
    registry_experiments: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    anchor_identities, anchor_warning = _resolve_e13_anchor_identities(
        experiment=experiment,
        variants=variants,
        registry_experiments=registry_experiments,
    )
    if not anchor_identities:
        return [], [str(anchor_warning or "E13 anchor resolution failed.")]

    cells: list[dict[str, Any]] = []
    for base in variants:
        if not bool(base.get("supported", False)):
            cells.append(_copy_variant(base))
            continue
        for anchor_identity in anchor_identities:
            row = _copy_variant(base)
            params = dict(row.get("params", {}))
            for key in (
                "subject",
                "train_subject",
                "test_subject",
                "model",
                "anchor_experiment_id",
            ):
                params.pop(key, None)
            for key in _E13_ANCHOR_PARAM_KEYS:
                params.pop(key, None)

            anchor_params = dict(anchor_identity.get("anchor_params", {}))
            for key, value in anchor_params.items():
                if value not in (None, ""):
                    params[key] = value
            params["model"] = "dummy"
            row["params"] = params

            cv_mode = _safe_text(params.get("cv"))
            if cv_mode == "within_subject_loso_session" and not _safe_text(params.get("subject")):
                row["supported"] = False
                row["blocked_reason"] = (
                    "E13 anchor inheritance failed to provide subject for cv='within_subject_loso_session'."
                )
                cells.append(row)
                continue
            if cv_mode == "frozen_cross_person_transfer":
                if not _safe_text(params.get("train_subject")) or not _safe_text(
                    params.get("test_subject")
                ):
                    row["supported"] = False
                    row["blocked_reason"] = (
                        "E13 anchor inheritance failed to provide train_subject/test_subject for "
                        "cv='frozen_cross_person_transfer'."
                    )
                    cells.append(row)
                    continue

            if cv_mode == "within_subject_loso_session":
                params.pop("train_subject", None)
                params.pop("test_subject", None)
            if cv_mode == "frozen_cross_person_transfer":
                params.pop("subject", None)

            anchor_experiment_id = _safe_text(anchor_identity.get("anchor_experiment_id"))
            anchor_template_id = _safe_text(anchor_identity.get("anchor_template_id"))
            anchor_analysis_label = _safe_text(anchor_identity.get("anchor_analysis_label"))
            baseline_group_id = _build_e13_baseline_group_id(
                template_id=_safe_text(row.get("template_id")) or "template",
                anchor_experiment_id=anchor_experiment_id,
                anchor_template_id=anchor_template_id,
                anchor_analysis_label=anchor_analysis_label,
            )

            row["factor_settings"].update(
                {
                    "anchor_experiment_id": anchor_experiment_id,
                    "anchor_template_id": anchor_template_id,
                    "anchor_analysis_type": _safe_text(anchor_identity.get("anchor_analysis_type")),
                    "anchor_subject": _safe_text(anchor_identity.get("anchor_subject")) or None,
                    "anchor_train_subject": _safe_text(anchor_identity.get("anchor_train_subject"))
                    or None,
                    "anchor_test_subject": _safe_text(anchor_identity.get("anchor_test_subject"))
                    or None,
                    "baseline_group_id": baseline_group_id,
                    "special_cell_kind": "confirmatory_dummy_baseline",
                }
            )
            row["design_metadata"].update(
                {
                    "special_cell_kind": "confirmatory_dummy_baseline",
                    "anchor_experiment_id": anchor_experiment_id,
                    "anchor_variant_id": anchor_template_id,
                    "anchor_analysis_type": _safe_text(anchor_identity.get("anchor_analysis_type")),
                    "anchor_analysis_label": anchor_analysis_label,
                    "anchor_subject": _safe_text(anchor_identity.get("anchor_subject")) or None,
                    "anchor_train_subject": _safe_text(anchor_identity.get("anchor_train_subject"))
                    or None,
                    "anchor_test_subject": _safe_text(anchor_identity.get("anchor_test_subject"))
                    or None,
                    "baseline_group_id": baseline_group_id,
                }
            )
            trial_id = str(row.get("trial_id")).strip() if row.get("trial_id") else None
            if trial_id:
                row["trial_id"] = (
                    f"{trial_id}__anchor_{anchor_experiment_id}_{anchor_template_id or 'template'}"
                )
            cell_id = str(row.get("cell_id")).strip() if row.get("cell_id") else None
            suffix = f"__anchor_{anchor_experiment_id}_{anchor_template_id or 'template'}"
            row["cell_id"] = (
                f"{cell_id}{suffix}" if cell_id else f"{row.get('template_id', 'cell')}{suffix}"
            )
            cells.append(row)
    return _reindex_variants(cells), []


def _resolve_e14_anchor_identities(
    *,
    experiment: dict[str, Any],
    variants: list[dict[str, Any]],
    registry_experiments: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], str | None]:
    experiments = list(registry_experiments or [])
    if not experiments:
        return [], "E14 anchor resolution failed: registry experiments are unavailable."

    hinted_anchor_ids: list[str] = []
    for variant in variants:
        params = variant.get("params")
        if not isinstance(params, dict):
            continue
        anchor_hint = _safe_text(params.get("anchor_experiment_id"))
        if anchor_hint and anchor_hint not in hinted_anchor_ids:
            hinted_anchor_ids.append(anchor_hint)

    experiments_by_id = {
        _safe_text(row.get("experiment_id")): dict(row)
        for row in experiments
        if isinstance(row, dict) and _safe_text(row.get("experiment_id"))
    }

    ordered_experiments: list[dict[str, Any]] = []
    seen_order_ids: set[str] = set()
    for experiment_id in hinted_anchor_ids + list(_E14_ANCHOR_PREFERRED_EXPERIMENT_IDS):
        experiment_key = str(experiment_id).strip()
        candidate = experiments_by_id.get(experiment_key)
        if candidate is None or experiment_key in seen_order_ids:
            continue
        ordered_experiments.append(candidate)
        seen_order_ids.add(experiment_key)

    anchors: list[dict[str, Any]] = []
    seen_analysis_keys: set[str] = set()
    for experiment_row in ordered_experiments:
        if not isinstance(experiment_row, dict):
            continue
        anchor_experiment_id = _safe_text(experiment_row.get("experiment_id"))
        if anchor_experiment_id != "E16":
            continue
        if not _is_executable_experiment(experiment_row):
            continue
        templates = list(experiment_row.get("variant_templates", []))
        for template in templates:
            if not isinstance(template, dict):
                continue
            if bool(template.get("supported", True)) is False:
                continue
            params = template.get("params")
            if not isinstance(params, dict):
                continue
            cv_mode = _safe_text(params.get("cv"))
            if cv_mode != "within_subject_loso_session":
                continue
            subject = _safe_text(params.get("subject"))
            if not subject:
                continue
            model_name = _safe_text(params.get("model"))
            if not model_name or not model_supports_linear_interpretability(model_name):
                continue

            anchor_params: dict[str, Any] = {
                key: params.get(key)
                for key in _E14_ANCHOR_PARAM_KEYS
                if params.get(key) not in (None, "")
            }
            analysis_key = _e14_anchor_analysis_key(anchor_params)
            if analysis_key in seen_analysis_keys:
                continue
            seen_analysis_keys.add(analysis_key)
            anchor_template_id = _safe_text(template.get("template_id"))
            analysis_label = f"{cv_mode}:{subject}"
            anchors.append(
                {
                    "anchor_experiment_id": anchor_experiment_id,
                    "anchor_experiment_title": _safe_text(experiment_row.get("title")),
                    "anchor_stage": _safe_text(experiment_row.get("stage")),
                    "anchor_template_id": anchor_template_id,
                    "anchor_subject": subject,
                    "anchor_cv": cv_mode,
                    "anchor_target": _safe_text(params.get("target")),
                    "anchor_model": model_name,
                    "anchor_feature_space": _safe_text(params.get("feature_space")),
                    "anchor_analysis_type": "within_person_loso",
                    "anchor_analysis_label": analysis_label,
                    "anchor_analysis_key": analysis_key,
                    "anchor_params": anchor_params,
                }
            )

    experiment_id = _safe_text(experiment.get("experiment_id")) or "E14"
    if not anchors:
        return (
            [],
            (
                f"{experiment_id} anchor resolution failed: no eligible E16 within-subject "
                "linear-interpretability anchors were found."
            ),
        )
    return anchors, None


def _build_e14_robustness_group_id(
    *,
    template_id: str,
    anchor_experiment_id: str,
    anchor_template_id: str,
    anchor_analysis_label: str,
) -> str:
    return (
        f"E14::{template_id}::anchor={anchor_experiment_id or 'unknown_anchor'}:"
        f"{anchor_template_id or 'unknown_template'}::{anchor_analysis_label or 'unknown_analysis'}"
    )


def _expand_e14_interpretability_stability_cells(
    *,
    experiment: dict[str, Any],
    variants: list[dict[str, Any]],
    registry_experiments: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    anchor_identities, anchor_warning = _resolve_e14_anchor_identities(
        experiment=experiment,
        variants=variants,
        registry_experiments=registry_experiments,
    )
    if not anchor_identities:
        return [], [str(anchor_warning or "E14 anchor resolution failed.")]

    cells: list[dict[str, Any]] = []
    for base in variants:
        if not bool(base.get("supported", False)):
            cells.append(_copy_variant(base))
            continue
        for anchor_identity in anchor_identities:
            row = _copy_variant(base)
            params = dict(row.get("params", {}))
            for key in (
                "subject",
                "train_subject",
                "test_subject",
                "model",
                "anchor_experiment_id",
            ):
                params.pop(key, None)
            for key in _E14_ANCHOR_PARAM_KEYS:
                params.pop(key, None)
            anchor_params = dict(anchor_identity.get("anchor_params", {}))
            for key, value in anchor_params.items():
                if value not in (None, ""):
                    params[key] = value
            row["params"] = params

            anchor_experiment_id = _safe_text(anchor_identity.get("anchor_experiment_id"))
            anchor_template_id = _safe_text(anchor_identity.get("anchor_template_id"))
            anchor_analysis_label = _safe_text(anchor_identity.get("anchor_analysis_label"))
            robustness_group_id = _build_e14_robustness_group_id(
                template_id=_safe_text(row.get("template_id")) or "template",
                anchor_experiment_id=anchor_experiment_id,
                anchor_template_id=anchor_template_id,
                anchor_analysis_label=anchor_analysis_label,
            )

            # E14 is a derived robustness family that reports anchor-run interpretability
            # stability; it should not trigger a redundant second model fit.
            row["supported"] = False
            row["blocked_reason"] = (
                "E14 interpretability_stability is derived from E16 artifacts and is not "
                "executed as a separate fit."
            )
            row["factor_settings"].update(
                {
                    "special_cell_kind": "interpretability_stability",
                    "anchor_experiment_id": anchor_experiment_id,
                    "anchor_template_id": anchor_template_id,
                    "anchor_analysis_type": _safe_text(anchor_identity.get("anchor_analysis_type")),
                    "anchor_analysis_label": anchor_analysis_label,
                    "anchor_subject": _safe_text(anchor_identity.get("anchor_subject")) or None,
                    "robustness_group_id": robustness_group_id,
                }
            )
            row["design_metadata"].update(
                {
                    "special_cell_kind": "interpretability_stability",
                    "anchor_experiment_id": anchor_experiment_id,
                    "anchor_variant_id": anchor_template_id,
                    "anchor_subject": _safe_text(anchor_identity.get("anchor_subject")) or None,
                    "anchor_analysis_type": _safe_text(anchor_identity.get("anchor_analysis_type")),
                    "anchor_analysis_label": anchor_analysis_label,
                    "robustness_group_id": robustness_group_id,
                }
            )
            trial_id = str(row.get("trial_id")).strip() if row.get("trial_id") else None
            if trial_id:
                row["trial_id"] = (
                    f"{trial_id}__anchor_{anchor_experiment_id}_{anchor_template_id or 'template'}"
                )
            cell_id = str(row.get("cell_id")).strip() if row.get("cell_id") else None
            suffix = f"__anchor_{anchor_experiment_id}_{anchor_template_id or 'template'}"
            row["cell_id"] = (
                f"{cell_id}{suffix}" if cell_id else f"{row.get('template_id', 'cell')}{suffix}"
            )
            cells.append(row)
    return _reindex_variants(cells), []


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
        "feature_space",
        "roi_spec_path",
        "preprocessing_strategy",
        "dimensionality_strategy",
        "pca_n_components",
        "pca_variance_ratio",
        "methodology_policy_name",
        "class_weight_policy",
        "tuning_search_space_id",
        "tuning_search_space_version",
        "tuning_inner_cv_scheme",
        "tuning_inner_group_field",
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


def _copy_variant(variant: dict[str, Any]) -> dict[str, Any]:
    copied = dict(variant)
    copied["params"] = (
        dict(variant.get("params", {})) if isinstance(variant.get("params"), dict) else {}
    )
    copied["factor_settings"] = (
        dict(variant.get("factor_settings", {}))
        if isinstance(variant.get("factor_settings"), dict)
        else {}
    )
    copied["fixed_controls"] = (
        dict(variant.get("fixed_controls", {}))
        if isinstance(variant.get("fixed_controls"), dict)
        else {}
    )
    copied["design_metadata"] = (
        dict(variant.get("design_metadata", {}))
        if isinstance(variant.get("design_metadata"), dict)
        else {}
    )
    return copied


def _reindex_variants(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    seen_trial_ids: set[str] = set()
    for index, variant in enumerate(variants, start=1):
        row = _copy_variant(variant)
        row["variant_index"] = int(index)
        trial_id = str(row.get("trial_id")).strip() if row.get("trial_id") else None
        if trial_id:
            candidate = trial_id
            suffix = 1
            while candidate in seen_trial_ids:
                suffix += 1
                candidate = f"{trial_id}__v{suffix:03d}"
            seen_trial_ids.add(candidate)
            row["trial_id"] = candidate
        resolved.append(row)
    return resolved


def _expand_e12_permutation_cells(
    *,
    experiment: dict[str, Any],
    variants: list[dict[str, Any]],
    n_permutations: int,
    seed: int = 42,
    registry_experiments: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if int(n_permutations) <= 0:
        return (
            [],
            ["E12 requires --n-permutations > 0 to materialize permutation chunks."],
        )

    anchor_identities, anchor_warning = _resolve_e12_anchor_identities(
        experiment=experiment,
        variants=variants,
        registry_experiments=registry_experiments,
    )
    if not anchor_identities:
        return [], [str(anchor_warning or "E12 anchor resolution failed.")]

    configured_chunk_size = _optional_int(experiment.get("permutation_chunk_size"))
    if configured_chunk_size is None or configured_chunk_size <= 0:
        configured_chunk_size = 50
    for candidate in variants:
        candidate_design_metadata = (
            dict(candidate.get("design_metadata", {}))
            if isinstance(candidate.get("design_metadata"), dict)
            else {}
        )
        candidate_chunk_size = _optional_int(candidate_design_metadata.get("permutation_chunk_size"))
        if candidate_chunk_size is not None and candidate_chunk_size > 0:
            configured_chunk_size = int(candidate_chunk_size)
            break

    chunk_size = int(configured_chunk_size)
    if int(n_permutations) < chunk_size:
        chunk_size = int(n_permutations)
    chunk_count = max(1, int(math.ceil(float(n_permutations) / float(chunk_size))))
    cells: list[dict[str, Any]] = []
    for base in variants:
        if not bool(base.get("supported", False)):
            cells.append(_copy_variant(base))
            continue
        for anchor_identity in anchor_identities:
            anchor_params = dict(anchor_identity.get("anchor_params", {}))
            anchor_experiment_id = _safe_text(anchor_identity.get("anchor_experiment_id"))
            anchor_template_id = _safe_text(anchor_identity.get("anchor_template_id"))
            anchor_analysis_label = _safe_text(anchor_identity.get("anchor_analysis_label"))

            base_row = _copy_variant(base)
            base_params = dict(base_row.get("params", {}))
            for key, value in anchor_params.items():
                if value not in (None, ""):
                    base_params[key] = value
            base_row["params"] = base_params
            cv_mode = _safe_text(base_params.get("cv"))
            if cv_mode == "within_subject_loso_session" and not _safe_text(
                base_params.get("subject")
            ):
                base_row["supported"] = False
                base_row["blocked_reason"] = (
                    "E12 anchor inheritance failed to provide subject for cv='within_subject_loso_session'."
                )
                cells.append(base_row)
                continue
            if cv_mode == "frozen_cross_person_transfer":
                if not _safe_text(base_params.get("train_subject")) or not _safe_text(
                    base_params.get("test_subject")
                ):
                    base_row["supported"] = False
                    base_row["blocked_reason"] = (
                        "E12 anchor inheritance failed to provide train_subject/test_subject for "
                        "cv='frozen_cross_person_transfer'."
                    )
                    cells.append(base_row)
                    continue

            permutation_group_id = _build_e12_permutation_group_id(
                params=base_params,
                template_id=_safe_text(base_row.get("template_id")) or "template",
                anchor_experiment_id=anchor_experiment_id,
                anchor_template_id=anchor_template_id,
                anchor_analysis_label=anchor_analysis_label,
            )
            for chunk_index in range(1, chunk_count + 1):
                start = int((chunk_index - 1) * chunk_size + 1)
                end = int(min(int(n_permutations), chunk_index * chunk_size))
                size = int(max(0, end - start + 1))
                row = _copy_variant(base_row)
                group_seed_offset = int(sum(ord(char) for char in str(permutation_group_id)) % 1000)
                row["seed"] = int(seed) + int(chunk_index * 1000) + int(group_seed_offset)
                row["n_permutations_override"] = size
                row["factor_settings"]["permutation_chunk_index"] = int(chunk_index)
                row["factor_settings"]["permutation_group_id"] = str(permutation_group_id)
                row["factor_settings"]["expected_chunk_count"] = int(chunk_count)
                row["factor_settings"]["total_permutations_requested"] = int(n_permutations)
                row["factor_settings"]["anchor_experiment_id"] = str(anchor_experiment_id)
                row["factor_settings"]["anchor_template_id"] = str(anchor_template_id)
                row["factor_settings"]["anchor_analysis_label"] = str(anchor_analysis_label)
                row["design_metadata"].update(
                    {
                        "special_cell_kind": "permutation_chunk",
                        "chunk_index": int(chunk_index),
                        "chunk_start": int(start),
                        "chunk_end": int(end),
                        "chunk_size": int(size),
                        "permutation_group_id": str(permutation_group_id),
                        "expected_chunk_count": int(chunk_count),
                        "total_permutations_requested": int(n_permutations),
                        "anchor_experiment_id": str(anchor_experiment_id),
                        "anchor_template_id": str(anchor_template_id),
                        "anchor_analysis_label": str(anchor_analysis_label),
                        "anchor_identity": {
                            "anchor_experiment_id": str(anchor_experiment_id),
                            "anchor_template_id": str(anchor_template_id),
                            "anchor_subject": _safe_text(anchor_identity.get("anchor_subject")),
                            "anchor_train_subject": _safe_text(
                                anchor_identity.get("anchor_train_subject")
                            ),
                            "anchor_test_subject": _safe_text(
                                anchor_identity.get("anchor_test_subject")
                            ),
                            "anchor_cv": _safe_text(anchor_identity.get("anchor_cv")),
                            "anchor_target": _safe_text(anchor_identity.get("anchor_target")),
                            "anchor_model": _safe_text(anchor_identity.get("anchor_model")),
                            "anchor_feature_space": _safe_text(
                                anchor_identity.get("anchor_feature_space")
                            )
                            or "whole_brain_masked",
                            "anchor_analysis_label": str(anchor_analysis_label),
                        },
                    }
                )
                trial_id = str(row.get("trial_id")).strip() if row.get("trial_id") else None
                if trial_id:
                    row["trial_id"] = f"{trial_id}__perm_chunk_{chunk_index:03d}"
                cell_id = str(row.get("cell_id")).strip() if row.get("cell_id") else None
                row["cell_id"] = (
                    f"{cell_id}__perm_chunk_{chunk_index:03d}"
                    if cell_id
                    else f"{row.get('template_id', 'cell')}__perm_chunk_{chunk_index:03d}"
                )
                cells.append(row)
    return _reindex_variants(cells), []


def _resolve_omitted_sessions(
    *,
    dataset_scope: dict[str, Any],
    subject: str,
    task: str,
    filter_modality: str | None,
) -> list[str]:
    modality_map = dataset_scope.get("sessions_by_subject_task_modality", {})
    if not isinstance(modality_map, dict):
        return []
    subject_map = modality_map.get(str(subject), {})
    if not isinstance(subject_map, dict):
        return []
    task_map = subject_map.get(str(task), {})
    if not isinstance(task_map, dict):
        return []

    if filter_modality:
        values = task_map.get(str(filter_modality), [])
        if not isinstance(values, list):
            return []
        return sorted(str(value) for value in values if str(value).strip())

    merged: set[str] = set()
    for raw_sessions in task_map.values():
        if not isinstance(raw_sessions, list):
            continue
        for value in raw_sessions:
            value_text = str(value).strip()
            if value_text:
                merged.add(value_text)
    return sorted(merged)


def _expand_e23_omitted_session_cells(
    *,
    variants: list[dict[str, Any]],
    dataset_scope: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    cells: list[dict[str, Any]] = []
    missing: list[str] = []
    default_subjects = [str(value) for value in dataset_scope.get("subjects", [])]
    default_tasks = [str(value) for value in dataset_scope.get("tasks", [])]

    for base in variants:
        if not bool(base.get("supported", False)):
            cells.append(_copy_variant(base))
            continue
        params = dict(base.get("params", {}))
        subject_values = (
            [str(params.get("subject"))] if params.get("subject") else list(default_subjects)
        )
        task_values = (
            [str(params.get("filter_task"))] if params.get("filter_task") else list(default_tasks)
        )
        filter_modality = (
            str(params.get("filter_modality")).strip() if params.get("filter_modality") else None
        )
        for subject in subject_values:
            for task in task_values:
                omitted_sessions = _resolve_omitted_sessions(
                    dataset_scope=dataset_scope,
                    subject=str(subject),
                    task=str(task),
                    filter_modality=filter_modality,
                )
                if not omitted_sessions:
                    missing.append(
                        f"subject={subject}, task={task}, modality={filter_modality or 'all'}"
                    )
                    continue
                for omitted_session in omitted_sessions:
                    row = _copy_variant(base)
                    row_params = dict(row.get("params", {}))
                    row_params["subject"] = str(subject)
                    row_params["filter_task"] = str(task)
                    row_params["omitted_session"] = str(omitted_session)
                    row["params"] = row_params
                    cv_mode = str(row_params.get("cv", "")).strip()
                    if cv_mode not in _RUN_EXPERIMENT_SUPPORTED_CV_MODES:
                        row["supported"] = False
                        row["blocked_reason"] = (
                            "E23 omitted-session cell is blocked: "
                            f"cv mode '{cv_mode}' is not supported by thesisml-run-experiment."
                        )
                    row["factor_settings"]["omitted_session"] = str(omitted_session)
                    row["design_metadata"].update(
                        {
                            "special_cell_kind": "session_influence_jackknife",
                            "omitted_session": str(omitted_session),
                        }
                    )
                    trial_id = str(row.get("trial_id")).strip() if row.get("trial_id") else None
                    suffix = f"__{subject}__{task}__omit_{omitted_session}"
                    if trial_id:
                        row["trial_id"] = f"{trial_id}{suffix}"
                    cell_id = str(row.get("cell_id")).strip() if row.get("cell_id") else None
                    row["cell_id"] = (
                        f"{cell_id}{suffix}"
                        if cell_id
                        else f"{row.get('template_id', 'cell')}{suffix}"
                    )
                    cells.append(row)

    if not cells:
        reason = (
            "E23 omitted-session units could not be derived safely from dataset scope: "
            + "; ".join(sorted(set(missing)))
        )
        return [], [reason]
    return _reindex_variants(cells), []


def _expand_e15_subset_sensitivity_cells(
    *,
    variants: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    cells: list[dict[str, Any]] = []
    warnings: list[str] = []
    full_control_keys: set[tuple[str, str, str]] = set()

    for base in variants:
        row = _copy_variant(base)
        if not bool(row.get("supported", False)):
            cells.append(row)
            continue

        params = dict(row.get("params", {}))
        cv_mode = _safe_text(params.get("cv"))
        subject = _safe_text(params.get("subject"))
        filter_task = _safe_text(params.get("filter_task"))
        filter_modality = _safe_text(params.get("filter_modality"))
        template_id = _safe_text(row.get("template_id")) or "template"

        if cv_mode not in _RUN_EXPERIMENT_SUPPORTED_CV_MODES:
            row["supported"] = False
            row["blocked_reason"] = (
                "E15 subset sensitivity cell is blocked: "
                f"cv mode '{cv_mode}' is not supported by thesisml-run-experiment."
            )
            cells.append(row)
            continue
        if cv_mode == "within_subject_loso_session" and not subject:
            row["supported"] = False
            row["blocked_reason"] = (
                "E15 subset sensitivity requires subject for cv='within_subject_loso_session'."
            )
            cells.append(row)
            continue
        if not filter_task:
            row["supported"] = False
            row["blocked_reason"] = "E15 subset sensitivity requires non-empty filter_task."
            cells.append(row)
            continue
        missing_locked_core = _missing_locked_core_keys(
            params,
            required_keys=_E15_LOCKED_CORE_REQUIRED_KEYS,
        )
        if missing_locked_core:
            row["supported"] = False
            row["blocked_reason"] = (
                "E15 subset sensitivity requires explicit locked-core parameters and must not "
                "fall back to model defaults. Missing keys: "
                + ", ".join(sorted(missing_locked_core))
            )
            cells.append(row)
            continue

        group_id = (
            f"E15::{template_id}::{subject or 'unknown_subject'}::"
            f"task={filter_task}::modality={filter_modality or 'all'}"
        )
        restricted_suffix = (
            f"__subject_{subject or 'unknown'}"
            f"__task_{filter_task}"
            f"__modality_{filter_modality or 'all'}"
        )
        row["factor_settings"].update(
            {
                "special_cell_kind": "subset_sensitivity",
                "subset_arm": "task_restricted",
                "subset_sensitivity_group_id": group_id,
            }
        )
        row["design_metadata"].update(
            {
                "special_cell_kind": "subset_sensitivity",
                "subset_arm": "task_restricted",
                "subset_sensitivity_group_id": group_id,
            }
        )
        trial_id = str(row.get("trial_id")).strip() if row.get("trial_id") else None
        if trial_id:
            row["trial_id"] = f"{trial_id}{restricted_suffix}"
        cell_id = str(row.get("cell_id")).strip() if row.get("cell_id") else None
        row["cell_id"] = (
            f"{cell_id}{restricted_suffix}" if cell_id else f"{template_id}{restricted_suffix}"
        )
        cells.append(row)

        full_control_key = (template_id, subject, filter_modality)
        if full_control_key in full_control_keys:
            continue
        full_control_keys.add(full_control_key)

        control = _copy_variant(base)
        control_params = dict(control.get("params", {}))
        control_params["subject"] = subject
        control_params.pop("filter_task", None)
        missing_control_locked_core = _missing_locked_core_keys(
            control_params,
            required_keys=_E15_LOCKED_CORE_REQUIRED_KEYS,
        )
        if missing_control_locked_core:
            control["supported"] = False
            control["blocked_reason"] = (
                "E15 full-control cell is missing explicit locked-core parameters and cannot be "
                "materialized safely. Missing keys: "
                + ", ".join(sorted(missing_control_locked_core))
            )
            cells.append(control)
            continue
        control["params"] = control_params
        control_group_id = (
            f"E15::{template_id}::{subject or 'unknown_subject'}::"
            f"task=full::modality={filter_modality or 'all'}"
        )
        control_suffix = (
            f"__subject_{subject or 'unknown'}__task_full__modality_{filter_modality or 'all'}"
        )
        control["factor_settings"].update(
            {
                "special_cell_kind": "subset_sensitivity",
                "subset_arm": "full_control",
                "subset_sensitivity_group_id": control_group_id,
            }
        )
        control["design_metadata"].update(
            {
                "special_cell_kind": "subset_sensitivity",
                "subset_arm": "full_control",
                "subset_sensitivity_group_id": control_group_id,
            }
        )
        trial_id = str(control.get("trial_id")).strip() if control.get("trial_id") else None
        if trial_id:
            control["trial_id"] = f"{trial_id}{control_suffix}"
        cell_id = str(control.get("cell_id")).strip() if control.get("cell_id") else None
        control["cell_id"] = (
            f"{cell_id}{control_suffix}" if cell_id else f"{template_id}{control_suffix}"
        )
        cells.append(control)

    if not cells:
        warnings.append("E15 subset sensitivity did not produce any materialized cells.")
    return _reindex_variants(cells), warnings


def _expand_e24_rerun_cells(
    *,
    variants: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    cells: list[dict[str, Any]] = []
    for base in variants:
        row = _copy_variant(base)
        row["sequential_only"] = True
        row["design_metadata"].update(
            {
                "special_cell_kind": "reproducibility_rerun",
                "sequential_only": True,
            }
        )
        cells.append(row)
    return _reindex_variants(cells), []


def materialize_experiment_cells(
    *,
    experiment: dict[str, Any],
    variants: list[dict[str, Any]],
    dataset_scope: dict[str, Any],
    n_permutations: int,
    seed: int = 42,
    registry_experiments: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    experiment_id = str(experiment.get("experiment_id"))
    if experiment_id == "E12":
        return _expand_e12_permutation_cells(
            experiment=experiment,
            variants=variants,
            n_permutations=int(n_permutations),
            seed=int(seed),
            registry_experiments=registry_experiments,
        )
    if experiment_id == "E13":
        return _expand_e13_dummy_baseline_cells(
            experiment=experiment,
            variants=variants,
            registry_experiments=registry_experiments,
        )
    if experiment_id == "E14":
        return _expand_e14_interpretability_stability_cells(
            experiment=experiment,
            variants=variants,
            registry_experiments=registry_experiments,
        )
    if experiment_id == "E15":
        return _expand_e15_subset_sensitivity_cells(
            variants=variants,
        )
    if experiment_id == "E23":
        return _expand_e23_omitted_session_cells(
            variants=variants,
            dataset_scope=dataset_scope,
        )
    if experiment_id == "E24":
        return _expand_e24_rerun_cells(variants=variants)
    return _reindex_variants([_copy_variant(variant) for variant in variants]), []


__all__ = [
    "expand_experiment_variants",
    "expand_template_variants",
    "materialize_experiment_cells",
    "variant_label",
]
