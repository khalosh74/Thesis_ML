from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_EXPERIMENT_REPORT,
    compute_config_hash,
    register_artifact,
)
from Thesis_ML.config.methodology import (
    ClassWeightPolicy,
    MethodologyPolicy,
    MethodologyPolicyName,
)
from Thesis_ML.config.metric_policy import extract_metric_value, validate_metric_name
from Thesis_ML.orchestration.reporting import build_dataset_subset_label
from Thesis_ML.orchestration.variant_expansion import variant_label


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except Exception:
        return None


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    value_text = str(value).strip()
    return value_text or None


def _resolve_methodology_params(params: dict[str, Any]) -> dict[str, Any]:
    raw_policy_name = _optional_str(params.get("methodology_policy_name"))
    raw_class_weight = _optional_str(params.get("class_weight_policy"))
    raw_tuning_space_id = _optional_str(params.get("tuning_search_space_id"))
    raw_tuning_space_version = _optional_str(params.get("tuning_search_space_version"))
    raw_tuning_inner_cv = _optional_str(params.get("tuning_inner_cv_scheme"))
    raw_tuning_inner_group = _optional_str(params.get("tuning_inner_group_field"))

    policy_name = raw_policy_name or MethodologyPolicyName.FIXED_BASELINES_ONLY.value
    class_weight_policy = raw_class_weight or ClassWeightPolicy.NONE.value
    tuning_enabled = policy_name == MethodologyPolicyName.GROUPED_NESTED_TUNING.value

    if tuning_enabled:
        required = {
            "tuning_search_space_id": raw_tuning_space_id,
            "tuning_search_space_version": raw_tuning_space_version,
            "tuning_inner_cv_scheme": raw_tuning_inner_cv,
            "tuning_inner_group_field": raw_tuning_inner_group,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "grouped_nested_tuning requires explicit template params: "
                + ", ".join(sorted(missing))
                + "."
            )

    try:
        validated = MethodologyPolicy(
            policy_name=MethodologyPolicyName(policy_name),
            class_weight_policy=ClassWeightPolicy(class_weight_policy),
            tuning_enabled=tuning_enabled,
            inner_cv_scheme=raw_tuning_inner_cv,
            inner_group_field=raw_tuning_inner_group,
            tuning_search_space_id=raw_tuning_space_id,
            tuning_search_space_version=raw_tuning_space_version,
        )
    except Exception as exc:
        raise ValueError(f"Invalid methodology parameters in variant params: {exc}") from exc

    return {
        "methodology_policy_name": validated.policy_name.value,
        "class_weight_policy": validated.class_weight_policy.value,
        "tuning_enabled": bool(validated.tuning_enabled),
        "tuning_search_space_id": validated.tuning_search_space_id,
        "tuning_search_space_version": validated.tuning_search_space_version,
        "tuning_inner_cv_scheme": validated.inner_cv_scheme,
        "tuning_inner_group_field": validated.inner_group_field,
    }


def build_command(
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    reports_root: Path,
    run_id: str,
    seed: int,
    n_permutations: int,
    params: dict[str, Any],
    start_section: str | None = None,
    end_section: str | None = None,
    base_artifact_id: str | None = None,
    reuse_policy: str | None = None,
    methodology_policy_name: str | None = None,
    class_weight_policy: str | None = None,
    tuning_search_space_id: str | None = None,
    tuning_search_space_version: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "Thesis_ML.experiments.run_experiment",
        "--index-csv",
        str(index_csv),
        "--data-root",
        str(data_root),
        "--cache-dir",
        str(cache_dir),
        "--target",
        str(params["target"]),
        "--model",
        str(params["model"]),
        "--cv",
        str(params["cv"]),
        "--seed",
        str(seed),
        "--run-id",
        run_id,
        "--reports-root",
        str(reports_root),
    ]
    if n_permutations > 0:
        command.extend(["--n-permutations", str(n_permutations)])
    if params.get("subject"):
        command.extend(["--subject", str(params["subject"])])
    if params.get("train_subject"):
        command.extend(["--train-subject", str(params["train_subject"])])
    if params.get("test_subject"):
        command.extend(["--test-subject", str(params["test_subject"])])
    if params.get("filter_task"):
        command.extend(["--filter-task", str(params["filter_task"])])
    if params.get("filter_modality"):
        command.extend(["--filter-modality", str(params["filter_modality"])])
    if params.get("feature_space"):
        command.extend(["--feature-space", str(params["feature_space"])])
    if params.get("roi_spec_path"):
        command.extend(["--roi-spec-path", str(params["roi_spec_path"])])
    if params.get("preprocessing_strategy"):
        command.extend(["--preprocessing-strategy", str(params["preprocessing_strategy"])])
    if params.get("dimensionality_strategy"):
        command.extend(["--dimensionality-strategy", str(params["dimensionality_strategy"])])
    if params.get("pca_n_components") is not None:
        command.extend(["--pca-n-components", str(params["pca_n_components"])])
    if params.get("pca_variance_ratio") is not None:
        command.extend(["--pca-variance-ratio", str(params["pca_variance_ratio"])])
    if methodology_policy_name:
        command.extend(["--methodology-policy", str(methodology_policy_name)])
    if class_weight_policy:
        command.extend(["--class-weight-policy", str(class_weight_policy)])
    if tuning_search_space_id:
        command.extend(["--tuning-search-space-id", str(tuning_search_space_id)])
    if tuning_search_space_version:
        command.extend(["--tuning-search-space-version", str(tuning_search_space_version)])
    if start_section:
        command.extend(["--start-section", str(start_section)])
    if end_section:
        command.extend(["--end-section", str(end_section)])
    if base_artifact_id:
        command.extend(["--base-artifact-id", str(base_artifact_id)])
    if reuse_policy:
        command.extend(["--reuse-policy", str(reuse_policy)])
    return command


def command_to_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def execute_variant(
    *,
    experiment: dict[str, Any],
    variant: dict[str, Any],
    campaign_id: str,
    experiment_root: Path,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    seed: int,
    n_permutations: int,
    dry_run: bool,
    run_experiment_fn: Callable[..., dict[str, Any]],
    artifact_registry_path: Path | None = None,
    code_ref: str | None = None,
) -> dict[str, Any]:
    experiment_id = str(experiment["experiment_id"])
    template_id = str(variant["template_id"])
    variant_index = int(variant["variant_index"])
    repeat_raw = variant.get("repeat_id")
    seed_raw = variant.get("seed")
    study_id = str(variant.get("study_id")).strip() if variant.get("study_id") else None
    trial_id = str(variant.get("trial_id")).strip() if variant.get("trial_id") else None
    cell_id = str(variant.get("cell_id")).strip() if variant.get("cell_id") else None
    repeat_id = _optional_int(repeat_raw)
    trial_seed = _optional_int(seed_raw)
    factor_settings = (
        dict(variant.get("factor_settings", {}))
        if isinstance(variant.get("factor_settings"), dict)
        else {}
    )
    fixed_controls = (
        dict(variant.get("fixed_controls", {}))
        if isinstance(variant.get("fixed_controls"), dict)
        else {}
    )
    design_metadata = (
        dict(variant.get("design_metadata", {}))
        if isinstance(variant.get("design_metadata"), dict)
        else {}
    )
    variant_id = trial_id or f"{template_id}__{variant_index:03d}"
    params = dict(variant.get("params", {}))
    params_snapshot = dict(params)
    supported = bool(variant.get("supported", False))
    blocked_reason = variant.get("blocked_reason")
    start_section = (
        str(variant.get("start_section")).strip() if variant.get("start_section") else None
    )
    end_section = str(variant.get("end_section")).strip() if variant.get("end_section") else None
    base_artifact_id = (
        str(variant.get("base_artifact_id")).strip() if variant.get("base_artifact_id") else None
    )
    reuse_policy = str(variant.get("reuse_policy")).strip() if variant.get("reuse_policy") else None
    search_space_id = (
        str(variant.get("search_space_id")).strip() if variant.get("search_space_id") else None
    )
    search_assignment = variant.get("search_assignment")

    run_token = variant_id.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")
    run_id = f"ds_{experiment_id}_{run_token}_{campaign_id}"
    reports_root = experiment_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    manifests_dir = experiment_root / "run_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    now_start = _utc_timestamp()
    command: list[str] | None = None
    command_text: str | None = None
    status = "planned"
    error: str | None = None
    result: dict[str, Any] | None = None

    effective_seed = int(trial_seed) if trial_seed is not None else int(seed)

    if not supported:
        status = "blocked"
    else:
        try:
            methodology_params = _resolve_methodology_params(params)
        except ValueError as exc:
            status = "blocked"
            blocked_reason = str(exc)
        else:
            command = build_command(
                index_csv=index_csv,
                data_root=data_root,
                cache_dir=cache_dir,
                reports_root=reports_root,
                run_id=run_id,
                seed=effective_seed,
                n_permutations=n_permutations,
                params=params,
                start_section=start_section,
                end_section=end_section,
                base_artifact_id=base_artifact_id,
                reuse_policy=reuse_policy,
                methodology_policy_name=methodology_params["methodology_policy_name"],
                class_weight_policy=methodology_params["class_weight_policy"],
                tuning_search_space_id=methodology_params["tuning_search_space_id"],
                tuning_search_space_version=methodology_params["tuning_search_space_version"],
            )
            command_text = command_to_text(command)
            if dry_run:
                status = "dry_run"
            else:
                try:
                    result = run_experiment_fn(
                        index_csv=index_csv,
                        data_root=data_root,
                        cache_dir=cache_dir,
                        target=str(params["target"]),
                        model=str(params["model"]),
                        cv=str(params["cv"]),
                        subject=(str(params["subject"]) if params.get("subject") else None),
                        train_subject=(
                            str(params["train_subject"]) if params.get("train_subject") else None
                        ),
                        test_subject=(
                            str(params["test_subject"]) if params.get("test_subject") else None
                        ),
                        seed=effective_seed,
                        filter_task=(
                            str(params["filter_task"]) if params.get("filter_task") else None
                        ),
                        filter_modality=(
                            str(params["filter_modality"])
                            if params.get("filter_modality")
                            else None
                        ),
                        feature_space=(
                            str(params["feature_space"])
                            if params.get("feature_space")
                            else "whole_brain_masked"
                        ),
                        roi_spec_path=(
                            str(params["roi_spec_path"]) if params.get("roi_spec_path") else None
                        ),
                        preprocessing_strategy=(
                            str(params["preprocessing_strategy"])
                            if params.get("preprocessing_strategy")
                            else None
                        ),
                        dimensionality_strategy=(
                            str(params["dimensionality_strategy"])
                            if params.get("dimensionality_strategy")
                            else "none"
                        ),
                        pca_n_components=(
                            int(params["pca_n_components"])
                            if params.get("pca_n_components") is not None
                            else None
                        ),
                        pca_variance_ratio=(
                            float(params["pca_variance_ratio"])
                            if params.get("pca_variance_ratio") is not None
                            else None
                        ),
                        n_permutations=n_permutations,
                        methodology_policy_name=methodology_params["methodology_policy_name"],
                        class_weight_policy=methodology_params["class_weight_policy"],
                        tuning_enabled=bool(methodology_params["tuning_enabled"]),
                        tuning_search_space_id=methodology_params["tuning_search_space_id"],
                        tuning_search_space_version=methodology_params[
                            "tuning_search_space_version"
                        ],
                        tuning_inner_cv_scheme=methodology_params["tuning_inner_cv_scheme"],
                        tuning_inner_group_field=methodology_params["tuning_inner_group_field"],
                        run_id=run_id,
                        reports_root=reports_root,
                        start_section=start_section,
                        end_section=end_section,
                        base_artifact_id=base_artifact_id,
                        reuse_policy=reuse_policy,
                    )
                    status = "completed"
                except Exception as exc:
                    status = "failed"
                    error = str(exc)

    now_end = _utc_timestamp()
    metrics = dict(result.get("metrics", {})) if result else {}
    raw_primary_metric_name = experiment.get("primary_metric")
    if raw_primary_metric_name is None or not str(raw_primary_metric_name).strip():
        raise ValueError(f"Experiment '{experiment_id}' is missing required primary_metric.")
    primary_metric_name = validate_metric_name(str(raw_primary_metric_name))
    primary_metric_value = extract_metric_value(
        metrics,
        primary_metric_name,
        require=False,
        payload_label=f"decision-support run '{run_id}' metrics",
    )

    manifest_payload = {
        "campaign_id": campaign_id,
        "experiment_id": experiment_id,
        "title": experiment.get("title"),
        "stage": experiment.get("stage"),
        "decision_id": experiment.get("decision_id"),
        "template_id": template_id,
        "variant_id": variant_id,
        "variant_label": variant_label(params),
        "status": status,
        "started_at": now_start,
        "finished_at": now_end,
        "command": command_text,
        "config_used": {
            "index_csv": str(index_csv.resolve()),
            "data_root": str(data_root.resolve()),
            "cache_dir": str(cache_dir.resolve()),
            "seed": int(effective_seed),
            "n_permutations": int(n_permutations),
            "start_section": start_section,
            "end_section": end_section,
            "base_artifact_id": base_artifact_id,
            "reuse_policy": reuse_policy,
            "search_space_id": search_space_id,
            "search_assignment": search_assignment,
            "params": params,
            "params_snapshot": params_snapshot,
            "study_id": study_id,
            "trial_id": trial_id,
            "cell_id": cell_id,
            "repeat_id": repeat_id,
            "factor_settings": factor_settings,
            "fixed_controls": fixed_controls,
            "design_metadata": design_metadata,
        },
        "dataset_subset": build_dataset_subset_label(params),
        "split_logic": params.get("cv"),
        "model": params.get("model"),
        "target_definition": params.get("target"),
        "feature_space": params.get("feature_space"),
        "roi_spec_path": params.get("roi_spec_path"),
        "preprocessing_strategy": params.get("preprocessing_strategy"),
        "dimensionality_strategy": params.get("dimensionality_strategy"),
        "pca_n_components": params.get("pca_n_components"),
        "pca_variance_ratio": params.get("pca_variance_ratio"),
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": primary_metric_value,
        "secondary_metrics": {
            "balanced_accuracy": _safe_float(metrics.get("balanced_accuracy")),
            "macro_f1": _safe_float(metrics.get("macro_f1")),
            "accuracy": _safe_float(metrics.get("accuracy")),
        },
        "artifacts": {
            "report_dir": result.get("report_dir") if result else None,
            "config_path": result.get("config_path") if result else None,
            "metrics_path": result.get("metrics_path") if result else None,
            "fold_metrics_path": result.get("fold_metrics_path") if result else None,
            "fold_splits_path": result.get("fold_splits_path") if result else None,
            "predictions_path": result.get("predictions_path") if result else None,
            "spatial_compatibility_report_path": (
                result.get("spatial_compatibility_report_path") if result else None
            ),
        },
        "warnings": [blocked_reason] if blocked_reason else [],
        "error": error,
        "design": {
            "study_id": study_id,
            "trial_id": trial_id,
            "cell_id": cell_id,
            "repeat_id": repeat_id,
            "factor_settings": factor_settings,
            "fixed_controls": fixed_controls,
            "design_metadata": design_metadata,
        },
    }

    manifest_path = manifests_dir / f"{variant_id}.json"
    manifest_path.write_text(f"{json.dumps(manifest_payload, indent=2)}\n", encoding="utf-8")

    orchestrator_artifact_id: str | None = None
    if artifact_registry_path is not None:
        run_artifact_ids: list[str] = []
        if isinstance(result, dict):
            artifact_ids_payload = result.get("artifact_ids")
            if isinstance(artifact_ids_payload, dict):
                run_artifact_ids = [
                    str(value) for value in artifact_ids_payload.values() if str(value).strip()
                ]
        orchestrator_artifact = register_artifact(
            registry_path=artifact_registry_path,
            artifact_type=ARTIFACT_TYPE_EXPERIMENT_REPORT,
            run_id=run_id,
            upstream_artifact_ids=run_artifact_ids,
            config_hash=compute_config_hash(
                {
                    "campaign_id": campaign_id,
                    "experiment_id": experiment_id,
                    "template_id": template_id,
                    "variant_id": variant_id,
                    "params": params,
                }
            ),
            code_ref=code_ref,
            path=manifest_path,
            status=status,
        )
        orchestrator_artifact_id = orchestrator_artifact.artifact_id

    record = {
        "experiment_id": experiment_id,
        "title": str(experiment.get("title", "")),
        "stage": str(experiment.get("stage", "")),
        "decision_id": str(experiment.get("decision_id", "")),
        "template_id": template_id,
        "variant_id": variant_id,
        "study_id": study_id,
        "trial_id": trial_id or variant_id,
        "cell_id": cell_id,
        "repeat_id": repeat_id,
        "variant_label": variant_label(params),
        "status": status,
        "target": params.get("target"),
        "cv": params.get("cv"),
        "model": params.get("model"),
        "subject": params.get("subject"),
        "train_subject": params.get("train_subject"),
        "test_subject": params.get("test_subject"),
        "filter_task": params.get("filter_task"),
        "filter_modality": params.get("filter_modality"),
        "feature_space": params.get("feature_space"),
        "roi_spec_path": params.get("roi_spec_path"),
        "preprocessing_strategy": params.get("preprocessing_strategy"),
        "dimensionality_strategy": params.get("dimensionality_strategy"),
        "pca_n_components": params.get("pca_n_components"),
        "pca_variance_ratio": params.get("pca_variance_ratio"),
        "start_section": start_section,
        "end_section": end_section,
        "base_artifact_id": base_artifact_id,
        "reuse_policy": reuse_policy,
        "search_space_id": search_space_id,
        "search_assignment": (
            json.dumps(search_assignment, sort_keys=True)
            if isinstance(search_assignment, dict)
            else (str(search_assignment) if search_assignment is not None else None)
        ),
        "factor_settings": factor_settings,
        "fixed_controls": fixed_controls,
        "design_metadata": design_metadata,
        "resolved_params": params,
        "params_snapshot": params_snapshot,
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": primary_metric_value,
        "balanced_accuracy": _safe_float(metrics.get("balanced_accuracy")),
        "macro_f1": _safe_float(metrics.get("macro_f1")),
        "accuracy": _safe_float(metrics.get("accuracy")),
        "run_id": run_id,
        "seed": int(effective_seed),
        "report_dir": result.get("report_dir") if result else None,
        "config_path": result.get("config_path") if result else None,
        "metrics_path": result.get("metrics_path") if result else None,
        "manifest_path": str(manifest_path.resolve()),
        "orchestrator_artifact_id": orchestrator_artifact_id,
        "blocked_reason": blocked_reason,
        "error": error,
        "command": command_text,
        "started_at": now_start,
        "finished_at": now_end,
        "n_folds": _safe_float(metrics.get("n_folds")),
        "notes": str(experiment.get("notes", "")),
    }
    return record


__all__ = ["build_command", "command_to_text", "execute_variant"]
