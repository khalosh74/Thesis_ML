from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.schema_versions import THESIS_PROTOCOL_SCHEMA_VERSION
from Thesis_ML.experiments.model_catalog import get_model_cost_entry, projected_runtime_seconds

_DEFAULT_PROTOCOL_PATH = Path("configs") / "protocols" / "thesis_confirmatory_v1.json"


class FrozenConfirmatoryBuildError(RuntimeError):
    """Raised when frozen confirmatory registry generation cannot proceed safely."""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a frozen confirmatory decision-support registry from reviewed preflight outputs."
        )
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--selection-bundle", type=Path, required=True)
    parser.add_argument("--scope-config", type=Path, required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=None,
        help="Optional explicit dataset index for confirmatory coverage validation.",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise FrozenConfirmatoryBuildError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FrozenConfirmatoryBuildError(f"{label} must be a JSON object: {path}")
    return payload


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _resolve_bundle_path(path: Path, *, campaign_root: Path) -> Path:
    if path.is_absolute():
        return path
    first = (campaign_root / path).resolve()
    if first.exists():
        return first
    return path.resolve()


def _resolve_scope_payload(scope_payload: dict[str, Any]) -> dict[str, Any]:
    required_keys = (
        "scope_id",
        "main_tasks",
        "main_modality",
        "main_target",
        "feature_space",
        "within_subjects",
        "transfer_pairs",
    )
    missing = [key for key in required_keys if key not in scope_payload]
    if missing:
        raise FrozenConfirmatoryBuildError(
            "Confirmatory scope is missing required keys: " + ", ".join(sorted(missing))
        )

    main_tasks = [str(value) for value in list(scope_payload.get("main_tasks") or [])]
    within_subjects = [str(value) for value in list(scope_payload.get("within_subjects") or [])]
    transfer_pairs_raw = list(scope_payload.get("transfer_pairs") or [])
    transfer_pairs: list[dict[str, str]] = []
    for row in transfer_pairs_raw:
        if not isinstance(row, dict):
            continue
        train_subject = _safe_text(row.get("train_subject"))
        test_subject = _safe_text(row.get("test_subject"))
        if not train_subject or not test_subject:
            continue
        transfer_pairs.append(
            {
                "train_subject": train_subject,
                "test_subject": test_subject,
            }
        )

    if not main_tasks:
        raise FrozenConfirmatoryBuildError("Confirmatory scope main_tasks must be non-empty.")
    if not within_subjects:
        raise FrozenConfirmatoryBuildError("Confirmatory scope within_subjects must be non-empty.")
    if not transfer_pairs:
        raise FrozenConfirmatoryBuildError("Confirmatory scope transfer_pairs must be non-empty.")
    feature_space = _safe_text(scope_payload.get("feature_space"))
    if not feature_space:
        raise FrozenConfirmatoryBuildError("Confirmatory scope feature_space must be non-empty.")

    return {
        "scope_id": _safe_text(scope_payload.get("scope_id")),
        "main_tasks": main_tasks,
        "main_modality": _safe_text(scope_payload.get("main_modality")),
        "main_target": _safe_text(scope_payload.get("main_target")),
        "feature_space": feature_space,
        "within_subjects": within_subjects,
        "transfer_pairs": transfer_pairs,
        "notes": dict(scope_payload.get("notes", {}))
        if isinstance(scope_payload.get("notes"), dict)
        else {},
    }


def _required_selected(bundle_payload: dict[str, Any]) -> dict[str, Any]:
    selected_payload = bundle_payload.get("selected")
    if not isinstance(selected_payload, dict):
        raise FrozenConfirmatoryBuildError("Selection bundle is missing object field: selected")

    required = (
        "target",
        "cv_within_subject",
        "cv_transfer",
        "model",
        "class_weight_policy",
        "methodology_policy_name",
        "dimensionality_strategy",
        "preprocessing_strategy",
    )
    missing = [key for key in required if selected_payload.get(key) in (None, "")]
    if missing:
        raise FrozenConfirmatoryBuildError(
            "Selection bundle selected payload is missing required keys: "
            + ", ".join(sorted(missing))
        )
    return dict(selected_payload)


def _extract_index_csv_from_run_config(config_path: Path) -> Path | None:
    if not config_path.exists() or not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    for container in (payload, payload.get("config_used"), payload.get("params")):
        if not isinstance(container, dict):
            continue
        value = container.get("index_csv")
        if isinstance(value, str) and value.strip():
            return Path(value)
    return None


def _infer_index_csv_from_bundle(
    *, campaign_root: Path, bundle_payload: dict[str, Any]
) -> Path | None:
    review_sources = bundle_payload.get("review_sources")
    if not isinstance(review_sources, dict):
        return None
    for path_text in review_sources.values():
        if not isinstance(path_text, str) or not path_text.strip():
            continue
        review_path = (campaign_root / path_text).resolve()
        if not review_path.exists() or not review_path.is_file():
            continue
        try:
            review_payload = json.loads(review_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(review_payload, dict):
            continue
        completed_runs = review_payload.get("completed_runs")
        if not isinstance(completed_runs, list):
            continue
        for row in completed_runs:
            if not isinstance(row, dict):
                continue
            config_path_text = row.get("config_path")
            if not isinstance(config_path_text, str) or not config_path_text.strip():
                continue
            config_path = Path(config_path_text)
            if not config_path.is_absolute():
                config_path = (campaign_root / config_path).resolve()
            inferred = _extract_index_csv_from_run_config(config_path)
            if inferred is None:
                continue
            if not inferred.is_absolute():
                inferred = (campaign_root / inferred).resolve()
            if inferred.exists():
                return inferred
    return None


def _validate_scope_coverage(
    *,
    index_csv: Path,
    scope: dict[str, Any],
) -> dict[str, Any]:
    if not index_csv.exists() or not index_csv.is_file():
        raise FrozenConfirmatoryBuildError(f"Coverage index_csv not found: {index_csv}")
    frame = pd.read_csv(index_csv)
    required_columns = {"subject", "task", "modality", str(scope["main_target"])}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise FrozenConfirmatoryBuildError(
            "Coverage validation index is missing required columns: " + ", ".join(missing_columns)
        )

    subjects_present = sorted(frame["subject"].astype(str).unique().tolist())
    tasks_present = sorted(frame["task"].astype(str).unique().tolist())
    modalities_present = sorted(frame["modality"].astype(str).unique().tolist())

    required_subjects = [str(value) for value in scope["within_subjects"]]
    required_tasks = [str(value) for value in scope["main_tasks"]]
    required_modality = str(scope["main_modality"])

    missing_subjects = [subject for subject in required_subjects if subject not in subjects_present]
    missing_tasks = [task for task in required_tasks if task not in tasks_present]
    modality_available = required_modality in modalities_present
    if missing_subjects or missing_tasks or not modality_available:
        messages: list[str] = []
        if missing_subjects:
            messages.append("missing subjects: " + ", ".join(sorted(missing_subjects)))
        if missing_tasks:
            messages.append("missing tasks: " + ", ".join(sorted(missing_tasks)))
        if not modality_available:
            messages.append(f"missing modality: {required_modality}")
        raise FrozenConfirmatoryBuildError(
            "Coverage validation failed for confirmatory scope: " + "; ".join(messages)
        )

    scoped = frame[
        frame["task"].astype(str).isin(set(required_tasks))
        & (frame["modality"].astype(str) == required_modality)
    ].copy()
    if scoped.empty:
        raise FrozenConfirmatoryBuildError(
            "Coverage validation failed: no rows match canonical task+modality scope."
        )

    within_missing: list[str] = []
    for subject in required_subjects:
        subset = scoped[scoped["subject"].astype(str) == subject]
        if subset.empty:
            within_missing.append(subject)
    if within_missing:
        raise FrozenConfirmatoryBuildError(
            "Coverage validation failed: missing within-subject scope rows for "
            + ", ".join(sorted(within_missing))
        )

    missing_pairs: list[str] = []
    for pair in scope["transfer_pairs"]:
        train_subject = str(pair["train_subject"])
        test_subject = str(pair["test_subject"])
        train_subset = scoped[scoped["subject"].astype(str) == train_subject]
        test_subset = scoped[scoped["subject"].astype(str) == test_subject]
        if train_subset.empty or test_subset.empty:
            missing_pairs.append(f"{train_subject}->{test_subject}")
    if missing_pairs:
        raise FrozenConfirmatoryBuildError(
            "Coverage validation failed: cannot form transfer coverage for "
            + ", ".join(sorted(missing_pairs))
        )

    return {
        "status": "passed",
        "index_csv": str(index_csv.resolve()),
        "subjects_present": subjects_present,
        "tasks_present": tasks_present,
        "modalities_present": modalities_present,
        "required_subjects": required_subjects,
        "required_tasks": required_tasks,
        "required_modality": required_modality,
        "within_cell_count": int(len(required_subjects)),
        "transfer_cell_count": int(len(scope["transfer_pairs"])),
    }


def _build_protocol_context(
    *,
    selected: dict[str, Any],
    protocol_payload: dict[str, Any],
    cv_mode: str,
    claim_ids: list[str],
    suite_id: str,
    target_name: str,
) -> dict[str, Any]:
    model_name = str(selected["model"])
    methodology_policy_name = str(selected["methodology_policy_name"])
    class_weight_policy = str(selected["class_weight_policy"])
    tuning_enabled = methodology_policy_name == "grouped_nested_tuning"
    if tuning_enabled:
        missing = [
            key
            for key in (
                "tuning_search_space_id",
                "tuning_search_space_version",
                "tuning_inner_cv_scheme",
                "tuning_inner_group_field",
            )
            if selected.get(key) in (None, "")
        ]
        if missing:
            raise FrozenConfirmatoryBuildError(
                "Selection bundle grouped_nested_tuning is missing tuning metadata: "
                + ", ".join(sorted(missing))
            )

    target_payload = protocol_payload.get("target", {})
    primary_analysis = protocol_payload.get("primary_analysis", {})
    controls_payload = protocol_payload.get("controls", {})
    subgroup_payload = protocol_payload.get("subgroups", {})
    multiplicity_payload = protocol_payload.get("multiplicity", {})

    primary_metric = str(primary_analysis.get("metric") or "balanced_accuracy")
    secondary_metrics = list(primary_analysis.get("secondary_metrics") or [])

    cost_entry = get_model_cost_entry(model_name)
    projected_seconds = projected_runtime_seconds(
        model_name=model_name,
        framework_mode=FrameworkMode.CONFIRMATORY,
        methodology_policy=methodology_policy_name,
        tuning_enabled=tuning_enabled,
    )

    artifact_requirements = sorted(
        {
            "config.json",
            "metrics.json",
            *[
                str(value)
                for value in list(controls_payload.get("required_artifacts") or [])
                if str(value).strip()
            ],
        }
    )

    context = {
        "framework_mode": FrameworkMode.CONFIRMATORY.value,
        "canonical_run": True,
        "protocol_id": "thesis_confirmatory_v1",
        "protocol_name": "thesis_confirmatory_v1",
        "protocol_version": str(protocol_payload.get("protocol_version") or "v1"),
        "protocol_schema_version": str(
            protocol_payload.get("protocol_schema_version") or THESIS_PROTOCOL_SCHEMA_VERSION
        ),
        "suite_id": str(suite_id),
        "claim_ids": [str(value) for value in claim_ids],
        "methodology_policy_name": methodology_policy_name,
        "class_weight_policy": class_weight_policy,
        "tuning_enabled": bool(tuning_enabled),
        "feature_recipe_id": "baseline_standard_scaler_v1",
        "preprocessing_strategy": str(selected["preprocessing_strategy"]),
        "emit_feature_qc_artifacts": True,
        "subgroup_reporting_enabled": bool(subgroup_payload.get("enabled", True)),
        "subgroup_dimensions": [
            str(value)
            for value in list(subgroup_payload.get("allowed") or ["subject", "task", "modality"])
        ],
        "subgroup_min_samples_per_group": int(subgroup_payload.get("min_samples_per_group", 1)),
        "primary_metric_aggregation": "mean_fold_scores",
        "metric_policy": {
            "primary_metric": str(primary_metric),
            "secondary_metrics": [str(value) for value in secondary_metrics],
            "decision_metric": str(primary_metric),
            "tuning_metric": str(primary_metric),
            "permutation_metric": str(primary_metric),
        },
        "data_policy": {},
        "model_cost_tier": str(cost_entry.cost_tier.value),
        "projected_runtime_seconds": int(projected_seconds),
        "required_run_metadata_fields": ["framework_mode", "canonical_run"],
        "artifact_requirements": artifact_requirements,
        "primary_metric": str(primary_metric),
        "controls": {
            "dummy_baseline_run": False,
            "permutation_metric": str(primary_metric),
        },
        "target_mapping_version": _safe_text(target_payload.get("mapping_version")),
        "target_mapping_hash": _safe_text(target_payload.get("mapping_hash")),
        "confirmatory_lock": {
            "protocol_id": "thesis_confirmatory_v1",
            "analysis_status": str(protocol_payload.get("analysis_status") or "locked"),
            "target_name": str(target_name),
            "target_source_column": str(target_payload.get("source_column") or "emotion"),
            "target_mapping_version": _safe_text(target_payload.get("mapping_version")),
            "target_mapping_hash": _safe_text(target_payload.get("mapping_hash")),
            "split": str(cv_mode),
            "primary_metric": str(primary_metric),
            "model_family": model_name,
            "hyperparameter_policy": ("grouped_nested_tuning" if tuning_enabled else "fixed"),
            "class_weight_policy": class_weight_policy,
            "subgroup_min_samples_per_group": int(subgroup_payload.get("min_samples_per_group", 1)),
            "subgroup_min_classes_per_group": int(subgroup_payload.get("min_classes_per_group", 1)),
            "subgroup_report_small_groups": bool(
                subgroup_payload.get("report_small_groups", False)
            ),
            "multiplicity_primary_hypotheses": int(
                multiplicity_payload.get("primary_hypotheses", 1)
            ),
            "multiplicity_primary_alpha": float(multiplicity_payload.get("primary_alpha", 0.05)),
            "multiplicity_secondary_policy": str(
                multiplicity_payload.get("secondary_policy", "descriptive_only")
            ),
            "multiplicity_exploratory_claims_allowed": bool(
                multiplicity_payload.get("exploratory_claims_allowed", False)
            ),
        },
    }

    if tuning_enabled:
        context["tuning_search_space_id"] = str(selected["tuning_search_space_id"])
        context["tuning_search_space_version"] = str(selected["tuning_search_space_version"])
        context["tuning_inner_cv_scheme"] = str(selected["tuning_inner_cv_scheme"])
        context["tuning_inner_group_field"] = str(selected["tuning_inner_group_field"])
    return context


def _build_variant_template(
    *,
    template_id: str,
    params: dict[str, Any],
    design_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "template_id": template_id,
        "supported": True,
        "params": params,
        "expand": {},
        "sections": [
            "dataset_selection",
            "feature_cache_build",
            "feature_matrix_load",
            "spatial_validation",
            "model_fit",
            "evaluation",
            "interpretability",
        ],
        "start_section": "dataset_selection",
        "end_section": "evaluation",
        "reuse_policy": "auto",
        "factor_settings": {},
        "fixed_controls": {},
        "design_metadata": design_metadata,
    }


def _write_markdown_report(path: Path, *, payload: dict[str, Any]) -> None:
    selected = payload["selected"]
    scope = payload["scope"]
    advisory = payload["advisory"]
    interpretation_boundary = payload.get("interpretation_boundary", {})
    lines = [
        "# Frozen Confirmatory Registry Report",
        "",
        "## Canonical Confirmatory Scope",
        f"- scope_id: {scope['scope_id']}",
        f"- main_tasks: {', '.join(scope['main_tasks'])}",
        f"- main_modality: {scope['main_modality']}",
        f"- main_target: {scope['main_target']}",
        f"- within_subjects: {', '.join(scope['within_subjects'])}",
        "- transfer_pairs:",
    ]
    for pair in scope["transfer_pairs"]:
        lines.append(f"  - {pair['train_subject']} -> {pair['test_subject']}")

    lines.extend(
        [
            "",
            "## Frozen Decisions",
            f"- target: {selected['target']}",
            f"- cv_within_subject: {selected['cv_within_subject']}",
            f"- cv_transfer: {selected['cv_transfer']}",
            f"- model: {selected['model']}",
            f"- class_weight_policy: {selected['class_weight_policy']}",
            f"- methodology_policy_name: {selected['methodology_policy_name']}",
            f"- feature_space: {selected['feature_space']}",
            f"- dimensionality_strategy: {selected['dimensionality_strategy']}",
            f"- preprocessing_strategy: {selected['preprocessing_strategy']}",
            "",
            "## Advisory Stages",
            f"- task_pooling (E02): {advisory.get('task_pooling')}",
            f"- modality_pooling (E03): {advisory.get('modality_pooling')}",
            "",
            "## Generated Final Cells",
            "- within_person_loso_sub-001",
            "- within_person_loso_sub-002",
            "- transfer_sub-001_to_sub-002",
            "- transfer_sub-002_to_sub-001",
            "",
            "## Protocol Binding",
            "- framework_mode: confirmatory",
            "- protocol_name: thesis_confirmatory_v1",
            f"- protocol_version: {payload['protocol_binding']['protocol_version']}",
            "",
            "## Selection and Validation Scope",
            "- Preflight experiments were used to select the final locked confirmatory configuration.",
            "- Final frozen confirmatory runs report that locked configuration under fixed settings.",
            "- Selection and confirmatory reporting both use the same overall project dataset.",
            (
                "- Interpretation boundary: stronger than ad hoc tuning, but weaker than "
                "independent external validation."
            ),
            f"- validation_scope: {interpretation_boundary.get('validation_scope', 'internal_project_dataset')}",
            "",
            "## Caveats",
            "- E02/E03 remain advisory and do not override canonical confirmatory scope.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    campaign_root = args.campaign_root.resolve()
    if not campaign_root.exists() or not campaign_root.is_dir():
        raise FrozenConfirmatoryBuildError(f"campaign root does not exist: {campaign_root}")

    bundle_path = _resolve_bundle_path(args.selection_bundle, campaign_root=campaign_root)
    scope_path = _resolve_bundle_path(args.scope_config, campaign_root=campaign_root)
    output_registry = args.output_registry.resolve()

    bundle_payload = _load_json_object(bundle_path, label="confirmatory selection bundle")
    scope_payload_raw = _load_json_object(scope_path, label="confirmatory scope config")
    scope = _resolve_scope_payload(scope_payload_raw)
    selected = _required_selected(bundle_payload)
    advisory_payload = bundle_payload.get("advisory")
    advisory = dict(advisory_payload) if isinstance(advisory_payload, dict) else {}
    scope_feature_space = str(scope["feature_space"])
    selected_feature_space = _safe_text(selected.get("feature_space"))
    if selected_feature_space and selected_feature_space != scope_feature_space:
        raise FrozenConfirmatoryBuildError(
            "Selection bundle feature_space does not match canonical confirmatory scope feature_space."
        )
    selected["feature_space"] = scope_feature_space

    if str(selected.get("target")) != str(scope["main_target"]):
        raise FrozenConfirmatoryBuildError(
            "Selection bundle target does not match canonical confirmatory scope target."
        )

    if str(bundle_payload.get("scope_id")) != str(scope["scope_id"]):
        raise FrozenConfirmatoryBuildError(
            "selection bundle scope_id does not match scope config scope_id."
        )
    if not bool(bundle_payload.get("freeze_ready")):
        raise FrozenConfirmatoryBuildError(
            "selection bundle freeze_ready=false; resolve preflight hard-lock reviews before freezing."
        )

    index_csv = args.index_csv.resolve() if args.index_csv is not None else None
    if index_csv is None:
        index_csv = _infer_index_csv_from_bundle(
            campaign_root=campaign_root,
            bundle_payload=bundle_payload,
        )
    if index_csv is not None:
        coverage_validation = _validate_scope_coverage(index_csv=index_csv, scope=scope)
    else:
        coverage_validation = {
            "status": "skipped_missing_index_csv",
            "reason": "index_csv not provided and could not be inferred from bundle review sources.",
        }

    protocol_payload = _load_json_object(
        _DEFAULT_PROTOCOL_PATH.resolve(), label="confirmatory protocol"
    )

    campaign_id = _safe_text(bundle_payload.get("campaign_id")) or campaign_root.name
    selected_model = str(selected["model"])
    selected_methodology = str(selected["methodology_policy_name"])
    selected_class_weight = str(selected["class_weight_policy"])

    common_params: dict[str, Any] = {
        "target": str(scope["main_target"]),
        "model": selected_model,
        "filter_modality": str(scope["main_modality"]),
        "feature_space": str(scope_feature_space),
        "dimensionality_strategy": str(selected["dimensionality_strategy"]),
        "preprocessing_strategy": str(selected["preprocessing_strategy"]),
        "methodology_policy_name": selected_methodology,
        "class_weight_policy": selected_class_weight,
        "framework_mode": FrameworkMode.CONFIRMATORY.value,
        "canonical_run": True,
    }
    for optional_key in (
        "tuning_search_space_id",
        "tuning_search_space_version",
        "tuning_inner_cv_scheme",
        "tuning_inner_group_field",
    ):
        if selected.get(optional_key) not in (None, ""):
            common_params[optional_key] = selected[optional_key]

    templates: list[dict[str, Any]] = []
    design_metadata_base = {
        "frozen_confirmatory": True,
        "scope_id": scope["scope_id"],
        "main_tasks": list(scope["main_tasks"]),
        "main_modality": str(scope["main_modality"]),
        "protocol_name": "thesis_confirmatory_v1",
    }

    for subject in scope["within_subjects"]:
        cv_mode = str(selected["cv_within_subject"])
        protocol_context = _build_protocol_context(
            selected=selected,
            protocol_payload=protocol_payload,
            cv_mode=cv_mode,
            claim_ids=["rq1_within_person"],
            suite_id="main_within_subject",
            target_name=str(scope["main_target"]),
        )
        params = dict(common_params)
        params["cv"] = cv_mode
        params["subject"] = str(subject)
        params["protocol_context"] = protocol_context
        params["scope_task_ids"] = list(scope["main_tasks"])
        template_id = f"within_person_loso_{subject}"
        templates.append(
            _build_variant_template(
                template_id=template_id,
                params=params,
                design_metadata={
                    **design_metadata_base,
                    "cell_kind": "within_subject",
                    "subject": str(subject),
                },
            )
        )

    for pair in scope["transfer_pairs"]:
        train_subject = str(pair["train_subject"])
        test_subject = str(pair["test_subject"])
        cv_mode = str(selected["cv_transfer"])
        protocol_context = _build_protocol_context(
            selected=selected,
            protocol_payload=protocol_payload,
            cv_mode=cv_mode,
            claim_ids=["rq2_transfer_directional"],
            suite_id="main_transfer_directional",
            target_name=str(scope["main_target"]),
        )
        params = dict(common_params)
        params["cv"] = cv_mode
        params["train_subject"] = train_subject
        params["test_subject"] = test_subject
        params["protocol_context"] = protocol_context
        params["scope_task_ids"] = list(scope["main_tasks"])
        template_id = f"transfer_{train_subject}_to_{test_subject}"
        templates.append(
            _build_variant_template(
                template_id=template_id,
                params=params,
                design_metadata={
                    **design_metadata_base,
                    "cell_kind": "transfer",
                    "train_subject": train_subject,
                    "test_subject": test_subject,
                },
            )
        )

    if len(templates) != 4:
        raise FrozenConfirmatoryBuildError(
            f"Frozen confirmatory registry must contain exactly 4 cells; got {len(templates)}."
        )

    registry_payload: dict[str, Any] = {
        "schema_version": "frozen-confirmatory-registry-v1",
        "description": (
            "Frozen confirmatory execution registry generated from reviewed preflight outputs."
        ),
        "campaign_id": campaign_id,
        "scope_id": scope["scope_id"],
        "selection_bundle_path": str(bundle_path.resolve()),
        "protocol_name": "thesis_confirmatory_v1",
        "framework_mode": FrameworkMode.CONFIRMATORY.value,
        "experiments": [
            {
                "experiment_id": "CFM01",
                "title": "Frozen confirmatory main package",
                "stage": "Stage 5 - Confirmatory analysis",
                "decision_id": "CFM_LOCK_V1",
                "manipulated_factor": "None; frozen confirmatory package",
                "primary_metric": "balanced_accuracy",
                "executable_now": True,
                "execution_status": "unknown",
                "blocked_reasons": [],
                "notes": "Generated from preflight confirmatory selection bundle.",
                "variant_templates": templates,
            }
        ],
    }

    output_registry.parent.mkdir(parents=True, exist_ok=True)
    output_registry.write_text(f"{json.dumps(registry_payload, indent=2)}\n", encoding="utf-8")

    manifest_path = output_registry.parent / f"frozen_confirmatory_manifest_{campaign_id}.json"
    report_path = output_registry.parent / f"frozen_confirmatory_report_{campaign_id}.md"

    protocol_version = str(protocol_payload.get("protocol_version") or "v1")
    interpretation_boundary = {
        "selection_reporting_relationship": "preflight_selected_locked_confirmatory",
        "validation_scope": "internal_project_dataset",
        "external_validation_equivalence": "not_equivalent",
        "interpretation_note": (
            "The frozen confirmatory registry is generated from reviewed preflight "
            "selection outputs. Selection and final reporting belong to the same "
            "overall project dataset, supporting methodological and dataset-specific "
            "conclusions rather than external validation claims."
        ),
    }
    manifest_payload = {
        "manifest_id": f"frozen_confirmatory_manifest_{campaign_id}",
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "campaign_id": campaign_id,
        "scope_id": scope["scope_id"],
        "freeze_allowed": True,
        "subject_coverage": list(scope["within_subjects"]),
        "transfer_direction_coverage": list(scope["transfer_pairs"]),
        "cell_counts": {
            "within_subject": 2,
            "transfer": 2,
            "total": 4,
        },
        "scope": scope,
        "selected": selected,
        "advisory": advisory,
        "protocol_binding": {
            "framework_mode": FrameworkMode.CONFIRMATORY.value,
            "protocol_name": "thesis_confirmatory_v1",
            "protocol_version": protocol_version,
            "protocol_schema_version": THESIS_PROTOCOL_SCHEMA_VERSION,
        },
        "selection_reporting_relationship": interpretation_boundary[
            "selection_reporting_relationship"
        ],
        "validation_scope": interpretation_boundary["validation_scope"],
        "external_validation_equivalence": interpretation_boundary[
            "external_validation_equivalence"
        ],
        "interpretation_note": interpretation_boundary["interpretation_note"],
        "coverage_validation": coverage_validation,
        "registry_path": str(output_registry.resolve()),
        "report_path": str(report_path.resolve()),
    }
    manifest_path.write_text(f"{json.dumps(manifest_payload, indent=2)}\n", encoding="utf-8")

    _write_markdown_report(
        report_path,
        payload={
            "scope": scope,
            "selected": selected,
            "advisory": advisory,
            "protocol_binding": {
                "protocol_version": protocol_version,
            },
            "interpretation_boundary": interpretation_boundary,
        },
    )

    outputs_payload = {
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "registry": str(output_registry.resolve()),
        "manifest": str(manifest_path.resolve()),
        "report": str(report_path.resolve()),
    }
    outputs_path = campaign_root / "preflight_reviews" / "frozen_confirmatory_outputs.json"
    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    outputs_path.write_text(f"{json.dumps(outputs_payload, indent=2)}\n", encoding="utf-8")

    print(f"Wrote frozen confirmatory registry: {output_registry}")
    print(f"Wrote frozen confirmatory manifest: {manifest_path}")
    print(f"Wrote frozen confirmatory report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
