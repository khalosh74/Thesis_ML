from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_EXPERIMENT_REPORT,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    compute_config_hash,
    register_artifact,
)
from Thesis_ML.orchestration.compiler import compile_registry_file
from Thesis_ML.orchestration.contracts import CompiledStudyManifest

STAGE_ORDER = [
    "Stage 1 - Target lock",
    "Stage 2 - Split lock",
    "Stage 3 - Model lock",
    "Stage 4 - Feature/preprocessing lock",
    "Stage 5 - Confirmatory analysis",
    "Stage 6 - Robustness analysis",
    "Stage 7 - Exploratory extension",
]

STAGE_SUMMARY_FILENAMES = {
    "Stage 1 - Target lock": "stage1_target_lock_summary",
    "Stage 2 - Split lock": "stage2_split_lock_summary",
    "Stage 3 - Model lock": "stage3_model_lock_summary",
    "Stage 4 - Feature/preprocessing lock": "stage4_feature_lock_summary",
    "Stage 5 - Confirmatory analysis": "stage5_confirmatory_summary",
    "Stage 6 - Robustness analysis": "stage6_robustness_summary",
    "Stage 7 - Exploratory extension": "stage7_exploratory_summary",
}

RUN_LOG_EXPORT_COLUMNS = [
    "Run_ID",
    "Experiment_ID",
    "Run_Date",
    "Dataset_Name",
    "Data_Subset",
    "Code_Commit_or_Version",
    "Config_File_or_Path",
    "Random_Seed",
    "Target",
    "Split_ID_or_Fold_Definition",
    "Model",
    "Feature_Set",
    "Run_Type",
    "Affects_Frozen_Pipeline",
    "Eligible_for_Method_Decision",
    "Primary_Metric_Value",
    "Secondary_Metric_1",
    "Secondary_Metric_2",
    "Robustness_Output_Summary",
    "Result_Summary",
    "Preliminary_Interpretation",
    "Reviewed",
    "Used_in_Thesis",
    "Artifact_Path",
    "Notes",
]

SUMMARY_COLUMNS = [
    "experiment_id",
    "title",
    "stage",
    "decision_id",
    "manipulated_factor",
    "primary_metric",
    "total_variants",
    "completed_variants",
    "failed_variants",
    "blocked_variants",
    "dry_run_variants",
    "best_variant_id",
    "best_primary_metric_value",
    "mean_primary_metric_value",
    "status",
    "notes",
]

VARIANT_EXPORT_COLUMNS = [
    "experiment_id",
    "title",
    "stage",
    "decision_id",
    "template_id",
    "variant_id",
    "variant_label",
    "status",
    "target",
    "cv",
    "model",
    "subject",
    "train_subject",
    "test_subject",
    "filter_task",
    "filter_modality",
    "start_section",
    "end_section",
    "base_artifact_id",
    "reuse_policy",
    "primary_metric_name",
    "primary_metric_value",
    "balanced_accuracy",
    "macro_f1",
    "accuracy",
    "run_id",
    "report_dir",
    "manifest_path",
    "blocked_reason",
    "error",
]


def run_experiment(**kwargs: Any) -> dict[str, Any]:
    from Thesis_ML.experiments.run_experiment import run_experiment as _run_experiment

    return _run_experiment(**kwargs)


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_str_list(values: pd.Series) -> list[str]:
    return sorted(values.dropna().astype(str).unique().tolist())


def _git_commit() -> str | None:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = process.stdout.strip()
    return commit or None


def _read_registry(path: Path) -> CompiledStudyManifest:
    return compile_registry_file(path)


def _collect_dataset_scope(
    index_csv: Path,
    subjects_filter: list[str] | None = None,
    tasks_filter: list[str] | None = None,
    modalities_filter: list[str] | None = None,
) -> dict[str, Any]:
    df = pd.read_csv(index_csv)
    required = {"subject", "task", "modality"}
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset index missing required columns for scope expansion: {missing}")

    subjects = _as_str_list(df["subject"])
    tasks = _as_str_list(df["task"])
    modalities = _as_str_list(df["modality"])

    if subjects_filter:
        selected = set(subjects_filter)
        subjects = [value for value in subjects if value in selected]
    if tasks_filter:
        selected = set(tasks_filter)
        tasks = [value for value in tasks if value in selected]
    if modalities_filter:
        selected = set(modalities_filter)
        modalities = [value for value in modalities if value in selected]

    ordered_pairs = [(train, test) for train in subjects for test in subjects if train != test]

    return {
        "subjects": subjects,
        "tasks": tasks,
        "modalities": modalities,
        "ordered_subject_pairs": ordered_pairs,
        "models_linear": ["ridge", "logreg", "linearsvc"],
    }


def _stage_sort_key(stage_name: str) -> int:
    try:
        return STAGE_ORDER.index(stage_name)
    except ValueError:
        return len(STAGE_ORDER)


def _experiment_sort_key(experiment: dict[str, Any]) -> tuple[int, int, str]:
    stage_index = _stage_sort_key(str(experiment.get("stage", "")))
    experiment_id = str(experiment.get("experiment_id", ""))
    numeric = 9999
    if experiment_id.startswith("E"):
        try:
            numeric = int(experiment_id[1:])
        except ValueError:
            numeric = 9999
    return (stage_index, numeric, experiment_id)


def _select_experiments(
    registry: CompiledStudyManifest,
    experiment_id: str | None,
    stage: str | None,
    run_all: bool,
) -> list[dict[str, Any]]:
    experiments = [experiment.model_dump(mode="python") for experiment in registry.experiments]
    if experiment_id:
        selected = [exp for exp in experiments if str(exp.get("experiment_id")) == experiment_id]
        if not selected:
            raise ValueError(f"Experiment '{experiment_id}' was not found in registry.")
        return sorted(selected, key=_experiment_sort_key)

    if stage:
        selected = [exp for exp in experiments if str(exp.get("stage")) == stage]
        if not selected:
            raise ValueError(f"No experiments found for stage '{stage}'.")
        return sorted(selected, key=_experiment_sort_key)

    if run_all:
        return sorted(experiments, key=_experiment_sort_key)

    raise ValueError("Select one of --experiment-id, --stage, or --all.")


def _variant_label(params: dict[str, Any]) -> str:
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


def _expand_template_variants(
    experiment: dict[str, Any],
    template: dict[str, Any],
    dataset_scope: dict[str, Any],
) -> list[dict[str, Any]]:
    template_id = str(template.get("template_id", "template"))
    supported = bool(template.get("supported", False))
    base_params = dict(template.get("params", {}))
    start_section = template.get("start_section")
    end_section = template.get("end_section")
    base_artifact_id = template.get("base_artifact_id")
    reuse_policy = template.get("reuse_policy")
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

    resolved: list[dict[str, Any]] = []
    for idx, row in enumerate(variants, start=1):
        resolved.append(
            {
                "template_id": template_id,
                "variant_index": idx,
                "params": row["params"],
                "supported": bool(row["supported"]),
                "blocked_reason": row["blocked_reason"],
                "start_section": start_section,
                "end_section": end_section,
                "base_artifact_id": base_artifact_id,
                "reuse_policy": reuse_policy,
            }
        )
    return resolved


def _expand_experiment_variants(
    experiment: dict[str, Any],
    dataset_scope: dict[str, Any],
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
                }
            ],
            [reason],
        )

    variants: list[dict[str, Any]] = []
    warnings: list[str] = []
    for template in templates:
        template_variants = _expand_template_variants(
            experiment=experiment,
            template=template,
            dataset_scope=dataset_scope,
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


def _build_command(
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
    if start_section:
        command.extend(["--start-section", str(start_section)])
    if end_section:
        command.extend(["--end-section", str(end_section)])
    if base_artifact_id:
        command.extend(["--base-artifact-id", str(base_artifact_id)])
    if reuse_policy:
        command.extend(["--reuse-policy", str(reuse_policy)])
    return command


def _command_to_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _build_dataset_subset_label(params: dict[str, Any]) -> str:
    parts: list[str] = []
    if params.get("subject"):
        parts.append(f"subject={params['subject']}")
    if params.get("train_subject") and params.get("test_subject"):
        parts.append(f"transfer={params['train_subject']}->{params['test_subject']}")
    if params.get("filter_task"):
        parts.append(f"task={params['filter_task']}")
    if params.get("filter_modality"):
        parts.append(f"modality={params['filter_modality']}")
    return ", ".join(parts) if parts else "full index after target cleanup"


def _extract_metric(metrics: dict[str, Any], name: str) -> float | None:
    if name in metrics:
        return _safe_float(metrics.get(name))
    if name == "balanced_accuracy":
        return _safe_float(metrics.get("balanced_accuracy"))
    if name == "macro_f1":
        return _safe_float(metrics.get("macro_f1"))
    if name == "accuracy":
        return _safe_float(metrics.get("accuracy"))
    return None


def _execute_variant(
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
    artifact_registry_path: Path | None = None,
    code_ref: str | None = None,
) -> dict[str, Any]:
    experiment_id = str(experiment["experiment_id"])
    template_id = str(variant["template_id"])
    variant_index = int(variant["variant_index"])
    variant_id = f"{template_id}__{variant_index:03d}"
    params = dict(variant.get("params", {}))
    supported = bool(variant.get("supported", False))
    blocked_reason = variant.get("blocked_reason")
    start_section = (
        str(variant.get("start_section")).strip() if variant.get("start_section") else None
    )
    end_section = str(variant.get("end_section")).strip() if variant.get("end_section") else None
    base_artifact_id = (
        str(variant.get("base_artifact_id")).strip()
        if variant.get("base_artifact_id")
        else None
    )
    reuse_policy = str(variant.get("reuse_policy")).strip() if variant.get("reuse_policy") else None

    run_id = f"ds_{experiment_id}_{template_id}_{variant_index:03d}_{campaign_id}"
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

    if not supported:
        status = "blocked"
    else:
        command = _build_command(
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=cache_dir,
            reports_root=reports_root,
            run_id=run_id,
            seed=seed,
            n_permutations=n_permutations,
            params=params,
            start_section=start_section,
            end_section=end_section,
            base_artifact_id=base_artifact_id,
            reuse_policy=reuse_policy,
        )
        command_text = _command_to_text(command)
        if dry_run:
            status = "dry_run"
        else:
            try:
                result = run_experiment(
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
                    seed=seed,
                    filter_task=(str(params["filter_task"]) if params.get("filter_task") else None),
                    filter_modality=(
                        str(params["filter_modality"]) if params.get("filter_modality") else None
                    ),
                    n_permutations=n_permutations,
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
    primary_metric_name = str(experiment.get("primary_metric", "balanced_accuracy"))
    primary_metric_value = _extract_metric(metrics, primary_metric_name)

    manifest_payload = {
        "campaign_id": campaign_id,
        "experiment_id": experiment_id,
        "title": experiment.get("title"),
        "stage": experiment.get("stage"),
        "decision_id": experiment.get("decision_id"),
        "template_id": template_id,
        "variant_id": variant_id,
        "variant_label": _variant_label(params),
        "status": status,
        "started_at": now_start,
        "finished_at": now_end,
        "command": command_text,
        "config_used": {
            "index_csv": str(index_csv.resolve()),
            "data_root": str(data_root.resolve()),
            "cache_dir": str(cache_dir.resolve()),
            "seed": int(seed),
            "n_permutations": int(n_permutations),
            "start_section": start_section,
            "end_section": end_section,
            "base_artifact_id": base_artifact_id,
            "reuse_policy": reuse_policy,
            "params": params,
        },
        "dataset_subset": _build_dataset_subset_label(params),
        "split_logic": params.get("cv"),
        "model": params.get("model"),
        "target_definition": params.get("target"),
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
        "variant_label": _variant_label(params),
        "status": status,
        "target": params.get("target"),
        "cv": params.get("cv"),
        "model": params.get("model"),
        "subject": params.get("subject"),
        "train_subject": params.get("train_subject"),
        "test_subject": params.get("test_subject"),
        "filter_task": params.get("filter_task"),
        "filter_modality": params.get("filter_modality"),
        "start_section": start_section,
        "end_section": end_section,
        "base_artifact_id": base_artifact_id,
        "reuse_policy": reuse_policy,
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": primary_metric_value,
        "balanced_accuracy": _safe_float(metrics.get("balanced_accuracy")),
        "macro_f1": _safe_float(metrics.get("macro_f1")),
        "accuracy": _safe_float(metrics.get("accuracy")),
        "run_id": run_id,
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


def _write_experiment_outputs(
    experiment: dict[str, Any],
    experiment_root: Path,
    variant_records: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    export_rows = []
    for row in variant_records:
        export_rows.append({column: row.get(column) for column in VARIANT_EXPORT_COLUMNS})
    pd.DataFrame(export_rows, columns=VARIANT_EXPORT_COLUMNS).to_csv(
        experiment_root / "experiment_variants.csv",
        index=False,
    )

    experiment_manifest = {
        "experiment_id": str(experiment.get("experiment_id")),
        "title": str(experiment.get("title", "")),
        "stage": str(experiment.get("stage", "")),
        "decision_id": str(experiment.get("decision_id", "")),
        "manipulated_factor": str(experiment.get("manipulated_factor", "")),
        "primary_metric": str(experiment.get("primary_metric", "balanced_accuracy")),
        "warnings": warnings,
        "variant_count": int(len(variant_records)),
        "completed_count": int(sum(1 for row in variant_records if row["status"] == "completed")),
        "failed_count": int(sum(1 for row in variant_records if row["status"] == "failed")),
        "blocked_count": int(sum(1 for row in variant_records if row["status"] == "blocked")),
        "dry_run_count": int(sum(1 for row in variant_records if row["status"] == "dry_run")),
        "variant_manifest_paths": [row["manifest_path"] for row in variant_records],
    }
    (experiment_root / "experiment_manifest.json").write_text(
        f"{json.dumps(experiment_manifest, indent=2)}\n",
        encoding="utf-8",
    )


def _summarize_by_experiment(
    experiments: list[dict[str, Any]],
    variant_records: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        experiment_id = str(experiment["experiment_id"])
        metric_name = str(experiment.get("primary_metric", "balanced_accuracy"))
        records = [row for row in variant_records if row["experiment_id"] == experiment_id]

        completed = [row for row in records if row["status"] == "completed"]
        failed = [row for row in records if row["status"] == "failed"]
        blocked = [row for row in records if row["status"] == "blocked"]
        dry_run = [row for row in records if row["status"] == "dry_run"]

        best_variant_id = None
        best_metric_value = None
        mean_metric_value = None
        if completed:
            metric_pairs = [
                (
                    row["variant_id"],
                    _safe_float(row.get("primary_metric_value")),
                )
                for row in completed
            ]
            metric_pairs = [item for item in metric_pairs if item[1] is not None]
            if metric_pairs:
                best_variant_id, best_metric_value = max(
                    metric_pairs, key=lambda item: float(item[1])
                )
                mean_metric_value = float(
                    sum(float(value) for _, value in metric_pairs) / len(metric_pairs)
                )

        if completed and not failed and not blocked:
            status = "completed"
        elif completed and (failed or blocked):
            status = "partial"
        elif dry_run and blocked and not completed:
            status = "dry_run_partial_blocked"
        elif dry_run and not completed:
            status = "dry_run"
        elif blocked and not completed:
            status = "blocked"
        elif failed and not completed:
            status = "failed"
        else:
            status = "not_executed"

        notes: list[str] = []
        if blocked:
            blocked_reasons = sorted(
                {str(row.get("blocked_reason")) for row in blocked if row.get("blocked_reason")}
            )
            if blocked_reasons:
                notes.append("blocked: " + "; ".join(blocked_reasons))
        if failed:
            notes.append(f"failed_variants={len(failed)}")

        rows.append(
            {
                "experiment_id": experiment_id,
                "title": str(experiment.get("title", "")),
                "stage": str(experiment.get("stage", "")),
                "decision_id": str(experiment.get("decision_id", "")),
                "manipulated_factor": str(experiment.get("manipulated_factor", "")),
                "primary_metric": metric_name,
                "total_variants": int(len(records)),
                "completed_variants": int(len(completed)),
                "failed_variants": int(len(failed)),
                "blocked_variants": int(len(blocked)),
                "dry_run_variants": int(len(dry_run)),
                "best_variant_id": best_variant_id,
                "best_primary_metric_value": best_metric_value,
                "mean_primary_metric_value": mean_metric_value,
                "status": status,
                "notes": " | ".join(notes),
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _write_stage_summaries(
    campaign_root: Path,
    variant_records: list[dict[str, Any]],
) -> list[Path]:
    created_files: list[Path] = []
    stage_columns = [
        "experiment_id",
        "title",
        "decision_id",
        "stage",
        "template_id",
        "variant_id",
        "variant_label",
        "status",
        "primary_metric_name",
        "primary_metric_value",
        "balanced_accuracy",
        "macro_f1",
        "accuracy",
        "report_dir",
        "blocked_reason",
        "error",
    ]
    df = pd.DataFrame(variant_records)
    if df.empty:
        df = pd.DataFrame(columns=stage_columns)

    for stage in STAGE_ORDER:
        stage_name = STAGE_SUMMARY_FILENAMES.get(stage, stage.replace(" ", "_").lower())
        csv_path = campaign_root / f"{stage_name}.csv"
        markdown_path = campaign_root / f"{stage_name}.md"
        stage_df = df[df.get("stage", pd.Series(dtype=str)) == stage].copy()
        if stage_df.empty:
            stage_df = pd.DataFrame(columns=stage_columns)
        else:
            stage_df = stage_df.reindex(columns=stage_columns)
        stage_df.to_csv(csv_path, index=False)

        lines = [f"# {stage}", ""]
        if stage_df.empty:
            lines.append("No variants were selected for this stage in this campaign.")
        else:
            lines.append("## What was compared")
            experiment_ids = sorted(stage_df["experiment_id"].astype(str).unique().tolist())
            lines.append(f"- Experiments: {', '.join(experiment_ids)}")
            lines.append(f"- Variants evaluated: {len(stage_df)}")
            lines.append("")
            lines.append("## Status summary")
            status_counts = stage_df["status"].value_counts().to_dict()
            for status_name in sorted(status_counts):
                lines.append(f"- {status_name}: {status_counts[status_name]}")
            lines.append("")
            lines.append("## Metric focus")
            lines.append(
                "- Primary metric tracked per experiment: balanced_accuracy (registry-defined)."
            )
            lines.append(
                "- Use corresponding CSV for exact per-variant evidence before locking decisions."
            )
            lines.append("")
            lines.append("## Decision linkage")
            decision_ids = sorted(stage_df["decision_id"].astype(str).unique().tolist())
            lines.append(f"- Informs Decision_Log IDs: {', '.join(decision_ids)}")

        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        created_files.extend([csv_path, markdown_path])

    return created_files


def _write_run_log_export(
    campaign_root: Path,
    variant_records: list[dict[str, Any]],
    dataset_name: str,
    seed: int,
    commit: str | None,
) -> Path:
    rows: list[dict[str, Any]] = []
    for row in variant_records:
        primary_value = _safe_float(row.get("primary_metric_value"))
        run_date = str(row.get("started_at", ""))[:10]
        notes = row.get("blocked_reason") or row.get("error") or ""
        rows.append(
            {
                "Run_ID": row.get("run_id"),
                "Experiment_ID": row.get("experiment_id"),
                "Run_Date": run_date,
                "Dataset_Name": dataset_name,
                "Data_Subset": _build_dataset_subset_label(row),
                "Code_Commit_or_Version": commit or "",
                "Config_File_or_Path": row.get("config_path") or "",
                "Random_Seed": seed,
                "Target": row.get("target"),
                "Split_ID_or_Fold_Definition": row.get("cv"),
                "Model": row.get("model"),
                "Feature_Set": "masked whole-brain voxel cache (current pipeline)",
                "Run_Type": "Decision-support",
                "Affects_Frozen_Pipeline": "Yes",
                "Eligible_for_Method_Decision": "Yes" if row.get("status") == "completed" else "No",
                "Primary_Metric_Value": primary_value,
                "Secondary_Metric_1": _safe_float(row.get("macro_f1")),
                "Secondary_Metric_2": _safe_float(row.get("accuracy")),
                "Robustness_Output_Summary": "",
                "Result_Summary": row.get("status"),
                "Preliminary_Interpretation": "",
                "Reviewed": "No",
                "Used_in_Thesis": "No",
                "Artifact_Path": row.get("report_dir") or row.get("manifest_path"),
                "Notes": notes,
            }
        )

    df = pd.DataFrame(rows, columns=RUN_LOG_EXPORT_COLUMNS)
    out_path = campaign_root / "run_log_export.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _decision_text_for_experiment(
    experiment: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[str]:
    primary_metric = str(experiment.get("primary_metric", "balanced_accuracy"))
    completed = [row for row in rows if row.get("status") == "completed"]
    blocked = [row for row in rows if row.get("status") == "blocked"]
    failed = [row for row in rows if row.get("status") == "failed"]
    dry_run = [row for row in rows if row.get("status") == "dry_run"]

    lines = [f"### {experiment['experiment_id']} - {experiment['title']}"]
    lines.append(f"- Decision linkage: {experiment.get('decision_id', 'n/a')}")
    lines.append(f"- Manipulated factor: {experiment.get('manipulated_factor', '')}")
    lines.append(f"- Held constant: {experiment.get('fixed_controls', '')}")
    lines.append(f"- Primary metric: {primary_metric}")
    if completed:
        metric_rows = [
            (row.get("variant_id"), _safe_float(row.get("primary_metric_value")))
            for row in completed
        ]
        metric_rows = [pair for pair in metric_rows if pair[1] is not None]
        if metric_rows:
            best_variant, best_value = max(metric_rows, key=lambda pair: float(pair[1]))
            lines.append(
                f"- Pattern favoring one option (executed variants only): "
                f"{best_variant} had best {primary_metric}={best_value:.4f}."
            )
        else:
            lines.append(
                "- Pattern favoring one option: completed runs exist, but primary metric "
                "was not available in the captured payload."
            )
    else:
        lines.append("- Pattern favoring one option: not available (no completed variants).")

    uncertainty_parts: list[str] = []
    if blocked:
        uncertainty_parts.append(f"blocked={len(blocked)}")
    if failed:
        uncertainty_parts.append(f"failed={len(failed)}")
    if dry_run:
        uncertainty_parts.append(f"dry_run={len(dry_run)}")
    if uncertainty_parts:
        lines.append("- Remaining uncertainty: " + ", ".join(uncertainty_parts))
    else:
        lines.append("- Remaining uncertainty: none flagged by orchestrator status layer.")
    lines.append("")
    return lines


def _write_decision_reports(
    campaign_root: Path,
    experiments: list[dict[str, Any]],
    variant_records: list[dict[str, Any]],
) -> tuple[Path, list[Path]]:
    record_map: dict[str, list[dict[str, Any]]] = {}
    for row in variant_records:
        record_map.setdefault(str(row["experiment_id"]), []).append(row)

    stage_markdowns: list[Path] = []
    lines: list[str] = [
        "# Decision-Support Recommendations",
        "",
        "This report summarizes decision-support evidence only (E01-E11).",
        "It does not include confirmatory-stage claims.",
        "",
    ]

    for stage in STAGE_ORDER:
        stage_experiments = [exp for exp in experiments if str(exp.get("stage")) == stage]
        if not stage_experiments:
            continue
        lines.append(f"## {stage}")
        lines.append("")
        stage_lines: list[str] = [f"# {stage}", ""]
        for experiment in stage_experiments:
            exp_lines = _decision_text_for_experiment(
                experiment=experiment,
                rows=record_map.get(str(experiment["experiment_id"]), []),
            )
            lines.extend(exp_lines)
            stage_lines.extend(exp_lines)

        stage_name = STAGE_SUMMARY_FILENAMES.get(stage, stage.replace(" ", "_").lower())
        stage_path = campaign_root / f"{stage_name}_decision_notes.md"
        stage_path.write_text("\n".join(stage_lines) + "\n", encoding="utf-8")
        stage_markdowns.append(stage_path)

    out_path = campaign_root / "decision_recommendations.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path, stage_markdowns


def _status_snapshot(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def run_decision_support_campaign(
    *,
    registry_path: Path,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    output_root: Path,
    experiment_id: str | None,
    stage: str | None,
    run_all: bool,
    seed: int,
    n_permutations: int,
    dry_run: bool,
    subjects_filter: list[str] | None = None,
    tasks_filter: list[str] | None = None,
    modalities_filter: list[str] | None = None,
    max_runs_per_experiment: int | None = None,
    dataset_name: str = "Internal BAS2",
) -> dict[str, Any]:
    registry = _read_registry(registry_path)
    selected_experiments = _select_experiments(
        registry=registry,
        experiment_id=experiment_id,
        stage=stage,
        run_all=run_all,
    )
    dataset_scope = _collect_dataset_scope(
        index_csv=index_csv,
        subjects_filter=subjects_filter,
        tasks_filter=tasks_filter,
        modalities_filter=modalities_filter,
    )

    campaign_id = _now_timestamp()
    campaign_root = output_root / "campaigns" / campaign_id
    campaign_root.mkdir(parents=True, exist_ok=True)

    commit = _git_commit()
    artifact_registry_path = output_root / "artifact_registry.sqlite3"
    all_variant_records: list[dict[str, Any]] = []
    blocked_experiments: list[dict[str, Any]] = []
    experiment_roots: dict[str, str] = {}

    for experiment in selected_experiments:
        exp_id = str(experiment["experiment_id"])
        experiment_root = output_root / exp_id / campaign_id
        experiment_root.mkdir(parents=True, exist_ok=True)
        experiment_roots[exp_id] = str(experiment_root.resolve())

        variants, warnings = _expand_experiment_variants(
            experiment=experiment,
            dataset_scope=dataset_scope,
            max_runs_per_experiment=max_runs_per_experiment,
        )
        variant_records: list[dict[str, Any]] = []
        for variant in variants:
            record = _execute_variant(
                experiment=experiment,
                variant=variant,
                campaign_id=campaign_id,
                experiment_root=experiment_root,
                index_csv=index_csv,
                data_root=data_root,
                cache_dir=cache_dir,
                seed=seed,
                n_permutations=n_permutations,
                dry_run=dry_run,
                artifact_registry_path=artifact_registry_path,
                code_ref=commit,
            )
            variant_records.append(record)
            all_variant_records.append(record)

        _write_experiment_outputs(
            experiment=experiment,
            experiment_root=experiment_root,
            variant_records=variant_records,
            warnings=warnings,
        )

        if variant_records and all(row["status"] == "blocked" for row in variant_records):
            blocked_reasons = sorted(
                {
                    str(row.get("blocked_reason"))
                    for row in variant_records
                    if row.get("blocked_reason")
                }
            )
            blocked_experiments.append(
                {
                    "experiment_id": exp_id,
                    "reasons": blocked_reasons,
                }
            )

    stage_summary_paths = _write_stage_summaries(
        campaign_root=campaign_root,
        variant_records=all_variant_records,
    )

    summary_df = _summarize_by_experiment(
        experiments=selected_experiments,
        variant_records=all_variant_records,
    )
    decision_summary_path = campaign_root / "decision_support_summary.csv"
    summary_df.to_csv(decision_summary_path, index=False)

    run_log_path = _write_run_log_export(
        campaign_root=campaign_root,
        variant_records=all_variant_records,
        dataset_name=dataset_name,
        seed=seed,
        commit=commit,
    )

    decision_report_path, stage_decision_paths = _write_decision_reports(
        campaign_root=campaign_root,
        experiments=selected_experiments,
        variant_records=all_variant_records,
    )

    campaign_metrics_artifact = register_artifact(
        registry_path=artifact_registry_path,
        artifact_type=ARTIFACT_TYPE_METRICS_BUNDLE,
        run_id=campaign_id,
        upstream_artifact_ids=[],
        config_hash=compute_config_hash(
            {
                "campaign_id": campaign_id,
                "seed": int(seed),
                "n_permutations": int(n_permutations),
                "dry_run": bool(dry_run),
                "selected_experiments": [str(exp["experiment_id"]) for exp in selected_experiments],
            }
        ),
        code_ref=commit,
        path=decision_summary_path,
        status="created",
    )

    campaign_manifest = {
        "campaign_id": campaign_id,
        "created_at": _utc_timestamp(),
        "registry_path": str(registry_path.resolve()),
        "selected_experiments": [str(exp["experiment_id"]) for exp in selected_experiments],
        "dataset_scope": dataset_scope,
        "seed": int(seed),
        "n_permutations": int(n_permutations),
        "dry_run": bool(dry_run),
        "status_counts": _status_snapshot(all_variant_records),
        "experiment_roots": experiment_roots,
        "exports": {
            "run_log_export": str(run_log_path.resolve()),
            "decision_support_summary": str(decision_summary_path.resolve()),
            "decision_recommendations": str(decision_report_path.resolve()),
            "stage_summaries": [str(path.resolve()) for path in stage_summary_paths],
            "stage_decision_notes": [str(path.resolve()) for path in stage_decision_paths],
        },
        "artifact_registry_path": str(artifact_registry_path.resolve()),
        "campaign_metrics_artifact_id": campaign_metrics_artifact.artifact_id,
        "blocked_experiments": blocked_experiments,
    }
    campaign_manifest_path = campaign_root / "campaign_manifest.json"
    campaign_manifest_path.write_text(
        f"{json.dumps(campaign_manifest, indent=2)}\n",
        encoding="utf-8",
    )

    if (
        not dry_run
        and experiment_id
        and blocked_experiments
        and all(
            row["status"] in {"blocked", "dry_run"}
            for row in all_variant_records
            if row["experiment_id"] == experiment_id
        )
    ):
        reasons = "; ".join(blocked_experiments[0]["reasons"]) if blocked_experiments else ""
        raise RuntimeError(
            f"Experiment '{experiment_id}' is not executable in current pipeline state. {reasons}"
        )

    return {
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root.resolve()),
        "selected_experiments": [str(exp["experiment_id"]) for exp in selected_experiments],
        "status_counts": _status_snapshot(all_variant_records),
        "blocked_experiments": blocked_experiments,
        "run_log_export_path": str(run_log_path.resolve()),
        "decision_support_summary_path": str(decision_summary_path.resolve()),
        "decision_recommendations_path": str(decision_report_path.resolve()),
        "campaign_manifest_path": str(campaign_manifest_path.resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automate thesis decision-support experiments (E01-E11) using the existing "
            "thesisml-run-experiment execution path."
        )
    )
    parser.add_argument(
        "--registry",
        default="decision_support_registry.json",
        help="Path to decision-support experiment registry JSON.",
    )
    parser.add_argument(
        "--index-csv",
        default=str(Path("Data") / "processed" / "dataset_index.csv"),
        help="Dataset index CSV used by the runner.",
    )
    parser.add_argument(
        "--data-root",
        default="Data",
        help="Root path for beta/mask paths in index CSV.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path("Data") / "processed" / "feature_cache"),
        help="Feature cache directory for runner.",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path("artifacts") / "decision_support"),
        help="Root folder for decision-support manifests and summaries.",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--experiment-id", help="Run one experiment ID (e.g., E01).")
    target_group.add_argument("--stage", help="Run one full stage (exact stage name).")
    target_group.add_argument("--all", action="store_true", help="Run all experiments in registry.")
    parser.add_argument("--seed", type=int, default=42, help="Seed passed to runner.")
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=0,
        help="Optional permutation rounds for each run.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject filter for variant expansion.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task filter for variant expansion.",
    )
    parser.add_argument(
        "--modalities",
        nargs="*",
        default=None,
        help="Optional modality filter for variant expansion.",
    )
    parser.add_argument(
        "--max-runs-per-experiment",
        type=int,
        default=None,
        help="Optional cap on executable variants per experiment.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Internal BAS2",
        help="Dataset label stored in workbook-export logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve variants and write manifests without invoking model runs.",
    )
    return parser


def _print_registry_status(registry: CompiledStudyManifest) -> None:
    experiments = sorted(
        [experiment.model_dump(mode="python") for experiment in registry.experiments],
        key=_experiment_sort_key,
    )
    executable = [exp for exp in experiments if bool(exp.get("executable_now"))]
    blocked = [exp for exp in experiments if not bool(exp.get("executable_now"))]

    print("Executable now:")
    for exp in executable:
        exp_id = str(exp.get("experiment_id"))
        status = str(exp.get("execution_status", "unknown"))
        print(f"  - {exp_id}: {exp.get('title')} [{status}]")

    print("Blocked now:")
    for exp in blocked:
        exp_id = str(exp.get("experiment_id"))
        reasons = exp.get("blocked_reasons", [])
        reason_text = "; ".join(str(reason) for reason in reasons) if reasons else "unspecified"
        print(f"  - {exp_id}: {exp.get('title')} -> {reason_text}")


def _print_stage1_commands(args: argparse.Namespace) -> None:
    base = [
        sys.executable,
        "run_decision_support_experiments.py",
        "--registry",
        str(args.registry),
        "--index-csv",
        str(args.index_csv),
        "--data-root",
        str(args.data_root),
        "--cache-dir",
        str(args.cache_dir),
        "--output-root",
        str(args.output_root),
        "--stage",
        "Stage 1 - Target lock",
        "--seed",
        str(args.seed),
    ]
    if args.n_permutations > 0:
        base.extend(["--n-permutations", str(args.n_permutations)])
    if args.max_runs_per_experiment:
        base.extend(["--max-runs-per-experiment", str(args.max_runs_per_experiment)])
    if args.subjects:
        base.extend(["--subjects", *args.subjects])
    if args.tasks:
        base.extend(["--tasks", *args.tasks])
    if args.modalities:
        base.extend(["--modalities", *args.modalities])

    command = _command_to_text(base)
    print("Stage 1 command:")
    print(f"  {command}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    registry_path = Path(args.registry)
    index_csv = Path(args.index_csv)
    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    output_root = Path(args.output_root)

    registry = _read_registry(registry_path)
    try:
        result = run_decision_support_campaign(
            registry_path=registry_path,
            index_csv=index_csv,
            data_root=data_root,
            cache_dir=cache_dir,
            output_root=output_root,
            experiment_id=args.experiment_id,
            stage=args.stage,
            run_all=bool(args.all),
            seed=args.seed,
            n_permutations=args.n_permutations,
            dry_run=bool(args.dry_run),
            subjects_filter=args.subjects,
            tasks_filter=args.tasks,
            modalities_filter=args.modalities,
            max_runs_per_experiment=args.max_runs_per_experiment,
            dataset_name=args.dataset_name,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        _print_registry_status(registry)
        _print_stage1_commands(args)
        return 2

    print(json.dumps(result, indent=2))
    _print_registry_status(registry)
    _print_stage1_commands(args)
    print(f"Outputs saved under: {Path(result['campaign_root'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
