from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

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


def _as_str_list(values: pd.Series) -> list[str]:
    return sorted(values.dropna().astype(str).unique().tolist())


def collect_dataset_scope(
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

    scoped_df = df.copy()
    if subjects_filter:
        scoped_df = scoped_df[scoped_df["subject"].astype(str).isin(set(subjects_filter))]
    if tasks_filter:
        scoped_df = scoped_df[scoped_df["task"].astype(str).isin(set(tasks_filter))]
    if modalities_filter:
        scoped_df = scoped_df[scoped_df["modality"].astype(str).isin(set(modalities_filter))]

    ordered_pairs = [(train, test) for train in subjects for test in subjects if train != test]
    sessions = _as_str_list(scoped_df["session"]) if "session" in scoped_df.columns else []
    sessions_by_subject_task_modality: dict[str, dict[str, dict[str, list[str]]]] = {}
    if "session" in scoped_df.columns:
        for _, row in scoped_df.iterrows():
            subject = str(row.get("subject", "")).strip()
            task = str(row.get("task", "")).strip()
            modality = str(row.get("modality", "")).strip()
            session = str(row.get("session", "")).strip()
            if not subject or not task or not modality or not session:
                continue
            subject_map = sessions_by_subject_task_modality.setdefault(subject, {})
            task_map = subject_map.setdefault(task, {})
            session_values = task_map.setdefault(modality, [])
            if session not in session_values:
                session_values.append(session)
        for subject_map in sessions_by_subject_task_modality.values():
            for task_map in subject_map.values():
                for modality_key, session_values in list(task_map.items()):
                    task_map[modality_key] = sorted(session_values)

    return {
        "subjects": subjects,
        "tasks": tasks,
        "modalities": modalities,
        "sessions": sessions,
        "sessions_by_subject_task_modality": sessions_by_subject_task_modality,
        "ordered_subject_pairs": ordered_pairs,
        "models_linear": ["ridge", "logreg", "linearsvc"],
    }


def stage_sort_key(stage_name: str) -> int:
    try:
        return STAGE_ORDER.index(stage_name)
    except ValueError:
        return len(STAGE_ORDER)


def experiment_sort_key(experiment: dict[str, Any]) -> tuple[int, int, str]:
    stage_index = stage_sort_key(str(experiment.get("stage", "")))
    experiment_id = str(experiment.get("experiment_id", ""))
    numeric = 9999
    if experiment_id.startswith("E"):
        try:
            numeric = int(experiment_id[1:])
        except ValueError:
            numeric = 9999
    return (stage_index, numeric, experiment_id)


def select_experiments(
    registry: CompiledStudyManifest,
    experiment_id: str | None,
    experiment_ids: list[str] | None,
    stage: str | None,
    run_all: bool,
) -> list[dict[str, Any]]:
    experiments = [experiment.model_dump(mode="python") for experiment in registry.experiments]
    if experiment_ids:
        requested_ids = {str(value).strip() for value in experiment_ids if str(value).strip()}
        selected = [
            exp for exp in experiments if str(exp.get("experiment_id")).strip() in requested_ids
        ]
        if not selected:
            raise ValueError(
                f"None of the requested experiments were found in registry: {sorted(requested_ids)}"
            )
        return sorted(selected, key=experiment_sort_key)

    if experiment_id:
        selected = [exp for exp in experiments if str(exp.get("experiment_id")) == experiment_id]
        if not selected:
            raise ValueError(f"Experiment '{experiment_id}' was not found in registry.")
        return sorted(selected, key=experiment_sort_key)

    if stage:
        selected = [exp for exp in experiments if str(exp.get("stage")) == stage]
        if not selected:
            raise ValueError(f"No experiments found for stage '{stage}'.")
        return sorted(selected, key=experiment_sort_key)

    if run_all:
        return sorted(experiments, key=experiment_sort_key)

    raise ValueError("Select one of --experiment-id, --experiment-ids, --stage, or --all.")


__all__ = [
    "STAGE_ORDER",
    "collect_dataset_scope",
    "experiment_sort_key",
    "select_experiments",
    "stage_sort_key",
]
