from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_WITHIN_CV = "within_subject_loso_session"
_TRANSFER_CV = "frozen_cross_person_transfer"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_scope(scope_payload: dict[str, Any]) -> dict[str, Any]:
    scope_id = _safe_text(scope_payload.get("scope_id"))
    within_subjects = [
        _safe_text(value) for value in list(scope_payload.get("within_subjects") or [])
    ]
    within_subjects = [value for value in within_subjects if value]
    transfer_pairs_raw = list(scope_payload.get("transfer_pairs") or [])
    transfer_pairs: list[tuple[str, str]] = []
    for row in transfer_pairs_raw:
        if not isinstance(row, dict):
            continue
        train_subject = _safe_text(row.get("train_subject"))
        test_subject = _safe_text(row.get("test_subject"))
        if train_subject and test_subject:
            transfer_pairs.append((train_subject, test_subject))

    if not scope_id:
        raise ValueError("Confirmatory scope is missing required key: scope_id")
    if not within_subjects:
        raise ValueError("Confirmatory scope requires non-empty within_subjects")
    if not transfer_pairs:
        raise ValueError("Confirmatory scope requires non-empty transfer_pairs")

    return {
        "scope_id": scope_id,
        "within_subjects": sorted(set(within_subjects)),
        "transfer_pairs": sorted(set(transfer_pairs)),
    }


def _resolve_exceptions(
    *,
    exceptions_payload: dict[str, Any] | None,
    scope_id: str,
) -> dict[str, Any]:
    if exceptions_payload is None:
        return {
            "deferred_within_subjects": [],
            "deferred_transfer_pairs": [],
            "source": None,
        }

    payload_scope_id = _safe_text(exceptions_payload.get("scope_id"))
    if payload_scope_id and payload_scope_id != scope_id:
        raise ValueError(
            "Confirmatory scope exceptions scope_id does not match scientific scope scope_id."
        )

    deferred_within = [
        _safe_text(value)
        for value in list(exceptions_payload.get("deferred_within_subjects") or [])
    ]
    deferred_within = sorted({value for value in deferred_within if value})

    deferred_transfer_raw = list(exceptions_payload.get("deferred_transfer_pairs") or [])
    deferred_transfer: list[tuple[str, str]] = []
    for row in deferred_transfer_raw:
        if not isinstance(row, dict):
            continue
        train_subject = _safe_text(row.get("train_subject"))
        test_subject = _safe_text(row.get("test_subject"))
        if train_subject and test_subject:
            deferred_transfer.append((train_subject, test_subject))

    return {
        "deferred_within_subjects": deferred_within,
        "deferred_transfer_pairs": sorted(set(deferred_transfer)),
        "source": _safe_text(exceptions_payload.get("source")) or None,
    }


def _is_confirmatory_experiment(experiment: dict[str, Any]) -> bool:
    stage = _safe_text(experiment.get("stage")).lower()
    if "confirmatory" in stage:
        return True
    templates = list(experiment.get("variant_templates") or [])
    for template in templates:
        if not isinstance(template, dict):
            continue
        params = template.get("params")
        if not isinstance(params, dict):
            continue
        if _safe_text(params.get("framework_mode")) == "confirmatory":
            return True
        canonical_run = params.get("canonical_run")
        if canonical_run is True:
            return True
    return False


def _is_executable(experiment: dict[str, Any]) -> bool:
    if bool(experiment.get("executable_now", True)) is False:
        return False
    status = _safe_text(experiment.get("execution_status")).lower()
    return status != "blocked"


def collect_runtime_confirmatory_anchors(
    runtime_registry_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    experiments = list(runtime_registry_payload.get("experiments") or [])
    anchors: list[dict[str, Any]] = []
    seen_labels: set[str] = set()

    for experiment in experiments:
        if not isinstance(experiment, dict):
            continue
        if not _is_executable(experiment):
            continue
        if not _is_confirmatory_experiment(experiment):
            continue
        experiment_id = _safe_text(experiment.get("experiment_id"))
        templates = list(experiment.get("variant_templates") or [])
        for template in templates:
            if not isinstance(template, dict):
                continue
            if bool(template.get("supported", True)) is False:
                continue
            params = template.get("params")
            if not isinstance(params, dict):
                continue
            cv = _safe_text(params.get("cv"))
            template_id = _safe_text(template.get("template_id"))

            if cv == _WITHIN_CV:
                subject = _safe_text(params.get("subject"))
                if not subject:
                    continue
                analysis_label = f"{_WITHIN_CV}:{subject}"
                transfer_pair: tuple[str, str] | None = None
            elif cv == _TRANSFER_CV:
                train_subject = _safe_text(params.get("train_subject"))
                test_subject = _safe_text(params.get("test_subject"))
                if not train_subject or not test_subject:
                    continue
                analysis_label = f"{_TRANSFER_CV}:{train_subject}->{test_subject}"
                transfer_pair = (train_subject, test_subject)
            else:
                continue

            if analysis_label in seen_labels:
                continue
            seen_labels.add(analysis_label)
            anchors.append(
                {
                    "analysis_label": analysis_label,
                    "analysis_type": (
                        "within_subject" if cv == _WITHIN_CV else "cross_person_transfer"
                    ),
                    "cv": cv,
                    "subject": _safe_text(params.get("subject")) or None,
                    "train_subject": _safe_text(params.get("train_subject")) or None,
                    "test_subject": _safe_text(params.get("test_subject")) or None,
                    "transfer_pair": transfer_pair,
                    "experiment_id": experiment_id,
                    "template_id": template_id,
                    "target": _safe_text(params.get("target")) or None,
                }
            )

    anchors.sort(key=lambda row: str(row.get("analysis_label") or ""))
    return anchors


def verify_confirmatory_scope_runtime_alignment(
    *,
    scope_config_path: Path,
    runtime_registry_path: Path,
    exceptions_config_path: Path | None = None,
) -> dict[str, Any]:
    scope_payload = _load_json_object(scope_config_path, label="confirmatory scope")
    runtime_payload = _load_json_object(
        runtime_registry_path,
        label="thesis runtime registry",
    )
    exceptions_payload = None
    if exceptions_config_path is not None and exceptions_config_path.exists():
        exceptions_payload = _load_json_object(
            exceptions_config_path,
            label="confirmatory scope exceptions",
        )

    scope = _resolve_scope(scope_payload)
    exceptions = _resolve_exceptions(
        exceptions_payload=exceptions_payload,
        scope_id=str(scope["scope_id"]),
    )
    runtime_anchors = collect_runtime_confirmatory_anchors(runtime_payload)

    scope_within = set(scope["within_subjects"])
    scope_transfer = set(scope["transfer_pairs"])
    deferred_within = set(exceptions["deferred_within_subjects"])
    deferred_transfer = set(exceptions["deferred_transfer_pairs"])

    runtime_within = {
        str(row["subject"])
        for row in runtime_anchors
        if str(row.get("cv")) == _WITHIN_CV and row.get("subject")
    }
    runtime_transfer = {
        tuple(row["transfer_pair"])
        for row in runtime_anchors
        if str(row.get("cv")) == _TRANSFER_CV and isinstance(row.get("transfer_pair"), tuple)
    }

    missing_within = sorted(scope_within - runtime_within - deferred_within)
    missing_transfer = sorted(scope_transfer - runtime_transfer - deferred_transfer)
    out_of_scope_within = sorted(runtime_within - scope_within)
    out_of_scope_transfer = sorted(runtime_transfer - scope_transfer)

    issues: list[dict[str, Any]] = []
    if missing_within:
        issues.append(
            {
                "code": "scope_within_missing_in_runtime",
                "message": "Scoped within-subject confirmatory analyses are missing from runtime.",
                "details": {
                    "missing_within_subjects": missing_within,
                },
            }
        )
    if missing_transfer:
        issues.append(
            {
                "code": "scope_transfer_missing_in_runtime",
                "message": "Scoped transfer confirmatory analyses are missing from runtime.",
                "details": {
                    "missing_transfer_pairs": [
                        {"train_subject": pair[0], "test_subject": pair[1]}
                        for pair in missing_transfer
                    ]
                },
            }
        )
    if out_of_scope_within:
        issues.append(
            {
                "code": "runtime_within_out_of_scope",
                "message": "Runtime includes within-subject confirmatory analyses outside scientific scope.",
                "details": {
                    "subjects": out_of_scope_within,
                },
            }
        )
    if out_of_scope_transfer:
        issues.append(
            {
                "code": "runtime_transfer_out_of_scope",
                "message": "Runtime includes transfer confirmatory analyses outside scientific scope.",
                "details": {
                    "pairs": [
                        {"train_subject": pair[0], "test_subject": pair[1]}
                        for pair in out_of_scope_transfer
                    ]
                },
            }
        )

    return {
        "passed": not issues,
        "scope_config_path": str(scope_config_path.resolve()),
        "runtime_registry_path": str(runtime_registry_path.resolve()),
        "exceptions_config_path": (
            str(exceptions_config_path.resolve())
            if exceptions_config_path is not None and exceptions_config_path.exists()
            else None
        ),
        "scope_id": scope["scope_id"],
        "scope_within_subjects": sorted(scope_within),
        "scope_transfer_pairs": [
            {"train_subject": pair[0], "test_subject": pair[1]} for pair in sorted(scope_transfer)
        ],
        "deferred_within_subjects": sorted(deferred_within),
        "deferred_transfer_pairs": [
            {"train_subject": pair[0], "test_subject": pair[1]}
            for pair in sorted(deferred_transfer)
        ],
        "runtime_anchor_set": runtime_anchors,
        "issues": issues,
    }


def build_confirmatory_control_coverage_rows(
    *,
    runtime_anchors: list[dict[str, Any]],
    e12_table_rows: list[dict[str, Any]],
    e13_table_rows: list[dict[str, Any]],
    e12_summary_json_path: str | None,
    e13_summary_json_path: str | None,
) -> list[dict[str, Any]]:
    e12_labels = {
        _safe_text(row.get("analysis_label"))
        for row in e12_table_rows
        if isinstance(row, dict) and _safe_text(row.get("analysis_label"))
    }
    e13_labels = {
        _safe_text(row.get("analysis_label"))
        for row in e13_table_rows
        if isinstance(row, dict) and _safe_text(row.get("analysis_label"))
    }

    rows: list[dict[str, Any]] = []
    for anchor in runtime_anchors:
        analysis_label = _safe_text(anchor.get("analysis_label"))
        rows.append(
            {
                "analysis_label": analysis_label,
                "analysis_type": _safe_text(anchor.get("analysis_type")),
                "cv": _safe_text(anchor.get("cv")),
                "subject": _safe_text(anchor.get("subject")) or None,
                "train_subject": _safe_text(anchor.get("train_subject")) or None,
                "test_subject": _safe_text(anchor.get("test_subject")) or None,
                "runtime_anchor_experiment_id": _safe_text(anchor.get("experiment_id")),
                "runtime_anchor_template_id": _safe_text(anchor.get("template_id")),
                "e12_covered": bool(analysis_label in e12_labels),
                "e13_covered": bool(analysis_label in e13_labels),
                "e12_summary_json_path": str(e12_summary_json_path)
                if e12_summary_json_path
                else None,
                "e13_summary_json_path": str(e13_summary_json_path)
                if e13_summary_json_path
                else None,
            }
        )
    rows.sort(key=lambda row: str(row.get("analysis_label") or ""))
    return rows


__all__ = [
    "build_confirmatory_control_coverage_rows",
    "collect_runtime_confirmatory_anchors",
    "verify_confirmatory_scope_runtime_alignment",
]
