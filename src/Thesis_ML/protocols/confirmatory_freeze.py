from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from Thesis_ML.config.metric_policy import SUPPORTED_CLASSIFICATION_METRICS
from Thesis_ML.config.paths import (
    DEFAULT_CONFIRMATORY_PROTOCOL_SCHEMA_PATH,
    DEFAULT_TARGET_CONFIGS_DIR,
)
from Thesis_ML.config.schema_versions import THESIS_PROTOCOL_SCHEMA_VERSION
from Thesis_ML.protocols.models import (
    ArtifactContract,
    ProtocolStatus,
    ThesisProtocol,
)

CONFIRMATORY_FREEZE_PROTOCOL_ID = "thesis_confirmatory_v1"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parsing failed for '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in '{path}'.")
    return payload


def _file_sha256(path: Path) -> str:
    # Normalize line endings before hashing so protocol mapping locks are stable
    # across LF/CRLF checkouts on different operating systems.
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _require_locked_analysis_status(payload: dict[str, Any], protocol_path: Path) -> None:
    analysis_status = str(payload.get("analysis_status", "")).strip().lower()
    if analysis_status != "locked":
        raise ValueError(
            "Confirmatory protocol preflight failed: analysis_status must be 'locked' "
            f"for '{protocol_path}'."
        )


def _resolve_target_mapping_path(mapping_version: str) -> Path:
    mapping_filename = f"{mapping_version}.json"
    return DEFAULT_TARGET_CONFIGS_DIR / mapping_filename


def _require_target_mapping_hash_match(payload: dict[str, Any], protocol_path: Path) -> dict[str, str]:
    target_payload = payload.get("target")
    if not isinstance(target_payload, dict):
        raise ValueError(
            "Confirmatory protocol preflight failed: target object is missing or invalid "
            f"in '{protocol_path}'."
        )
    mapping_version = str(target_payload.get("mapping_version", "")).strip()
    expected_hash = str(target_payload.get("mapping_hash", "")).strip().lower()
    if not mapping_version:
        raise ValueError(
            "Confirmatory protocol preflight failed: target.mapping_version must be present "
            f"in '{protocol_path}'."
        )
    mapping_path = _resolve_target_mapping_path(mapping_version)
    if not mapping_path.exists():
        raise ValueError(
            "Confirmatory protocol preflight failed: target mapping file was not found: "
            f"{mapping_path}"
        )
    actual_hash = _file_sha256(mapping_path)
    if actual_hash != expected_hash:
        raise ValueError(
            "Confirmatory protocol preflight failed: target.mapping_hash mismatch. "
            f"expected={expected_hash}, actual={actual_hash}, mapping_file='{mapping_path}'."
        )
    return {
        "mapping_version": mapping_version,
        "mapping_file": str(mapping_path.resolve()),
        "mapping_hash": actual_hash,
    }


def _validate_schema_node(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str,
    errors: list[str],
) -> None:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        type_valid = {
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
        }.get(schema_type, True)
        if not type_valid:
            errors.append(
                f"{path}: expected type '{schema_type}', got '{type(value).__name__}'."
            )
            return

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected const value {schema['const']!r}, got {value!r}.")

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path}: value {value!r} is not in enum {enum_values!r}.")

    pattern = schema.get("pattern")
    if isinstance(pattern, str) and isinstance(value, str):
        if re.fullmatch(pattern, value) is None:
            errors.append(f"{path}: value {value!r} does not match pattern {pattern!r}.")

    minimum = schema.get("minimum")
    if isinstance(minimum, (int, float)) and isinstance(value, (int, float)):
        if float(value) < float(minimum):
            errors.append(f"{path}: value {value!r} is below minimum {minimum!r}.")

    if isinstance(value, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    errors.append(f"{path}: missing required property '{key}'.")

        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key not in value or not isinstance(child_schema, dict):
                    continue
                _validate_schema_node(
                    value[key],
                    child_schema,
                    path=f"{path}.{key}",
                    errors=errors,
                )

            if schema.get("additionalProperties") is False:
                allowed_keys = set(properties.keys())
                for key in value:
                    if key not in allowed_keys:
                        errors.append(f"{path}: unexpected property '{key}'.")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"{path}: expected at least {min_items} items, got {len(value)}.")

        if schema.get("uniqueItems") is True:
            seen: set[str] = set()
            for idx, item in enumerate(value):
                marker = json.dumps(item, sort_keys=True, separators=(",", ":"))
                if marker in seen:
                    errors.append(f"{path}[{idx}]: duplicate item not allowed.")
                seen.add(marker)

        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                _validate_schema_node(
                    item,
                    item_schema,
                    path=f"{path}[{idx}]",
                    errors=errors,
                )


def validate_confirmatory_freeze_preflight(
    protocol_path: Path | str,
    *,
    schema_path: Path | str = DEFAULT_CONFIRMATORY_PROTOCOL_SCHEMA_PATH,
) -> dict[str, Any]:
    resolved_protocol_path = Path(protocol_path)
    resolved_schema_path = Path(schema_path)
    if not resolved_schema_path.exists():
        raise FileNotFoundError(
            f"Confirmatory schema file was not found: {resolved_schema_path}"
        )

    payload = _load_json_object(resolved_protocol_path)
    schema_payload = _load_json_object(resolved_schema_path)

    errors: list[str] = []
    _validate_schema_node(
        payload,
        schema_payload,
        path="$",
        errors=errors,
    )
    if errors:
        preview = "\n".join(f"- {line}" for line in errors[:20])
        raise ValueError(
            "Confirmatory protocol preflight schema validation failed for "
            f"'{resolved_protocol_path}':\n{preview}"
        )
    _require_locked_analysis_status(payload, resolved_protocol_path)
    mapping_validation = _require_target_mapping_hash_match(payload, resolved_protocol_path)
    payload["__mapping_validation"] = mapping_validation
    return payload


def build_confirmatory_lock_context(
    payload: dict[str, Any],
    *,
    source_path: Path,
) -> dict[str, Any]:
    target = dict(payload.get("target", {}))
    primary_analysis = dict(payload.get("primary_analysis", {}))
    dataset_contract = dict(payload.get("dataset_contract", {}))
    controls = dict(payload.get("controls", {}))
    subgroups = dict(payload.get("subgroups", {}))
    multiplicity = dict(payload.get("multiplicity", {}))
    interpretation_limits = dict(payload.get("interpretation_limits", {}))
    reporting = dict(payload.get("reporting", {}))
    mapping_validation = dict(payload.get("__mapping_validation", {}))

    return {
        "source_protocol_path": str(source_path.resolve()),
        "protocol_id": str(payload.get("protocol_id")),
        "analysis_status": str(payload.get("analysis_status")),
        "target_name": str(target.get("name")),
        "target_source_column": str(target.get("source_column")),
        "target_mapping_version": str(target.get("mapping_version")),
        "target_mapping_hash": str(target.get("mapping_hash")),
        "target_mapping_file": str(mapping_validation.get("mapping_file", "")),
        "target_mapping_hash_verified": str(mapping_validation.get("mapping_hash", "")),
        "split": str(primary_analysis.get("split")),
        "primary_metric": str(primary_analysis.get("metric")),
        "model_family": str(primary_analysis.get("model_family")),
        "hyperparameter_policy": str(primary_analysis.get("hyperparameter_policy")),
        "class_weight_policy": str(primary_analysis.get("class_weight_policy")),
        "dataset_fingerprint_required": bool(
            dataset_contract.get("dataset_fingerprint_required", True)
        ),
        "required_index_columns": [
            str(value) for value in list(dataset_contract.get("allowed_index_columns", []))
        ],
        "minimum_subjects": int(dataset_contract.get("minimum_subjects", 1)),
        "minimum_sessions_per_subject": int(
            dataset_contract.get("minimum_sessions_per_subject", 2)
        ),
        "dummy_baseline_required": bool(controls.get("dummy_baseline", True)),
        "permutation_required": bool(controls.get("permutation_test", True)),
        "minimum_permutations": int(controls.get("n_permutations", 0)),
        "allowed_subgroup_axes": [str(value) for value in list(subgroups.get("allowed", []))],
        "subgroup_interpretation": str(subgroups.get("interpretation", "descriptive_only")),
        "subgroup_min_samples_per_group": int(subgroups.get("min_samples_per_group", 1)),
        "subgroup_min_classes_per_group": int(subgroups.get("min_classes_per_group", 1)),
        "subgroup_report_small_groups": bool(subgroups.get("report_small_groups", False)),
        "multiplicity_primary_hypotheses": int(
            multiplicity.get("primary_hypotheses", 1)
        ),
        "multiplicity_primary_alpha": float(multiplicity.get("primary_alpha", 0.05)),
        "multiplicity_secondary_policy": str(
            multiplicity.get("secondary_policy", "descriptive_only")
        ),
        "multiplicity_exploratory_claims_allowed": bool(
            multiplicity.get("exploratory_claims_allowed", False)
        ),
        "interpretation_limits": {
            str(key): bool(value) for key, value in interpretation_limits.items()
        },
        "reporting_require_protocol_id": bool(reporting.get("require_protocol_id", True)),
        "reporting_require_dataset_fingerprint": bool(
            reporting.get("require_dataset_fingerprint", True)
        ),
        "reporting_require_mapping_hash": bool(reporting.get("require_mapping_hash", True)),
        "reporting_require_interpretation_limits": bool(
            reporting.get("require_interpretation_limits", True)
        ),
        "reporting_require_deviation_log": bool(
            reporting.get("require_deviation_log", True)
        ),
    }


def adapt_confirmatory_freeze_to_thesis_protocol(
    payload: dict[str, Any],
    *,
    source_path: Path,
) -> ThesisProtocol:
    primary_analysis = dict(payload.get("primary_analysis", {}))
    target = dict(payload.get("target", {}))
    dataset_contract = dict(payload.get("dataset_contract", {}))
    controls = dict(payload.get("controls", {}))
    subgroups = dict(payload.get("subgroups", {}))
    secondary_analyses = list(payload.get("secondary_analyses", []))
    confirmatory_lock = build_confirmatory_lock_context(payload, source_path=source_path)

    primary_metric = str(primary_analysis.get("metric", "balanced_accuracy"))
    requested_secondary_metrics = [
        str(value)
        for value in list(primary_analysis.get("secondary_metrics", ["macro_f1", "accuracy"]))
    ]
    secondary_metrics = [
        metric_name
        for metric_name in requested_secondary_metrics
        if metric_name in SUPPORTED_CLASSIFICATION_METRICS and metric_name != primary_metric
    ]
    if not secondary_metrics:
        secondary_metrics = ["macro_f1", "accuracy"]
    secondary_metrics = list(dict.fromkeys(secondary_metrics))
    model_family = str(primary_analysis.get("model_family", "ridge"))
    hyperparameter_policy = str(primary_analysis.get("hyperparameter_policy", "fixed"))
    class_weight_policy = str(primary_analysis.get("class_weight_policy", "none"))
    seed = int(primary_analysis.get("seed", 42))

    methodology_policy_name = (
        "fixed_baselines_only"
        if hyperparameter_policy == "fixed"
        else "grouped_nested_tuning"
    )
    tuning_enabled = methodology_policy_name == "grouped_nested_tuning"

    primary_suite_id = "confirmatory_primary_within_subject"
    secondary_suite_id = "confirmatory_secondary_cross_person_transfer"
    include_secondary = any(
        isinstance(item, dict)
        and str(item.get("name")) == "frozen_cross_person_transfer"
        and str(item.get("status")) in {"secondary", "official_secondary"}
        for item in secondary_analyses
    )
    suite_ids_for_controls = [primary_suite_id]
    if include_secondary:
        suite_ids_for_controls.append(secondary_suite_id)

    adapted_payload: dict[str, Any] = {
        "protocol_schema_version": THESIS_PROTOCOL_SCHEMA_VERSION,
        "framework_mode": "confirmatory",
        "protocol_id": str(payload["protocol_id"]),
        "protocol_version": str(payload["protocol_version"]),
        "status": ProtocolStatus.LOCKED.value,
        "description": "Adapted confirmatory freeze protocol.",
        "confirmatory_lock": confirmatory_lock,
        "notes": (
            "Auto-adapted from confirmatory freeze protocol "
            f"source='{source_path}'"
        ),
        "scientific_contract": {
            "sample_unit": str(payload.get("sample_unit", "beta_event")),
            "target": str(target.get("name", "coarse_affect")),
            "label_policy": (
                f"{target.get('mapping_version', 'unknown')}:"
                f"{target.get('mapping_hash', 'unknown')}"
            ),
            "primary_metric": primary_metric,
            "secondary_metrics": secondary_metrics,
            "seed_policy": {
                "global_seed": seed,
                "per_suite_overrides_allowed": False,
            },
        },
        "split_policy": {
            "primary_mode": str(
                primary_analysis.get("split", "within_subject_loso_session")
            ),
            "secondary_mode": "frozen_cross_person_transfer",
            "grouping_field": "session",
            "transfer_constraints": (
                "secondary analysis is descriptive-only"
                if include_secondary
                else "secondary analysis disabled"
            ),
        },
        "model_policy": {
            "selection_strategy": (
                "fixed_baselines"
                if methodology_policy_name == "fixed_baselines_only"
                else "nested_tuned"
            ),
            "models": [model_family],
            "tuning_enabled": tuning_enabled,
            "nested_grouped_cv": tuning_enabled,
            "class_weight_policy": class_weight_policy,
        },
        "methodology_policy": {
            "policy_name": methodology_policy_name,
            "class_weight_policy": class_weight_policy,
            "tuning_enabled": tuning_enabled,
            "inner_cv_scheme": (
                "grouped_leave_one_group_out" if tuning_enabled else None
            ),
            "inner_group_field": ("session" if tuning_enabled else None),
            "tuning_search_space_id": (f"{model_family}_grouped_nested_v1" if tuning_enabled else None),
            "tuning_search_space_version": ("v1" if tuning_enabled else None),
        },
        "metric_policy": {
            "primary_metric": primary_metric,
            "secondary_metrics": secondary_metrics,
        },
        "subgroup_reporting_policy": {
            "enabled": bool(subgroups.get("enabled", True)),
            "subgroup_dimensions": [
                str(value) for value in list(subgroups.get("allowed", ["subject", "task", "modality"]))
            ],
            "min_samples_per_group": int(subgroups.get("min_samples_per_group", 20)),
        },
        "data_policy": {
            "class_balance": {
                "enabled": True,
                "axes": ["overall", "subject", "session", "task", "modality"],
                "min_class_fraction_warning": 0.05,
                "min_class_fraction_blocking": None,
            },
            "missingness": {
                "enabled": True,
                "max_missing_fraction_warning": 0.1,
                "max_missing_fraction_blocking": None,
            },
            "leakage": {
                "enabled": True,
                "fail_on_duplicate_sample_id": True,
                "warn_on_duplicate_beta_path": True,
                "fail_on_duplicate_beta_path": False,
                "fail_on_subject_overlap_for_transfer": True,
                "fail_on_cv_group_overlap": True,
            },
            "external_validation": {
                "enabled": False,
                "mode": "compatibility_only",
                "require_compatible": False,
                "require_for_official_runs": False,
                "datasets": [],
            },
            "required_index_columns": [
                str(value)
                for value in list(dataset_contract.get("allowed_index_columns", []))
            ],
            "intended_use": (
                "Frozen confirmatory internal validation for thesis_confirmatory_v1."
            ),
            "not_intended_use": [
                "External generalization claims from internal confirmatory runs.",
                "Causal or clinical claims.",
            ],
            "known_limitations": [
                "External validation is compatibility-only in this phase."
            ],
        },
        "evidence_policy": {
            "repeat_evaluation": {
                "repeat_count": 3,
                "seed_stride": 1000,
            },
            "confidence_intervals": {
                "method": "grouped_bootstrap_percentile",
                "confidence_level": 0.95,
                "n_bootstrap": 1000,
                "seed": 2026,
            },
            "paired_comparisons": {
                "method": "paired_sign_flip_permutation",
                "n_permutations": 5000,
                "alpha": 0.05,
                "require_significant_win": False,
            },
            "permutation": {
                "alpha": float(confirmatory_lock["multiplicity_primary_alpha"]),
                "minimum_permutations": int(controls.get("n_permutations", 1000)),
                "require_pass_for_validity": False,
            },
            "calibration": {
                "enabled": True,
                "n_bins": 10,
                "require_probabilities_for_validity": False,
            },
            "required_package": {
                "require_dummy_baseline": bool(controls.get("dummy_baseline", True)),
                "require_permutation_control": bool(controls.get("permutation_test", True)),
                "require_untuned_baseline_if_tuning": bool(tuning_enabled),
            },
        },
        "control_policy": {
            "dummy_baseline": {
                "enabled": bool(controls.get("dummy_baseline", True)),
                "suites": list(suite_ids_for_controls),
            },
            "permutation": {
                "enabled": bool(controls.get("permutation_test", True)),
                "metric": primary_metric,
                "n_permutations": int(controls.get("n_permutations", 1000)),
                "suites": list(suite_ids_for_controls),
            },
        },
        "interpretability_policy": {
            "enabled": False,
            "suites": [],
            "modes": [],
            "models": [],
            "supporting_evidence_only": True,
        },
        "sensitivity_policy": {
            "role": (
                "official_secondary_analyses" if include_secondary else "exploratory_only"
            ),
            "suites": ([secondary_suite_id] if include_secondary else []),
        },
        "artifact_contract": ArtifactContract().model_dump(mode="json"),
        "official_run_suites": [
            {
                "suite_id": primary_suite_id,
                "description": "Primary confirmatory within-subject evaluation.",
                "enabled": True,
                "suite_type": "primary",
                "claim_ids": ["confirmatory_primary_claim"],
                "split_mode": "within_subject_loso_session",
                "models": [model_family],
                "subject_source": "all_from_index",
                "subjects": [],
                "transfer_pair_source": "all_ordered_pairs_from_index",
                "transfer_pairs": [],
                "controls_required": False,
                "interpretability_requested": False,
            }
        ],
    }

    if include_secondary:
        adapted_payload["official_run_suites"].append(
            {
                "suite_id": secondary_suite_id,
                "description": "Secondary frozen cross-person transfer evaluation.",
                "enabled": True,
                "suite_type": "secondary",
                "claim_ids": ["confirmatory_secondary_transfer_claim"],
                "split_mode": "frozen_cross_person_transfer",
                "models": [model_family],
                "subject_source": "all_from_index",
                "subjects": [],
                "transfer_pair_source": "all_ordered_pairs_from_index",
                "transfer_pairs": [],
                "controls_required": False,
                "interpretability_requested": False,
            }
        )

    return ThesisProtocol.model_validate(adapted_payload)


__all__ = [
    "CONFIRMATORY_FREEZE_PROTOCOL_ID",
    "adapt_confirmatory_freeze_to_thesis_protocol",
    "build_confirmatory_lock_context",
    "validate_confirmatory_freeze_preflight",
]
