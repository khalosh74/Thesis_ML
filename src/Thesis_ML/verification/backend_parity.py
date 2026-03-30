from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.base import clone

ParityCategory = Literal["exact", "near_exact"]


@dataclass(frozen=True)
class ParityResult:
    passed: bool
    category: ParityCategory
    agreement: float | None
    max_abs_difference: float | None
    mismatches: tuple[str, ...]
    timing_metadata: dict[str, Any]


_TIMING_SUFFIXES = (
    "_seconds",
    "_time",
    "_ms",
)
_TIMING_KEYWORDS = (
    "elapsed",
    "latency",
    "duration",
    "timing",
)


def _is_timing_key(key: str) -> bool:
    normalized = str(key).strip().lower()
    if not normalized:
        return False
    if normalized.endswith(_TIMING_SUFFIXES):
        return True
    return any(token in normalized for token in _TIMING_KEYWORDS)


def _to_numpy(value: Any) -> np.ndarray:
    return np.asarray(value)


def _safe_max_abs_diff(left: Any, right: Any) -> float | None:
    left_arr = _to_numpy(left)
    right_arr = _to_numpy(right)
    if left_arr.shape != right_arr.shape:
        return None
    if left_arr.size == 0:
        return 0.0
    try:
        return float(np.max(np.abs(left_arr.astype(np.float64) - right_arr.astype(np.float64))))
    except Exception:
        return None


def _prediction_agreement(left: Any, right: Any) -> float | None:
    left_arr = _to_numpy(left).astype(str, copy=False)
    right_arr = _to_numpy(right).astype(str, copy=False)
    if left_arr.shape != right_arr.shape:
        return None
    if left_arr.size == 0:
        return 1.0
    return float(np.mean(left_arr == right_arr))


def _parse_best_params(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return {str(key): item for key, item in payload.items()}
    return None


def _timing_metadata_from_dict(payload: dict[str, Any]) -> dict[str, Any]:
    timing: dict[str, Any] = {}
    for key, value in payload.items():
        if _is_timing_key(str(key)):
            timing[str(key)] = value
    return timing


def compare_estimator_fit_predict_parity(
    *,
    reference_estimator: Any,
    candidate_estimator: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    category: ParityCategory = "exact",
    near_exact_prediction_agreement: float = 0.999,
    near_exact_score_tolerance: float = 1e-9,
    near_exact_proba_tolerance: float = 1e-9,
) -> ParityResult:
    reference = clone(reference_estimator)
    candidate = clone(candidate_estimator)

    reference.fit(x_train, y_train)
    candidate.fit(x_train, y_train)

    mismatches: list[str] = []
    max_abs_difference = 0.0

    reference_pred = np.asarray(reference.predict(x_eval)).astype(str, copy=False)
    candidate_pred = np.asarray(candidate.predict(x_eval)).astype(str, copy=False)
    agreement = _prediction_agreement(reference_pred, candidate_pred)
    if agreement is None:
        mismatches.append("prediction_shape_mismatch")
    elif category == "exact" and agreement != 1.0:
        mismatches.append("prediction_values_mismatch")
    elif category == "near_exact" and agreement < float(near_exact_prediction_agreement):
        mismatches.append("prediction_agreement_below_threshold")

    if hasattr(reference, "decision_function") and hasattr(candidate, "decision_function"):
        reference_scores = np.asarray(reference.decision_function(x_eval), dtype=np.float64)
        candidate_scores = np.asarray(candidate.decision_function(x_eval), dtype=np.float64)
        score_diff = _safe_max_abs_diff(reference_scores, candidate_scores)
        if score_diff is None:
            mismatches.append("decision_score_shape_mismatch")
        else:
            max_abs_difference = max(max_abs_difference, float(score_diff))
            threshold = 0.0 if category == "exact" else float(near_exact_score_tolerance)
            if float(score_diff) > threshold:
                mismatches.append("decision_scores_exceed_tolerance")

    if hasattr(reference, "predict_proba") and hasattr(candidate, "predict_proba"):
        reference_proba = np.asarray(reference.predict_proba(x_eval), dtype=np.float64)
        candidate_proba = np.asarray(candidate.predict_proba(x_eval), dtype=np.float64)
        proba_diff = _safe_max_abs_diff(reference_proba, candidate_proba)
        if proba_diff is None:
            mismatches.append("predict_proba_shape_mismatch")
        else:
            max_abs_difference = max(max_abs_difference, float(proba_diff))
            threshold = 0.0 if category == "exact" else float(near_exact_proba_tolerance)
            if float(proba_diff) > threshold:
                mismatches.append("predict_proba_exceed_tolerance")

    reference_runtime = {}
    candidate_runtime = {}
    if hasattr(reference, "get_backend_runtime_metadata"):
        metadata = reference.get_backend_runtime_metadata()
        if isinstance(metadata, dict):
            reference_runtime = _timing_metadata_from_dict(metadata)
    if hasattr(candidate, "get_backend_runtime_metadata"):
        metadata = candidate.get_backend_runtime_metadata()
        if isinstance(metadata, dict):
            candidate_runtime = _timing_metadata_from_dict(metadata)

    return ParityResult(
        passed=not mismatches,
        category=category,
        agreement=(float(agreement) if agreement is not None else None),
        max_abs_difference=float(max_abs_difference),
        mismatches=tuple(mismatches),
        timing_metadata={
            "reference": reference_runtime,
            "candidate": candidate_runtime,
        },
    )


def compare_tuning_parity(
    *,
    reference_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    category: ParityCategory = "exact",
    near_exact_score_tolerance: float = 1e-12,
) -> ParityResult:
    mismatches: list[str] = []

    reference_best_params = _parse_best_params(
        reference_payload.get("best_params", reference_payload.get("best_params_json"))
    )
    candidate_best_params = _parse_best_params(
        candidate_payload.get("best_params", candidate_payload.get("best_params_json"))
    )
    if reference_best_params != candidate_best_params:
        mismatches.append("best_params_mismatch")

    reference_score = reference_payload.get("best_score")
    candidate_score = candidate_payload.get("best_score")
    max_abs_difference = 0.0
    if isinstance(reference_score, (int, float)) and isinstance(candidate_score, (int, float)):
        score_diff = abs(float(reference_score) - float(candidate_score))
        max_abs_difference = max(max_abs_difference, score_diff)
        threshold = 0.0 if category == "exact" else float(near_exact_score_tolerance)
        if score_diff > threshold:
            mismatches.append("best_score_exceeds_tolerance")

    reference_cv = reference_payload.get("cv_results")
    candidate_cv = candidate_payload.get("cv_results")
    if isinstance(reference_cv, dict) and isinstance(candidate_cv, dict):
        reference_mean = np.asarray(reference_cv.get("mean_test_score", []), dtype=np.float64)
        candidate_mean = np.asarray(candidate_cv.get("mean_test_score", []), dtype=np.float64)
        cv_diff = _safe_max_abs_diff(reference_mean, candidate_mean)
        if cv_diff is None:
            mismatches.append("cv_mean_test_score_shape_mismatch")
        else:
            max_abs_difference = max(max_abs_difference, float(cv_diff))
            threshold = 0.0 if category == "exact" else float(near_exact_score_tolerance)
            if float(cv_diff) > threshold:
                mismatches.append("cv_mean_test_score_exceeds_tolerance")

    return ParityResult(
        passed=not mismatches,
        category=category,
        agreement=1.0 if not mismatches else 0.0,
        max_abs_difference=float(max_abs_difference),
        mismatches=tuple(mismatches),
        timing_metadata={
            "reference": _timing_metadata_from_dict(reference_payload),
            "candidate": _timing_metadata_from_dict(candidate_payload),
        },
    )


def compare_permutation_parity(
    *,
    reference_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    category: ParityCategory = "exact",
    near_exact_score_tolerance: float = 1e-12,
) -> ParityResult:
    mismatches: list[str] = []
    max_abs_difference = 0.0

    for key in (
        "metric_name",
        "primary_metric_aggregation",
        "n_permutations",
        "observed_score",
        "p_value",
    ):
        if key not in reference_payload or key not in candidate_payload:
            continue
        if isinstance(reference_payload[key], (int, float)) and isinstance(
            candidate_payload[key], (int, float)
        ):
            diff = abs(float(reference_payload[key]) - float(candidate_payload[key]))
            max_abs_difference = max(max_abs_difference, diff)
            threshold = 0.0 if category == "exact" else float(near_exact_score_tolerance)
            if diff > threshold:
                mismatches.append(f"{key}_exceeds_tolerance")
        elif reference_payload[key] != candidate_payload[key]:
            mismatches.append(f"{key}_mismatch")

    reference_null_scores = np.asarray(reference_payload.get("null_scores", []), dtype=np.float64)
    candidate_null_scores = np.asarray(candidate_payload.get("null_scores", []), dtype=np.float64)
    null_diff = _safe_max_abs_diff(reference_null_scores, candidate_null_scores)
    if null_diff is None:
        mismatches.append("null_scores_shape_mismatch")
    else:
        max_abs_difference = max(max_abs_difference, float(null_diff))
        threshold = 0.0 if category == "exact" else float(near_exact_score_tolerance)
        if float(null_diff) > threshold:
            mismatches.append("null_scores_exceed_tolerance")

    return ParityResult(
        passed=not mismatches,
        category=category,
        agreement=1.0 if not mismatches else 0.0,
        max_abs_difference=float(max_abs_difference),
        mismatches=tuple(mismatches),
        timing_metadata={
            "reference": _timing_metadata_from_dict(reference_payload),
            "candidate": _timing_metadata_from_dict(candidate_payload),
        },
    )


def compare_tuned_null_parity(
    *,
    reference_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    category: ParityCategory = "exact",
    near_exact_score_tolerance: float = 1e-12,
) -> ParityResult:
    base_result = compare_permutation_parity(
        reference_payload=reference_payload,
        candidate_payload=candidate_payload,
        category=category,
        near_exact_score_tolerance=near_exact_score_tolerance,
    )
    mismatches = list(base_result.mismatches)

    for key in (
        "tuning_reapplied_under_null",
        "null_matches_confirmatory_setup",
        "null_tuning_search_space_id",
        "null_tuning_search_space_version",
        "null_inner_cv_scheme",
        "null_inner_group_field",
    ):
        if key in reference_payload or key in candidate_payload:
            if reference_payload.get(key) != candidate_payload.get(key):
                mismatches.append(f"{key}_mismatch")

    return ParityResult(
        passed=not mismatches,
        category=category,
        agreement=base_result.agreement,
        max_abs_difference=base_result.max_abs_difference,
        mismatches=tuple(mismatches),
        timing_metadata=base_result.timing_metadata,
    )


__all__ = [
    "ParityCategory",
    "ParityResult",
    "compare_estimator_fit_predict_parity",
    "compare_permutation_parity",
    "compare_tuned_null_parity",
    "compare_tuning_parity",
]
