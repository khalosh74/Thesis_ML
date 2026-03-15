from __future__ import annotations

import numpy as np
import pytest

from Thesis_ML.config.metric_policy import (
    classification_metric_score,
    enforce_primary_metric_alignment,
    extract_metric_value,
    metric_bundle,
    metric_higher_is_better,
    metric_scorer,
    resolve_effective_metric_policy,
    validate_metric_name,
)
from Thesis_ML.orchestration.contracts import ExperimentSpec, SearchSpaceSpec
from Thesis_ML.orchestration.result_aggregation import aggregate_variant_records


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("balanced_accuracy", "balanced_accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("macro_f1", "macro_f1"),
        ("macro f1", "macro_f1"),
        ("accuracy", "accuracy"),
    ],
)
def test_metric_name_validation_normalizes_supported_aliases(
    raw_name: str,
    expected: str,
) -> None:
    assert validate_metric_name(raw_name) == expected


def test_metric_registry_resolves_scores_scorers_and_direction_consistently() -> None:
    y_true = np.asarray(["neg", "pos", "neg", "pos"])
    y_pred = np.asarray(["neg", "neg", "neg", "pos"])

    for metric_name in ("balanced_accuracy", "macro_f1", "accuracy"):
        score = classification_metric_score(y_true, y_pred, metric_name)
        bundle_score = metric_bundle(y_true, y_pred, metric_names=[metric_name])[metric_name]
        scorer = metric_scorer(metric_name)

        assert bundle_score == pytest.approx(score)
        assert scorer._score_func(y_true, y_pred) == pytest.approx(score)
        assert metric_higher_is_better(metric_name) is True


def test_effective_metric_policy_alignment_rejects_drift() -> None:
    effective = resolve_effective_metric_policy(
        primary_metric="balanced_accuracy",
        secondary_metrics=["macro_f1", "accuracy"],
        decision_metric="macro_f1",
        tuning_metric="balanced_accuracy",
        permutation_metric="balanced_accuracy",
    )
    with pytest.raises(ValueError, match="metric policy drift detected"):
        enforce_primary_metric_alignment(effective, context="unit-test")


def test_extract_metric_value_requires_explicit_metric_presence() -> None:
    payload = {"primary_metric_name": "balanced_accuracy", "primary_metric_value": 0.73}
    assert extract_metric_value(payload, "balanced_accuracy", require=True) == pytest.approx(0.73)

    with pytest.raises(ValueError, match="missing required metric 'balanced_accuracy'"):
        extract_metric_value({}, "balanced_accuracy", require=True, payload_label="run metrics")


def test_contract_metric_fields_reject_invalid_metric_names() -> None:
    with pytest.raises(ValueError, match="Unsupported metric"):
        ExperimentSpec.model_validate(
            {
                "experiment_id": "E-metric-invalid",
                "title": "Invalid metric experiment",
                "stage": "Stage 1",
                "primary_metric": "invalid_metric",
            }
        )

    with pytest.raises(ValueError, match="Unsupported metric"):
        SearchSpaceSpec.model_validate(
            {
                "search_space_id": "SS-invalid-metric",
                "enabled": False,
                "objective_metric": "invalid_metric",
                "dimensions": [],
            }
        )


def test_aggregation_rejects_missing_primary_metric_name_without_fallback() -> None:
    with pytest.raises(ValueError, match="missing required primary_metric_name"):
        aggregate_variant_records(
            [
                {
                    "status": "completed",
                    "run_id": "run_missing_metric_name",
                    "model": "ridge",
                    "cv": "within_subject_loso_session",
                    "target": "coarse_affect",
                    "primary_metric_value": 0.61,
                }
            ],
            top_k=1,
        )
