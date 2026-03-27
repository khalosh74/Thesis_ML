import json
from dataclasses import dataclass
from pathlib import Path

from Thesis_ML.protocols.claim_evaluator import evaluate_claim_outcomes
from Thesis_ML.protocols.loader import load_protocol
from Thesis_ML.protocols.models import ProtocolRunResult


@dataclass
class _ClaimManifestStub:
    claim_to_run_map: dict[str, list[str]]


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_evaluate_claim_outcomes_primary_supported(tmp_path):
    protocol = load_protocol(Path("configs/protocols/thesis_canonical_v1.json"))

    primary_claim_id = protocol.success_criteria.primary_claim_id
    primary_claim = next(c for c in protocol.claims if c.claim_id == primary_claim_id)
    primary_suite_id = primary_claim.suite_ids[0]
    run_id = "run-primary-supported"

    metrics_path = _write_json(
        tmp_path / "metrics_supported.json",
        {
            "primary_metric_value": 0.71,
            "permutation_test": {
                "passes_threshold": True,
            },
        },
    )

    compiled_manifest = _ClaimManifestStub(
        claim_to_run_map={
            primary_claim_id: [run_id],
        }
    )

    run_results = [
        ProtocolRunResult.model_validate(
            {
                "run_id": run_id,
                "suite_id": primary_suite_id,
                "status": "completed",
                "metrics_path": metrics_path,
            }
        )
    ]

    reporting_contract = {
        "controls_status": {
            "controls_valid_for_confirmatory": True,
        },
        "deviations_from_protocol": {
            "science_critical_deviation_detected": False,
        },
    }

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=compiled_manifest,
        run_results=run_results,
        reporting_contract=reporting_contract,
    )

    assert payload["primary_claim_verdict"] == "supported"


def test_evaluate_claim_outcomes_primary_not_supported(tmp_path):
    protocol = load_protocol(Path("configs/protocols/thesis_canonical_v1.json"))

    primary_claim_id = protocol.success_criteria.primary_claim_id
    primary_claim = next(c for c in protocol.claims if c.claim_id == primary_claim_id)
    primary_suite_id = primary_claim.suite_ids[0]
    run_id = "run-primary-not-supported"

    metrics_path = _write_json(
        tmp_path / "metrics_not_supported.json",
        {
            "primary_metric_value": 0.61,
            "permutation_test": {
                "passes_threshold": False,
            },
        },
    )

    compiled_manifest = _ClaimManifestStub(
        claim_to_run_map={
            primary_claim_id: [run_id],
        }
    )

    run_results = [
        ProtocolRunResult.model_validate(
            {
                "run_id": run_id,
                "suite_id": primary_suite_id,
                "status": "completed",
                "metrics_path": metrics_path,
            }
        )
    ]

    reporting_contract = {
        "controls_status": {
            "controls_valid_for_confirmatory": True,
        },
        "deviations_from_protocol": {
            "science_critical_deviation_detected": False,
        },
    }

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=compiled_manifest,
        run_results=run_results,
        reporting_contract=reporting_contract,
    )

    assert payload["primary_claim_verdict"] == "not_supported"


def test_evaluate_claim_outcomes_primary_inconclusive(tmp_path):
    protocol = load_protocol(Path("configs/protocols/thesis_canonical_v1.json"))

    primary_claim_id = protocol.success_criteria.primary_claim_id

    compiled_manifest = _ClaimManifestStub(
        claim_to_run_map={
            primary_claim_id: [],
        }
    )

    run_results = []

    reporting_contract = {
        "controls_status": {
            "controls_valid_for_confirmatory": True,
        },
        "deviations_from_protocol": {
            "science_critical_deviation_detected": False,
        },
    }

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=compiled_manifest,
        run_results=run_results,
        reporting_contract=reporting_contract,
    )

    assert payload["primary_claim_verdict"] == "inconclusive"


def test_evaluate_claim_outcomes_primary_invalid_missing_controls(tmp_path):
    protocol = load_protocol(Path("configs/protocols/thesis_canonical_v1.json"))

    primary_claim_id = protocol.success_criteria.primary_claim_id
    primary_claim = next(c for c in protocol.claims if c.claim_id == primary_claim_id)
    primary_suite_id = primary_claim.suite_ids[0]
    run_id = "run-primary-invalid"

    metrics_path = _write_json(
        tmp_path / "metrics_invalid.json",
        {
            "primary_metric_value": 0.73,
            "permutation_test": {
                "passes_threshold": True,
            },
        },
    )

    compiled_manifest = _ClaimManifestStub(
        claim_to_run_map={
            primary_claim_id: [run_id],
        }
    )

    run_results = [
        ProtocolRunResult.model_validate(
            {
                "run_id": run_id,
                "suite_id": primary_suite_id,
                "status": "completed",
                "metrics_path": metrics_path,
            }
        )
    ]

    reporting_contract = {
        "controls_status": {
            "controls_valid_for_confirmatory": False,
        },
        "deviations_from_protocol": {
            "science_critical_deviation_detected": False,
        },
    }

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=compiled_manifest,
        run_results=run_results,
        reporting_contract=reporting_contract,
    )

    assert payload["primary_claim_verdict"] == "invalid"


def test_evaluate_claim_outcomes_secondary_transfer_never_primary(tmp_path):
    protocol = load_protocol(Path("configs/protocols/thesis_canonical_v1.json"))

    primary_claim_id = protocol.success_criteria.primary_claim_id
    primary_claim = next(c for c in protocol.claims if c.claim_id == primary_claim_id)
    secondary_claim = next(
        c for c in protocol.claims if c.category.value == "cross_person_transfer"
    )

    primary_run_id = "run-primary"
    secondary_run_id = "run-secondary"

    primary_metrics_path = _write_json(
        tmp_path / "metrics_primary.json",
        {
            "primary_metric_value": 0.70,
            "permutation_test": {
                "passes_threshold": False,
            },
        },
    )

    secondary_metrics_path = _write_json(
        tmp_path / "metrics_secondary.json",
        {
            "primary_metric_value": 0.99,
            "permutation_test": {
                "passes_threshold": True,
            },
        },
    )

    compiled_manifest = _ClaimManifestStub(
        claim_to_run_map={
            primary_claim.claim_id: [primary_run_id],
            secondary_claim.claim_id: [secondary_run_id],
        }
    )

    run_results = [
        ProtocolRunResult.model_validate(
            {
                "run_id": primary_run_id,
                "suite_id": primary_claim.suite_ids[0],
                "status": "completed",
                "metrics_path": primary_metrics_path,
            }
        ),
        ProtocolRunResult.model_validate(
            {
                "run_id": secondary_run_id,
                "suite_id": secondary_claim.suite_ids[0],
                "status": "completed",
                "metrics_path": secondary_metrics_path,
            }
        ),
    ]

    reporting_contract = {
        "controls_status": {
            "controls_valid_for_confirmatory": True,
        },
        "deviations_from_protocol": {
            "science_critical_deviation_detected": False,
        },
    }

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=compiled_manifest,
        run_results=run_results,
        reporting_contract=reporting_contract,
    )

    assert payload["primary_claim_verdict"] == "not_supported"

    secondary_entry = next(
        entry for entry in payload["claims"] if entry["claim_id"] == secondary_claim.claim_id
    )
    assert secondary_entry["verdict"] != "supported"
