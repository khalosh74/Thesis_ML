import json
from dataclasses import dataclass, field
from pathlib import Path

from Thesis_ML.protocols.claim_evaluator import evaluate_claim_outcomes
from Thesis_ML.protocols.loader import load_protocol
from Thesis_ML.protocols.models import ClaimCategory, ClaimRole, ProtocolRunResult


@dataclass
class _RunControlsStub:
    dummy_baseline_run: bool = False


@dataclass
class _RunSpecStub:
    run_id: str
    suite_id: str
    model: str
    cv_mode: str = "within_subject_loso_session"
    subject: str | None = "sub-001"
    train_subject: str | None = None
    test_subject: str | None = None
    repeat_id: int = 1
    repeat_count: int = 1
    evidence_run_role: str = "primary"
    controls: _RunControlsStub = field(default_factory=_RunControlsStub)


@dataclass
class _ClaimManifestStub:
    claim_to_run_map: dict[str, list[str]]
    runs: list[_RunSpecStub]


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _reporting_contract() -> dict[str, object]:
    return {
        "controls_status": {"controls_valid_for_confirmatory": True},
        "deviations_from_protocol": {"science_critical_deviation_detected": False},
    }


def _load_protocol_and_claims() -> tuple[object, object, object]:
    protocol = load_protocol(Path("configs/protocols/thesis_canonical_nested_v2.json"))
    primary_claim = next(
        claim
        for claim in protocol.claims
        if claim.claim_id == protocol.success_criteria.primary_claim_id
    )
    control_claim = next(
        claim
        for claim in protocol.claims
        if (
            claim.role == ClaimRole.SUPPORTING
            and claim.category == ClaimCategory.CONTROL_EVIDENCE
            and claim.estimand_scope == primary_claim.estimand_scope
        )
    )
    return protocol, primary_claim, control_claim


def _run_result(run_id: str, suite_id: str, metrics_path: str) -> ProtocolRunResult:
    return ProtocolRunResult.model_validate(
        {
            "run_id": run_id,
            "suite_id": suite_id,
            "status": "success",
            "metrics_path": metrics_path,
        }
    )


def _primary_entry(payload: dict) -> dict:
    return next(entry for entry in payload["claims"] if entry["claim_id"] == payload["primary_claim_id"])


def _manifest(primary_claim: object, control_claim: object, runs: list[_RunSpecStub]) -> _ClaimManifestStub:
    control_run_ids = [
        run.run_id for run in runs if run.suite_id == control_claim.suite_ids[0]
    ]
    primary_run_ids = [
        run.run_id for run in runs if run.suite_id == primary_claim.suite_ids[0]
    ]
    return _ClaimManifestStub(
        claim_to_run_map={
            primary_claim.claim_id: primary_run_ids,
            control_claim.claim_id: control_run_ids,
        },
        runs=runs,
    )


def test_supported_when_matched_dummy_exists_and_model_strictly_beats_dummy(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "primary_sub001_r1"
    dummy_run_id = "dummy_sub001_r1"

    manifest = _manifest(
        primary_claim,
        control_claim,
        [
            _RunSpecStub(
                run_id=primary_run_id,
                suite_id=primary_claim.suite_ids[0],
                model="ridge",
            ),
            _RunSpecStub(
                run_id=dummy_run_id,
                suite_id=control_claim.suite_ids[0],
                model="dummy",
                controls=_RunControlsStub(dummy_baseline_run=True),
            ),
        ],
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                _write_json(tmp_path / "primary_supported.json", {"primary_metric_value": 0.71}),
            ),
            _run_result(
                dummy_run_id,
                control_claim.suite_ids[0],
                _write_json(tmp_path / "dummy_supported.json", {"primary_metric_value": 0.50}),
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "supported"
    summary = _primary_entry(payload)["control_evidence_summary"]
    assert summary["compared_primary_run_ids"] == [primary_run_id]
    assert summary["matched_dummy_run_ids"] == [dummy_run_id]
    assert summary["baseline_fail_run_ids"] == []


def test_not_supported_when_model_equals_dummy(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "primary_equal_dummy"
    dummy_run_id = "dummy_equal_dummy"
    manifest = _manifest(
        primary_claim,
        control_claim,
        [
            _RunSpecStub(run_id=primary_run_id, suite_id=primary_claim.suite_ids[0], model="ridge"),
            _RunSpecStub(
                run_id=dummy_run_id,
                suite_id=control_claim.suite_ids[0],
                model="dummy",
                controls=_RunControlsStub(dummy_baseline_run=True),
            ),
        ],
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                _write_json(tmp_path / "primary_equal.json", {"primary_metric_value": 0.60}),
            ),
            _run_result(
                dummy_run_id,
                control_claim.suite_ids[0],
                _write_json(tmp_path / "dummy_equal.json", {"primary_metric_value": 0.60}),
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "not_supported"
    assert _primary_entry(payload)["control_evidence_summary"]["baseline_fail_run_ids"] == [
        primary_run_id
    ]


def test_not_supported_when_model_is_below_dummy(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "primary_below_dummy"
    dummy_run_id = "dummy_below_dummy"
    manifest = _manifest(
        primary_claim,
        control_claim,
        [
            _RunSpecStub(run_id=primary_run_id, suite_id=primary_claim.suite_ids[0], model="ridge"),
            _RunSpecStub(
                run_id=dummy_run_id,
                suite_id=control_claim.suite_ids[0],
                model="dummy",
                controls=_RunControlsStub(dummy_baseline_run=True),
            ),
        ],
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                _write_json(tmp_path / "primary_below.json", {"primary_metric_value": 0.55}),
            ),
            _run_result(
                dummy_run_id,
                control_claim.suite_ids[0],
                _write_json(tmp_path / "dummy_below.json", {"primary_metric_value": 0.70}),
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "not_supported"
    assert _primary_entry(payload)["control_evidence_summary"]["baseline_fail_run_ids"] == [
        primary_run_id
    ]


def test_invalid_when_dummy_baseline_evidence_required_but_missing(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "primary_missing_dummy"
    manifest = _manifest(
        primary_claim,
        control_claim,
        [_RunSpecStub(run_id=primary_run_id, suite_id=primary_claim.suite_ids[0], model="ridge")],
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                _write_json(tmp_path / "primary_missing_dummy.json", {"primary_metric_value": 0.80}),
            )
        ],
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "invalid"
    assert _primary_entry(payload)["control_evidence_summary"]["missing_dummy_runs"] == [primary_run_id]


def test_regression_unrelated_dummy_runs_do_not_satisfy_requirement(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "primary_sub001"
    unrelated_dummy_run_id = "dummy_sub999"
    manifest = _manifest(
        primary_claim,
        control_claim,
        [
            _RunSpecStub(
                run_id=primary_run_id,
                suite_id=primary_claim.suite_ids[0],
                model="ridge",
                subject="sub-001",
            ),
            _RunSpecStub(
                run_id=unrelated_dummy_run_id,
                suite_id=control_claim.suite_ids[0],
                model="dummy",
                subject="sub-999",
                controls=_RunControlsStub(dummy_baseline_run=True),
            ),
        ],
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                _write_json(tmp_path / "primary_unrelated_dummy.json", {"primary_metric_value": 0.75}),
            ),
            _run_result(
                unrelated_dummy_run_id,
                control_claim.suite_ids[0],
                _write_json(tmp_path / "dummy_unrelated_dummy.json", {"primary_metric_value": 0.10}),
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "invalid"
    assert _primary_entry(payload)["control_evidence_summary"]["missing_dummy_runs"] == [primary_run_id]


def test_regression_untuned_baseline_runs_do_not_count_as_primary_claim_evidence(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    untuned_run_id = "primary_untuned"
    dummy_run_id = "dummy_present"
    manifest = _manifest(
        primary_claim,
        control_claim,
        [
            _RunSpecStub(
                run_id=untuned_run_id,
                suite_id=primary_claim.suite_ids[0],
                model="ridge",
                evidence_run_role="untuned_baseline",
            ),
            _RunSpecStub(
                run_id=dummy_run_id,
                suite_id=control_claim.suite_ids[0],
                model="dummy",
                controls=_RunControlsStub(dummy_baseline_run=True),
            ),
        ],
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                untuned_run_id,
                primary_claim.suite_ids[0],
                _write_json(tmp_path / "untuned_primary.json", {"primary_metric_value": 0.99}),
            ),
            _run_result(
                dummy_run_id,
                control_claim.suite_ids[0],
                _write_json(tmp_path / "untuned_dummy.json", {"primary_metric_value": 0.10}),
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "inconclusive"
