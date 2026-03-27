from __future__ import annotations

from pathlib import Path

import pytest

from Thesis_ML.comparisons.compiler import compile_comparison
from Thesis_ML.comparisons.loader import load_comparison_spec
from Thesis_ML.experiments.backend_registry import resolve_backend_support
from Thesis_ML.experiments.comparison_contract import (
    ComparisonContractSignature,
    validate_contract_signature_parity,
)
from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy
from Thesis_ML.experiments.model_admission import (
    admitted_models_for_framework,
    official_gpu_only_backend_pairs,
    official_gpu_only_model_backend_allowed,
    official_max_both_gpu_lane_backend_pairs,
    official_max_both_gpu_lane_eligible,
)
from Thesis_ML.experiments.model_catalog import model_catalog_snapshot
from Thesis_ML.experiments.model_factory import ALL_MODEL_NAMES
from Thesis_ML.experiments.model_registry import get_model_spec, registered_model_names
from Thesis_ML.protocols.compiler import compile_protocol
from Thesis_ML.protocols.loader import load_protocol


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _demo_index_csv() -> Path:
    return _repo_root() / "demo_data" / "synthetic_v1" / "dataset_index.csv"


def _comparison_grouped_nested_v1_path() -> Path:
    return (
        _repo_root() / "configs" / "comparisons" / "model_family_grouped_nested_comparison_v1.json"
    )


def _comparison_grouped_nested_v2_path() -> Path:
    return (
        _repo_root() / "configs" / "comparisons" / "model_family_grouped_nested_comparison_v2.json"
    )


def _protocol_grouped_nested_v1_path() -> Path:
    return _repo_root() / "configs" / "protocols" / "thesis_canonical_nested_v1.json"


def _protocol_grouped_nested_v2_path() -> Path:
    return _repo_root() / "configs" / "protocols" / "thesis_canonical_nested_v2.json"


def _cpu_policy() -> ResolvedComputePolicy:
    return ResolvedComputePolicy(
        hardware_mode_requested="cpu_only",
        hardware_mode_effective="cpu_only",
        requested_backend_family="sklearn_cpu",
        effective_backend_family="sklearn_cpu",
        gpu_device_id=None,
        gpu_device_name=None,
        gpu_device_total_memory_mb=None,
        deterministic_compute=False,
        allow_backend_fallback=False,
        backend_stack_id="sklearn_cpu_reference_v1",
        backend_fallback_used=False,
        backend_fallback_reason=None,
    )


def _gpu_policy() -> ResolvedComputePolicy:
    return ResolvedComputePolicy(
        hardware_mode_requested="gpu_only",
        hardware_mode_effective="gpu_only",
        requested_backend_family="torch_gpu",
        effective_backend_family="torch_gpu",
        gpu_device_id=0,
        gpu_device_name="synthetic_gpu",
        gpu_device_total_memory_mb=4096,
        deterministic_compute=True,
        allow_backend_fallback=False,
        backend_stack_id="torch_gpu_reference_v1",
        backend_fallback_used=False,
        backend_fallback_reason=None,
    )


def _signature(*, class_weight_policy: str = "none") -> ComparisonContractSignature:
    return ComparisonContractSignature(
        target="coarse_affect",
        data_slice_semantics=("within_subject_loso_session", "all_from_index"),
        split_policy=("within_subject_loso_session", "session"),
        feature_recipe_id="baseline_standard_scaler_v1",
        metric_policy=("balanced_accuracy", "macro_f1", "accuracy"),
        methodology_policy=(
            "grouped_nested_tuning",
            "True",
            "grouped_leave_one_group_out",
            "session",
            "official-linear-grouped-nested-v2",
            "2.0.0",
        ),
        tuning_budget_semantics=(
            "official-linear-grouped-nested-v2",
            "2.0.0",
            "grouped_leave_one_group_out",
            "session",
            "True",
        ),
        repeat_policy=(3, 1000),
        control_policy=("False", "balanced_accuracy", "0", "False"),
        class_weight_policy=class_weight_policy,
        deterministic_compute_requirements=(True, True),
        backend_parity_semantics=("sklearn_cpu", "torch_gpu"),
    )


def test_registry_is_authoritative_for_supported_model_set() -> None:
    expected = {"dummy", "ridge", "logreg", "linearsvc", "xgboost"}
    assert set(registered_model_names()) == expected
    assert set(ALL_MODEL_NAMES) == expected
    assert set(model_catalog_snapshot().keys()) == expected


def test_official_model_admission_is_centralized_and_conservative() -> None:
    assert set(admitted_models_for_framework("confirmatory")) == {
        "dummy",
        "ridge",
        "logreg",
        "linearsvc",
    }
    assert set(admitted_models_for_framework("locked_comparison")) == {
        "dummy",
        "ridge",
        "logreg",
        "linearsvc",
    }
    assert "xgboost" in set(admitted_models_for_framework("exploratory"))

    assert official_gpu_only_backend_pairs("confirmatory") == (("ridge", "torch_gpu"),)
    assert official_gpu_only_backend_pairs("locked_comparison") == (("ridge", "torch_gpu"),)
    assert official_max_both_gpu_lane_backend_pairs("locked_comparison") == (
        ("ridge", "torch_gpu"),
    )

    assert official_gpu_only_model_backend_allowed(
        framework_mode="confirmatory",
        model_name="ridge",
        backend_family="torch_gpu",
    )
    assert not official_gpu_only_model_backend_allowed(
        framework_mode="confirmatory",
        model_name="logreg",
        backend_family="torch_gpu",
    )
    assert official_max_both_gpu_lane_eligible(
        framework_mode="locked_comparison",
        model_name="ridge",
        backend_family="torch_gpu",
    )
    assert not official_max_both_gpu_lane_eligible(
        framework_mode="locked_comparison",
        model_name="linearsvc",
        backend_family="sklearn_cpu",
    )


def test_backend_routing_derives_from_registry_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "Thesis_ML.experiments.backend_registry.xgboost_cpu_support_status",
        lambda: (True, None),
    )
    monkeypatch.setattr(
        "Thesis_ML.experiments.backend_registry.xgboost_gpu_support_status",
        lambda gpu_device_id=None: (True, None),
    )
    policies = {
        "sklearn_cpu": _cpu_policy(),
        "torch_gpu": _gpu_policy(),
    }

    for model_name in registered_model_names():
        spec = get_model_spec(model_name)
        for compute_backend_family, policy in policies.items():
            support = resolve_backend_support(model_name, policy)
            binding = spec.backend_binding_for_compute_family(compute_backend_family)
            if binding is None:
                assert support.supported is False
                continue
            assert support.supported is True
            assert support.backend_id == binding.backend_id


def test_fairness_contract_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="Mismatched fields: class_weight_policy"):
        validate_contract_signature_parity(
            signatures_by_id={
                "ridge_variant": _signature(class_weight_policy="none"),
                "linearsvc_variant": _signature(class_weight_policy="balanced"),
            },
            context="locked_comparison",
        )


def test_grouped_nested_v2_configs_compile_and_v1_configs_remain_loadable() -> None:
    comparison_v1 = load_comparison_spec(_comparison_grouped_nested_v1_path())
    comparison_v2 = load_comparison_spec(_comparison_grouped_nested_v2_path())
    assert comparison_v1.methodology_policy.tuning_search_space_id == "linear-grouped-nested-v1"
    assert (
        comparison_v2.methodology_policy.tuning_search_space_id
        == "official-linear-grouped-nested-v2"
    )
    comparison_manifest = compile_comparison(
        comparison_v2,
        index_csv=_demo_index_csv(),
        variant_ids=["ridge"],
    )
    assert comparison_manifest.runs

    protocol_v1 = load_protocol(_protocol_grouped_nested_v1_path())
    protocol_v2 = load_protocol(_protocol_grouped_nested_v2_path())
    assert protocol_v1.methodology_policy.tuning_search_space_id == "linear-grouped-nested-v1"
    assert protocol_v2.methodology_policy.tuning_search_space_id == "official-linear-grouped-nested-v2"
    protocol_manifest = compile_protocol(
        protocol_v2,
        index_csv=_demo_index_csv(),
        suite_ids=["primary_within_subject"],
    )
    assert protocol_manifest.runs
