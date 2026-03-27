from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.features.preprocessing import (
    BASELINE_STANDARD_SCALER_RECIPE_ID,
    FEATURE_RECIPE_IDS,
)

MODEL_REGISTRY_VERSION = "model-registry-v1"

OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID = "official-linear-grouped-nested-v2"
OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION = "2.0.0"
LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID = "linear-grouped-nested-v1"
LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION = "1.0.0"
EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_ID = "exploratory-xgboost-grouped-nested-v1"
EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_VERSION = "1.0.0"

ModelFamily = Literal["linear", "control", "tree_boosting"]
ModelStatus = Literal["official", "exploratory_only"]
ModelCostTierName = Literal[
    "official_fast",
    "official_allowed",
    "benchmark_expensive",
    "exploratory_only",
]
ClassWeightPolicyName = Literal["none", "balanced"]
ComputeBackendFamily = Literal["sklearn_cpu", "torch_gpu"]
ResolvedBackendFamily = Literal["sklearn_cpu", "torch_gpu", "xgboost_cpu", "xgboost_gpu"]
StageRouteToken = Literal[
    "cpu_reference",
    "torch_ridge",
    "torch_logreg",
    "xgboost_cpu",
    "xgboost_gpu",
    "generic",
    "control_skip",
    "linearsvc_specialized",
    "logreg_specialized",
    "reference",
    "ridge_gpu_preferred",
]


@dataclass(frozen=True)
class ModelBackendBinding:
    compute_backend_family: ComputeBackendFamily
    backend_family: ResolvedBackendFamily
    backend_id: str


@dataclass(frozen=True)
class ModelTuningPolicy:
    supports_tuning: bool
    default_search_space_id: str | None
    default_search_space_version: str | None
    allowed_search_space_ids: tuple[str, ...]
    official_search_space_ids: tuple[str, ...]
    policy_reference: str


@dataclass(frozen=True)
class ModelOfficialAdmission:
    status: ModelStatus
    locked_comparison_allowed: bool
    confirmatory_allowed: bool
    locked_comparison_gpu_only_backend_families: tuple[str, ...]
    locked_comparison_max_both_gpu_lane_backend_families: tuple[str, ...]
    confirmatory_gpu_only_backend_families: tuple[str, ...]
    deterministic_compute_required_for_official_gpu: bool
    allow_backend_fallback_for_official: bool


@dataclass(frozen=True)
class ModelCapabilities:
    linear_interpretability: bool
    probability_outputs: bool
    decision_function_outputs: bool


@dataclass(frozen=True)
class ModelSpec:
    logical_name: str
    model_family: ModelFamily
    preprocess_kind: str
    cost_tier: ModelCostTierName
    projected_runtime_seconds_by_mode: dict[str, int]
    grouped_nested_runtime_multiplier: float
    timeout_override_seconds: int | None
    requires_explicit_comparison_spec: bool
    supported_class_weight_policies: tuple[ClassWeightPolicyName, ...]
    allowed_feature_recipe_ids: tuple[str, ...]
    backend_bindings: tuple[ModelBackendBinding, ...]
    default_compute_backend_preference: ComputeBackendFamily
    tuning_policy: ModelTuningPolicy
    official_admission: ModelOfficialAdmission
    capabilities: ModelCapabilities
    model_fit_route: tuple[StageRouteToken, ...]
    tuning_route: tuple[StageRouteToken, ...]
    permutation_route: tuple[StageRouteToken, ...]
    notes: str | None = None

    def backend_binding_for_compute_family(
        self,
        compute_backend_family: ComputeBackendFamily | str,
    ) -> ModelBackendBinding | None:
        normalized = str(compute_backend_family).strip()
        for binding in self.backend_bindings:
            if binding.compute_backend_family == normalized:
                return binding
        return None


def _validate_runtime_map(runtime_map: dict[str, int]) -> dict[str, int]:
    required_modes = {
        FrameworkMode.EXPLORATORY.value,
        FrameworkMode.LOCKED_COMPARISON.value,
        FrameworkMode.CONFIRMATORY.value,
    }
    missing = sorted(required_modes - set(runtime_map.keys()))
    if missing:
        raise ValueError(
            "Model registry runtime map is missing required framework modes: "
            + ", ".join(missing)
        )
    resolved: dict[str, int] = {}
    for mode_name, seconds_raw in runtime_map.items():
        seconds = int(seconds_raw)
        if seconds <= 0:
            raise ValueError(
                f"Model registry runtime map for mode '{mode_name}' must be > 0 seconds."
            )
        resolved[str(mode_name)] = seconds
    return resolved


def _linear_tuning_policy() -> ModelTuningPolicy:
    return ModelTuningPolicy(
        supports_tuning=True,
        default_search_space_id=OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
        default_search_space_version=OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
        allowed_search_space_ids=(
            OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
            LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
        ),
        official_search_space_ids=(OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,),
        policy_reference="grouped_nested_tuning",
    )


def _xgboost_tuning_policy() -> ModelTuningPolicy:
    return ModelTuningPolicy(
        supports_tuning=True,
        default_search_space_id=EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_ID,
        default_search_space_version=EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_VERSION,
        allowed_search_space_ids=(
            EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_ID,
            LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
        ),
        official_search_space_ids=(),
        policy_reference="grouped_nested_tuning",
    )


def _no_tuning_policy() -> ModelTuningPolicy:
    return ModelTuningPolicy(
        supports_tuning=False,
        default_search_space_id=None,
        default_search_space_version=None,
        allowed_search_space_ids=(),
        official_search_space_ids=(),
        policy_reference="fixed_baselines_only",
    )


def _official_admission(
    *,
    locked_comparison_gpu_only_backend_families: tuple[str, ...] = (),
    locked_comparison_max_both_gpu_lane_backend_families: tuple[str, ...] = (),
    confirmatory_gpu_only_backend_families: tuple[str, ...] = (),
) -> ModelOfficialAdmission:
    return ModelOfficialAdmission(
        status="official",
        locked_comparison_allowed=True,
        confirmatory_allowed=True,
        locked_comparison_gpu_only_backend_families=locked_comparison_gpu_only_backend_families,
        locked_comparison_max_both_gpu_lane_backend_families=(
            locked_comparison_max_both_gpu_lane_backend_families
        ),
        confirmatory_gpu_only_backend_families=confirmatory_gpu_only_backend_families,
        deterministic_compute_required_for_official_gpu=True,
        allow_backend_fallback_for_official=False,
    )


def _exploratory_only_admission() -> ModelOfficialAdmission:
    return ModelOfficialAdmission(
        status="exploratory_only",
        locked_comparison_allowed=False,
        confirmatory_allowed=False,
        locked_comparison_gpu_only_backend_families=(),
        locked_comparison_max_both_gpu_lane_backend_families=(),
        confirmatory_gpu_only_backend_families=(),
        deterministic_compute_required_for_official_gpu=True,
        allow_backend_fallback_for_official=False,
    )


def _build_registry() -> dict[str, ModelSpec]:
    all_feature_recipes = tuple(FEATURE_RECIPE_IDS)
    registry = {
        "ridge": ModelSpec(
            logical_name="ridge",
            model_family="linear",
            preprocess_kind="standard_scaler",
            cost_tier="official_fast",
            projected_runtime_seconds_by_mode=_validate_runtime_map(
                {
                    FrameworkMode.EXPLORATORY.value: 15 * 60,
                    FrameworkMode.LOCKED_COMPARISON.value: 25 * 60,
                    FrameworkMode.CONFIRMATORY.value: 20 * 60,
                }
            ),
            grouped_nested_runtime_multiplier=1.25,
            timeout_override_seconds=None,
            requires_explicit_comparison_spec=False,
            supported_class_weight_policies=("none", "balanced"),
            allowed_feature_recipe_ids=all_feature_recipes,
            backend_bindings=(
                ModelBackendBinding(
                    compute_backend_family="sklearn_cpu",
                    backend_family="sklearn_cpu",
                    backend_id="cpu_reference",
                ),
                ModelBackendBinding(
                    compute_backend_family="torch_gpu",
                    backend_family="torch_gpu",
                    backend_id="torch_ridge_gpu_v2",
                ),
            ),
            default_compute_backend_preference="sklearn_cpu",
            tuning_policy=_linear_tuning_policy(),
            official_admission=_official_admission(
                locked_comparison_gpu_only_backend_families=("torch_gpu",),
                locked_comparison_max_both_gpu_lane_backend_families=("torch_gpu",),
                confirmatory_gpu_only_backend_families=("torch_gpu",),
            ),
            capabilities=ModelCapabilities(
                linear_interpretability=True,
                probability_outputs=False,
                decision_function_outputs=True,
            ),
            model_fit_route=("torch_ridge", "cpu_reference"),
            tuning_route=("generic",),
            permutation_route=("ridge_gpu_preferred", "reference"),
            notes="Linear baseline with stable runtime envelope.",
        ),
        "logreg": ModelSpec(
            logical_name="logreg",
            model_family="linear",
            preprocess_kind="standard_scaler",
            cost_tier="benchmark_expensive",
            projected_runtime_seconds_by_mode=_validate_runtime_map(
                {
                    FrameworkMode.EXPLORATORY.value: 35 * 60,
                    FrameworkMode.LOCKED_COMPARISON.value: 70 * 60,
                    FrameworkMode.CONFIRMATORY.value: 55 * 60,
                }
            ),
            grouped_nested_runtime_multiplier=1.25,
            timeout_override_seconds=120 * 60,
            requires_explicit_comparison_spec=True,
            supported_class_weight_policies=("none", "balanced"),
            allowed_feature_recipe_ids=all_feature_recipes,
            backend_bindings=(
                ModelBackendBinding(
                    compute_backend_family="sklearn_cpu",
                    backend_family="sklearn_cpu",
                    backend_id="cpu_reference",
                ),
                ModelBackendBinding(
                    compute_backend_family="torch_gpu",
                    backend_family="torch_gpu",
                    backend_id="torch_logreg_gpu_v1",
                ),
            ),
            default_compute_backend_preference="sklearn_cpu",
            tuning_policy=_linear_tuning_policy(),
            official_admission=_official_admission(),
            capabilities=ModelCapabilities(
                linear_interpretability=True,
                probability_outputs=True,
                decision_function_outputs=True,
            ),
            model_fit_route=("cpu_reference", "torch_logreg"),
            tuning_route=("logreg_specialized", "generic"),
            permutation_route=("reference",),
            notes=(
                "Expensive benchmark model that remains official but requires explicit comparison "
                "declaration."
            ),
        ),
        "linearsvc": ModelSpec(
            logical_name="linearsvc",
            model_family="linear",
            preprocess_kind="standard_scaler",
            cost_tier="official_allowed",
            projected_runtime_seconds_by_mode=_validate_runtime_map(
                {
                    FrameworkMode.EXPLORATORY.value: 20 * 60,
                    FrameworkMode.LOCKED_COMPARISON.value: 35 * 60,
                    FrameworkMode.CONFIRMATORY.value: 30 * 60,
                }
            ),
            grouped_nested_runtime_multiplier=1.25,
            timeout_override_seconds=None,
            requires_explicit_comparison_spec=False,
            supported_class_weight_policies=("none", "balanced"),
            allowed_feature_recipe_ids=all_feature_recipes,
            backend_bindings=(
                ModelBackendBinding(
                    compute_backend_family="sklearn_cpu",
                    backend_family="sklearn_cpu",
                    backend_id="cpu_reference",
                ),
            ),
            default_compute_backend_preference="sklearn_cpu",
            tuning_policy=_linear_tuning_policy(),
            official_admission=_official_admission(),
            capabilities=ModelCapabilities(
                linear_interpretability=True,
                probability_outputs=False,
                decision_function_outputs=True,
            ),
            model_fit_route=("cpu_reference",),
            tuning_route=("linearsvc_specialized", "generic"),
            permutation_route=("reference",),
            notes="Allowed official model with moderate runtime.",
        ),
        "dummy": ModelSpec(
            logical_name="dummy",
            model_family="control",
            preprocess_kind="standard_scaler",
            cost_tier="official_fast",
            projected_runtime_seconds_by_mode=_validate_runtime_map(
                {
                    FrameworkMode.EXPLORATORY.value: 60,
                    FrameworkMode.LOCKED_COMPARISON.value: 90,
                    FrameworkMode.CONFIRMATORY.value: 90,
                }
            ),
            grouped_nested_runtime_multiplier=1.0,
            timeout_override_seconds=None,
            requires_explicit_comparison_spec=False,
            supported_class_weight_policies=("none", "balanced"),
            allowed_feature_recipe_ids=all_feature_recipes,
            backend_bindings=(
                ModelBackendBinding(
                    compute_backend_family="sklearn_cpu",
                    backend_family="sklearn_cpu",
                    backend_id="cpu_reference",
                ),
            ),
            default_compute_backend_preference="sklearn_cpu",
            tuning_policy=_no_tuning_policy(),
            official_admission=_official_admission(),
            capabilities=ModelCapabilities(
                linear_interpretability=False,
                probability_outputs=True,
                decision_function_outputs=False,
            ),
            model_fit_route=("cpu_reference",),
            tuning_route=("control_skip",),
            permutation_route=("reference",),
            notes="Control baseline with negligible runtime.",
        ),
        "xgboost": ModelSpec(
            logical_name="xgboost",
            model_family="tree_boosting",
            preprocess_kind="standard_scaler",
            cost_tier="exploratory_only",
            projected_runtime_seconds_by_mode=_validate_runtime_map(
                {
                    FrameworkMode.EXPLORATORY.value: 45 * 60,
                    FrameworkMode.LOCKED_COMPARISON.value: 90 * 60,
                    FrameworkMode.CONFIRMATORY.value: 90 * 60,
                }
            ),
            grouped_nested_runtime_multiplier=1.25,
            timeout_override_seconds=None,
            requires_explicit_comparison_spec=False,
            supported_class_weight_policies=("none",),
            allowed_feature_recipe_ids=(BASELINE_STANDARD_SCALER_RECIPE_ID,),
            backend_bindings=(
                ModelBackendBinding(
                    compute_backend_family="sklearn_cpu",
                    backend_family="xgboost_cpu",
                    backend_id="xgboost_cpu_reference_v1",
                ),
                ModelBackendBinding(
                    compute_backend_family="torch_gpu",
                    backend_family="xgboost_gpu",
                    backend_id="xgboost_gpu_reference_v1",
                ),
            ),
            default_compute_backend_preference="sklearn_cpu",
            tuning_policy=_xgboost_tuning_policy(),
            official_admission=_exploratory_only_admission(),
            capabilities=ModelCapabilities(
                linear_interpretability=False,
                probability_outputs=True,
                decision_function_outputs=False,
            ),
            model_fit_route=("xgboost_gpu", "xgboost_cpu"),
            tuning_route=("generic",),
            permutation_route=("reference",),
            notes="Exploratory-only gradient boosting family.",
        ),
    }

    # Lightweight integrity checks to catch registry construction errors early.
    backend_ids: set[str] = set()
    for name, spec in registry.items():
        # Ensure the dict key matches the declared logical name
        if spec.logical_name != name:
            raise ValueError(
                f"Model registry key '{name}' does not match spec.logical_name '{spec.logical_name}'."
            )

        # Ensure at least one backend binding exists
        if not spec.backend_bindings:
            raise ValueError(f"Model '{name}' has no backend_bindings defined.")

        # Ensure default compute backend preference exists among bindings
        compute_families = {str(b.compute_backend_family).strip() for b in spec.backend_bindings}
        if spec.default_compute_backend_preference not in compute_families:
            raise ValueError(
                f"Model '{name}' default_compute_backend_preference='{spec.default_compute_backend_preference}' "
                f"is not present in backend_bindings compute_backend_family={sorted(compute_families)}."
            )

        # Ensure backend IDs are present and unique across the registry
        for b in spec.backend_bindings:
            if not str(b.backend_id).strip():
                raise ValueError(f"Model '{name}' has a backend_binding with an empty backend_id.")
            if b.backend_id in backend_ids:
                raise ValueError(
                    f"Duplicate backend_id '{b.backend_id}' found in model_registry for model '{name}'."
                )
            backend_ids.add(b.backend_id)

    return registry


MODEL_REGISTRY: dict[str, ModelSpec] = _build_registry()


def get_model_spec(model_name: str) -> ModelSpec:
    normalized = str(model_name).strip().lower()
    spec = MODEL_REGISTRY.get(normalized)
    if spec is None:
        allowed = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{model_name}'. Allowed values: {allowed}.")
    return spec


def iter_model_specs() -> tuple[ModelSpec, ...]:
    return tuple(MODEL_REGISTRY[name] for name in MODEL_REGISTRY)


def registered_model_names() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY.keys())


def official_model_names() -> tuple[str, ...]:
    return tuple(
        spec.logical_name
        for spec in iter_model_specs()
        if spec.official_admission.status == "official"
    )


def exploratory_only_model_names() -> tuple[str, ...]:
    return tuple(
        spec.logical_name
        for spec in iter_model_specs()
        if spec.official_admission.status == "exploratory_only"
    )


def official_linear_model_names() -> tuple[str, ...]:
    return tuple(
        spec.logical_name
        for spec in iter_model_specs()
        if spec.model_family == "linear" and spec.official_admission.status == "official"
    )


def control_model_names() -> tuple[str, ...]:
    return tuple(spec.logical_name for spec in iter_model_specs() if spec.model_family == "control")


def default_batch_model_names() -> tuple[str, ...]:
    return official_linear_model_names()


def models_supporting_compute_backend_family(
    compute_backend_family: ComputeBackendFamily | str,
) -> tuple[str, ...]:
    normalized = str(compute_backend_family).strip()
    supported: list[str] = []
    for spec in iter_model_specs():
        if spec.backend_binding_for_compute_family(normalized) is not None:
            supported.append(spec.logical_name)
    return tuple(supported)


def model_registry_snapshot() -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for spec in iter_model_specs():
        payload[spec.logical_name] = {
            "logical_name": spec.logical_name,
            "model_family": spec.model_family,
            "preprocess_kind": spec.preprocess_kind,
            "cost_tier": spec.cost_tier,
            "projected_runtime_seconds_by_mode": dict(spec.projected_runtime_seconds_by_mode),
            "grouped_nested_runtime_multiplier": float(spec.grouped_nested_runtime_multiplier),
            "timeout_override_seconds": spec.timeout_override_seconds,
            "requires_explicit_comparison_spec": bool(spec.requires_explicit_comparison_spec),
            "supported_class_weight_policies": list(spec.supported_class_weight_policies),
            "allowed_feature_recipe_ids": list(spec.allowed_feature_recipe_ids),
            "backend_bindings": [
                {
                    "compute_backend_family": binding.compute_backend_family,
                    "backend_family": binding.backend_family,
                    "backend_id": binding.backend_id,
                }
                for binding in spec.backend_bindings
            ],
            "default_compute_backend_preference": spec.default_compute_backend_preference,
            "tuning_policy": {
                "supports_tuning": bool(spec.tuning_policy.supports_tuning),
                "default_search_space_id": spec.tuning_policy.default_search_space_id,
                "default_search_space_version": spec.tuning_policy.default_search_space_version,
                "allowed_search_space_ids": list(spec.tuning_policy.allowed_search_space_ids),
                "official_search_space_ids": list(spec.tuning_policy.official_search_space_ids),
                "policy_reference": spec.tuning_policy.policy_reference,
            },
            "official_admission": {
                "status": spec.official_admission.status,
                "locked_comparison_allowed": bool(spec.official_admission.locked_comparison_allowed),
                "confirmatory_allowed": bool(spec.official_admission.confirmatory_allowed),
                "locked_comparison_gpu_only_backend_families": list(
                    spec.official_admission.locked_comparison_gpu_only_backend_families
                ),
                "locked_comparison_max_both_gpu_lane_backend_families": list(
                    spec.official_admission.locked_comparison_max_both_gpu_lane_backend_families
                ),
                "confirmatory_gpu_only_backend_families": list(
                    spec.official_admission.confirmatory_gpu_only_backend_families
                ),
                "deterministic_compute_required_for_official_gpu": bool(
                    spec.official_admission.deterministic_compute_required_for_official_gpu
                ),
                "allow_backend_fallback_for_official": bool(
                    spec.official_admission.allow_backend_fallback_for_official
                ),
            },
            "capabilities": {
                "linear_interpretability": bool(spec.capabilities.linear_interpretability),
                "probability_outputs": bool(spec.capabilities.probability_outputs),
                "decision_function_outputs": bool(spec.capabilities.decision_function_outputs),
            },
            "model_fit_route": list(spec.model_fit_route),
            "tuning_route": list(spec.tuning_route),
            "permutation_route": list(spec.permutation_route),
            "notes": spec.notes,
        }
    return payload


__all__ = [
    "BASELINE_STANDARD_SCALER_RECIPE_ID",
    "EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_ID",
    "EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_VERSION",
    "LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID",
    "LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION",
    "MODEL_REGISTRY",
    "MODEL_REGISTRY_VERSION",
    "ModelBackendBinding",
    "ModelCapabilities",
    "ModelOfficialAdmission",
    "ModelSpec",
    "ModelTuningPolicy",
    "OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID",
    "OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION",
    "control_model_names",
    "default_batch_model_names",
    "exploratory_only_model_names",
    "get_model_spec",
    "iter_model_specs",
    "model_registry_snapshot",
    "models_supporting_compute_backend_family",
    "official_linear_model_names",
    "official_model_names",
    "registered_model_names",
]
