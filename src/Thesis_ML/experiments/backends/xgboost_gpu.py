from __future__ import annotations

import importlib
from time import perf_counter
from typing import Any

import numpy as np

XGBOOST_GPU_BACKEND_ID = "xgboost_gpu_reference_v1"


def _import_xgboost_module() -> Any:
    try:
        return importlib.import_module("xgboost")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "XGBoost GPU backend requested but xgboost is not installed. "
            "Install the optional extra with `pip install .[xgboost]`."
        ) from exc


def _xgboost_supports_device_parameter(xgboost_module: Any) -> bool:
    estimator_cls = getattr(xgboost_module, "XGBClassifier", None)
    if estimator_cls is None:
        return False
    try:
        params = estimator_cls().get_params(deep=False)
    except Exception:
        return False
    return "device" in params


def _coerce_truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "on", "enabled"}


def xgboost_gpu_support_status(*, gpu_device_id: int | None) -> tuple[bool, str | None]:
    if gpu_device_id is None:
        return False, "gpu_device_id_missing"
    if int(gpu_device_id) < 0:
        return False, "gpu_device_id_must_be_non_negative"

    try:
        xgboost_module = _import_xgboost_module()
    except RuntimeError as exc:
        return False, str(exc)

    build_info_fn = getattr(xgboost_module, "build_info", None)
    if not callable(build_info_fn):
        return False, "xgboost_build_info_unavailable_for_gpu_validation"

    try:
        build_info = build_info_fn()
    except Exception as exc:  # pragma: no cover - defensive probe guard
        return False, f"xgboost_build_info_probe_failed:{exc.__class__.__name__}"

    if not isinstance(build_info, dict):
        return False, "xgboost_build_info_invalid"

    use_cuda = build_info.get("USE_CUDA")
    if not _coerce_truthy_flag(use_cuda):
        return False, "xgboost_cuda_support_unavailable"

    return True, None


def _validate_xgboost_class_weight_policy(class_weight_policy: str) -> None:
    normalized = str(class_weight_policy).strip().lower()
    if normalized == "none":
        return
    if normalized == "balanced":
        raise ValueError(
            "XGBoost exploratory backend currently supports class_weight_policy='none' only."
        )
    raise ValueError(
        f"Unsupported class_weight_policy '{class_weight_policy}'. Allowed values: none, balanced."
    )


try:
    _xgboost = importlib.import_module("xgboost")
    _XGBClassifierBase = getattr(_xgboost, "XGBClassifier", None)
except Exception:  # pragma: no cover - optional dependency import guard
    _xgboost = None
    _XGBClassifierBase = None


if _XGBClassifierBase is not None:

    class XGBoostGpuClassifier(_XGBClassifierBase):
        backend_id: str = XGBOOST_GPU_BACKEND_ID
        backend_family: str = "xgboost_gpu"

        def fit(self, x_matrix: np.ndarray, y_labels: np.ndarray, **kwargs: Any) -> Any:
            fit_start = perf_counter()
            result = super().fit(x_matrix, y_labels, **kwargs)
            fit_elapsed = float(perf_counter() - fit_start)
            params = self.get_params(deep=False)
            self.backend_runtime_metadata_ = {
                "backend_id": self.backend_id,
                "backend_family": self.backend_family,
                "backend_device": str(params.get("device", "gpu")),
                "xgboost_fit_elapsed_seconds": fit_elapsed,
                "xgboost_tree_method": str(params.get("tree_method", "hist")),
                "xgboost_device": str(params.get("device", "gpu")),
            }
            return result

        def get_backend_runtime_metadata(self) -> dict[str, Any]:
            metadata = getattr(self, "backend_runtime_metadata_", None)
            return dict(metadata) if isinstance(metadata, dict) else {}

else:

    class XGBoostGpuClassifier:  # pragma: no cover - optional dependency stub
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "XGBoost GPU backend requested but xgboost is not installed. "
                "Install the optional extra with `pip install .[xgboost]`."
            )


def make_xgboost_gpu_estimator(
    *,
    seed: int,
    class_weight_policy: str = "none",
    gpu_device_id: int = 0,
    deterministic_compute: bool = False,
) -> XGBoostGpuClassifier:
    _validate_xgboost_class_weight_policy(class_weight_policy)
    xgboost_module = _import_xgboost_module()
    supported, reason = xgboost_gpu_support_status(gpu_device_id=int(gpu_device_id))
    if not supported:
        raise RuntimeError(
            "XGBoost GPU backend is unavailable for the current environment: "
            f"{reason or 'unknown_reason'}."
        )
    if _XGBClassifierBase is None:
        raise RuntimeError("xgboost.XGBClassifier is unavailable in the current environment.")

    estimator_kwargs: dict[str, Any] = {
        "random_state": int(seed),
        "n_jobs": 1,
        "verbosity": 0,
        "eval_metric": "logloss",
    }
    if _xgboost_supports_device_parameter(xgboost_module):
        estimator_kwargs["tree_method"] = "hist"
        estimator_kwargs["device"] = f"cuda:{int(gpu_device_id)}"
    else:
        estimator_kwargs["tree_method"] = "gpu_hist"

    estimator = XGBoostGpuClassifier(**estimator_kwargs)
    estimator.deterministic_compute = bool(deterministic_compute)
    estimator.gpu_device_id = int(gpu_device_id)
    return estimator


__all__ = [
    "XGBOOST_GPU_BACKEND_ID",
    "XGBoostGpuClassifier",
    "make_xgboost_gpu_estimator",
    "xgboost_gpu_support_status",
]
