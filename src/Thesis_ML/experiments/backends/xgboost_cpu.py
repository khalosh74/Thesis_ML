from __future__ import annotations

import importlib
from time import perf_counter
from typing import Any

import numpy as np

XGBOOST_CPU_BACKEND_ID = "xgboost_cpu_reference_v1"


def _import_xgboost_module() -> Any:
    try:
        return importlib.import_module("xgboost")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "XGBoost backend requested but xgboost is not installed. "
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


def xgboost_cpu_support_status() -> tuple[bool, str | None]:
    try:
        _import_xgboost_module()
    except RuntimeError as exc:
        return False, str(exc)
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

    class XGBoostCpuClassifier(_XGBClassifierBase):
        backend_id: str = XGBOOST_CPU_BACKEND_ID
        backend_family: str = "xgboost_cpu"
        backend_device: str = "cpu"

        def fit(self, x_matrix: np.ndarray, y_labels: np.ndarray, **kwargs: Any) -> Any:
            fit_start = perf_counter()
            result = super().fit(x_matrix, y_labels, **kwargs)
            fit_elapsed = float(perf_counter() - fit_start)
            self.backend_runtime_metadata_ = {
                "backend_id": self.backend_id,
                "backend_family": self.backend_family,
                "backend_device": self.backend_device,
                "xgboost_fit_elapsed_seconds": fit_elapsed,
                "xgboost_tree_method": str(self.get_params(deep=False).get("tree_method", "hist")),
                "xgboost_device": str(self.get_params(deep=False).get("device", "cpu")),
            }
            return result

        def get_backend_runtime_metadata(self) -> dict[str, Any]:
            metadata = getattr(self, "backend_runtime_metadata_", None)
            return dict(metadata) if isinstance(metadata, dict) else {}

else:

    class XGBoostCpuClassifier:  # pragma: no cover - optional dependency stub
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "XGBoost backend requested but xgboost is not installed. "
                "Install the optional extra with `pip install .[xgboost]`."
            )



def make_xgboost_cpu_estimator(
    *,
    seed: int,
    class_weight_policy: str = "none",
) -> XGBoostCpuClassifier:
    _validate_xgboost_class_weight_policy(class_weight_policy)
    xgboost_module = _import_xgboost_module()
    if _XGBClassifierBase is None:
        raise RuntimeError("xgboost.XGBClassifier is unavailable in the current environment.")

    estimator_kwargs: dict[str, Any] = {
        "random_state": int(seed),
        "n_jobs": 1,
        "tree_method": "hist",
        "verbosity": 0,
        "eval_metric": "logloss",
    }
    if _xgboost_supports_device_parameter(xgboost_module):
        estimator_kwargs["device"] = "cpu"

    return XGBoostCpuClassifier(**estimator_kwargs)


__all__ = [
    "XGBOOST_CPU_BACKEND_ID",
    "XGBoostCpuClassifier",
    "make_xgboost_cpu_estimator",
    "xgboost_cpu_support_status",
]
