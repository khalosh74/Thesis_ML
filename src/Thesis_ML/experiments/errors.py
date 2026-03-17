from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ThesisMLError(Exception):
    """Base structured exception for machine-readable run failure metadata."""

    message: str
    code: str = "thesisml_error"
    stage: str = "runtime"
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


class OfficialContractValidationError(ThesisMLError):
    def __init__(
        self,
        message: str,
        *,
        stage: str = "preflight_validation",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="official_contract_validation_error",
            stage=stage,
            details=details,
        )


class OfficialArtifactContractError(ThesisMLError):
    def __init__(
        self,
        message: str,
        *,
        stage: str = "artifact_validation",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="official_artifact_contract_error",
            stage=stage,
            details=details,
        )


class ReproducibilityValidationError(ThesisMLError):
    def __init__(
        self,
        message: str,
        *,
        stage: str = "reproducibility",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="reproducibility_validation_error",
            stage=stage,
            details=details,
        )


def exception_failure_payload(
    exc: Exception,
    *,
    default_stage: str = "runtime",
) -> dict[str, Any]:
    if isinstance(exc, ThesisMLError):
        return {
            "error_code": str(exc.code),
            "error_type": type(exc).__name__,
            "failure_stage": str(exc.stage),
            "error_details": dict(exc.details or {}),
        }
    return {
        "error_code": "unhandled_exception",
        "error_type": type(exc).__name__,
        "failure_stage": str(default_stage),
        "error_details": {},
    }


__all__ = [
    "OfficialArtifactContractError",
    "OfficialContractValidationError",
    "ReproducibilityValidationError",
    "exception_failure_payload",
    "ThesisMLError",
]
