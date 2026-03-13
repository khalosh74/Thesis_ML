"""Experiment runner utilities.

Expose the historical ``run_experiment`` API lazily to avoid importing heavy
runtime dependencies at package import time.
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_experiment"]


def run_experiment(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from Thesis_ML.experiments.run_experiment import run_experiment as _run_experiment

    return _run_experiment(*args, **kwargs)
