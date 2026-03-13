from __future__ import annotations

import json
import subprocess
import sys


def _run_import_probe(source: str) -> dict[str, bool]:
    process = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(process.stdout.strip().splitlines()[-1])


def test_compiler_and_contracts_import_without_runner_runtime() -> None:
    probe = _run_import_probe(
        """
import importlib
import json
import sys

importlib.import_module("Thesis_ML.orchestration.contracts")
importlib.import_module("Thesis_ML.orchestration.compiler")
importlib.import_module("Thesis_ML.orchestration")

print(json.dumps({
    "run_experiment_loaded": "Thesis_ML.experiments.run_experiment" in sys.modules,
    "nibabel_loaded": "nibabel" in sys.modules,
}))
"""
    )
    assert probe["run_experiment_loaded"] is False
    assert probe["nibabel_loaded"] is False


def test_decision_support_import_is_lazy_for_runner_runtime() -> None:
    probe = _run_import_probe(
        """
import importlib
import json
import sys

importlib.import_module("Thesis_ML.orchestration.decision_support")

print(json.dumps({
    "run_experiment_loaded": "Thesis_ML.experiments.run_experiment" in sys.modules,
    "nibabel_loaded": "nibabel" in sys.modules,
}))
"""
    )
    assert probe["run_experiment_loaded"] is False
    assert probe["nibabel_loaded"] is False
