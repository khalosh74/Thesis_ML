from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

from Thesis_ML.orchestration import decision_support

_SHIM_PATH = Path(__file__).resolve().parents[1] / "run_decision_support_experiments.py"
_SHIM_SPEC = importlib.util.spec_from_file_location("run_decision_support_experiments", _SHIM_PATH)
if _SHIM_SPEC is None or _SHIM_SPEC.loader is None:
    raise RuntimeError(f"Unable to load shim module from {_SHIM_PATH}")
shim = importlib.util.module_from_spec(_SHIM_SPEC)
_SHIM_SPEC.loader.exec_module(shim)


def test_shim_main_forwards_to_packaged_orchestrator(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        seen["argv"] = argv
        return 7

    monkeypatch.setattr(shim._decision_support, "main", fake_main)
    assert shim.main(["--all", "--dry-run"]) == 7
    assert seen["argv"] == ["--all", "--dry-run"]


def test_shim_exposes_campaign_function() -> None:
    assert shim.run_decision_support_campaign is decision_support.run_decision_support_campaign


def test_shim_emits_deprecation_notice_to_stderr(monkeypatch, capsys) -> None:
    monkeypatch.setattr(shim._decision_support, "main", lambda argv=None: 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert shim.main(["--all"]) == 0
    stderr = capsys.readouterr().err
    assert "deprecated" in stderr.lower()
    assert "thesisml-run-decision-support" in stderr
    assert any("deprecated" in str(item.message).lower() for item in caught)
