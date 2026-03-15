from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_verify_repro_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "verify_official_reproducibility.py"
    spec = importlib.util.spec_from_file_location("verify_official_reproducibility", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_repro_uses_default_protocol_config_when_omitted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_verify_repro_module()
    captured: dict[str, Path] = {}

    def _stub_run_protocol_once(**kwargs):
        captured["config_path"] = Path(kwargs["protocol_path"])
        return {
            "n_failed": 0,
            "protocol_output_dir": str(tmp_path / "protocol_runs" / "thesis-canonical__1.0.0"),
        }

    monkeypatch.setattr(module, "_run_protocol_once", _stub_run_protocol_once)
    monkeypatch.setattr(
        module,
        "compare_official_outputs",
        lambda **_: {
            "passed": True,
            "left": {},
            "right": {},
            "mismatches": [],
        },
    )

    exit_code = module.main(
        [
            "--mode",
            "protocol",
            "--index-csv",
            "dummy_index.csv",
            "--data-root",
            "dummy_data_root",
            "--cache-dir",
            "dummy_cache",
            "--suite",
            "primary_controls",
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 0
    assert captured["config_path"] == Path(module.DEFAULT_THESIS_PROTOCOL_PATH)


def test_verify_repro_uses_default_comparison_config_when_omitted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_verify_repro_module()
    captured: dict[str, Path] = {}

    def _stub_run_comparison_once(**kwargs):
        captured["config_path"] = Path(kwargs["comparison_path"])
        return {
            "n_failed": 0,
            "comparison_output_dir": str(
                tmp_path / "comparison_runs" / "model-family-within-subject__1.0.0"
            ),
        }

    monkeypatch.setattr(module, "_run_comparison_once", _stub_run_comparison_once)
    monkeypatch.setattr(
        module,
        "compare_official_outputs",
        lambda **_: {
            "passed": True,
            "left": {},
            "right": {},
            "mismatches": [],
        },
    )

    exit_code = module.main(
        [
            "--mode",
            "comparison",
            "--index-csv",
            "dummy_index.csv",
            "--data-root",
            "dummy_data_root",
            "--cache-dir",
            "dummy_cache",
            "--variant",
            "ridge",
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 0
    assert captured["config_path"] == Path(module.DEFAULT_COMPARISON_SPEC_PATH)
