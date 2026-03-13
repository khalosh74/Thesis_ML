from __future__ import annotations

from pathlib import Path
from typing import Any

from Thesis_ML.orchestration import campaign_runner


def test_campaign_runner_uses_facade_runner_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_engine(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(campaign_runner, "_engine_run_decision_support_campaign", _fake_engine)

    result = campaign_runner.run_decision_support_campaign(
        registry_path=tmp_path / "registry.json",
        index_csv=tmp_path / "index.csv",
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "outputs",
        experiment_id=None,
        stage=None,
        run_all=True,
        seed=42,
        n_permutations=0,
        dry_run=True,
    )

    assert result == {"ok": True}
    assert captured["run_experiment_fn"] is campaign_runner.run_experiment
