from __future__ import annotations

from dataclasses import dataclass

from Thesis_ML.orchestration.experiment_selection import select_experiments


@dataclass
class _ExperimentRow:
    payload: dict[str, object]

    def model_dump(self, mode: str = "python") -> dict[str, object]:
        _ = mode
        return dict(self.payload)


@dataclass
class _Registry:
    experiments: list[_ExperimentRow]


def _registry() -> _Registry:
    return _Registry(
        experiments=[
            _ExperimentRow(
                {
                    "experiment_id": "E12",
                    "stage": "Stage 6 - Robustness analysis",
                }
            ),
            _ExperimentRow(
                {
                    "experiment_id": "E13",
                    "stage": "Stage 6 - Robustness analysis",
                }
            ),
            _ExperimentRow(
                {
                    "experiment_id": "E15",
                    "stage": "Stage 6 - Robustness analysis",
                }
            ),
            _ExperimentRow(
                {
                    "experiment_id": "E20",
                    "stage": "Stage 6 - Robustness analysis",
                }
            ),
        ]
    )


def test_select_experiments_supports_multi_id_selection() -> None:
    selected = select_experiments(
        registry=_registry(),  # type: ignore[arg-type]
        experiment_id=None,
        experiment_ids=["E15", "E12", "E13"],
        stage=None,
        run_all=False,
    )

    assert [str(row["experiment_id"]) for row in selected] == ["E12", "E13", "E15"]


def test_select_experiments_rejects_unknown_multi_id_set() -> None:
    try:
        select_experiments(
            registry=_registry(),  # type: ignore[arg-type]
            experiment_id=None,
            experiment_ids=["E99"],
            stage=None,
            run_all=False,
        )
    except ValueError as exc:
        assert "requested experiments" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown experiment_ids")
