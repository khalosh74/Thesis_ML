from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_dependent_rerun_registry_replaces_ridge_only(tmp_path: Path) -> None:
    source_registry = {
        "schema_version": "test",
        "experiments": [
            {
                "experiment_id": "E06",
                "variant_templates": [
                    {
                        "template_id": "e06_a",
                        "params": {"model": "ridge", "other": 1},
                        "search_space_id": "SS",
                    }
                ],
            },
            {
                "experiment_id": "E07",
                "variant_templates": [
                    {
                        "template_id": "e07_a",
                        "params": {"model": "ridge", "class_weight_policy": "none"},
                    }
                ],
            },
            {
                "experiment_id": "E08",
                "variant_templates": [
                    {
                        "template_id": "e08_a",
                        "params": {
                            "model": "ridge",
                            "methodology_policy_name": "fixed_baselines_only",
                        },
                    }
                ],
            },
        ],
        "study_reviews": [],
    }

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(source_registry, indent=2) + "\n", encoding="utf-8")

    output_path = tmp_path / "derived.json"
    script = _load_script_module(Path("scripts") / "prepare_dependent_rerun_registry.py")

    exit_code = script.main(
        [
            "--registry",
            str(registry_path),
            "--selected-model",
            "logreg",
            "--experiments",
            "E07",
            "E08",
            "--output-registry",
            str(output_path),
        ]
    )
    assert exit_code == 0
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert [entry["experiment_id"] for entry in payload["experiments"]] == ["E07", "E08"]

    e07 = payload["experiments"][0]
    e08 = payload["experiments"][1]
    assert e07["variant_templates"][0]["params"]["model"] == "logreg"
    assert e08["variant_templates"][0]["params"]["model"] == "logreg"

    # Non-model fields should be preserved exactly.
    assert e07["variant_templates"][0]["params"]["class_weight_policy"] == "none"
    assert (
        e08["variant_templates"][0]["params"]["methodology_policy_name"]
        == "fixed_baselines_only"
    )


def test_prepare_dependent_rerun_registry_writes_valid_json(tmp_path: Path) -> None:
    source_registry = {
        "schema_version": "test",
        "experiments": [
            {
                "experiment_id": "E07",
                "variant_templates": [{"template_id": "e07", "params": {"model": "ridge"}}],
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(source_registry) + "\n", encoding="utf-8")
    output_path = tmp_path / "derived.json"

    script = _load_script_module(Path("scripts") / "prepare_dependent_rerun_registry.py")
    exit_code = script.main(
        [
            "--registry",
            str(registry_path),
            "--selected-model",
            "linearsvc",
            "--experiments",
            "E07",
            "--output-registry",
            str(output_path),
        ]
    )
    assert exit_code == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["experiments"][0]["variant_templates"][0]["params"]["model"] == "linearsvc"
