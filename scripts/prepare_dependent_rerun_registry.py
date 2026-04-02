from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_DEFAULT_REGISTRY = Path("configs") / "decision_support_registry_revised_execution.json"


class DependentRerunRegistryError(RuntimeError):
    """Raised when rerun registry derivation inputs are invalid."""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a temporary registry for E07/E08 reruns when Stage 3 chooses a non-ridge model."
    )
    parser.add_argument(
        "--selected-model",
        type=str,
        required=True,
        help="Model selected by Stage 3 (for example: logreg).",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiments to clone into derived registry (for example: E07 E08).",
    )
    parser.add_argument(
        "--output-registry",
        type=Path,
        required=True,
        help="Output path for derived registry JSON.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=_DEFAULT_REGISTRY,
        help="Source revised registry JSON (default: configs/decision_support_registry_revised_execution.json).",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise DependentRerunRegistryError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DependentRerunRegistryError(f"{label} must be a JSON object: {path}")
    return payload


def _replace_ridge_model_in_experiment(experiment_payload: dict[str, Any], selected_model: str) -> dict[str, Any]:
    updated = json.loads(json.dumps(experiment_payload))
    templates = updated.get("variant_templates")
    if not isinstance(templates, list):
        return updated
    for template in templates:
        if not isinstance(template, dict):
            continue
        params = template.get("params")
        if not isinstance(params, dict):
            continue
        model_name = params.get("model")
        if isinstance(model_name, str) and model_name.strip().lower() == "ridge":
            params["model"] = str(selected_model)
    return updated


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    selected_model = str(args.selected_model).strip()
    if not selected_model:
        raise DependentRerunRegistryError("--selected-model must be non-empty.")

    experiment_ids = [str(value).strip().upper() for value in args.experiments if str(value).strip()]
    if not experiment_ids:
        raise DependentRerunRegistryError("--experiments must include at least one experiment ID.")

    if selected_model.lower() == "ridge":
        print("Selected model is ridge; no dependent rerun registry is required.")
        return 0

    source_registry = _load_json_object(args.registry.resolve(), label="source registry")
    experiments_payload = source_registry.get("experiments")
    if not isinstance(experiments_payload, list):
        raise DependentRerunRegistryError("source registry is missing 'experiments' list.")

    by_id: dict[str, dict[str, Any]] = {}
    for entry in experiments_payload:
        if not isinstance(entry, dict):
            continue
        exp_id = str(entry.get("experiment_id") or "").strip().upper()
        if not exp_id:
            continue
        by_id[exp_id] = entry

    missing = [exp_id for exp_id in experiment_ids if exp_id not in by_id]
    if missing:
        raise DependentRerunRegistryError(
            "Requested experiment(s) not found in source registry: " + ", ".join(sorted(missing))
        )

    derived = json.loads(json.dumps(source_registry))
    derived_experiments: list[dict[str, Any]] = []
    for exp_id in experiment_ids:
        derived_experiments.append(
            _replace_ridge_model_in_experiment(by_id[exp_id], selected_model=selected_model)
        )

    derived["experiments"] = derived_experiments

    output_path = args.output_registry.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(derived, indent=2)}\n", encoding="utf-8")

    print(f"Wrote dependent rerun registry: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
