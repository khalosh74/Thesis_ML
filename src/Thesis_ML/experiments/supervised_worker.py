from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from Thesis_ML.experiments.timeout_watchdog import execute_supervised_worker_payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Internal supervised worker for executing a single experiment run."
    )
    parser.add_argument("--input", required=True, help="Path to run kwargs JSON.")
    parser.add_argument("--output", required=True, help="Path to worker result JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        _write_json(
            output_path,
            {
                "ok": False,
                "error": "Invalid worker input payload; expected JSON object.",
                "failure_payload": {
                    "error_code": "worker_input_invalid",
                    "error_type": "ValueError",
                    "failure_stage": "watchdog_worker",
                    "error_details": {},
                },
            },
        )
        return 2

    run_kwargs = payload.get("run_kwargs")
    worker_payload = execute_supervised_worker_payload(run_kwargs=run_kwargs)  # type: ignore[arg-type]
    _write_json(output_path, worker_payload)
    return 0 if bool(worker_payload.get("ok", False)) else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
