from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_step(command: list[str], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _run_shell_step(command: str, *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        shell=True,
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RC-1 release-candidate gate runner for hygiene, artifact checks, and reproducibility."
    )
    parser.add_argument(
        "--summary-out",
        default="outputs/release/rc1_gate_summary.json",
        help="Where to write RC gate summary JSON.",
    )
    parser.add_argument(
        "--run-ruff",
        action="store_true",
        help="Run ruff check as part of RC gate.",
    )
    parser.add_argument(
        "--run-pytest",
        action="store_true",
        help="Run full pytest suite as part of RC gate.",
    )
    parser.add_argument(
        "--run-performance-smoke",
        action="store_true",
        help="Run scripts/performance_smoke.py.",
    )
    parser.add_argument(
        "--verify-official-dir",
        action="append",
        default=[],
        help="Official output directory to verify with scripts/verify_official_artifacts.py (repeatable).",
    )
    parser.add_argument(
        "--repro-command",
        default="",
        help=(
            "Optional full command string to execute reproducibility verification "
            "(for example: 'python scripts/verify_official_reproducibility.py ...')."
        ),
    )
    parser.add_argument(
        "--confirmatory-ready-dir",
        default="",
        help=(
            "Optional confirmatory output directory to verify with "
            "scripts/verify_confirmatory_ready.py."
        ),
    )
    parser.add_argument(
        "--confirmatory-ready-summary-out",
        default="",
        help="Optional path to write confirmatory-ready summary JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    steps: list[dict[str, Any]] = []

    steps.append(_run_step(["python", "scripts/release_hygiene_check.py"], cwd=REPO_ROOT))

    if args.run_ruff:
        steps.append(
            _run_step(
                ["python", "-m", "ruff", "check", "src", "tests", "scripts"],
                cwd=REPO_ROOT,
            )
        )

    if args.run_pytest:
        steps.append(_run_step(["python", "-m", "pytest", "-q"], cwd=REPO_ROOT))

    if args.run_performance_smoke:
        steps.append(
            _run_step(
                [
                    "python",
                    "scripts/performance_smoke.py",
                    "--output",
                    "outputs/performance/performance_smoke_summary.json",
                ],
                cwd=REPO_ROOT,
            )
        )

    for directory in list(args.verify_official_dir):
        steps.append(
            _run_step(
                [
                    "python",
                    "scripts/verify_official_artifacts.py",
                    "--output-dir",
                    str(directory),
                ],
                cwd=REPO_ROOT,
            )
        )

    if args.confirmatory_ready_dir:
        command = [
            "python",
            "scripts/verify_confirmatory_ready.py",
            "--output-dir",
            str(args.confirmatory_ready_dir),
        ]
        if args.confirmatory_ready_summary_out:
            command.extend(
                [
                    "--summary-out",
                    str(args.confirmatory_ready_summary_out),
                ]
            )
        steps.append(_run_step(command, cwd=REPO_ROOT))

    if args.repro_command:
        steps.append(_run_shell_step(args.repro_command, cwd=REPO_ROOT))

    passed = all(int(step["returncode"]) == 0 for step in steps)
    summary = {
        "passed": bool(passed),
        "steps": steps,
    }

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(f"{json.dumps(summary, indent=2)}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not passed:
        print("RC-1 release gate failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
