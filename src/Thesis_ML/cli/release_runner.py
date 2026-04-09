from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from Thesis_ML.release.models import RunClass
from Thesis_ML.release.runner import run_release


class _ReleaseTerminalEventPrinter:
    def __init__(self, *, verbose: bool, live_progress: bool) -> None:
        self._verbose = bool(verbose)
        self._live_progress = bool(live_progress)
        self._total_runs: int | None = None
        self._completed_run_ids: set[str] = set()
        self._status_counts: dict[str, int] = {
            "success": 0,
            "failed": 0,
            "timed_out": 0,
            "skipped_due_to_policy": 0,
            "planned": 0,
            "unknown": 0,
        }

    def __call__(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        event_name = str(event.get("event", "")).strip()
        if not event_name:
            return
        self._update_total_runs(event)
        if self._verbose:
            self._emit_verbose(event_name=event_name, event=event)
        if self._live_progress:
            self._emit_live_progress(event_name=event_name, event=event)

    def _update_total_runs(self, event: dict[str, Any]) -> None:
        for key in ("n_runs", "n_compiled_runs"):
            value = event.get(key)
            if isinstance(value, int) and value >= 0:
                self._total_runs = int(value)
                return

    def _emit_verbose(self, *, event_name: str, event: dict[str, Any]) -> None:
        run_id = str(event.get("run_id", "")).strip()
        status = str(event.get("status", "")).strip()
        if event_name == "release_runner.start":
            message = (
                "release start "
                f"run_class={event.get('run_class')} "
                f"release_ref={event.get('release_ref')}"
            )
        elif event_name == "release_runner.validation_passed":
            message = (
                "release validated "
                f"release_id={event.get('release_id')} "
                f"version={event.get('release_version')}"
            )
        elif event_name == "release_runner.scope_compiled":
            message = (
                "scope compiled "
                f"rows={event.get('selected_row_count')} "
                f"sha256={event.get('selected_sample_ids_sha256')}"
            )
        elif event_name in {"parallel_execution.run_admitted", "parallel_execution.run_completed"}:
            message = (
                f"{event_name} run_id={run_id} "
                f"status={status or 'n/a'} lane={event.get('resource_lane_hint')}"
            )
        else:
            message = f"{event_name} {json.dumps(event, sort_keys=True)}"
        print(f"[thesisml] {message}", file=sys.stderr, flush=True)

    def _emit_live_progress(self, *, event_name: str, event: dict[str, Any]) -> None:
        if event_name not in {"parallel_execution.run_completed", "runtime_protocol.run_completed"}:
            return
        run_id = str(event.get("run_id", "")).strip()
        if not run_id or run_id in self._completed_run_ids:
            return
        self._completed_run_ids.add(run_id)
        status_raw = str(event.get("status", "unknown")).strip().lower()
        status = status_raw if status_raw in self._status_counts else "unknown"
        self._status_counts[status] = int(self._status_counts.get(status, 0)) + 1
        completed = len(self._completed_run_ids)
        total_text = str(self._total_runs) if self._total_runs is not None else "?"
        print(
            "[thesisml-live] "
            f"completed={completed}/{total_text} "
            f"success={self._status_counts['success']} "
            f"failed={self._status_counts['failed']} "
            f"timed_out={self._status_counts['timed_out']} "
            f"skipped={self._status_counts['skipped_due_to_policy']} "
            f"planned={self._status_counts['planned']}",
            file=sys.stderr,
            flush=True,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run release-authorized thesis execution for scratch/exploratory/candidate classes. "
            "Official outputs are promotion-only."
        )
    )
    parser.add_argument("--release", required=True, help="Release bundle path or release alias.")
    parser.add_argument(
        "--dataset-manifest",
        required=True,
        help="Dataset instance manifest JSON path.",
    )
    parser.add_argument(
        "--run-class",
        required=True,
        choices=[RunClass.SCRATCH.value, RunClass.EXPLORATORY.value, RunClass.CANDIDATE.value],
        help="Run class to materialize. official is promotion-only and rejected here.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing run directory.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Forward resume behavior to underlying execution for allowed run classes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile and write artifacts without executing worker runs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit stage and per-run execution events to stderr.",
    )
    parser.add_argument(
        "--live-progress",
        action="store_true",
        help="Emit rolling completion counters to stderr while runs execute.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    event_callback = None
    if bool(args.verbose) or bool(args.live_progress):
        event_callback = _ReleaseTerminalEventPrinter(
            verbose=bool(args.verbose),
            live_progress=bool(args.live_progress),
        )

    try:
        result = run_release(
            release_ref=args.release,
            dataset_manifest_path=args.dataset_manifest,
            run_class=RunClass(args.run_class),
            force=bool(args.force),
            resume=bool(args.resume),
            dry_run=bool(args.dry_run),
            command_line=[
                "thesisml-run-release",
                "--release",
                str(args.release),
                "--dataset-manifest",
                str(args.dataset_manifest),
                "--run-class",
                str(args.run_class),
            ]
            + (["--force"] if args.force else [])
            + (["--resume"] if args.resume else [])
            + (["--dry-run"] if args.dry_run else [])
            + (["--verbose"] if args.verbose else [])
            + (["--live-progress"] if args.live_progress else []),
            event_callback=event_callback,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
