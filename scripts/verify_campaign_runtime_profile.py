from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from Thesis_ML.verification.campaign_runtime_profile import verify_campaign_runtime_profile
from time import perf_counter

from Thesis_ML.experiments.progress import ProgressEvent

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run precheck-only runtime profiling for confirmatory/comparison campaign plans "
            "and estimate campaign ETA by phase/model/cohort."
        )
    )
    parser.add_argument("--index-csv", required=True, help="Dataset index CSV path.")
    parser.add_argument("--data-root", required=True, help="Dataset data root path.")
    parser.add_argument("--cache-dir", required=True, help="Feature cache directory path.")
    parser.add_argument(
        "--confirmatory-protocol",
        required=True,
        help="Confirmatory protocol JSON path.",
    )
    parser.add_argument(
        "--comparison-spec",
        action="append",
        required=True,
        help="Comparison spec JSON path. Repeat for multiple specs.",
    )
    parser.add_argument(
        "--profile-root",
        required=True,
        help="Output root for runtime profiling precheck runs.",
    )
    parser.add_argument(
        "--summary-out",
        default="",
        help="Optional JSON summary output path.",
    )
    parser.add_argument(
        "--hardware-mode",
        choices=["cpu_only", "gpu_only", "max_both"],
        default="gpu_only",
        help="Compute hardware mode for profiling runs.",
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=0,
        help="CUDA device id to use for profiling runs.",
    )
    parser.add_argument(
        "--deterministic-compute",
        action="store_true",
        help="Enable deterministic GPU execution for profiling runs.",
    )
    parser.add_argument(
        "--allow-backend-fallback",
        action="store_true",
        help="Allow fallback to CPU if GPU is unavailable in exploratory profiling.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Disable live progress messages on stderr.",
    )
    return parser

class _ConsoleProgressReporter:
    def __init__(self) -> None:
        self._started = perf_counter()
        self._last_print_at = 0.0

    @staticmethod
    def _humanize_seconds(total_seconds: float | None) -> str:
        if total_seconds is None:
            return "unknown"
        seconds = max(0, int(round(float(total_seconds))))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def __call__(self, event: ProgressEvent) -> None:
        now = perf_counter()

        # Throttle noisy permutation updates.
        if (
            event.stage == "permutation"
            and event.completed_units is not None
            and event.total_units is not None
            and event.completed_units not in {0.0, event.total_units}
            and (now - self._last_print_at) < 0.5
        ):
            return

        elapsed = now - self._started
        fraction = None
        eta_seconds = None
        remaining_units = None

        if (
            event.completed_units is not None
            and event.total_units is not None
            and event.total_units > 0
        ):
            completed = float(event.completed_units)
            total = float(event.total_units)
            remaining_units = max(0.0, total - completed)
            if completed > 0.0:
                fraction = min(max(completed / total, 0.0), 1.0)
                eta_seconds = elapsed * (1.0 - fraction) / fraction

        percent_text = "--.-%"
        done_text = "?"
        left_text = "?"
        if fraction is not None and event.total_units is not None and event.completed_units is not None:
            percent_text = f"{100.0 * fraction:5.1f}%"
            done_text = f"{int(event.completed_units)}/{int(event.total_units)}"
            left_text = str(int(remaining_units or 0.0))

        phase = event.metadata.get("phase")
        model = event.metadata.get("model")
        section = event.metadata.get("section")
        fold_index = event.metadata.get("fold_index")
        total_folds = event.metadata.get("total_folds")
        lane = event.metadata.get("assigned_compute_lane")
        backend = event.metadata.get("effective_backend_family")
        hardware = event.metadata.get("hardware_mode_effective")

        suffix_parts: list[str] = []
        if phase is not None:
            suffix_parts.append(f"phase={phase}")
        if model is not None:
            suffix_parts.append(f"model={model}")
        if hardware is not None:
            suffix_parts.append(f"hw={hardware}")
        if lane is not None:
            suffix_parts.append(f"lane={lane}")
        if backend is not None:
            suffix_parts.append(f"backend={backend}")
        if section is not None:
            suffix_parts.append(f"section={section}")
        if fold_index is not None and total_folds is not None:
            suffix_parts.append(f"fold={fold_index}/{total_folds}")

        suffix = " | ".join(suffix_parts)

        line = (
            f"[{event.stage}] {event.message} | done {done_text} | left {left_text} "
            f"| elapsed {self._humanize_seconds(elapsed)} "
            f"| eta {self._humanize_seconds(eta_seconds)} "
            f"| {percent_text}"
        )
        if suffix:
            line = f"{line} | {suffix}"

        print(line, file=sys.stderr, flush=True)
        self._last_print_at = now

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    progress_callback = None if args.quiet_progress else _ConsoleProgressReporter()

    summary = verify_campaign_runtime_profile(
        index_csv=Path(args.index_csv),
        data_root=Path(args.data_root),
        cache_dir=Path(args.cache_dir),
        confirmatory_protocol=Path(args.confirmatory_protocol),
        comparison_specs=[Path(path) for path in list(args.comparison_spec)],
        profile_root=Path(args.profile_root),
        hardware_mode=args.hardware_mode,
        gpu_device_id=args.gpu_device_id,
        deterministic_compute=bool(args.deterministic_compute),
        allow_backend_fallback=bool(args.allow_backend_fallback),
        progress_callback=progress_callback,
    )

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(f"{json.dumps(summary, indent=2)}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not bool(summary.get("passed", False)):
        print("Campaign runtime profile precheck failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
