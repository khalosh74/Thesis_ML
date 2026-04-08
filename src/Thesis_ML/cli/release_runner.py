from __future__ import annotations

import argparse
import json

from Thesis_ML.release.models import RunClass
from Thesis_ML.release.runner import run_release


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

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
            + (["--dry-run"] if args.dry_run else []),
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

