from __future__ import annotations

import argparse
import json

from Thesis_ML.release.validator import validate_dataset_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a dataset instance manifest against strict dataset contract checks."
    )
    parser.add_argument(
        "--dataset-manifest",
        required=True,
        help="Path to dataset manifest JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = validate_dataset_manifest(args.dataset_manifest)
    print(json.dumps(summary, indent=2))
    return 0 if bool(summary.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())

