from __future__ import annotations

import argparse
import json

from Thesis_ML.release.promotion import promote_candidate_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a verified candidate release run to the official singleton for its release_id."
        )
    )
    parser.add_argument(
        "--candidate-run",
        required=True,
        help="Candidate run path or candidate run_id.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result = promote_candidate_run(args.candidate_run)
    except Exception as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

