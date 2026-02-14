"""CLI entrypoint to run the deterministic baseline pipeline."""

from __future__ import annotations

import argparse
import json

from Thesis_ML.pipelines.baseline import run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Thesis_ML synthetic baseline experiment.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument("--n-samples", type=int, default=500, help="Number of synthetic samples.")
    parser.add_argument("--n-features", type=int, default=12, help="Number of synthetic features.")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.3,
        help="Gaussian noise standard deviation.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction reserved for testing.",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Output directory for metrics.json (default: reports).",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Output directory for baseline_model.npz (default: models).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_baseline(
        seed=args.seed,
        n_samples=args.n_samples,
        n_features=args.n_features,
        noise_std=args.noise_std,
        test_fraction=args.test_fraction,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
    )

    summary = {
        "metrics": result["metrics"],
        "metrics_path": result["metrics_path"],
        "model_path": result["model_path"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
