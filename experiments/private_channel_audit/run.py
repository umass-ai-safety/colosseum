from __future__ import annotations

"""Entry point for the prompted-JIRA private-channel audit A/B experiment."""

import argparse
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.collusion.run import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the private-channel audit A/B experiment."
    )
    parser.add_argument(
        "--config",
        default="experiments/private_channel_audit/configs/jira_prompted_simple.yaml",
        help="Experiment YAML configuration.",
    )
    parser.add_argument("--out-dir", default=None, help="Override output root directory.")
    parser.add_argument(
        "--resume-dir",
        default=None,
        help=(
            "Existing experiment root to reuse. Fully completed, exactly matching "
            "runs are skipped; missing runs are executed there."
        ),
    )
    parser.add_argument(
        "--max-concurrent-runs",
        type=int,
        default=None,
        help="Maximum number of runs to execute in parallel.",
    )
    args = parser.parse_args()
    root = asyncio.run(
        run_from_config(
            args.config,
            out_dir=args.out_dir,
            resume_dir=args.resume_dir,
            max_concurrent_runs=args.max_concurrent_runs,
        )
    )
    print(f"Wrote results to: {root}")


if __name__ == "__main__":
    main()
