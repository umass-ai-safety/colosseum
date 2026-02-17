from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from experiments.common.plotting.io_utils import (
    ensure_dir,
    infer_labels_from_sweep_dir,
    write_csv,
    write_json,
)
from experiments.common.plotting.logging_utils import configure_basic_logging
from experiments.common.plotting.load_runs import load_runs

from .build_tables import build_tables
from .plot_sweep import plot_sweep_metrics


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate plots for an emergent_misalignment sweep directory."
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Path like experiments/emergent_misalignment/outputs/<tag>/<ts>/runs/<model>/<sweep_name>",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/emergent_misalignment/plots_outputs/<tag>/<ts>/<model>/<sweep_name>)",
    )
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    runs, meta = load_runs(sweep_dir)
    tables = build_tables(runs)

    labels = infer_labels_from_sweep_dir(sweep_dir)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path("experiments/emergent_misalignment/plots_outputs")
        / labels.experiment_tag
        / labels.timestamp
        / labels.model_label
        / labels.sweep_name
    )
    ensure_dir(out_dir)

    write_json(
        out_dir / "meta.json",
        {
            "meta": meta,
            "counts": {
                "runs": len(runs),
                "run_rows": len(tables.run_rows),
                "agent_rows": len(tables.agent_rows),
            },
        },
        sort_keys=True,
    )
    write_csv(out_dir / "run_rows.csv", tables.run_rows)
    write_csv(out_dir / "agent_rows.csv", tables.agent_rows)

    sweep_out = out_dir / "sweep"
    ensure_dir(sweep_out)
    plot_sweep_metrics(
        run_rows=tables.run_rows, agent_rows=tables.agent_rows, out_dir=sweep_out
    )

    return 0


if __name__ == "__main__":
    configure_basic_logging()
    raise SystemExit(main())
