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

from .build_tables import Tables, build_tables
from .plot_sweep import build_run_stats, plot_sweep_summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the sweep summary plot for a sweep directory."
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Path like experiments/network_influence/outputs/<tag>/<ts>/runs/<model>/<sweep_name>",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/network_influence/plots_outputs/<tag>/<ts>/<model>/<sweep_name>)",
    )
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    runs, meta = load_runs(sweep_dir)
    tables: Tables = build_tables(runs)
    run_stats = build_run_stats(
        run_rows=tables.run_rows,
        target_rows=tables.target_rows,
        agent_rows=tables.agent_rows,
    )

    labels = infer_labels_from_sweep_dir(sweep_dir)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path("experiments/network_influence/plots_outputs")
        / labels.experiment_tag
        / labels.timestamp
        / labels.model_label
        / labels.sweep_name
    )
    ensure_dir(out_dir)

    # Persist data tables for reproducibility/debugging.
    write_json(
        out_dir / "meta.json",
        {
            "meta": meta,
            "plot_config": {"exclude_target_from_propagation": True},
            "counts": {
                "runs": len(runs),
                "run_rows": len(tables.run_rows),
                "target_rows": len(tables.target_rows),
                "agent_rows": len(tables.agent_rows),
            },
        },
        sort_keys=True,
    )
    write_csv(out_dir / "run_rows.csv", tables.run_rows)
    write_csv(out_dir / "target_rows.csv", tables.target_rows)
    write_csv(out_dir / "run_stats.csv", run_stats)

    sweep_out = out_dir / "sweep"
    ensure_dir(sweep_out)
    plot_sweep_summary(
        run_stats,
        out_path=sweep_out / "summary_mean.png",
    )

    return 0


if __name__ == "__main__":
    configure_basic_logging()
    raise SystemExit(main())
