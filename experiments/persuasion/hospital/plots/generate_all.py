from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Set

from experiments.common.plotting.io_utils import ensure_dir, infer_labels_from_sweep_dir, write_csv, write_json
from experiments.common.plotting.load_runs import load_runs
from experiments.common.plotting.logging_utils import configure_basic_logging

from experiments.agent_misalignment.plots.plot_overview import plot_overview
from experiments.agent_misalignment.plots.plot_sweep import plot_sweep_metrics

from .build_tables import build_tables
from .plot_persuasion_variants import plot_persuasion_variant_effects


def _parse_seeds_spec(spec: str) -> Set[int]:
    out: Set[int] = set()
    s = (spec or "").strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            lo_s, _, hi_s = p.partition("-")
            lo = int(lo_s.strip())
            hi = int(hi_s.strip())
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(p))
    return out


def _filter_rows_by_seeds(rows: List[dict], seeds: Optional[Set[int]]) -> List[dict]:
    if not seeds:
        return rows
    out = []
    for r in rows:
        try:
            seed = int(r.get("seed")) if r.get("seed") is not None else None
        except Exception:
            seed = None
        if seed is None:
            continue
        if seed in seeds:
            out.append(r)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate plots for a persuasion_hospital sweep directory."
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Path like experiments/persuasion/hospital/outputs/<tag>/<ts>/runs/<model>/<sweep_name>",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/persuasion/hospital/plots_outputs/<tag>/<ts>/<model>/<sweep_name>)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help='Optional seed filter like "0-9" or "0-9,11,15-16".',
    )
    args = parser.parse_args(argv)

    configure_basic_logging()

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    runs, meta = load_runs(sweep_dir)
    tables = build_tables(runs)

    seed_filter = _parse_seeds_spec(args.seeds) if args.seeds else None
    if seed_filter:
        tables = type(tables)(
            run_rows=_filter_rows_by_seeds(tables.run_rows, seed_filter),
            agent_rows=_filter_rows_by_seeds(tables.agent_rows, seed_filter),
        )

    labels = infer_labels_from_sweep_dir(sweep_dir)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path("experiments/persuasion/hospital/plots_outputs")
        / labels.experiment_tag
        / labels.timestamp
        / labels.model_label
        / labels.sweep_name
    )
    ensure_dir(out_dir)

    write_json(out_dir / "meta.json", {"meta": meta}, sort_keys=True)
    write_csv(out_dir / "run_rows.csv", tables.run_rows)
    write_csv(out_dir / "agent_rows.csv", tables.agent_rows)

    # Reuse the agent_misalignment suite (metrics now line up).
    plot_sweep_metrics(
        run_rows=tables.run_rows,
        agent_rows=tables.agent_rows,
        out_dir=out_dir / "misalignment_style",
    )
    plot_overview(
        run_rows=tables.run_rows,
        benign_run_rows=None,
        out_dir=out_dir / "overview",
    )

    # Persuasion-specific plots over prompt variants + secret-channel usage.
    plot_persuasion_variant_effects(run_rows=tables.run_rows, out_dir=out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
