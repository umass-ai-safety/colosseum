from __future__ import annotations

"""Generate the small set of supported collusion plots for a sweep directory.

This entrypoint mirrors `experiments.network_influence.plots.generate_all` but for
the collusion experiment.

It generates:
- `by_topology/<topo>/sweep/prompt_variants__mean_sem__cX__reward-<metric>.png`
- `sweep/topology_comparison__optimality_gap_and_judge__cX.png`
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe

from experiments.collusion.plots.common import default_out_dir
from experiments.collusion.plots.plot_sweep import (
    _canonical_topology_key,
    _select_colluder_count,
    build_rows,
    plot_collusion_hist_bars_se_by_topology,
    plot_combined_six_bars_by_topology,
)
from experiments.common.plotting.io_utils import ensure_dir, safe_load_json
from experiments.common.plotting.logging_utils import configure_basic_logging
from experiments.common.plotting.load_runs import load_runs


logger = logging.getLogger(__name__)


def _topology_title_overrides_from_config(
    *, sweep_dir: Path, titles_config_path: Optional[Path]
) -> Dict[str, str]:
    """
    Best-effort mapping from topology -> pretty title (including parameters).

    Used to annotate combined cross-topology plots. Keys are canonicalized
    (underscored) topology names (e.g., "watts_strogatz").
    """
    title_by_topology: Dict[str, str] = {}
    try:
        cfg_path = (
            titles_config_path
            if titles_config_path is not None
            else (sweep_dir.parent.parent.parent / "config.json")
        )
        cfg = safe_load_json(cfg_path) if cfg_path is not None else None

        edge_prob = None
        watts_k = None
        watts_rewire_prob = None
        ba_m = None
        if isinstance(cfg, dict):
            cn = cfg.get("communication_network")
            if isinstance(cn, dict):
                try:
                    edge_prob = float(cn.get("edge_prob")) if cn.get("edge_prob") is not None else None
                except Exception:
                    edge_prob = None
                try:
                    k_raw = cn.get("k", cn.get("nearest_neighbors", cn.get("num_neighbors")))
                    watts_k = int(k_raw) if k_raw is not None else None
                except Exception:
                    watts_k = None
                try:
                    p_raw = cn.get("rewire_prob", cn.get("rewiring_prob", cn.get("beta")))
                    watts_rewire_prob = float(p_raw) if p_raw is not None else None
                except Exception:
                    watts_rewire_prob = None
                try:
                    m_raw = cn.get("m", cn.get("edges_per_node", cn.get("num_edges_to_attach")))
                    ba_m = int(m_raw) if m_raw is not None else None
                except Exception:
                    ba_m = None

        if edge_prob is not None:
            p_str = (f"{float(edge_prob):.3f}").rstrip("0").rstrip(".")
            title_by_topology["erdos_renyi"] = f"Erdős–Rényi (p={p_str})"

        if watts_k is not None or watts_rewire_prob is not None:
            parts: List[str] = []
            if watts_k is not None:
                parts.append(f"k={int(watts_k)}")
            if watts_rewire_prob is not None:
                p_str = (f"{float(watts_rewire_prob):.3f}").rstrip("0").rstrip(".")
                parts.append(f"p={p_str}")
            if parts:
                title_by_topology["watts_strogatz"] = "Watts–Strogatz (" + ", ".join(parts) + ")"

        if ba_m is not None:
            title_by_topology["barabasi_albert"] = f"Barabási–Albert (m={int(ba_m)})"
    except Exception:
        return {}
    return title_by_topology


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate collusion sweep plots (bars + cross-topology comparison)."
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Path like experiments/collusion/outputs/<tag>/<ts>/runs/<model_label>/<sweep_name>",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/collusion/plots_outputs/<tag>/<ts>/<model_label>/<sweep_name>)",
    )
    parser.add_argument(
        "--colluder-count",
        type=int,
        default=None,
        help="Coalition size to plot (default: max colluder_count present).",
    )
    parser.add_argument(
        "--baseline-prompt-variant",
        type=str,
        default="control",
        help="Prompt variant used in baseline runs (secret_channel_enabled=false).",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include runs where status != 'complete' (not recommended).",
    )
    parser.add_argument(
        "--reward-metric",
        type=str,
        default="joint_reward_ratio",
        choices=["joint_reward_ratio", "achieved_over_optimal", "regret_ratio", "optimality_gap", "regret"],
        help="Reward metric to include in the per-topology bars plot.",
    )
    parser.add_argument(
        "--compute-optimal",
        action="store_true",
        help="If optimal_summary.json is missing, compute and write it (no API calls).",
    )
    parser.add_argument(
        "--prefer-repaired",
        action="store_true",
        help="Prefer *_repaired.json artifacts when present.",
    )
    parser.add_argument(
        "--titles-config",
        type=str,
        default=None,
        help="Optional config.json/yaml to read topology parameter labels (ER p, WS k/p, BA m).",
    )
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    runs, _ = load_runs(sweep_dir, prefer_repaired=bool(args.prefer_repaired))

    # Optional: compute optimal_summary.json for Jira runs so `optimality_gap` is available.
    if bool(args.compute_optimal):
        from experiments.collusion.plots.generate_jira_regret_report import (
            _compute_and_write_optimal_summary as _compute_optimal_summary,  # noqa: WPS433
        )

        for run in runs:
            run_dir = getattr(run, "run_dir", None)
            if not isinstance(run_dir, Path):
                continue
            if (run_dir / "optimal_summary.json").exists():
                continue
            _compute_optimal_summary(run_dir)

    rows: List[Dict[str, Any]] = build_rows(runs)
    out_dir = default_out_dir(sweep_dir=sweep_dir, requested_out_dir=args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "sweep")

    colluder_count = _select_colluder_count(rows, args.colluder_count)
    baseline_variant = str(args.baseline_prompt_variant or "control").strip() or "control"

    titles_config_path = (
        Path(str(args.titles_config)).expanduser().resolve() if args.titles_config else None
    )
    title_by_topology = _topology_title_overrides_from_config(
        sweep_dir=sweep_dir, titles_config_path=titles_config_path
    )

    # 1) Per-topology bar plot (mean ± SEM) for all prompt variants.
    plot_collusion_hist_bars_se_by_topology(
        rows=rows,
        out_dir=out_dir,
        colluder_count=int(colluder_count),
        baseline_variant=baseline_variant,
        include_incomplete=bool(args.include_incomplete),
        reward_metric=str(args.reward_metric),
    )

    # 2) Single combined figure comparing topologies (optimality gap + judge).
    compare_rows = list(rows)
    if "complete" not in {_canonical_topology_key(r.get("topology")) for r in rows}:
        sibling_complete = sweep_dir.parent / "complete"
        if sibling_complete.exists() and sibling_complete.is_dir():
            extra_runs, _ = load_runs(
                sibling_complete, prefer_repaired=bool(args.prefer_repaired)
            )
            compare_rows.extend(build_rows(extra_runs))

    plot_combined_six_bars_by_topology(
        rows=compare_rows,
        metric_key="optimality_gap",
        out_path=out_dir
        / "sweep"
        / f"topology_comparison__optimality_gap_and_judge__c{colluder_count}.png",
        colluder_count=int(colluder_count),
        baseline_variant=baseline_variant,
        include_incomplete=bool(args.include_incomplete),
        title_by_topology=title_by_topology or None,
    )

    return 0


if __name__ == "__main__":
    configure_basic_logging()
    raise SystemExit(main())
