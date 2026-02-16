"""Sweep plotting for the network influence experiment.

This module intentionally contains only the code needed to generate `summary_mean.png`
via `experiments.network_influence.plots.generate_all`.

Plot semantics:
- solid line: `joint_reward_ratio` (environment-provided performance metric)
- dashed line: `prop_misinfo_mean` (misinformation propagation among non-adversaries, excluding the victim)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.common.plotting.io_utils import (
    as_float,
    as_int,
    ensure_dir,
    finite,
    groupby,
    mean,
    safe_load_json,
    sem,
    sorted_unique,
)
from experiments.common.plotting.logging_utils import log_saved_plot
from experiments.common.plotting.style import apply_default_style


logger = logging.getLogger(__name__)

_TOPOLOGY_PALETTE = [
    "#264653",  # Charcoal Blue
    "#2a9d8f",  # Verdigris
    "#8ab17d",  # Muted Olive
    "#e9c46a",  # Jasmine
    "#f4a261",  # Sandy Brown
    "#e76f51",  # Burnt Peach
]


def _format_param_float(x: float) -> str:
    s = (f"{float(x):.3f}").rstrip("0").rstrip(".")
    return s if s else "0"


def _finite_int(values: List[Any]) -> List[int]:
    out: List[int] = []
    for v in values:
        if v is None:
            continue
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _pretty_topology_label(
    topo: Any,
    rows: List[Dict[str, Any]],
    *,
    cn_params: Optional[Dict[str, Any]] = None,
    abbreviate_random: bool = True,
) -> str:
    s = str(topo or "").strip()
    if not s:
        return s

    key = s.lower().replace("-", "_").replace(" ", "_")
    alias_to_canonical = {"er": "erdos_renyi", "ws": "watts_strogatz", "ba": "barabasi_albert"}
    canonical = alias_to_canonical.get(key, key)

    full_names = {
        "erdos_renyi": "Erdős–Rényi",
        "barabasi_albert": "Barabási–Albert",
        "watts_strogatz": "Watts–Strogatz",
    }
    abbrev_names = {
        "erdos_renyi": "ER",
        "barabasi_albert": "BA",
        "watts_strogatz": "WS",
    }
    if abbreviate_random and canonical in abbrev_names:
        base = abbrev_names[canonical]
    else:
        base = full_names.get(canonical, s.replace("_", " ").title())

    if canonical == "erdos_renyi":
        ps = finite([r.get("cn_edge_prob") for r in rows])
        if ps:
            p_unique = sorted({round(float(p), 6) for p in ps})
            if len(p_unique) == 1:
                return f"{base} (p={_format_param_float(p_unique[0])})"
        if cn_params:
            p = as_float(cn_params.get("edge_prob"))
            if p is not None:
                return f"{base} (p={_format_param_float(p)})"
        return base

    if canonical == "watts_strogatz":
        ks = _finite_int([r.get("cn_k") for r in rows])
        ps = finite([r.get("cn_rewire_prob") for r in rows])
        parts: List[str] = []
        if ks:
            k_unique = sorted({int(k) for k in ks})
            if len(k_unique) == 1:
                parts.append(f"k={k_unique[0]}")
        if ps:
            p_unique = sorted({round(float(p), 6) for p in ps})
            if len(p_unique) == 1:
                parts.append(f"p={_format_param_float(p_unique[0])}")
        if cn_params:
            if not ks:
                k = as_int(cn_params.get("k"))
                if k is not None:
                    parts.append(f"k={int(k)}")
            if not ps:
                p = as_float(cn_params.get("rewire_prob"))
                if p is not None:
                    parts.append(f"p={_format_param_float(p)}")
        if parts:
            return f"{base} ({', '.join(parts)})"
        return base

    if canonical == "barabasi_albert":
        ms = _finite_int([r.get("cn_m") for r in rows])
        if ms:
            m_unique = sorted({int(m) for m in ms})
            if len(m_unique) == 1:
                return f"{base} (m={m_unique[0]})"
        if cn_params:
            m = as_int(cn_params.get("m"))
            if m is not None:
                return f"{base} (m={int(m)})"
        return base

    return base


def _ordered_topologies(topologies: List[Any]) -> List[Any]:
    preferred = [
        "path",
        "star",
        "complete",
        "watts_strogatz",
        "barabasi_albert",
        "erdos_renyi",
    ]
    preferred_idx = {name: i for i, name in enumerate(preferred)}

    def _key(t: Any) -> Tuple[int, str]:
        s = str(t or "")
        idx = preferred_idx.get(s, preferred_idx.get(s.lower(), 10_000))
        return (idx, s)

    return sorted(topologies, key=_key)


def _mean_sem_series(
    rows: List[Dict[str, Any]], *, adv_counts: List[Any], y_key: str
) -> Tuple[List[float], List[float], List[float]]:
    by_adv = groupby(rows, ("adversary_count",))
    xs: List[float] = []
    ys_mean: List[float] = []
    ys_sem: List[float] = []
    for a in adv_counts:
        ys = finite([r.get(y_key) for r in by_adv.get((a,), [])])
        if not ys:
            continue
        xs.append(float(a))
        ys_mean.append(mean(ys))
        ys_sem.append(sem(ys))
    return xs, ys_mean, ys_sem


def _plot_mean_with_sem_band(
    *,
    ax: Any,
    rows: List[Dict[str, Any]],
    adv_counts: List[Any],
    y_key: str,
    color: Any,
    linestyle: str = "-",
    marker: str = "o",
    band_alpha: float = 0.22,
    linewidth: float = 3.0,
    markersize: float = 8.8,
) -> None:
    xs, ys_mean, ys_sem = _mean_sem_series(rows, adv_counts=adv_counts, y_key=y_key)
    if not xs:
        return

    y = np.array(ys_mean, dtype=float)
    yerr = np.array(ys_sem, dtype=float)
    ax.fill_between(
        xs,
        (y - yerr).tolist(),
        (y + yerr).tolist(),
        color=color,
        alpha=band_alpha,
        linewidth=0.0,
    )
    ax.plot(
        xs,
        ys_mean,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        alpha=0.95,
    )


def _apply_fixed_y_grid(ax: Any) -> None:
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, alpha=0.9)


def _load_sweep_communication_network_params(
    run_stats: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Best-effort load of the sweep-level config.json to recover communication_network params.
    Needed because run_config.json files may omit these.
    """
    for r in run_stats:
        run_dir = r.get("run_dir")
        if not run_dir:
            continue
        try:
            sweep_dir = Path(str(run_dir)).expanduser().resolve().parent
        except Exception:
            sweep_dir = Path(str(run_dir)).parent

        # Search upwards for config.json (expects it at the timestamp folder).
        for parent in [sweep_dir, *list(sweep_dir.parents)[:8]]:
            cfg_path = parent / "config.json"
            if not cfg_path.exists():
                continue
            cfg = safe_load_json(cfg_path)
            if not isinstance(cfg, dict):
                continue
            cn = cfg.get("communication_network")
            if isinstance(cn, dict) and cn:
                return cn
    return {}


def build_run_stats(
    *,
    run_rows: List[Dict[str, Any]],
    target_rows: List[Dict[str, Any]],
    agent_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Produces one row per run_id with:
      - misinformation propagation mean across targets (computed over non-adversaries, excluding the target agent)
    """
    by_run_targets = groupby(target_rows, ("run_id",))
    by_run_agents = groupby(agent_rows, ("run_id", "target_agent"))

    def _prop_rate_from_agents(
        agents: List[Dict[str, Any]],
        *,
        target_agent: str,
        belief_key: str,
    ) -> float:
        denom = 0
        num = 0
        for a in agents:
            if str(a.get("role")) == "adversary":
                continue
            if str(a.get("agent_name")) == str(target_agent):
                continue
            denom += 1
            if bool(a.get(belief_key, False)):
                num += 1
        return (num / denom) if denom else 0.0

    out: List[Dict[str, Any]] = []
    for rr in run_rows:
        run_id = rr.get("run_id")
        if not run_id:
            continue

        targets = by_run_targets.get((run_id,), [])
        prop_vals: List[float] = []
        for t in targets:
            ta = t.get("target_agent")
            if not ta:
                continue
            agents = by_run_agents.get((run_id, ta), [])
            if agents:
                prop_vals.append(
                    _prop_rate_from_agents(
                        agents,
                        target_agent=str(ta),
                        belief_key="believes_misinformation",
                    )
                )
            else:
                # Fallback (should be rare): approximate using target-level counts,
                # assuming the target does not believe misinformation about themselves.
                total_agents = t.get("total_agents")
                adv_count = t.get("adversary_count")
                mis_believers = t.get("believes_misinformation_non_adversary")
                if total_agents is None or adv_count is None or mis_believers is None:
                    continue
                try:
                    non_adv = int(total_agents) - int(adv_count)
                    denom = non_adv - 1
                    if denom <= 0:
                        prop_vals.append(0.0)
                    else:
                        rate = float(int(mis_believers)) / float(denom)
                        prop_vals.append(max(0.0, min(1.0, rate)))
                except Exception:
                    continue

        row = dict(rr)
        row.update({"prop_misinfo_mean": mean(prop_vals) if prop_vals else None})
        out.append(row)
    return out


def plot_sweep_summary(
    run_stats: List[Dict[str, Any]],
    *,
    out_path: Path,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    topologies_present = sorted_unique(run_stats, "topology")
    if not topologies_present:
        return

    adv_counts = sorted_unique(run_stats, "adversary_count")
    prop_key = "prop_misinfo_mean"

    cn_params = _load_sweep_communication_network_params(run_stats)
    topologies = _ordered_topologies(topologies_present)
    color_by_topo: Dict[Any, Any] = {}
    for i, topo in enumerate(topologies):
        color_by_topo[topo] = _TOPOLOGY_PALETTE[i % len(_TOPOLOGY_PALETTE)]

    deterministic_order = ["path", "star", "complete"]
    random_order = ["watts_strogatz", "barabasi_albert", "erdos_renyi"]
    top_row = [t for t in deterministic_order if t in topologies_present]
    bottom_row = [t for t in random_order if t in topologies_present]
    extras = [t for t in topologies if t not in top_row and t not in bottom_row]
    bottom_row.extend(extras)

    ncols = max(len(top_row), len(bottom_row))
    if ncols <= 0:
        return

    title_fs = 19
    label_fs = 17
    tick_fs = 15
    legend_fs = 15
    summary_linewidth = 3.0

    fig, axes = plt.subplots(
        nrows=2,
        ncols=ncols,
        figsize=(4.2 * ncols * 0.7, 6.2 * 0.95),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(2, ncols)

    for row in range(2):
        row_topos = top_row if row == 0 else bottom_row
        for col in range(ncols):
            ax = axes[row, col]
            if col >= len(row_topos):
                ax.axis("off")
                continue
            topo = row_topos[col]
            subset = [r for r in run_stats if r.get("topology") == topo]
            color = color_by_topo.get(topo, "#4c72b0")

            _plot_mean_with_sem_band(
                ax=ax,
                rows=subset,
                adv_counts=adv_counts,
                y_key="joint_reward_ratio",
                color=color,
                linestyle="-",
                marker="o",
                band_alpha=0.22,
                linewidth=summary_linewidth,
            )
            _plot_mean_with_sem_band(
                ax=ax,
                rows=subset,
                adv_counts=adv_counts,
                y_key=prop_key,
                color=color,
                linestyle="--",
                marker="o",
                band_alpha=0.14,
                linewidth=summary_linewidth,
            )

            ax.set_title(
                _pretty_topology_label(
                    topo,
                    subset,
                    cn_params=cn_params,
                    abbreviate_random=True,
                ),
                fontsize=title_fs,
            )
            ax.set_ylim(0.0, 1.05)
            _apply_fixed_y_grid(ax)
            ax.tick_params(axis="both", labelsize=tick_fs)

            if adv_counts:
                ax.set_xticks(
                    [float(a) for a in adv_counts],
                    [str(a) for a in adv_counts],
                )
            if row == 1:
                ax.set_xlabel("Adversary Count", fontsize=label_fs)

    metric_handles = [
        Line2D([0], [0], color="#111827", linewidth=summary_linewidth, linestyle="-"),
        Line2D([0], [0], color="#111827", linewidth=summary_linewidth, linestyle="--"),
    ]
    fig.legend(
        metric_handles,
        ["Joint Reward Ratio", "Misinformation Propagation Rate"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        frameon=False,
        fontsize=legend_fs,
    )

    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.97))
    fig.subplots_adjust(wspace=0.10, hspace=0.30)
    fig.savefig(out_path, bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)
