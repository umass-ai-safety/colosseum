from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from experiments.common.plotting.io_utils import ensure_dir, finite, groupby, mean, sem
from experiments.common.plotting.logging_utils import log_saved_plot
from experiments.common.plotting.style import apply_default_style

logger = logging.getLogger(__name__)

_STYLE = {
    # Paper-friendly defaults (reasonable in LaTeX figures).
    "font.size": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 26,
}


def _unique_non_null(rows: List[Dict[str, Any]], key: str) -> List[Any]:
    seen = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        if v not in seen:
            seen.append(v)
    return seen


def _infer_x_key(run_rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Picks a sweep axis based on what varies in the data.

    Returns (x_key, x_label).
    """
    if len(_unique_non_null(run_rows, "adversary_count")) > 1:
        return ("adversary_count", "Adversary Count")
    if len(_unique_non_null(run_rows, "target_role")) > 1:
        return ("target_role", "Target Role")
    if len(_unique_non_null(run_rows, "num_agents")) > 1:
        return ("num_agents", "Number of Agents")
    return ("run_id", "Run")


def _is_numeric(values: Iterable[Any]) -> bool:
    for v in values:
        if v is None:
            continue
        try:
            float(v)
        except Exception:
            return False
    return True


def _plot_metric_by_x(
    run_rows: List[Dict[str, Any]],
    *,
    x_key: str,
    x_label: str,
    y_key: str,
    y_label: str,
    out_path: Path,
) -> None:
    apply_default_style(plt)
    plt.rcParams.update(_STYLE)
    ensure_dir(out_path.parent)

    xs = [r.get(x_key) for r in run_rows]
    if not xs:
        return
    numeric_x = _is_numeric(xs)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    if numeric_x:
        by_x = groupby(run_rows, (x_key,))
        x_vals = sorted([x for x in by_x.keys() if x[0] is not None], key=lambda t: float(t[0]))
        for x_t in x_vals:
            x = x_t[0]
            rows = by_x.get((x,), [])
            ys = finite([r.get(y_key) for r in rows])
            if not ys:
                continue
            # scatter
            for i, r in enumerate(rows):
                y = r.get(y_key)
                if y is None:
                    continue
                ax.scatter(float(x), float(y), s=22, alpha=0.55, color="#4c78a8")
            ax.errorbar(
                [float(x)],
                [mean(ys)],
                yerr=[sem(ys)],
                fmt="o",
                ms=6,
                color="black",
                capsize=3,
                zorder=5,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    else:
        cats = [str(v) for v in _unique_non_null(run_rows, x_key)]
        if not cats:
            return
        by_x = groupby(run_rows, (x_key,))
        means: List[float] = []
        errs: List[float] = []
        for c in cats:
            ys = finite([r.get(y_key) for r in by_x.get((c,), [])])
            means.append(mean(ys) if ys else 0.0)
            errs.append(sem(ys) if ys else 0.0)
        ax.bar(range(len(cats)), means, yerr=errs, capsize=3, color="#4c78a8", alpha=0.85)
        # overlay points
        for i, c in enumerate(cats):
            rows = by_x.get((c,), [])
            for r in rows:
                y = r.get(y_key)
                if y is None:
                    continue
                ax.scatter(i, float(y), s=22, alpha=0.55, color="#222222")
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=30, ha="right")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def plot_distance_effects(
    agent_rows: List[Dict[str, Any]],
    *,
    out_path: Path,
) -> None:
    """
    Aggregates non-adversary agents across runs and plots:
      - suspicion rate by distance to nearest adversary
      - correct attribution rate by distance to nearest adversary
    """
    apply_default_style(plt)
    plt.rcParams.update(_STYLE)
    ensure_dir(out_path.parent)

    rows = [
        r
        for r in agent_rows
        if str(r.get("role") or "") != "adversary"
        and r.get("distance_to_nearest_adversary") is not None
    ]
    if not rows:
        return

    by_d = groupby(rows, ("distance_to_nearest_adversary",))
    ds = sorted([int(k[0]) for k in by_d.keys() if k[0] is not None])
    if not ds:
        return

    susp_rates: List[float] = []
    corr_rates: List[float] = []
    for d in ds:
        bucket = by_d.get((d,), [])
        denom = len(bucket)
        if denom <= 0:
            susp_rates.append(0.0)
            corr_rates.append(0.0)
            continue
        susp = sum(1 for r in bucket if bool(r.get("suspects_foul_play")))
        corr = sum(1 for r in bucket if bool(r.get("correctly_blames_adversary")))
        susp_rates.append(susp / denom)
        corr_rates.append(corr / denom)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(ds, susp_rates, marker="o", label="Suspicion rate (non-adversary)")
    ax.plot(ds, corr_rates, marker="o", label="Correct attribution rate (non-adversary)")
    ax.set_xlabel("Distance to Nearest Adversary")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def plot_sweep_metrics(
    *,
    run_rows: List[Dict[str, Any]],
    agent_rows: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    x_key, x_label = _infer_x_key(run_rows)

    # Core outcomes
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="joint_reward",
        y_label="Joint Reward",
        out_path=out_dir / "joint_reward.pdf",
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="resource_failures_total",
        y_label="Total Resource Failures",
        out_path=out_dir / "resource_failures_total.pdf",
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="failed_patients_count",
        y_label="Failed Patients",
        out_path=out_dir / "failed_patients_count.pdf",
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="converged_patients",
        y_label="Converged Patients",
        out_path=out_dir / "converged_patients.pdf",
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="meeting_outcome_variables_assigned",
        y_label="Variables Assigned",
        out_path=out_dir / "variables_assigned.pdf",
    )

    # Belief / judge metrics
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="suspicion_rate_non_adversary",
        y_label="Suspicion Rate (Non-adversary)",
        out_path=out_dir / "suspicion_rate_non_adversary.pdf",
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="correct_attribution_rate_non_adversary",
        y_label="Correct Attribution Rate (Non-adversary)",
        out_path=out_dir / "correct_attribution_rate_non_adversary.pdf",
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="avg_distance_to_adversary_suspicious_agents",
        y_label="Avg Distance to Adversary (Suspicious Non-adversaries)",
        out_path=out_dir / "avg_distance_suspicious.pdf",
    )

    # Communication proxies
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="total_messages",
        y_label="Total Messages",
        out_path=out_dir / "total_messages.pdf",
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="complaint_messages",
        y_label="Complaint Messages",
        out_path=out_dir / "complaint_messages.pdf",
    )

    # Graph metrics (key logged fields)
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="graph_density",
        y_label="Graph Density",
        out_path=out_dir / "graph_density.pdf",
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="graph_avg_degree",
        y_label="Graph Avg Degree",
        out_path=out_dir / "graph_avg_degree.pdf",
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="graph_avg_clustering",
        y_label="Graph Avg Clustering",
        out_path=out_dir / "graph_avg_clustering.pdf",
    )

    # Coalition advantage (if present)
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="coalition_minus_noncoalition_avg",
        y_label="Coalition Advantage (Avg Reward Diff)",
        out_path=out_dir / "coalition_advantage.pdf",
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="coalition_reward_avg",
        y_label="Coalition Avg Reward",
        out_path=out_dir / "coalition_reward_avg.pdf",
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="noncoalition_reward_avg",
        y_label="Non-coalition Avg Reward",
        out_path=out_dir / "noncoalition_reward_avg.pdf",
    )

    # Distance effects (aggregated across runs)
    plot_distance_effects(agent_rows, out_path=out_dir / "distance_effects.pdf")
