from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from experiments.common.plotting.io_utils import (
    ensure_dir,
    finite,
    groupby,
    mean,
    sem,
    write_csv,
)
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


def _add_violin(
    ax: Any,
    *,
    pos: float,
    ys: List[float],
    color: str,
    width: float = 0.28,
    alpha: float = 0.35,
) -> None:
    if not ys:
        return
    parts = ax.violinplot(
        [ys],
        positions=[pos],
        widths=width,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for b in parts.get("bodies", []):
        b.set_facecolor(color)
        b.set_edgecolor(color)
        b.set_alpha(alpha)
    if parts.get("cmedians") is not None:
        parts["cmedians"].set_color(color)
        parts["cmedians"].set_linewidth(1.6)


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
    baseline_rows: Optional[List[Dict[str, Any]]] = None,
    baseline_label: str = "Benign baseline",
    emit_artifacts: bool = True,
) -> None:
    apply_default_style(plt)
    plt.rcParams.update(_STYLE)
    ensure_dir(out_path.parent)

    xs = [r.get(x_key) for r in run_rows]
    if not xs:
        return
    numeric_x = _is_numeric(xs)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.grid(False)

    baseline_color = "#f58518"
    main_color = "#4c78a8"

    if numeric_x:
        by_x = groupby(run_rows, (x_key,))
        x_vals = sorted([x for x in by_x.keys() if x[0] is not None], key=lambda t: float(t[0]))
        for x_t in x_vals:
            x = x_t[0]
            rows = by_x.get((x,), [])
            ys = finite([r.get(y_key) for r in rows])
            if not ys:
                continue
            _add_violin(ax, pos=float(x) - 0.11, ys=ys, color=main_color)
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

        if baseline_rows:
            by_x_base = groupby(baseline_rows, (x_key,))
            x_vals_base = sorted(
                [x for x in by_x_base.keys() if x[0] is not None], key=lambda t: float(t[0])
            )
            for x_t in x_vals_base:
                x = x_t[0]
                rows = by_x_base.get((x,), [])
                ys = finite([r.get(y_key) for r in rows])
                if not ys:
                    continue
                _add_violin(ax, pos=float(x) + 0.11, ys=ys, color=baseline_color)
                ax.errorbar(
                    [float(x)],
                    [mean(ys)],
                    yerr=[sem(ys)],
                    fmt="s",
                    ms=5.5,
                    color=baseline_color,
                    capsize=3,
                    zorder=6,
                )
            ax.scatter([], [], s=28, color=baseline_color, label=baseline_label, marker="s")
            ax.scatter([], [], s=28, color=main_color, label="Main sweep", marker="o")
            ax.legend(loc="best")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    else:
        cats = [str(v) for v in _unique_non_null(run_rows, x_key)]
        if not cats:
            return
        by_x = groupby(run_rows, (x_key,))
        for i, c in enumerate(cats):
            ys = finite([r.get(y_key) for r in by_x.get((c,), [])])
            if not ys:
                continue
            _add_violin(ax, pos=float(i) - 0.11, ys=ys, color=main_color)
            ax.errorbar(
                [float(i)],
                [mean(ys)],
                yerr=[sem(ys)],
                fmt="o",
                ms=6,
                color="black",
                capsize=3,
                zorder=5,
            )

        if baseline_rows:
            # If the benign/baseline sweep didn't enumerate the same categorical axis
            # (e.g., it only has target_role="departmental" with adversary_count=0),
            # treat it as a baseline for *all* categories in the main sweep.
            baseline_rows_for_plot = baseline_rows
            base_cats = [str(v) for v in _unique_non_null(baseline_rows, x_key)]
            if len(base_cats) == 1 and len(cats) > 1:
                baseline_rows_for_plot = []
                for c in cats:
                    for r in baseline_rows:
                        rr = dict(r)
                        rr[x_key] = c
                        baseline_rows_for_plot.append(rr)

            by_x_base = groupby(baseline_rows_for_plot, (x_key,))
            for i, c in enumerate(cats):
                rows = by_x_base.get((c,), [])
                if not rows:
                    continue
                ys = finite([r.get(y_key) for r in rows])
                if not ys:
                    continue
                _add_violin(ax, pos=float(i) + 0.11, ys=ys, color=baseline_color)
                ax.errorbar(
                    [float(i)],
                    [mean(ys)],
                    yerr=[sem(ys)],
                    fmt="s",
                    ms=5.5,
                    color=baseline_color,
                    capsize=3,
                    zorder=6,
                )
            ax.scatter([], [], s=28, color=baseline_color, label=baseline_label, marker="s")
            ax.scatter([], [], s=28, color=main_color, label="Main sweep", marker="o")
            ax.legend(loc="best")

        ax.set_xticks(range(len(cats)))
        rot = 0 if max((len(c) for c in cats), default=0) <= 12 else 90
        ax.set_xticklabels(cats, rotation=rot, ha="right" if rot else "center")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)

    if not emit_artifacts:
        return

    csv_path = out_path.with_suffix(".csv")
    script_path = out_path.with_name(out_path.stem + "__replot.py")

    baseline_expanded = False
    baseline_rows_for_plot = baseline_rows
    if not numeric_x and baseline_rows:
        base_cats = [str(v) for v in _unique_non_null(baseline_rows, x_key)]
        cats = [str(v) for v in _unique_non_null(run_rows, x_key)]
        if len(base_cats) == 1 and len(cats) > 1:
            baseline_expanded = True
            baseline_rows_for_plot = []
            for c in cats:
                for r in baseline_rows:
                    rr = dict(r)
                    rr[x_key] = c
                    baseline_rows_for_plot.append(rr)

    def _point_rows(rows: List[Dict[str, Any]], *, group: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in rows:
            x = r.get(x_key)
            y = r.get(y_key)
            if x is None or y is None:
                continue
            out.append(
                {
                    "group": group,
                    "plot_x_key": x_key,
                    "plot_y_key": y_key,
                    "plot_x_label": x_label,
                    "plot_y_label": y_label,
                    "baseline_label": baseline_label if group == "baseline" else "",
                    "baseline_expanded": bool(baseline_expanded) if group == "baseline" else False,
                    x_key: x,
                    y_key: y,
                    "run_id": r.get("run_id"),
                    "seed": r.get("seed"),
                    "strategy": r.get("strategy"),
                    "target_role": r.get("target_role"),
                    "adversary_count": r.get("adversary_count"),
                    "num_agents": r.get("num_agents"),
                }
            )
        return out

    rows_out: List[Dict[str, Any]] = []
    rows_out.extend(_point_rows(run_rows, group="main"))
    if baseline_rows_for_plot:
        rows_out.extend(_point_rows(list(baseline_rows_for_plot), group="baseline"))
    write_csv(csv_path, rows_out)

    script = f"""\
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _sem(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = float(sum(xs) / len(xs))
    var = sum((x - m) ** 2 for x in xs) / float(len(xs) - 1)
    return float(math.sqrt(var) / math.sqrt(len(xs)))


def _finite(xs: List[Any]) -> List[float]:
    out: List[float] = []
    for x in xs:
        f = _as_float(x)
        if f is None:
            continue
        if math.isfinite(f):
            out.append(float(f))
    return out


def _is_numeric(values: List[Any]) -> bool:
    for v in values:
        if v is None:
            continue
        try:
            float(v)
        except Exception:
            return False
    return True


def _group(rows: List[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    out: Dict[Any, List[Dict[str, Any]]] = {{}}
    for r in rows:
        out.setdefault(r.get(key), []).append(r)
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / {csv_path.name!r}

    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))

    if not rows:
        raise SystemExit("No rows in CSV")

    x_key = rows[0].get("plot_x_key") or {x_key!r}
    y_key = rows[0].get("plot_y_key") or {y_key!r}
    x_label = rows[0].get("plot_x_label") or {x_label!r}
    y_label = rows[0].get("plot_y_label") or {y_label!r}

    main_rows = [r for r in rows if (r.get("group") or "") == "main"]
    base_rows = [r for r in rows if (r.get("group") or "") == "baseline"]

    numeric_x = _is_numeric([r.get(x_key) for r in main_rows])

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.grid(False)

    baseline_color = "#f58518"
    main_color = "#4c78a8"

    def _violin(pos: float, vals: List[float], color: str, alpha: float = 0.35) -> None:
        if not vals:
            return
        parts = ax.violinplot(
            [vals],
            positions=[pos],
            widths=0.28,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for b in parts.get("bodies", []):
            b.set_facecolor(color)
            b.set_edgecolor(color)
            b.set_alpha(alpha)
        if parts.get("cmedians") is not None:
            parts["cmedians"].set_color(color)
            parts["cmedians"].set_linewidth(1.6)

    if numeric_x:
        by_x = _group(main_rows, x_key)
        x_vals = sorted([x for x in by_x.keys() if x is not None], key=lambda t: float(t))
        for x in x_vals:
            ys = _finite([r.get(y_key) for r in by_x.get(x, [])])
            if not ys:
                continue
            _violin(float(x) - 0.11, ys, main_color)
            ax.errorbar([float(x)], [_mean(ys)], yerr=[_sem(ys)], fmt="o", ms=6, color="black", capsize=3, zorder=5)

        if base_rows:
            by_xb = _group(base_rows, x_key)
            xb_vals = sorted([x for x in by_xb.keys() if x is not None], key=lambda t: float(t))
            for x in xb_vals:
                ys = _finite([r.get(y_key) for r in by_xb.get(x, [])])
                if not ys:
                    continue
                _violin(float(x) + 0.11, ys, baseline_color)
                ax.errorbar([float(x)], [_mean(ys)], yerr=[_sem(ys)], fmt="s", ms=5.5, color=baseline_color, capsize=3, zorder=6)
            ax.scatter([], [], s=28, color=baseline_color, label=rows[0].get("baseline_label") or "Benign baseline", marker="s")
            ax.scatter([], [], s=28, color=main_color, label="Main sweep", marker="o")
            ax.legend(loc="best")

        ax.set_xlabel(str(x_label))
        ax.set_ylabel(str(y_label))
    else:
        cats: List[str] = []
        seen = set()
        for r in main_rows:
            c = str(r.get(x_key))
            if c in seen:
                continue
            seen.add(c)
            cats.append(c)
        if not cats:
            raise SystemExit("No categories")

        by_c = _group(main_rows, x_key)
        for i, c in enumerate(cats):
            ys = _finite([r.get(y_key) for r in by_c.get(c, [])])
            if not ys:
                continue
            _violin(float(i) - 0.11, ys, main_color)
            ax.errorbar([float(i)], [_mean(ys)], yerr=[_sem(ys)], fmt="o", ms=6, color="black", capsize=3, zorder=5)

        if base_rows:
            by_cb = _group(base_rows, x_key)
            for i, c in enumerate(cats):
                ys = _finite([r.get(y_key) for r in by_cb.get(c, [])])
                if not ys:
                    continue
                _violin(float(i) + 0.11, ys, baseline_color)
                ax.errorbar([float(i)], [_mean(ys)], yerr=[_sem(ys)], fmt="s", ms=5.5, color=baseline_color, capsize=3, zorder=6)
            ax.scatter([], [], s=28, color=baseline_color, label=rows[0].get("baseline_label") or "Benign baseline", marker="s")
            ax.scatter([], [], s=28, color=main_color, label="Main sweep", marker="o")
            ax.legend(loc="best")

        ax.set_xticks(list(range(len(cats))))
        ax.set_xticklabels(
            cats,
            rotation=0 if max((len(c) for c in cats), default=0) <= 12 else 90,
            ha="right",
        )
        ax.set_xlabel(str(x_label))
        ax.set_ylabel(str(y_label))

    fig.tight_layout()
    out_pdf = here / {out_path.name!r}
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
"""
    script_path.write_text(script, encoding="utf-8")


def plot_distance_effects(
    agent_rows: List[Dict[str, Any]],
    *,
    out_path: Path,
    emit_artifacts: bool = True,
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
    ax.grid(False)
    ax.plot(
        ds,
        susp_rates,
        marker="o",
        linewidth=3.0,
        markersize=8,
        label="Suspicion rate (non-adversary)",
    )
    ax.plot(
        ds,
        corr_rates,
        marker="o",
        linewidth=3.0,
        markersize=8,
        label="Correct attribution rate (non-adversary)",
    )
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
    baseline_run_rows: Optional[List[Dict[str, Any]]] = None,
    baseline_agent_rows: Optional[List[Dict[str, Any]]] = None,
    baseline_label: str = "Benign baseline",
) -> None:
    ensure_dir(out_dir)
    x_key, x_label = _infer_x_key(run_rows)

    # Some metrics are undefined in benign runs (no coalition). For plotting, treat
    # coalition advantage as 0 when adversary_count == 0.
    baseline_run_rows_coalition_adv = baseline_run_rows
    if baseline_run_rows:
        patched: List[Dict[str, Any]] = []
        for r in baseline_run_rows:
            rr = dict(r)
            try:
                ac = int(rr.get("adversary_count") or 0)
            except Exception:
                ac = 0
            if ac == 0 and rr.get("coalition_minus_noncoalition_avg") is None:
                rr["coalition_minus_noncoalition_avg"] = 0.0
            if ac == 0 and rr.get("coalition_minus_noncoalition_avg_hospital_adjusted") is None:
                rr["coalition_minus_noncoalition_avg_hospital_adjusted"] = 0.0
            patched.append(rr)
        baseline_run_rows_coalition_adv = patched

    baseline_run_rows_coalition_regret = baseline_run_rows
    if baseline_run_rows:
        patched: List[Dict[str, Any]] = []
        for r in baseline_run_rows:
            rr = dict(r)
            try:
                ac = int(rr.get("adversary_count") or 0)
            except Exception:
                ac = 0
            if ac == 0 and rr.get("coalition_reward_regret") is None:
                rr["coalition_reward_regret"] = 0.0
            if ac == 0 and rr.get("coalition_reward_regret_hospital_adjusted") is None:
                rr["coalition_reward_regret_hospital_adjusted"] = 0.0
            patched.append(rr)
        baseline_run_rows_coalition_regret = patched

    # Core outcomes
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="joint_reward",
        y_label="Joint Reward",
        out_path=out_dir / "joint_reward.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="joint_reward_regret",
        y_label="Overall Regret",
        out_path=out_dir / "overall_regret.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="resource_failures_total",
        y_label="Total Resource Failures",
        out_path=out_dir / "resource_failures_total.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="failed_patients_count",
        y_label="Failed Patients",
        out_path=out_dir / "failed_patients_count.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="converged_patients",
        y_label="Converged Patients",
        out_path=out_dir / "converged_patients.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="meeting_outcome_variables_assigned",
        y_label="Variables Assigned",
        out_path=out_dir / "variables_assigned.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )

    # Belief / judge metrics
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="suspicion_rate_non_adversary",
        y_label="Suspicion Rate (Non-adversary)",
        out_path=out_dir / "suspicion_rate_non_adversary.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="correct_attribution_rate_non_adversary",
        y_label="Correct Attribution Rate (Non-adversary)",
        out_path=out_dir / "correct_attribution_rate_non_adversary.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="avg_distance_to_adversary_suspicious_agents",
        y_label="Avg Distance to Adversary (Suspicious Non-adversaries)",
        out_path=out_dir / "avg_distance_suspicious.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )

    # Communication proxies
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="total_messages",
        y_label="Total Messages",
        out_path=out_dir / "total_messages.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="complaint_messages",
        y_label="Complaint Messages",
        out_path=out_dir / "complaint_messages.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )

    # Graph metrics (key logged fields)
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="graph_density",
        y_label="Graph Density",
        out_path=out_dir / "graph_density.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="graph_avg_degree",
        y_label="Graph Avg Degree",
        out_path=out_dir / "graph_avg_degree.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="graph_avg_clustering",
        y_label="Graph Avg Clustering",
        out_path=out_dir / "graph_avg_clustering.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )

    # Coalition advantage (if present)
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="coalition_minus_noncoalition_avg",
        y_label="Coalition Advantage",
        out_path=out_dir / "coalition_advantage.pdf",
        baseline_rows=baseline_run_rows_coalition_adv,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="coalition_minus_noncoalition_avg_hospital_adjusted",
        y_label="Coalition Advantage",
        out_path=out_dir / "coalition_advantage_hospital_adjusted.pdf",
        baseline_rows=baseline_run_rows_coalition_adv,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="coalition_minus_noncoalition_avg_hospital_adjusted_normalized",
        y_label="Coalition Advantage",
        out_path=out_dir / "coalition_advantage_hospital_adjusted_normalized.pdf",
        baseline_rows=baseline_run_rows_coalition_adv,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="coalition_reward_regret",
        y_label="Coalition Regret",
        out_path=out_dir / "coalition_regret.pdf",
        baseline_rows=baseline_run_rows_coalition_regret,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="coalition_reward_regret_hospital_adjusted",
        y_label="Coalition Regret",
        out_path=out_dir / "coalition_regret_hospital_adjusted.pdf",
        baseline_rows=baseline_run_rows_coalition_regret,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="coalition_reward_avg",
        y_label="Coalition Avg Reward",
        out_path=out_dir / "coalition_reward_avg.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="noncoalition_reward_avg",
        y_label="Non-coalition Avg Reward",
        out_path=out_dir / "noncoalition_reward_avg.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )

    # Distance effects (aggregated across runs)
    plot_distance_effects(agent_rows, out_path=out_dir / "distance_effects.pdf")

    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="hospital_holding_cost_total",
        y_label="Holding Cost",
        out_path=out_dir / "holding_cost_total.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="hospital_resource_failure_penalty_total",
        y_label="Resource Failure Penalty",
        out_path=out_dir / "resource_failure_penalty_total.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="agent_resource_failure_penalty_total",
        y_label="Resource Failure Penalty",
        out_path=out_dir / "resource_failure_penalty_total_agent_attributed.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="hospital_missed_step_penalty_total",
        y_label="Missed-Step Penalty",
        out_path=out_dir / "missed_step_penalty_total.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
