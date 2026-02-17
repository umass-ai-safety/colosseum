from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from experiments.common.plotting.io_utils import (
    ensure_dir,
    finite,
    groupby,
    mean,
    sem,
    sanitize_filename,
    write_csv,
)
from experiments.common.plotting.logging_utils import log_saved_plot
from experiments.common.plotting.style import apply_default_style

logger = logging.getLogger(__name__)

_STYLE = {
    "font.size": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
}


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _norm_role(value: Any) -> str:
    return str(value) if value is not None else "None"


def _strategy_type(value: Any) -> str:
    """
    Coarsen strategy names for plotting.
    """
    s = (str(value) if value is not None else "").strip()
    if not s or s == "none":
        return "benign"
    if s == "covert_max_coalition_reward":
        return "covert"
    if s == "destructive_max_coalition_reward":
        return "destructive_max"
    if s == "destructive_no_reward_preservation":
        return "destructive_no_preservation"
    return s


def _strategy_label(strategy_type: str) -> str:
    return {
        "benign": "Benign",
        "covert": "Covert",
        "destructive_max": "Des (MCR)",
        "destructive_no_preservation": "Des (NRP)",
    }.get(strategy_type, strategy_type)


def _aggregate_metric(
    rows: List[Dict[str, Any]],
    *,
    metric_key: str,
    default_when_missing: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], int]:
    ys = finite([r.get(metric_key) for r in rows])
    if not ys and default_when_missing is not None and rows:
        ys = [float(default_when_missing)]
    if not ys:
        return None, None, 0
    return mean(ys), sem(ys), len(ys)


def _filter_rows(
    rows: List[Dict[str, Any]],
    *,
    num_agents: int,
    target_role: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if _as_int(r.get("num_agents")) != int(num_agents):
            continue
        if _norm_role(r.get("target_role")) != str(target_role):
            continue
        out.append(r)
    return out


def _filter_adv_count(rows: List[Dict[str, Any]], adversary_count: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if _as_int(r.get("adversary_count")) != int(adversary_count):
            continue
        out.append(r)
    return out


def _bar_across_strategies(
    *,
    run_rows: List[Dict[str, Any]],
    benign_rows: List[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    num_agents: int,
    target_role: str,
    adversary_count: int,
    out_path: Path,
    emit_artifacts: bool = True,
) -> None:
    """
    Bar plot across strategy types for a fixed adversary_count, plus a benign baseline bar.
    """
    apply_default_style(plt)
    plt.rcParams.update(_STYLE)
    ensure_dir(out_path.parent)

    main = _filter_rows(run_rows, num_agents=num_agents, target_role=target_role)
    main = _filter_adv_count(main, adversary_count)
    base = _filter_rows(benign_rows, num_agents=num_agents, target_role=target_role)
    base = _filter_adv_count(base, 0)

    order = ["benign", "covert", "destructive_max", "destructive_no_preservation"]
    colors = {
        "benign": "#f58518",
        "covert": "#4c78a8",
        "destructive_max": "#e45756",
        "destructive_no_preservation": "#72b7b2",
    }

    grouped_main = groupby(main, ("strategy",))
    # Baseline grouped separately (strategy is often "none" in benign config)
    baseline_mean, baseline_sem, baseline_n = _aggregate_metric(base, metric_key=metric_key)

    xs: List[int] = []
    heights: List[float] = []
    yerrs: List[float] = []
    labels: List[str] = []
    bar_colors: List[str] = []

    artifact_rows: List[Dict[str, Any]] = []

    for i, stype in enumerate(order):
        rows_for_stype: List[Dict[str, Any]] = []
        if stype == "benign":
            rows_for_stype = base
        else:
            # Pull from all raw strategies that map to this type.
            for (raw_strategy,), rs in grouped_main.items():
                if _strategy_type(raw_strategy) == stype:
                    rows_for_stype.extend(rs)

        # Artifact: raw points used for this bar.
        for r in rows_for_stype:
            v = r.get(metric_key)
            if v is None:
                continue
            artifact_rows.append(
                {
                    "plot_type": "bar_across_strategies",
                    "plot_metric_key": metric_key,
                    "plot_metric_label": metric_label,
                    "num_agents": num_agents,
                    "target_role": target_role,
                    "adversary_count": adversary_count,
                    "strategy_type": stype,
                    "raw_strategy": r.get("strategy"),
                    "group": "benign" if stype == "benign" else "main",
                    "value": v,
                    "run_id": r.get("run_id"),
                    "seed": r.get("seed"),
                }
            )

        default_when_missing = None
        if stype == "benign" and metric_key == "coalition_reward_regret":
            default_when_missing = 0.0
        m, e, n = _aggregate_metric(
            rows_for_stype,
            metric_key=metric_key,
            default_when_missing=default_when_missing,
        )
        if m is None:
            continue
        xs.append(i)
        heights.append(float(m))
        yerrs.append(float(e or 0.0))
        labels.append(_strategy_label(stype))
        bar_colors.append(colors.get(stype, "#999999"))

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.grid(False)
    ax.bar(xs, heights, yerr=yerrs, capsize=3, color=bar_colors, alpha=0.9)
    # Make zero-height bars visible (e.g., benign regret is exactly 0).
    for x, h, c in zip(xs, heights, bar_colors):
        if abs(float(h)) < 1e-12:
            ax.plot([x], [0.0], marker="_", markersize=18, color=c, zorder=6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel(metric_label)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)

    if not emit_artifacts:
        return

    csv_path = out_path.with_suffix(".csv")
    script_path = out_path.with_name(out_path.stem + "__replot.py")
    write_csv(csv_path, artifact_rows)

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


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / {csv_path.name!r}
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    if not rows:
        raise SystemExit("No rows in CSV")

    metric_label = str(rows[0].get("plot_metric_label") or {metric_label!r})
    # Preserve ordering used in the main plot.
    order = ["benign", "covert", "destructive_max", "destructive_no_preservation"]

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.grid(False)

    xs: List[int] = []
    heights: List[float] = []
    yerrs: List[float] = []
    labels: List[str] = []
    colors = {{
        "benign": "#f58518",
        "covert": "#4c78a8",
        "destructive_max": "#e45756",
        "destructive_no_preservation": "#72b7b2",
    }}

    for i, stype in enumerate(order):
        vals = _finite([r.get("value") for r in rows if str(r.get("strategy_type")) == stype])
        if not vals:
            continue
        xs.append(i)
        heights.append(float(_mean(vals) or 0.0))
        yerrs.append(float(_sem(vals)))
        labels.append(stype.replace("_", " "))
        ax.scatter([float(i)] * len(vals), vals, s=16, alpha=0.25, color=colors.get(stype, "#999999"))

    ax.bar(xs, heights, yerr=yerrs, capsize=3, color=[colors.get(order[i], "#999999") for i in xs], alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel(metric_label)

    fig.tight_layout()
    out_pdf = here / {out_path.name!r}
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
"""
    script_path.write_text(script, encoding="utf-8")


def _lines_over_adversary_count(
    *,
    run_rows: List[Dict[str, Any]],
    benign_rows: List[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    num_agents: int,
    target_role: str,
    adv_counts: Sequence[int],
    out_path: Path,
    emit_artifacts: bool = True,
) -> None:
    """
    Line plot: x=adversary_count (including 0 benign), separate line per strategy type.
    """
    apply_default_style(plt)
    plt.rcParams.update(_STYLE)
    ensure_dir(out_path.parent)

    main = _filter_rows(run_rows, num_agents=num_agents, target_role=target_role)
    base = _filter_rows(benign_rows, num_agents=num_agents, target_role=target_role)

    # Build points per strategy type.
    strategies = ["benign", "covert", "destructive_max", "destructive_no_preservation"]
    colors = {
        "benign": "#f58518",
        "covert": "#4c78a8",
        "destructive_max": "#e45756",
        "destructive_no_preservation": "#72b7b2",
    }

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.grid(False)

    for stype in strategies:
        xs: List[int] = []
        ys: List[float] = []
        es: List[float] = []

        for ac in adv_counts:
            if stype == "benign":
                if ac != 0:
                    continue
                rows = _filter_adv_count(base, 0)
            else:
                rows = _filter_adv_count(main, ac)
                rows = [r for r in rows if _strategy_type(r.get("strategy")) == stype]

            default_when_missing = None
            if stype == "benign" and metric_key == "coalition_reward_regret":
                default_when_missing = 0.0
            m, e, n = _aggregate_metric(
                rows,
                metric_key=metric_key,
                default_when_missing=default_when_missing,
            )
            if m is None:
                continue
            xs.append(int(ac))
            ys.append(float(m))
            es.append(float(e or 0.0))

        if not xs:
            continue
        ax.errorbar(
            xs,
            ys,
            yerr=es,
            marker="o",
            linewidth=3.0,
            markersize=8,
            elinewidth=2.0,
            capthick=2.0,
            capsize=3,
            color=colors.get(stype, "#999999"),
            label=_strategy_label(stype),
        )

    ax.set_xlabel("Adversary Count")
    ax.set_ylabel(metric_label)
    ax.set_xticks(sorted(set(int(a) for a in adv_counts)))
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)

    if not emit_artifacts:
        return

    # Artifact: raw per-run points used for the line plot.
    csv_path = out_path.with_suffix(".csv")
    script_path = out_path.with_name(out_path.stem + "__replot.py")

    artifact_rows: List[Dict[str, Any]] = []
    for stype in strategies:
        for ac in adv_counts:
            if stype == "benign":
                if int(ac) != 0:
                    continue
                rows = _filter_adv_count(base, 0)
            else:
                rows = _filter_adv_count(main, int(ac))
                rows = [r for r in rows if _strategy_type(r.get("strategy")) == stype]

            default_when_missing = None
            if stype == "benign" and metric_key == "coalition_reward_regret":
                default_when_missing = 0.0

            for r in rows:
                v = r.get(metric_key)
                if v is None:
                    continue
                artifact_rows.append(
                    {
                        "plot_type": "lines_over_adversary_count",
                        "plot_metric_key": metric_key,
                        "plot_metric_label": metric_label,
                        "num_agents": num_agents,
                        "target_role": target_role,
                        "strategy_type": stype,
                        "adversary_count": int(ac),
                        "raw_strategy": r.get("strategy"),
                        "group": "benign" if stype == "benign" else "main",
                        "value": v,
                        "run_id": r.get("run_id"),
                        "seed": r.get("seed"),
                    }
                )

    write_csv(csv_path, artifact_rows)

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


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / {csv_path.name!r}
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    if not rows:
        raise SystemExit("No rows in CSV")

    metric_label = str(rows[0].get("plot_metric_label") or {metric_label!r})

    strategies = ["benign", "covert", "destructive_max", "destructive_no_preservation"]
    colors = {{
        "benign": "#f58518",
        "covert": "#4c78a8",
        "destructive_max": "#e45756",
        "destructive_no_preservation": "#72b7b2",
    }}

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.grid(False)

    for stype in strategies:
        pts = [r for r in rows if str(r.get("strategy_type")) == stype]
        if not pts:
            continue
        advs = sorted(set(int(float(r.get("adversary_count") or 0)) for r in pts))
        xs: List[int] = []
        ys: List[float] = []
        es: List[float] = []
        for ac in advs:
            vals = _finite([r.get("value") for r in pts if int(float(r.get("adversary_count") or 0)) == int(ac)])
            if not vals:
                continue
            xs.append(int(ac))
            ys.append(float(_mean(vals) or 0.0))
            es.append(float(_sem(vals)))
        if not xs:
            continue
        ax.errorbar(
            xs,
            ys,
            yerr=es,
            marker="o",
            linewidth=3.0,
            markersize=8,
            elinewidth=2.0,
            capthick=2.0,
            capsize=3,
            color=colors.get(stype, "#999999"),
            label=stype.replace("_", " "),
        )

    ax.set_xlabel("Adversary Count")
    ax.set_ylabel(metric_label)
    ax.legend(loc="best")

    fig.tight_layout()
    out_pdf = here / {out_path.name!r}
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
"""
    script_path.write_text(script, encoding="utf-8")


@dataclass(frozen=True)
class OverviewSpec:
    adv_counts: Tuple[int, ...] = (0, 2, 4, 8)


def plot_overview(
    *,
    run_rows: List[Dict[str, Any]],
    benign_run_rows: Optional[List[Dict[str, Any]]],
    out_dir: Path,
    spec: OverviewSpec = OverviewSpec(),
) -> None:
    """
    Produces:
      - bar plots across strategy types (per adv_count in {2,4,8} + benign baseline)
      - line plots over adversary_count (0,2,4,8) with separate lines per strategy type
    Stratified by:
      - target_role (departmental vs Resource_Provisioner)
      - num_agents (requested sweep value)
    """
    ensure_dir(out_dir)
    benign_run_rows = benign_run_rows or []

    roles = sorted({_norm_role(r.get("target_role")) for r in run_rows if r.get("target_role") is not None})
    num_agents_vals = sorted({int(r.get("num_agents")) for r in run_rows if r.get("num_agents") is not None})

    if not roles or not num_agents_vals:
        return

    for role in roles:
        role_dir = out_dir / "by_target_role" / sanitize_filename(role)
        ensure_dir(role_dir)

        for n in num_agents_vals:
            n_dir = role_dir / f"n{int(n)}"
            ensure_dir(n_dir)

            # Line plots across adversary counts for this role + n.
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="joint_reward",
                metric_label="Joint Reward",
                num_agents=int(n),
                target_role=role,
                adv_counts=spec.adv_counts,
                out_path=n_dir / "joint_reward_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_minus_noncoalition_avg",
                metric_label="Coalition Advantage",
                num_agents=int(n),
                target_role=role,
                adv_counts=spec.adv_counts,
                out_path=n_dir / "coalition_advantage_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_minus_noncoalition_hospital_avg",
                metric_label="Coalition Advantage",
                num_agents=int(n),
                target_role=role,
                adv_counts=spec.adv_counts,
                out_path=n_dir / "coalition_advantage_hospital_level_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_minus_noncoalition_hospital_avg_actual",
                metric_label="Coalition Advantage",
                num_agents=int(n),
                target_role=role,
                adv_counts=spec.adv_counts,
                out_path=n_dir / "coalition_advantage_hospital_level_actual_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_minus_noncoalition_avg_hospital_adjusted",
                metric_label="Coalition Advantage",
                num_agents=int(n),
                target_role=role,
                adv_counts=spec.adv_counts,
                out_path=n_dir / "coalition_advantage_hospital_adjusted_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_minus_noncoalition_avg_hospital_adjusted_normalized",
                metric_label="Coalition Advantage",
                num_agents=int(n),
                target_role=role,
                adv_counts=spec.adv_counts,
                out_path=n_dir
                / "coalition_advantage_hospital_adjusted_normalized_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_minus_noncoalition_hospital_avg_hospital_adjusted_normalized",
                metric_label="Coalition Advantage",
                num_agents=int(n),
                target_role=role,
                adv_counts=spec.adv_counts,
                out_path=n_dir
                / "coalition_advantage_hospital_level_hospital_adjusted_normalized_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="joint_reward_regret",
                metric_label="Overall Regret",
                num_agents=int(n),
                target_role=role,
                adv_counts=spec.adv_counts,
                out_path=n_dir / "overall_regret_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_reward_regret",
                metric_label="Coalition Regret",
                num_agents=int(n),
                target_role=role,
                adv_counts=spec.adv_counts,
                out_path=n_dir / "coalition_regret_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_reward_regret_hospital_adjusted",
                metric_label="Coalition Regret",
                num_agents=int(n),
                target_role=role,
                adv_counts=spec.adv_counts,
                out_path=n_dir / "coalition_regret_hospital_adjusted_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="agent_resource_failure_penalty_total",
                metric_label="Resource Failure Penalty",
                num_agents=int(n),
                target_role=role,
                adv_counts=spec.adv_counts,
                out_path=n_dir / "resource_failure_penalty_total_agent_attributed_vs_adv_count.pdf",
            )

            # Bar plots across strategy types, per adversary count (2/4/8).
            for ac in [a for a in spec.adv_counts if a != 0]:
                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="joint_reward",
                    metric_label="Joint Reward",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir / f"joint_reward_by_strategy_adv{int(ac)}.pdf",
                )
                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_minus_noncoalition_avg",
                    metric_label="Coalition Advantage",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir / f"coalition_advantage_by_strategy_adv{int(ac)}.pdf",
                )
                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_minus_noncoalition_hospital_avg",
                    metric_label="Coalition Advantage",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir
                    / f"coalition_advantage_hospital_level_by_strategy_adv{int(ac)}.pdf",
                )
                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_minus_noncoalition_hospital_avg_actual",
                    metric_label="Coalition Advantage",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir
                    / f"coalition_advantage_hospital_level_actual_by_strategy_adv{int(ac)}.pdf",
                )
                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_minus_noncoalition_avg_hospital_adjusted",
                    metric_label="Coalition Advantage",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir
                    / f"coalition_advantage_hospital_adjusted_by_strategy_adv{int(ac)}.pdf",
                )
                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_minus_noncoalition_avg_hospital_adjusted_normalized",
                    metric_label="Coalition Advantage",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir
                    / f"coalition_advantage_hospital_adjusted_normalized_by_strategy_adv{int(ac)}.pdf",
                )
                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_minus_noncoalition_hospital_avg_hospital_adjusted_normalized",
                    metric_label="Coalition Advantage",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir
                    / f"coalition_advantage_hospital_level_hospital_adjusted_normalized_by_strategy_adv{int(ac)}.pdf",
                )

                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="hospital_holding_cost_total",
                    metric_label="Holding Cost",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir / f"holding_cost_total_by_strategy_adv{int(ac)}.pdf",
                )
                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="hospital_resource_failure_penalty_total",
                    metric_label="Resource Failure Penalty",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir
                    / f"resource_failure_penalty_total_by_strategy_adv{int(ac)}.pdf",
                )
                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="agent_resource_failure_penalty_total",
                    metric_label="Resource Failure Penalty",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir
                    / f"resource_failure_penalty_total_agent_attributed_by_strategy_adv{int(ac)}.pdf",
                )
                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="hospital_missed_step_penalty_total",
                    metric_label="Missed-Step Penalty",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir
                    / f"missed_step_penalty_total_by_strategy_adv{int(ac)}.pdf",
                )
                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="joint_reward_regret",
                    metric_label="Overall Regret",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir / f"overall_regret_by_strategy_adv{int(ac)}.pdf",
                )
                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="joint_reward_regret_normalized",
                    metric_label="Overall Regret",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir
                    / f"overall_regret_normalized_by_strategy_adv{int(ac)}.pdf",
                )
                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_reward_regret",
                    metric_label="Coalition Regret",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir / f"coalition_regret_by_strategy_adv{int(ac)}.pdf",
                )
                _bar_across_strategies(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_reward_regret_hospital_adjusted",
                    metric_label="Coalition Regret",
                    num_agents=int(n),
                    target_role=role,
                    adversary_count=int(ac),
                    out_path=n_dir
                    / f"coalition_regret_hospital_adjusted_by_strategy_adv{int(ac)}.pdf",
                )
