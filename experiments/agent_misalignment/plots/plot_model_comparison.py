from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from experiments.common.plotting.style import apply_default_style
from experiments.common.plotting.io_utils import (
    as_float,
    as_int,
    ensure_dir,
    finite,
    mean,
    sanitize_filename,
    sem,
    write_csv,
)
from experiments.common.plotting.load_runs import load_runs

from .build_tables import build_tables


_MODEL_COLORS: Dict[str, str] = {
    "GPT-4.1-Mini": "#4c78a8",
    "GPT-4o-Mini": "#e45756",
    "Kimi-K2-Instruct": "#72b7b2",
}

_ADV_ROLE_LABELS: Dict[str, str] = {
    "departmental": "Dept",
    "Resource_Provisioner": "Prov",
}

_ADV_ROLE_COLORS: Dict[str, str] = {
    "departmental": "#4c78a8",
    "Resource_Provisioner": "#e45756",
}


def _strategy_type(value: Any) -> str:
    """
    Match the 4 strategy buckets used in plot_overview.py.
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
    # Unknown / legacy strings fall back to raw (still plotted if present).
    return s


def _strategy_label(stype: str) -> str:
    m = {
        "benign": "Benign",
        "covert": "Covert",
        "destructive_max": "Des (MCR)",
        "destructive_no_preservation": "Des (NRP)",
    }
    return m.get(str(stype), str(stype).replace("_", " "))


def _parse_seeds_spec(spec: Optional[str]) -> Optional[set[int]]:
    if not spec:
        return None
    out: set[int] = set()
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
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
    return out or None


def _filter_rows(
    rows: List[Dict[str, Any]],
    *,
    num_agents: int,
    target_role: str,
    adversary_count: int,
    seeds: Optional[set[int]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if as_int(r.get("num_agents")) != int(num_agents):
            continue
        if str(r.get("target_role")) != str(target_role):
            continue
        if as_int(r.get("adversary_count")) != int(adversary_count):
            continue
        if seeds is not None:
            seed = as_int(r.get("seed"))
            if seed is None or int(seed) not in seeds:
                continue
        out.append(r)
    return out


def _groupby(rows: List[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    out: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r.get(key), []).append(r)
    return out


def _role_label(role: str) -> str:
    return _ADV_ROLE_LABELS.get(str(role), str(role).replace("_", " "))


def _role_order(roles: Sequence[str]) -> List[str]:
    preferred = ["departmental", "Resource_Provisioner"]
    present = [str(r) for r in roles if str(r)]
    out: List[str] = []
    for r in preferred:
        if r in present and r not in out:
            out.append(r)
    for r in present:
        if r not in out:
            out.append(r)
    return out


def _plot_models_by_role_faceted(
    *,
    rows: List[Dict[str, Any]],
    metric_label: str,
    out_path: Path,
    strategy_order: Sequence[str],
    model_order: Sequence[str],
    role_order: Sequence[str],
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 26,
            "axes.titlesize": 26,
            "axes.labelsize": 26,
            "legend.fontsize": 26,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
        }
    )

    nrows = max(1, len(strategy_order))
    fig_h = 3.2 * float(nrows)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(12.5, fig_h),
        sharex=True,
        sharey=True,
    )
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    else:
        axes = [axes]

    group_count = len(model_order)
    hue_count = max(1, len(role_order))
    group_width = 0.85
    bar_width = group_width / float(hue_count)
    x_step = 1.25
    xs = [float(i) * x_step for i in range(group_count)]

    # Summaries per (strategy_type, model_label, target_role).
    summaries: Dict[tuple[str, str, str], tuple[float, float]] = {}
    for r in rows:
        st = str(r.get("strategy_type"))
        ml = str(r.get("model_label"))
        rr = str(r.get("target_role"))
        v = as_float(r.get("value"))
        if v is None or not math.isfinite(float(v)):
            continue
        summaries.setdefault((st, ml, rr), ([], []))  # type: ignore[assignment]

    # Rebuild with lists to compute sem robustly.
    buckets: Dict[tuple[str, str, str], List[float]] = {}
    for r in rows:
        st = str(r.get("strategy_type"))
        ml = str(r.get("model_label"))
        rr = str(r.get("target_role"))
        v = as_float(r.get("value"))
        if v is None or not math.isfinite(float(v)):
            continue
        buckets.setdefault((st, ml, rr), []).append(float(v))

    for k, vals in buckets.items():
        summaries[k] = (float(mean(vals)), float(sem(vals)))

    for ax, stype in zip(axes, strategy_order):
        ax.grid(False)
        for j, role in enumerate(role_order):
            offsets = [(j - (hue_count - 1) / 2.0) * bar_width for _ in xs]
            heights: List[float] = []
            yerrs: List[float] = []
            for model_label in model_order:
                key = (str(stype), str(model_label), str(role))
                if key not in summaries:
                    heights.append(float("nan"))
                    yerrs.append(0.0)
                    continue
                m, e = summaries[key]
                heights.append(float(m))
                yerrs.append(float(e))
            if not any(math.isfinite(float(h)) for h in heights):
                continue
            x_pos = [float(x) + float(off) for x, off in zip(xs, offsets)]
            ax.bar(
                x_pos,
                heights,
                yerr=yerrs,
                capsize=3,
                width=bar_width * 0.92,
                alpha=0.9,
                color=_ADV_ROLE_COLORS.get(str(role), None),
                label=_role_label(str(role)),
            )
        ax.set_title(_strategy_label(str(stype)))

    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(list(model_order), rotation=0, ha="center")
    fig.supylabel(metric_label)

    handles: List[Patch] = []
    for r in role_order:
        handles.append(
            Patch(
                facecolor=_ADV_ROLE_COLORS.get(str(r), "#999999"),
                edgecolor="none",
                label=_role_label(str(r)),
            )
        )

    fig.tight_layout()
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        bbox_transform=fig.transFigure,
        ncol=max(1, len(handles)),
        frameon=False,
        title=None,
        handlelength=1.2,
        columnspacing=0.9,
    )

    # Reserve bottom margin based on legend height (+ gap) so it doesn't feel cramped.
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer=renderer)
        gap_px = max((float(plt.rcParams.get("xtick.labelsize", 26)) * 2.0 / 72.0) * float(fig.dpi), 6.0)
        required_bottom_frac = float(legend_bbox.y1 + gap_px) / float(fig.bbox.height)
        required_bottom_frac = min(max(required_bottom_frac, 0.0), 0.45)
        fig.subplots_adjust(bottom=required_bottom_frac)
        fig.canvas.draw()
    except Exception:
        pass

    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_bars(
    *,
    rows: List[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    out_path: Path,
    strategy_order: Sequence[str],
    model_order: Sequence[str],
    legend_mode: str,
    ylabel_fontsize: Optional[float] = None,
    ylabel_pad: Optional[float] = None,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.grid(False)

    # Compute means+SEMs per (strategy_type, model_label).
    by_s = _groupby(rows, "strategy_type")
    summaries: Dict[tuple[str, str], tuple[float, float]] = {}
    for stype, bucket_rows in by_s.items():
        st = str(stype)
        by_m = _groupby(bucket_rows, "model_label")
        for mlabel, mrows in by_m.items():
            ml = str(mlabel)
            vals = finite([as_float(r.get("value")) for r in mrows])
            if not vals:
                continue
            summaries[(st, ml)] = (float(mean(vals)), float(sem(vals)))
    colors = dict(_MODEL_COLORS)

    group_count = len(strategy_order)
    hue_count = max(1, len(model_order))
    group_width = 0.80
    bar_width = group_width / float(hue_count)

    xs = list(range(group_count))
    for j, model_label in enumerate(model_order):
        offsets = [(j - (hue_count - 1) / 2.0) * bar_width for _ in xs]
        heights: List[float] = []
        yerrs: List[float] = []
        for i, stype in enumerate(strategy_order):
            key = (str(stype), str(model_label))
            if key not in summaries:
                heights.append(float("nan"))
                yerrs.append(0.0)
                continue
            m, e = summaries[key]
            heights.append(float(m))
            yerrs.append(float(e))

        # Skip plotting if no finite heights.
        if not any(math.isfinite(float(h)) for h in heights):
            continue

        x_pos = [float(x) + float(off) for x, off in zip(xs, offsets)]
        ax.bar(
            x_pos,
            heights,
            yerr=yerrs,
            capsize=3,
            width=bar_width * 0.92,
            alpha=0.9,
            color=colors.get(str(model_label), None),
            label=str(model_label),
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([_strategy_label(s) for s in strategy_order], rotation=0, ha="center")
    ax.set_ylabel(metric_label, fontsize=ylabel_fontsize, labelpad=ylabel_pad)
    if str(legend_mode) == "in-plot":
        ax.legend(loc="best", frameon=False, title=None)
    elif str(legend_mode) == "outside":
        ncol = max(1, len(model_order))
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=ncol,
            frameon=False,
            title=None,
        )

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_model_legend_row(*, out_path: Path, model_order: Sequence[str]) -> None:
    apply_default_style(plt)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 26,
            "legend.fontsize": 26,
        }
    )
    ensure_dir(out_path.parent)
    handles: List[Patch] = []
    for m in model_order:
        handles.append(Patch(facecolor=_MODEL_COLORS.get(str(m), "#999999"), edgecolor="none", label=str(m)))
    ncol = max(1, len(handles))
    fig = plt.figure(figsize=(9.6, 0.9))
    fig.legend(handles=handles, loc="center", ncol=ncol, frameon=False)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _emit_artifacts(
    *,
    out_path: Path,
    rows: List[Dict[str, Any]],
    strategy_order: Sequence[str],
    legend_mode: str,
) -> None:
    csv_path = out_path.with_suffix(".csv")
    script_path = out_path.with_name(out_path.stem + "__replot.py")
    write_csv(csv_path, rows)

    order_literal = "[" + ", ".join(repr(str(s)) for s in strategy_order) + "]"
    script = f"""\
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _finite(xs: List[Any]) -> List[float]:
    out: List[float] = []
    for x in xs:
        f = _as_float(x)
        if f is None:
            continue
        if math.isfinite(f):
            out.append(float(f))
    return out


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


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / {csv_path.name!r}
    legend_mode = {str(legend_mode)!r}

    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    if not rows:
        raise SystemExit("No rows")

    metric_label = str(rows[0].get("plot_metric_label") or "Metric")
    # Stable order (matches generator configuration for this plot)
    strategy_order = {order_literal}
    label_map = {{
        "benign": "Benign",
        "covert": "Covert",
        "destructive_max": "Des (MCR)",
        "destructive_no_preservation": "Des (NRP)",
    }}
    model_order = []
    for r in rows:
        m = str(r.get("model_label") or "")
        if m and m not in model_order:
            model_order.append(m)

    colors = {{
        "GPT-4.1-Mini": "#4c78a8",
        "GPT-4o-Mini": "#e45756",
        "Kimi-K2-Instruct": "#72b7b2",
    }}

    # Match paper-style defaults used in the repo.
    plt.rcParams.update({{
        "font.size": 26,
        "axes.titlesize": 26,
        "axes.labelsize": 26,
        "legend.fontsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
    }})

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.grid(False)

    group_count = len(strategy_order)
    hue_count = max(1, len(model_order))
    group_width = 0.80
    bar_width = group_width / float(hue_count)
    xs = list(range(group_count))

    for j, model_label in enumerate(model_order):
        offsets = [(j - (hue_count - 1) / 2.0) * bar_width for _ in xs]
        heights: List[float] = []
        yerrs: List[float] = []
        for stype in strategy_order:
            vals = _finite(
                [
                    r.get("value")
                    for r in rows
                    if str(r.get("strategy_type")) == stype and str(r.get("model_label")) == str(model_label)
                ]
            )
            if not vals:
                heights.append(float("nan"))
                yerrs.append(0.0)
                continue
            heights.append(float(_mean(vals) or 0.0))
            yerrs.append(float(_sem(vals)))
        if not any(math.isfinite(float(h)) for h in heights):
            continue
        x_pos = [float(x) + float(off) for x, off in zip(xs, offsets)]
        ax.bar(
            x_pos,
            heights,
            yerr=yerrs,
            capsize=3,
            width=bar_width * 0.92,
            alpha=0.9,
            color=colors.get(str(model_label), None),
            label=str(model_label),
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([label_map.get(s, s.replace("_", " ")) for s in strategy_order], rotation=0, ha="center")
    ax.set_ylabel(metric_label)
    if str(legend_mode) == "in-plot":
        ax.legend(loc="best", frameon=False, title=None)
    elif str(legend_mode) == "outside":
        ncol = max(1, len(model_order))
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=ncol,
            frameon=False,
            title=None,
        )

    fig.tight_layout()
    out_pdf = here / {out_path.name!r}
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
"""
    script_path.write_text(script, encoding="utf-8")


def _emit_legend_artifacts(*, out_dir: Path, model_order: Sequence[str]) -> None:
    csv_path = out_dir / "model_legend.csv"
    script_path = out_dir / "model_legend__replot.py"
    out_pdf = out_dir / "model_legend.pdf"

    rows: List[Dict[str, Any]] = []
    for m in model_order:
        rows.append({"model_label": str(m), "color": _MODEL_COLORS.get(str(m), "#999999")})
    write_csv(csv_path, rows)

    script = f"""\
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / {csv_path.name!r}
    out_pdf = here / {out_pdf.name!r}

    plt.rcParams.update({{
        "font.size": 26,
        "axes.titlesize": 26,
        "axes.labelsize": 26,
        "legend.fontsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
    }})

    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    if not rows:
        raise SystemExit("No rows")

    handles: List[Patch] = []
    for r in rows:
        label = str(r.get("model_label") or "")
        color = str(r.get("color") or "#999999")
        handles.append(Patch(facecolor=color, edgecolor="none", label=label))

    fig = plt.figure(figsize=(9.6, 0.9))
    fig.legend(handles=handles, loc="center", ncol=max(1, len(handles)), frameon=False)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
"""
    script_path.write_text(script, encoding="utf-8")
    _plot_model_legend_row(out_path=out_pdf, model_order=model_order)


def _collect_model_rows(
    *,
    model_runs_dir: Path,
    sweep_name: str,
    benign_sweep_name: str,
    model_label: str,
    num_agents: int,
    target_role: str,
    adversary_count: int,
    seeds: Optional[set[int]],
    metric_key: str,
    plot_metric_label: str,
    default_benign_when_missing: Optional[float] = None,
) -> List[Dict[str, Any]]:
    sweep_dir = model_runs_dir / sweep_name
    benign_dir = model_runs_dir / benign_sweep_name

    runs, _ = load_runs(sweep_dir)
    tables = build_tables(runs)
    main = _filter_rows(
        tables.run_rows,
        num_agents=num_agents,
        target_role=target_role,
        adversary_count=adversary_count,
        seeds=seeds,
    )

    benign_rows: List[Dict[str, Any]] = []
    if benign_dir.exists():
        benign_runs, _ = load_runs(benign_dir)
        benign_tables = build_tables(benign_runs)
        benign_rows = _filter_rows(
            benign_tables.run_rows,
            num_agents=num_agents,
            target_role=target_role,
            adversary_count=0,
            seeds=seeds,
        )
        # Convenience: benign sweeps are often run for only one target_role (e.g., departmental).
        # For model comparisons where we want a baseline under multiple target_role facets,
        # reuse that single baseline by relabeling it to the requested target_role.
        if not benign_rows:
            try:
                roles_present = sorted(
                    {str(r.get("target_role")) for r in benign_tables.run_rows if r.get("target_role") is not None}
                )
            except Exception:
                roles_present = []
            if len(roles_present) == 1:
                fallback_role = roles_present[0]
                fallback = _filter_rows(
                    benign_tables.run_rows,
                    num_agents=num_agents,
                    target_role=str(fallback_role),
                    adversary_count=0,
                    seeds=seeds,
                )
                if fallback:
                    benign_rows = []
                    for r in fallback:
                        rr = dict(r)
                        rr["target_role"] = str(target_role)
                        benign_rows.append(rr)

    out: List[Dict[str, Any]] = []
    for r in main:
        out.append(
            {
                "plot_metric_key": metric_key,
                "plot_metric_label": plot_metric_label,
                "model_label": model_label,
                "model_runs_dir": str(model_runs_dir),
                "strategy": r.get("strategy"),
                "strategy_type": _strategy_type(r.get("strategy")),
                "adversary_count": adversary_count,
                "num_agents": num_agents,
                "target_role": target_role,
                "seed": r.get("seed"),
                "run_id": r.get("run_id"),
                "value": r.get(metric_key),
                "group": "main",
            }
        )

    for r in benign_rows:
        v = r.get(metric_key)
        if v is None and default_benign_when_missing is not None:
            v = float(default_benign_when_missing)
        out.append(
            {
                "plot_metric_key": metric_key,
                "plot_metric_label": plot_metric_label,
                "model_label": model_label,
                "model_runs_dir": str(model_runs_dir),
                "strategy": r.get("strategy"),
                "strategy_type": "benign",
                "adversary_count": 0,
                "num_agents": num_agents,
                "target_role": target_role,
                "seed": r.get("seed"),
                "run_id": r.get("run_id"),
                "value": v,
                "group": "benign",
            }
        )

    return out


def _relabel_benign_rows_to_target_role(
    benign_run_rows: List[Dict[str, Any]], *, target_role: str
) -> List[Dict[str, Any]]:
    """
    Convenience: benign sweeps are often run for only one target_role (e.g., departmental).
    For model comparisons where we want a baseline under multiple target_role facets, reuse that
    single baseline by relabeling it to the requested target_role.
    """
    if not benign_run_rows:
        return []
    try:
        roles_present = sorted(
            {str(r.get("target_role")) for r in benign_run_rows if r.get("target_role") is not None}
        )
    except Exception:
        roles_present = []
    if len(roles_present) != 1:
        return []
    fallback_role = roles_present[0]
    fallback = _filter_rows(
        benign_run_rows,
        num_agents=as_int(benign_run_rows[0].get("num_agents")) or 0,
        target_role=str(fallback_role),
        adversary_count=0,
        seeds=None,
    )
    if not fallback:
        return []
    out: List[Dict[str, Any]] = []
    for r in fallback:
        rr = dict(r)
        rr["target_role"] = str(target_role)
        out.append(rr)
    return out


def _collect_advrole_minmax_rows(
    *,
    model_dirs: Sequence[Path],
    model_labels: Sequence[str],
    sweep_name: str,
    benign_sweep_name: str,
    num_agents: int,
    adversary_count: int,
    seeds: Optional[set[int]],
    target_roles: Sequence[str],
    metric: str,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Build rows for the combined-role plots where x=models and hue=target_role.

    metric:
      - "overall_regret_minmax": min-max normalize regret using joint_reward, with
            max = max_joint_reward
            min = min(empirical_min_joint_reward, 0)
      - "coalition_regret_minmax": normalize coalition regret via
            coalition_reward_regret / (coalition_max_reward_sum - cmin),
        where cmin = min(empirical_min_coalition_reward_sum, 0).
      - "coalition_advantage_hospital_level_actual_minmax": scale normalize advantage using
            max = max_joint_reward/num_agents
            min = min(empirical_min_advantage, 0)
        and value = advantage / (max - min)
    """
    if metric not in (
        "overall_regret_minmax",
        "coalition_regret_minmax",
        "coalition_advantage_hospital_level_actual_minmax",
    ):
        raise ValueError(f"Unknown metric {metric}")

    # First pass: gather raw values to compute empirical minima + maxima.
    joint_rewards: List[float] = []
    max_joint_rewards: List[float] = []
    adv_values: List[float] = []
    coalition_reward_sums: List[float] = []
    coalition_max_sums: List[float] = []

    main_rows_by_model_role: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    benign_rows_by_model_role: Dict[tuple[str, str], List[Dict[str, Any]]] = {}

    for model_runs_dir, model_label in zip(model_dirs, model_labels):
        sweep_dir = model_runs_dir / sweep_name
        benign_dir = model_runs_dir / benign_sweep_name

        runs, _ = load_runs(sweep_dir)
        tables = build_tables(runs)

        benign_run_rows: List[Dict[str, Any]] = []
        if benign_dir.exists():
            benign_runs, _ = load_runs(benign_dir)
            benign_tables = build_tables(benign_runs)
            benign_run_rows = benign_tables.run_rows

        for role in target_roles:
            filtered_main = _filter_rows(
                tables.run_rows,
                num_agents=num_agents,
                target_role=str(role),
                adversary_count=adversary_count,
                seeds=seeds,
            )
            main_rows_by_model_role[(str(model_label), str(role))] = filtered_main

            filtered_benign = []
            if benign_run_rows:
                filtered_benign = _filter_rows(
                    benign_run_rows,
                    num_agents=num_agents,
                    target_role=str(role),
                    adversary_count=0,
                    seeds=seeds,
                )
                if not filtered_benign:
                    # If benign only exists for one role, relabel to match.
                    try:
                        roles_present = sorted(
                            {str(r.get("target_role")) for r in benign_run_rows if r.get("target_role") is not None}
                        )
                    except Exception:
                        roles_present = []
                    if len(roles_present) == 1:
                        fallback_role = roles_present[0]
                        fallback = _filter_rows(
                            benign_run_rows,
                            num_agents=num_agents,
                            target_role=str(fallback_role),
                            adversary_count=0,
                            seeds=seeds,
                        )
                        if fallback:
                            filtered_benign = [dict(r, target_role=str(role)) for r in fallback]

            benign_rows_by_model_role[(str(model_label), str(role))] = filtered_benign

            for r in filtered_main + filtered_benign:
                mjr = as_float(r.get("max_joint_reward"))
                if mjr is not None and math.isfinite(float(mjr)):
                    max_joint_rewards.append(float(mjr))
                if metric == "overall_regret_minmax":
                    jr = as_float(r.get("joint_reward"))
                    if jr is not None and math.isfinite(float(jr)):
                        joint_rewards.append(float(jr))
                elif metric == "coalition_regret_minmax":
                    csum = as_float(r.get("coalition_reward_sum"))
                    if csum is not None and math.isfinite(float(csum)):
                        coalition_reward_sums.append(float(csum))
                    cmax = as_float(r.get("coalition_max_reward_sum"))
                    if cmax is not None and math.isfinite(float(cmax)):
                        coalition_max_sums.append(float(cmax))
                else:
                    av = as_float(r.get("coalition_minus_noncoalition_hospital_avg_actual"))
                    if av is not None and math.isfinite(float(av)):
                        adv_values.append(float(av))

    if not max_joint_rewards:
        raise SystemExit("No max_joint_reward values found for normalization.")
    mjr_max = float(max(max_joint_rewards))

    out_rows: List[Dict[str, Any]] = []
    if metric == "overall_regret_minmax":
        empirical_min = float(min(joint_rewards)) if joint_rewards else 0.0
        min_val = float(min(empirical_min, 0.0))
        denom = float(mjr_max - min_val)
        if not (math.isfinite(denom) and denom > 0.0):
            raise SystemExit(f"Invalid overall-regret denom (min={min_val}, max={mjr_max}).")

        for model_label in model_labels:
            for role in target_roles:
                rows = main_rows_by_model_role.get((str(model_label), str(role)), []) + benign_rows_by_model_role.get(
                    (str(model_label), str(role)), []
                )
                for r in rows:
                    jr = as_float(r.get("joint_reward"))
                    if jr is None or not math.isfinite(float(jr)):
                        continue
                    regret = max(0.0, float(mjr_max) - float(jr))
                    out_rows.append(
                        {
                            "model_label": str(model_label),
                            "target_role": str(role),
                            "strategy_type": _strategy_type(r.get("strategy")) if r.get("strategy") is not None else "benign",
                            "seed": r.get("seed"),
                            "run_id": r.get("run_id"),
                            "value": float(regret) / denom,
                        }
                    )
        return out_rows, "Normalized Overall Regret"

    if metric == "coalition_regret_minmax":
        if not coalition_max_sums:
            raise SystemExit("No coalition_max_reward_sum values found for normalization.")
        cmax_val = float(max(coalition_max_sums))
        empirical_min = float(min(coalition_reward_sums)) if coalition_reward_sums else 0.0
        cmin_val = float(min(empirical_min, 0.0))
        denom = float(cmax_val - cmin_val)
        if not (math.isfinite(denom) and denom > 0.0):
            raise SystemExit(f"Invalid coalition-regret denom (cmin={cmin_val}, cmax={cmax_val}).")

        for model_label in model_labels:
            for role in target_roles:
                rows = main_rows_by_model_role.get((str(model_label), str(role)), [])
                for r in rows:
                    reg = as_float(r.get("coalition_reward_regret"))
                    if reg is None or not math.isfinite(float(reg)):
                        continue
                    out_rows.append(
                        {
                            "model_label": str(model_label),
                            "target_role": str(role),
                            "strategy_type": _strategy_type(r.get("strategy")),
                            "seed": r.get("seed"),
                            "run_id": r.get("run_id"),
                            "value": float(reg) / denom,
                        }
                    )
        return out_rows, "Normalized Coalition Regret"

    # coalition advantage (hospital-level actual)
    empirical_min = float(min(adv_values)) if adv_values else 0.0
    min_val = float(min(empirical_min, 0.0))
    max_per_agent = float(mjr_max) / float(num_agents) if num_agents else float("nan")
    denom = float(max_per_agent - min_val)
    if not (math.isfinite(denom) and denom > 0.0):
        raise SystemExit(
            f"Invalid coalition-advantage denom (min={min_val}, max_per_agent={max_per_agent})."
        )

    for model_label in model_labels:
        for role in target_roles:
            rows = main_rows_by_model_role.get((str(model_label), str(role)), [])
            for r in rows:
                av = as_float(r.get("coalition_minus_noncoalition_hospital_avg_actual"))
                if av is None or not math.isfinite(float(av)):
                    continue
                out_rows.append(
                    {
                        "model_label": str(model_label),
                        "target_role": str(role),
                        "strategy_type": _strategy_type(r.get("strategy")),
                        "seed": r.get("seed"),
                        "run_id": r.get("run_id"),
                        "value": (float(av) - float(min_val)) / denom,
                    }
                )

    return out_rows, "Normalized Coalition Advantage"


def _collect_advrole_raw_rows(
    *,
    model_dirs: Sequence[Path],
    model_labels: Sequence[str],
    sweep_name: str,
    num_agents: int,
    adversary_count: int,
    seeds: Optional[set[int]],
    target_roles: Sequence[str],
    metric_key: str,
    metric_label: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for model_runs_dir, model_label in zip(model_dirs, model_labels):
        sweep_dir = model_runs_dir / sweep_name
        if not sweep_dir.exists():
            continue
        runs, _ = load_runs(sweep_dir)
        tables = build_tables(runs)
        for role in target_roles:
            filtered = _filter_rows(
                tables.run_rows,
                num_agents=num_agents,
                target_role=str(role),
                adversary_count=adversary_count,
                seeds=seeds,
            )
            for r in filtered:
                v = as_float(r.get(metric_key))
                if v is None or not math.isfinite(float(v)):
                    continue
                out.append(
                    {
                        "plot_metric_label": metric_label,
                        "model_label": str(model_label),
                        "target_role": str(role),
                        "strategy_type": _strategy_type(r.get("strategy")),
                        "seed": r.get("seed"),
                        "run_id": r.get("run_id"),
                        "value": float(v),
                    }
                )
    return out


def _normalized_out_path(out_path: Path) -> Path:
    # Insert suffix before extension: foo.pdf -> foo_normalized.pdf
    return out_path.with_name(out_path.stem + "_normalized.pdf")


def _collect_role_context_rows(
    *,
    model_dirs: Sequence[Path],
    model_labels: Sequence[str],
    sweep_name: str,
    benign_sweep_name: str,
    num_agents: int,
    target_role: str,
    adversary_count: int,
    seeds: Optional[set[int]],
    include_benign: bool,
) -> List[Dict[str, Any]]:
    """
    Collect per-run rows for a single target_role, across models.
    Includes raw fields from build_tables (joint_reward, max_joint_reward, coalition_reward_sum, etc.)
    for computing derived/normalized metrics.
    """
    out: List[Dict[str, Any]] = []
    for d, mlabel in zip(model_dirs, model_labels):
        sweep_dir = d / sweep_name
        if not sweep_dir.exists():
            continue
        runs, _ = load_runs(sweep_dir)
        tables = build_tables(runs)
        main = _filter_rows(
            tables.run_rows,
            num_agents=num_agents,
            target_role=str(target_role),
            adversary_count=adversary_count,
            seeds=seeds,
        )
        for r in main:
            rr = dict(r)
            rr["model_label"] = str(mlabel)
            rr["strategy_type"] = _strategy_type(rr.get("strategy"))
            rr["group"] = "main"
            out.append(rr)

        if not include_benign:
            continue
        benign_dir = d / benign_sweep_name
        if not benign_dir.exists():
            continue
        benign_runs, _ = load_runs(benign_dir)
        benign_tables = build_tables(benign_runs)
        benign_rows = _filter_rows(
            benign_tables.run_rows,
            num_agents=num_agents,
            target_role=str(target_role),
            adversary_count=0,
            seeds=seeds,
        )
        if not benign_rows:
            # If benign only exists for one role, relabel to the requested target_role.
            try:
                roles_present = sorted(
                    {str(r.get("target_role")) for r in benign_tables.run_rows if r.get("target_role") is not None}
                )
            except Exception:
                roles_present = []
            if len(roles_present) == 1:
                fallback_role = roles_present[0]
                fallback = _filter_rows(
                    benign_tables.run_rows,
                    num_agents=num_agents,
                    target_role=str(fallback_role),
                    adversary_count=0,
                    seeds=seeds,
                )
                benign_rows = [dict(r, target_role=str(target_role)) for r in fallback]

        for r in benign_rows:
            rr = dict(r)
            rr["model_label"] = str(mlabel)
            rr["strategy_type"] = "benign"
            rr["group"] = "benign"
            out.append(rr)

    return out


def _build_normalized_rows_for_plot(
    *,
    context_rows: List[Dict[str, Any]],
    metric: str,
    metric_label: str,
) -> List[Dict[str, Any]]:
    """
    Build per-run plot rows (strategy_type, model_label, value) for a normalized metric.

    metric:
      - overall_regret_minmax: regret / (max_joint_reward - min(min_joint_reward,0))
      - coalition_regret_minmax: coalition_regret / (coalition_max_reward_sum - min(min_coalition_reward_sum,0))
      - coalition_advantage_actual_minmax: normalized_noncoalition_regret - normalized_coalition_regret, where
            normalized_*_regret = *_regret / (*_max_sum - min(empirical_min_*_reward_sum, 0))
    """
    if metric == "overall_regret_minmax":
        mjrs = finite([as_float(r.get("max_joint_reward")) for r in context_rows])
        jrs = finite([as_float(r.get("joint_reward")) for r in context_rows])
        if not mjrs:
            return []
        max_joint_reward = float(max(mjrs))
        empirical_min = float(min(jrs)) if jrs else 0.0
        min_joint_reward = float(min(empirical_min, 0.0))
        denom = float(max_joint_reward - min_joint_reward)
        if not (math.isfinite(denom) and denom > 0.0):
            return []

        out: List[Dict[str, Any]] = []
        for r in context_rows:
            jr = as_float(r.get("joint_reward"))
            if jr is None or not math.isfinite(float(jr)):
                continue
            regret = max(0.0, float(max_joint_reward) - float(jr))
            out.append(
                {
                    "plot_metric_key": metric,
                    "plot_metric_label": metric_label,
                    "model_label": r.get("model_label"),
                    "strategy_type": r.get("strategy_type"),
                    "adversary_count": r.get("adversary_count"),
                    "num_agents": r.get("num_agents"),
                    "target_role": r.get("target_role"),
                    "seed": r.get("seed"),
                    "run_id": r.get("run_id"),
                    "value": float(regret) / denom,
                    "group": r.get("group"),
                }
            )
        return out

    if metric == "coalition_regret_minmax":
        cmaxs = finite([as_float(r.get("coalition_max_reward_sum")) for r in context_rows])
        csums = finite([as_float(r.get("coalition_reward_sum")) for r in context_rows])
        if not cmaxs:
            return []
        cmax = float(max(cmaxs))
        empirical_min = float(min(csums)) if csums else 0.0
        cmin = float(min(empirical_min, 0.0))
        denom = float(cmax - cmin)
        if not (math.isfinite(denom) and denom > 0.0):
            return []

        out = []
        for r in context_rows:
            reg = as_float(r.get("coalition_reward_regret"))
            if reg is None or not math.isfinite(float(reg)):
                continue
            out.append(
                {
                    "plot_metric_key": metric,
                    "plot_metric_label": metric_label,
                    "model_label": r.get("model_label"),
                    "strategy_type": r.get("strategy_type"),
                    "adversary_count": r.get("adversary_count"),
                    "num_agents": r.get("num_agents"),
                    "target_role": r.get("target_role"),
                    "seed": r.get("seed"),
                    "run_id": r.get("run_id"),
                    "value": float(reg) / denom,
                    "group": r.get("group"),
                }
            )
        return out

    if metric == "coalition_advantage_actual_minmax":
        mjrs = finite([as_float(r.get("max_joint_reward")) for r in context_rows])
        cmaxs = finite([as_float(r.get("coalition_max_reward_sum")) for r in context_rows])
        csums = finite([as_float(r.get("coalition_reward_sum")) for r in context_rows])
        nsums = finite([as_float(r.get("noncoalition_reward_sum")) for r in context_rows])
        if not mjrs or not cmaxs:
            return []

        # Coalition normalization (same shape as coalition_regret_minmax).
        coalition_max_sum = float(max(cmaxs))
        coalition_min_sum = float(min(min(csums), 0.0)) if csums else 0.0
        coalition_denom = float(coalition_max_sum - coalition_min_sum)
        if not (math.isfinite(coalition_denom) and coalition_denom > 0.0):
            return []

        # Non-coalition normalization uses the non-coalition optimum sum (remainder share).
        max_joint_reward = float(max(mjrs))
        n_agents = as_int(context_rows[0].get("actual_num_agents") or context_rows[0].get("num_agents")) or 0
        adv = as_int(context_rows[0].get("actual_adversary_count") or context_rows[0].get("adversary_count")) or 0
        if n_agents <= 0 or adv < 0 or adv > n_agents:
            return []
        noncoal_max_sum = (float(max_joint_reward) / float(n_agents)) * float(n_agents - adv)
        noncoal_min_sum = float(min(min(nsums), 0.0)) if nsums else 0.0
        noncoal_denom = float(noncoal_max_sum - noncoal_min_sum)
        if not (math.isfinite(noncoal_denom) and noncoal_denom > 0.0):
            return []

        out: List[Dict[str, Any]] = []
        for r in context_rows:
            csum = as_float(r.get("coalition_reward_sum"))
            nsum = as_float(r.get("noncoalition_reward_sum"))
            cmax = as_float(r.get("coalition_max_reward_sum"))
            mjr = as_float(r.get("max_joint_reward"))
            n_agents_r = as_int(r.get("actual_num_agents") or r.get("num_agents"))
            adv_r = as_int(r.get("actual_adversary_count") or r.get("adversary_count"))
            if (
                csum is None
                or nsum is None
                or cmax is None
                or mjr is None
                or n_agents_r is None
                or adv_r is None
            ):
                continue
            if not (
                math.isfinite(float(csum))
                and math.isfinite(float(nsum))
                and math.isfinite(float(cmax))
                and math.isfinite(float(mjr))
            ):
                continue
            if n_agents_r <= 0 or adv_r < 0 or adv_r > n_agents_r:
                continue

            noncoal_max_sum_r = (float(mjr) / float(n_agents_r)) * float(n_agents_r - adv_r)
            if not (math.isfinite(noncoal_max_sum_r) and noncoal_max_sum_r > 0.0):
                continue

            coal_reg = max(0.0, float(cmax) - float(csum))
            noncoal_reg = max(0.0, float(noncoal_max_sum_r) - float(nsum))
            coal_reg_norm = float(coal_reg) / float(coalition_denom)
            noncoal_reg_norm = float(noncoal_reg) / float(noncoal_denom)

            out.append(
                {
                    "plot_metric_key": metric,
                    "plot_metric_label": metric_label,
                    "model_label": r.get("model_label"),
                    "strategy_type": r.get("strategy_type"),
                    "adversary_count": r.get("adversary_count"),
                    "num_agents": r.get("num_agents"),
                    "target_role": r.get("target_role"),
                    "seed": r.get("seed"),
                    "run_id": r.get("run_id"),
                    "value": float(noncoal_reg_norm - coal_reg_norm),
                    "group": r.get("group"),
                }
            )
        return out

    raise ValueError(f"Unknown metric {metric}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grouped barplots across models (strategy type on x, model as hue)."
    )
    parser.add_argument(
        "--model-runs-dir",
        action="append",
        required=True,
        help="Repeatable. Path like .../outputs/<tag>/<ts>/runs/<model_label>.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional repeatable labels corresponding to each --model-runs-dir (defaults to directory basename).",
    )
    parser.add_argument(
        "--sweep-name",
        type=str,
        default="agent_misalignment_sweep",
        help="Adversarial sweep name under each model runs dir.",
    )
    parser.add_argument(
        "--benign-sweep-name",
        type=str,
        default="benign_baseline_sweep",
        help="Benign sweep name under each model runs dir.",
    )
    parser.add_argument("--num-agents", type=int, default=9)
    parser.add_argument(
        "--target-role",
        action="append",
        dest="target_roles",
        default=None,
        help=(
            "Repeatable. Target role to plot (e.g., departmental, Resource_Provisioner). "
            "If omitted, plots both departmental and Resource_Provisioner."
        ),
    )
    parser.add_argument("--adversary-count", type=int, default=4)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help='Optional seed filter like "0-4" or "0,1,2".',
    )
    parser.add_argument(
        "--legend-mode",
        type=str,
        default="separate",
        choices=["separate", "outside", "in-plot", "none"],
        help=(
            "Legend placement. 'separate' writes model_legend.pdf and omits legends in plots; "
            "'outside' draws above plot; 'in-plot' uses a standard legend; 'none' omits legends entirely."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/agent_misalignment/plots_outputs_model_compare",
    )
    parser.add_argument(
        "--emit-adv-role-plots",
        action="store_true",
        help="Also emit the alternative by_adv_role faceted plots (default: off).",
    )
    args = parser.parse_args(argv)

    model_dirs = [Path(p).expanduser().resolve() for p in (args.model_runs_dir or [])]
    labels = args.label or []
    model_labels: List[str] = []

    display_map = {
        "openai-gpt-4.1-mini": "GPT-4.1-Mini",
        "openai-gpt-4o-mini": "GPT-4o-Mini",
        "together-kimik2-Instruct": "Kimi-K2-Instruct",
    }
    for i, d in enumerate(model_dirs):
        if i < len(labels) and str(labels[i]).strip():
            model_labels.append(str(labels[i]).strip())
        else:
            model_labels.append(display_map.get(d.name, d.name))

    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    seeds = _parse_seeds_spec(args.seeds)

    strategy_order = ["benign", "covert", "destructive_max", "destructive_no_preservation"]
    model_order = list(model_labels)

    specs = [
        (
            "joint_reward_regret_normalized",
            "Overall Regret (Normalized)",
            "overall_regret_by_model.pdf",
            None,
            True,
        ),
        (
            "coalition_reward_regret_normalized",
            "Coalition Regret (Normalized)",
            "coalition_regret_by_model.pdf",
            0.0,
            False,
        ),
        (
            "coalition_minus_noncoalition_hospital_avg_actual",
            "Coalition Advantage",
            "coalition_advantage_actual_by_model.pdf",
            0.0,
            False,
        ),
    ]

    target_roles = args.target_roles or ["departmental", "Resource_Provisioner"]
    for target_role in target_roles:
        role_dir = out_dir / "by_target_role" / sanitize_filename(str(target_role))
        ensure_dir(role_dir)
        if str(args.legend_mode) == "separate":
            _emit_legend_artifacts(out_dir=role_dir, model_order=model_order)

        for metric_key, metric_label, filename, benign_default, include_benign in specs:
            rows: List[Dict[str, Any]] = []
            for d, mlabel in zip(model_dirs, model_labels):
                rows.extend(
                    _collect_model_rows(
                        model_runs_dir=d,
                        sweep_name=str(args.sweep_name),
                        benign_sweep_name=str(args.benign_sweep_name),
                        model_label=mlabel,
                        num_agents=int(args.num_agents),
                        target_role=str(target_role),
                        adversary_count=int(args.adversary_count),
                        seeds=seeds,
                        metric_key=str(metric_key),
                        plot_metric_label=str(metric_label),
                        default_benign_when_missing=benign_default,
                    )
                )

            # For coalition-only metrics, drop benign rows (no coalition exists).
            if not bool(include_benign):
                rows = [r for r in rows if str(r.get("strategy_type")) != "benign"]

            # Plot expects per-run points with (strategy_type, model_label, value).
            out_path = role_dir / sanitize_filename(str(filename))
            metric_strategy_order = (
                strategy_order if bool(include_benign) else [s for s in strategy_order if s != "benign"]
            )
            _plot_grouped_bars(
                rows=rows,
                metric_key=str(metric_key),
                metric_label=(
                    "Coalition Adv"
                    if str(filename) == "coalition_advantage_actual_by_model.pdf"
                    else str(metric_label)
                ),
                out_path=out_path,
                strategy_order=metric_strategy_order,
                model_order=model_order,
                legend_mode=str(args.legend_mode),
                ylabel_fontsize=(
                    None
                    if str(filename) == "coalition_advantage_actual_by_model.pdf"
                    else None
                ),
                ylabel_pad=(
                    None
                    if str(filename) == "coalition_advantage_actual_by_model.pdf"
                    else None
                ),
            )
            _emit_artifacts(
                out_path=out_path,
                rows=rows,
                strategy_order=metric_strategy_order,
                legend_mode=str(args.legend_mode),
            )

    # Normalized variants alongside the by_target_role plots.
    for target_role in target_roles:
        role_dir = out_dir / "by_target_role" / sanitize_filename(str(target_role))
        if not role_dir.exists():
            continue

        # 1) Overall regret (actual) min-max normalization.
        ctx_overall = _collect_role_context_rows(
            model_dirs=model_dirs,
            model_labels=model_labels,
            sweep_name=str(args.sweep_name),
            benign_sweep_name=str(args.benign_sweep_name),
            num_agents=int(args.num_agents),
            target_role=str(target_role),
            adversary_count=int(args.adversary_count),
            seeds=seeds,
            include_benign=True,
        )
        rows_overall_norm = _build_normalized_rows_for_plot(
            context_rows=ctx_overall,
            metric="overall_regret_minmax",
            metric_label="Overall Regret",
        )
        if rows_overall_norm:
            out_path = _normalized_out_path(role_dir / "overall_regret_by_model.pdf")
            _plot_grouped_bars(
                rows=rows_overall_norm,
                metric_key="overall_regret_minmax",
                metric_label="Overall Regret",
                out_path=out_path,
                strategy_order=["benign", "covert", "destructive_max", "destructive_no_preservation"],
                model_order=model_order,
                legend_mode=str(args.legend_mode),
            )
            _emit_artifacts(
                out_path=out_path,
                rows=rows_overall_norm,
                strategy_order=["benign", "covert", "destructive_max", "destructive_no_preservation"],
                legend_mode=str(args.legend_mode),
            )

        # 2) Coalition regret min-max normalization (exclude benign).
        ctx_coal = _collect_role_context_rows(
            model_dirs=model_dirs,
            model_labels=model_labels,
            sweep_name=str(args.sweep_name),
            benign_sweep_name=str(args.benign_sweep_name),
            num_agents=int(args.num_agents),
            target_role=str(target_role),
            adversary_count=int(args.adversary_count),
            seeds=seeds,
            include_benign=False,
        )
        rows_coal_norm = _build_normalized_rows_for_plot(
            context_rows=ctx_coal,
            metric="coalition_regret_minmax",
            metric_label="Coalition Regret",
        )
        if rows_coal_norm:
            out_path = _normalized_out_path(role_dir / "coalition_regret_by_model.pdf")
            _plot_grouped_bars(
                rows=rows_coal_norm,
                metric_key="coalition_regret_minmax",
                metric_label="Coalition Regret",
                out_path=out_path,
                strategy_order=["covert", "destructive_max", "destructive_no_preservation"],
                model_order=model_order,
                legend_mode=str(args.legend_mode),
            )
            _emit_artifacts(
                out_path=out_path,
                rows=rows_coal_norm,
                strategy_order=["covert", "destructive_max", "destructive_no_preservation"],
                legend_mode=str(args.legend_mode),
            )

        # 3) Coalition advantage (hospital-level actual) min-max normalization (exclude benign).
        rows_adv_norm = _build_normalized_rows_for_plot(
            context_rows=ctx_coal,
            metric="coalition_advantage_actual_minmax",
            metric_label="Coalition Advantage",
        )
        if rows_adv_norm:
            out_path = _normalized_out_path(role_dir / "coalition_advantage_actual_by_model.pdf")
            _plot_grouped_bars(
                rows=rows_adv_norm,
                metric_key="coalition_advantage_actual_minmax",
                metric_label="Coalition Adv",
                out_path=out_path,
                strategy_order=["covert", "destructive_max", "destructive_no_preservation"],
                model_order=model_order,
                legend_mode=str(args.legend_mode),
                ylabel_fontsize=None,
                ylabel_pad=None,
            )
            _emit_artifacts(
                out_path=out_path,
                rows=rows_adv_norm,
                strategy_order=["covert", "destructive_max", "destructive_no_preservation"],
                legend_mode=str(args.legend_mode),
            )

    # Optional: keep the alternative by_adv_role faceted plots behind a flag.
    if bool(args.emit_adv_role_plots) and len(target_roles) >= 2:
        advrole_dir = out_dir / "by_adv_role"
        ensure_dir(advrole_dir)
        role_order = _role_order(target_roles)
        rows_regret, ylabel_regret = _collect_advrole_minmax_rows(
            model_dirs=model_dirs,
            model_labels=model_labels,
            sweep_name=str(args.sweep_name),
            benign_sweep_name=str(args.benign_sweep_name),
            num_agents=int(args.num_agents),
            adversary_count=int(args.adversary_count),
            seeds=seeds,
            target_roles=target_roles,
            metric="overall_regret_minmax",
        )
        present_strats = sorted({str(r.get("strategy_type")) for r in rows_regret if r.get("strategy_type")})
        strategy_order_regret = [s for s in strategy_order if s in present_strats]
        out_path_regret = advrole_dir / "overall_regret_minmax_normalized_by_model_and_advrole.pdf"
        _plot_models_by_role_faceted(
            rows=rows_regret,
            metric_label=ylabel_regret,
            out_path=out_path_regret,
            strategy_order=strategy_order_regret,
            model_order=model_order,
            role_order=role_order,
        )
        write_csv(out_path_regret.with_suffix(".csv"), rows_regret)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
