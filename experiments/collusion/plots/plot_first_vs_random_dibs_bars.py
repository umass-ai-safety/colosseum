from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch


ConditionKey = Tuple[bool, str]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_mean_results(path_or_dir: str) -> Path:
    p = Path(path_or_dir).expanduser()
    if p.is_dir():
        return p / "mean_results.json"
    return p


def _resolve_output_dir(path_or_dir: str) -> Path:
    p = Path(path_or_dir).expanduser()
    if p.is_file():
        return p.parent
    return p


def _extract_agent_order_label(output_dir: Path) -> str:
    cfg_path = output_dir / "config.json"
    if not cfg_path.exists():
        return "unknown"
    try:
        cfg = _load_json(cfg_path)
    except Exception:
        return "unknown"
    collusion_cfg = (cfg.get("experiment") or {}).get("collusion") or {}
    val = str(collusion_cfg.get("agent_order") or "unknown")
    return val.strip().lower() or "unknown"


def _condition_value(
    mean_results: Dict[str, Any],
    *,
    condition: str,
    metric_mean_key: str,
    metric_sd_key: str,
) -> Tuple[float, float, int]:
    res = (mean_results.get("results") or {}).get(condition) or {}
    n = int(res.get("n_complete") or 0)
    mean_val = res.get(metric_mean_key)
    sd_val = res.get(metric_sd_key)
    mean_f = float(mean_val) if isinstance(mean_val, (int, float)) else float("nan")
    sd_f = float(sd_val) if isinstance(sd_val, (int, float)) else float("nan")
    # Plot SEM (visually similar to other report plots).
    sem = sd_f / math.sqrt(n) if (n > 0 and math.isfinite(sd_f)) else float("nan")
    return mean_f, sem, n


def _iter_run_dirs(output_dir: Path) -> Iterable[Path]:
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return
    for cfg_path in sorted(runs_dir.rglob("run_config.json")):
        yield cfg_path.parent


def _read_metrics(run_dir: Path) -> Dict[str, Any]:
    try:
        return _load_json(run_dir / "metrics.json")
    except Exception:
        return {}


def _is_complete_status(status: Any) -> bool:
    return str(status or "").strip().lower() in {"complete", "completed", "success"}


def _finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        if isinstance(v, bool) or v is None:
            continue
        try:
            f = float(v)
        except Exception:
            continue
        if not math.isfinite(f):
            continue
        out.append(float(f))
    return out


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pp = max(0.0, min(100.0, float(p))) / 100.0
    k = pp * (len(sorted_vals) - 1)
    i0 = int(math.floor(k))
    i1 = int(math.ceil(k))
    if i0 == i1:
        return float(sorted_vals[i0])
    w = float(k - i0)
    return float(sorted_vals[i0] * (1.0 - w) + sorted_vals[i1] * w)


def _robust_range(values: List[float]) -> Tuple[float, float]:
    """
    Robust min/max scaling based on the pooled distribution (percentile range),
    mirroring the "use the entire distribution" philosophy in our collusion plots.
    """
    vals = sorted(_finite(values))
    if not vals:
        return 0.0, 1.0
    if len(vals) == 1:
        center = float(vals[0])
        pad = max(1e-6, abs(center) * 0.1)
        return center - pad, center + pad

    lo = float(_percentile(vals, 1))
    hi = float(_percentile(vals, 99))
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        lo = float(vals[0])
        hi = float(vals[-1])
    if lo == hi:
        pad = max(1e-6, abs(lo) * 0.1)
        lo -= pad
        hi += pad
    else:
        # Mirror histogram x-range padding to reduce saturation at exactly 0/1.
        pad = max(1e-6, 0.06 * abs(hi - lo))
        lo -= pad
        hi += pad
    return float(lo), float(hi)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def _norm_one(value: float, *, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    if hi == lo:
        return 0.5
    return _clamp01((float(value) - float(lo)) / (float(hi) - float(lo)))


def _sample_std(values: List[float]) -> float:
    vals = _finite(values)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return 0.0
    m = float(sum(vals) / len(vals))
    var = sum((x - m) ** 2 for x in vals) / float(len(vals) - 1)
    return float(var**0.5)


def _sem(values: List[float]) -> float:
    vals = _finite(values)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return 0.0
    return float(_sample_std(vals) / math.sqrt(len(vals)))


def _add_group_bracket(
    ax: plt.Axes,
    *,
    x0: float,
    x1: float,
    y: float,
    label: str,
    cap_height: float = 0.16,
    label_pad: float = 0.12,
) -> None:
    """Draw a bracket in a dedicated annotation axis (no tick-label collisions)."""
    h = float(cap_height)
    ax.plot(
        [x0, x0, x1, x1],
        [y, y - h, y - h, y],
        color="black",
        lw=1.8,
        clip_on=False,
    )
    ax.text(
        (x0 + x1) / 2.0,
        y - h - float(label_pad),
        label,
        ha="center",
        va="top",
        fontsize=22,
        color="black",
        clip_on=False,
    )


def _apply_style() -> None:
    # Match the paper-ish style of existing collusion plots while keeping labels readable.
    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.labelsize": 25,
            "axes.titlesize": 30,
            "xtick.labelsize": 23,
            "ytick.labelsize": 25,
            "legend.fontsize": 20,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot coalition regret/advantage bars for colluders-first (first dibs) vs random ordering."
        )
    )
    parser.add_argument(
        "--first-dibs",
        default="experiments/collusion/outputs/collusion_topologies_sweep/20260211-145555",
        help="Output dir (or mean_results.json path) for colluders-first runs.",
    )
    parser.add_argument(
        "--random-dibs",
        default="experiments/collusion/outputs/collusion_topologies_sweep/20260211-163504",
        help="Output dir (or mean_results.json path) for random-order runs.",
    )
    parser.add_argument(
        "--out",
        default="experiments/collusion/plots_outputs/first_vs_random_dibs_bars.pdf",
        help="Output image path.",
    )
    args = parser.parse_args()

    first_dir = Path(args.first_dibs).expanduser()
    rand_dir = Path(args.random_dibs).expanduser()
    first_results_path = _resolve_mean_results(args.first_dibs)
    rand_results_path = _resolve_mean_results(args.random_dibs)

    if not first_results_path.exists():
        raise FileNotFoundError(f"Missing: {first_results_path}")
    if not rand_results_path.exists():
        raise FileNotFoundError(f"Missing: {rand_results_path}")

    # Keep the mean_results.json loads for sanity/debug, but compute plotted values from
    # the full run distribution (metrics.json) so normalization uses the entire distribution.
    first = _load_json(first_results_path)
    rand = _load_json(rand_results_path)

    first_out_dir = _resolve_output_dir(args.first_dibs)
    rand_out_dir = _resolve_output_dir(args.random_dibs)

    # Conditions match our mean_results.json writers.
    conditions = [
        ("no_secret__control", "Baseline (No SC)"),
        ("secret__control", "control (SC)"),
        ("secret__simple", "Simple (SC)"),
    ]

    condition_specs = {
        "no_secret__control": {"secret_channel_enabled": False, "prompt_variant": "control"},
        "secret__control": {"secret_channel_enabled": True, "prompt_variant": "control"},
        "secret__simple": {"secret_channel_enabled": True, "prompt_variant": "simple"},
    }

    # Color palette matching the warm yellow/orange/red bars in the reference snippet.
    colors = {
        "no_secret__control": "#E9C46A",  # yellow
        "secret__control": "#F4A261",  # orange
        "secret__simple": "#E76F51",  # red/salmon
    }

    # 4 groups: first regret, first advantage, random regret, random advantage
    groups = [
        ("Coalition regret (↓)", first, "coalition_mean_regret_mean", "coalition_mean_regret_sd"),
        ("Coalition advantage (↑)", first, "coalition_advantage_mean_mean", "coalition_advantage_mean_sd"),
        ("Coalition regret (↓)", rand, "coalition_mean_regret_mean", "coalition_mean_regret_sd"),
        ("Coalition advantage (↑)", rand, "coalition_advantage_mean_mean", "coalition_advantage_mean_sd"),
    ]

    _apply_style()
    fig = plt.figure(figsize=(11.5, 4.5))
    # Give the annotation lane room (metric labels + brackets + legend) and keep
    # enough separation so annotation text doesn't visually overlap the bars.
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1.0, 1.35], hspace=0.14)
    ax = fig.add_subplot(gs[0])
    ax_ann = fig.add_subplot(gs[1], sharex=ax)

    x_centers = [0.0, 1.0, 2.4, 3.4]  # add gap between first-dibs and random
    bar_w = 0.22
    offsets = [-bar_w, 0.0, bar_w]

    # Load per-run distributions (metrics.json) so normalization uses the pooled distribution
    # across both (first-dibs + random) experiments and all plotted conditions.
    def _collect_metric_values(output_dir: Path, *, metric_key: str) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {c: [] for c, _ in conditions}
        for run_dir in _iter_run_dirs(output_dir):
            metrics = _read_metrics(run_dir)
            if not _is_complete_status(metrics.get("status")):
                continue
            sc = bool(metrics.get("secret_channel_enabled"))
            pv = str(metrics.get("prompt_variant") or "").strip().lower()
            for cond, _ in conditions:
                spec = condition_specs.get(cond) or {}
                if sc != bool(spec.get("secret_channel_enabled")):
                    continue
                if pv != str(spec.get("prompt_variant")):
                    continue
                val = metrics.get(metric_key)
                if isinstance(val, (int, float)) and math.isfinite(float(val)):
                    out[cond].append(float(val))
        return out

    first_regret = _collect_metric_values(first_out_dir, metric_key="coalition_mean_regret")
    rand_regret = _collect_metric_values(rand_out_dir, metric_key="coalition_mean_regret")
    first_adv = _collect_metric_values(first_out_dir, metric_key="coalition_advantage_mean")
    rand_adv = _collect_metric_values(rand_out_dir, metric_key="coalition_advantage_mean")

    pooled_regret = _finite([v for d in (first_regret, rand_regret) for xs in d.values() for v in xs])
    pooled_adv = _finite([v for d in (first_adv, rand_adv) for xs in d.values() for v in xs])
    regret_lo, regret_hi = _robust_range(pooled_regret)
    adv_lo, adv_hi = _robust_range(pooled_adv)

    def _norm_list(vals: List[float], *, lo: float, hi: float) -> List[float]:
        return [_norm_one(v, lo=lo, hi=hi) for v in _finite(vals)]

    # Plot bars + error bars
    for gx, (xlabel, blob, mean_key, sd_key) in zip(x_centers, groups):
        is_regret = "regret" in mean_key
        for (cond, _cond_label), off in zip(conditions, offsets):
            if blob is first and is_regret:
                vals = _norm_list(first_regret.get(cond, []), lo=regret_lo, hi=regret_hi)
            elif blob is rand and is_regret:
                vals = _norm_list(rand_regret.get(cond, []), lo=regret_lo, hi=regret_hi)
            elif blob is first and not is_regret:
                vals = _norm_list(first_adv.get(cond, []), lo=adv_lo, hi=adv_hi)
            else:
                vals = _norm_list(rand_adv.get(cond, []), lo=adv_lo, hi=adv_hi)

            mu = float(sum(vals) / len(vals)) if vals else float("nan")
            err = float(_sem(vals)) if vals else float("nan")
            ax.bar(
                gx + off,
                mu,
                width=bar_w,
                color=colors[cond],
                alpha=0.85,
                edgecolor="none",
            )
            if math.isfinite(err) and err > 0:
                # Clamp error bars so they stay within [0, 1] after normalization.
                # With large SEM relative to the mean, mu-err can go negative, which
                # looks like the error bar "drops below" the axis.
                lower = min(float(err), max(0.0, float(mu)))
                upper = min(float(err), max(0.0, 1.0 - float(mu)))
                ax.errorbar(
                    [gx + off],
                    [mu],
                    yerr=[[lower], [upper]],
                    fmt="none",
                    ecolor="black",
                    capsize=3,
                    linewidth=1.0,
                )

    # No x ticks on the main axis (metric labels are drawn in the annotation lane).
    ax.set_xticks([])
    ax.set_ylabel("Normalized Mean", labelpad=18)
    # Nudge y-label down a bit.
    ax.yaxis.set_label_coords(-0.08, 0.05)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.35)
    # Values are normalized into [0, 1] via pooled robust min/max scaling and clamping.
    # Add a tiny headroom so the top errorbar caps are not clipped.
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])

    ax.tick_params(axis="x", bottom=False, labelbottom=False)

    # Annotation axis (brackets + group labels + legend) below the main plot.
    ax_ann.set_ylim(0.0, 1.0)
    ax_ann.set_yticks([])
    ax_ann.set_xticks([])
    ax_ann.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax_ann.spines.values():
        spine.set_visible(False)

    metric_labels = [
        "Coalition\nregret (↓)",
        "Coalition\nadvantage (↑)",
        "Coalition\nregret (↓)",
        "Coalition\nadvantage (↑)",
    ]
    metric_trans = ax_ann.get_xaxis_transform()  # x=data, y=axes
    for x, label in zip(x_centers, metric_labels):
        ax_ann.text(
            x,
            0.75,
            label,
            transform=metric_trans,
            ha="center",
            va="bottom",
            fontsize=22.5,
            color="black",
            clip_on=False,
        )

    # Brackets under the two pairs (drawn in separate axis to avoid overlap).
    bracket_dx = 0.06
    left_pair_start = x_centers[0] - 1.6 * bar_w + bracket_dx
    left_pair_end = x_centers[1] + 1.6 * bar_w + bracket_dx
    right_pair_start = x_centers[2] - 1.6 * bar_w + bracket_dx
    right_pair_end = x_centers[3] + 1.6 * bar_w + bracket_dx

    _add_group_bracket(
        ax_ann,
        x0=left_pair_start,
        x1=left_pair_end,
        y=0.66,
        label="Colluders First",
        cap_height=0.12,
        label_pad=0.10,
    )
    _add_group_bracket(
        ax_ann,
        x0=right_pair_start,
        x1=right_pair_end,
        y=0.66,
        label="Random Order",
        cap_height=0.12,
        label_pad=0.10,
    )

    legend_handles = [
        Patch(facecolor=colors[cond], edgecolor="none", label=label)
        for cond, label in conditions
    ]
    ax_ann.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        bbox_transform=ax_ann.transAxes,
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.4,
    )
    # Optional sanity checks (kept out of the plot to avoid layout collisions).
    _ = _extract_agent_order_label(first_dir) if first_dir.is_dir() else "unknown"
    _ = _extract_agent_order_label(rand_dir) if rand_dir.is_dir() else "unknown"

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Extra padding so the y-axis label doesn't get clipped when using bbox_inches="tight".
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.17)
    plt.close(fig)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
