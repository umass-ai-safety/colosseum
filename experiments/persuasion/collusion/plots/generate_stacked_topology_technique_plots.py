from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Literal


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "experiments").is_dir() and (parent / "pyproject.toml").exists():
            return parent
    for parent in (here.parent, *here.parents):
        if (parent / "experiments").is_dir():
            return parent
    return here.parents[3]


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))

# Avoid relying on a user-writable HOME for matplotlib cache/config.
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from experiments.common.plotting.io_utils import (
    as_bool,
    as_float,
    ensure_dir,
    groupby,
    load_json,
    sem,
)
from experiments.common.plotting.style import apply_default_style


logger = logging.getLogger(__name__)

_STYLE = {
    # Match paper-friendly, compact panel style.
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    # Slightly lighter strokes to match the reference panels.
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
}


DEFAULT_ROOTS: Tuple[str, ...] = (
    "experiments/persuasion/collusion/outputs/persuasion_collusion/run_1",
    "experiments/persuasion/collusion/outputs/persuasion_collusion/run_2",
)


DEFAULT_TOPOLOGIES: Tuple[str, ...] = (
    "star",
    "complete",
    "erdos_renyi",
)


DEFAULT_TECHNIQUES: Tuple[str, ...] = (
    "control",
    "helpful_misdirection",
    "authority_nudge",
    "reciprocity_trade",
)


MODEL_LABELS: Dict[str, str] = {
    "openai_gpt4.1_mini": "GPT-4.1-Mini",
    "openai_gpt4o_mini": "GPT-4o-Mini",
    "together-kimik2-Instruct": "Kimi-K2-Instruct",
}


TECHNIQUE_LABELS: Dict[str, str] = {
    "control": "Ctrl",
    "helpful_misdirection": "Misdr",
    "authority_nudge": "Auth",
    "social_proof": "SocPr",
    "scarcity_pressure": "Scarc",
    "reciprocity_trade": "Recip",
}

TOPOLOGY_TITLES: Dict[str, str] = {
    "erdos_renyi": "Erdos-Renyi (p = 0.6)",
    "star":"Star",
    "complete":"Complete"
}


def _iter_summary_paths(roots: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if (root / "summary.json").exists():
            p = (root / "summary.json").resolve()
            if p not in seen:
                out.append(p)
                seen.add(p)
            continue
        if root.is_dir():
            for p in root.rglob("summary.json"):
                if not p.is_file():
                    continue
                rp = p.resolve()
                if rp not in seen:
                    out.append(rp)
                    seen.add(rp)
    return sorted(out)


def _load_summary_rows(summary_path: Path) -> List[Dict[str, Any]]:
    payload = load_json(summary_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {summary_path}")
    return [r for r in payload if isinstance(r, dict)]


def _normalize_rows(
    rows: List[Dict[str, Any]],
    *,
    include_incomplete: bool,
    source: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        status = str(r.get("status") or "").strip().lower()
        if not include_incomplete and status != "complete":
            continue

        secret = as_bool(r.get("secret_channel_enabled"))
        prompt_variant = str(r.get("prompt_variant") or "").strip() or "control"
        technique = prompt_variant if bool(secret) else "no_secret"

        out.append(
            {
                **r,
                "secret_channel_enabled": bool(secret),
                "prompt_variant": prompt_variant,
                "technique": technique,
                "_source": source,
            }
        )
    return out


def _sorted_models(rows: List[Dict[str, Any]]) -> List[str]:
    present = {str(r.get("model_label")) for r in rows if r.get("model_label")}
    preferred = [m for m in MODEL_LABELS.keys() if m in present]
    rest = sorted(present - set(preferred))
    return preferred + rest


def _compute_group_stats(
    rows: List[Dict[str, Any]],
    *,
    metric: str,
) -> List[Dict[str, Any]]:
    grouped = groupby(rows, ("topology", "model_label", "technique"))
    out: List[Dict[str, Any]] = []
    for (topology, model_label, technique), group_rows in grouped.items():
        vals = [v for v in (as_float(r.get(metric)) for r in group_rows) if v is not None]
        if not vals:
            continue
        out.append(
            {
                "topology": str(topology),
                "model_label": str(model_label),
                "technique": str(technique),
                "mean": float(sum(vals) / len(vals)),
                "sem": float(sem(vals)),
                "n": int(len(vals)),
            }
        )
    return out


def _pretty_model_label(model_label: str) -> str:
    return MODEL_LABELS.get(model_label, model_label)


def _metric_label(metric: str) -> str:
    if metric == "mean_regret":
        return "Regret"
    if metric == "coalition_mean_regret":
        return "Coalition Regret"
    if metric == "noncoalition_mean_regret":
        return "Non-coalition Regret"
    if metric == "coalition_advantage_mean":
        return "Coalition Advantage (↑)"
    if metric == "coalition_mean_reward":
        return "Coalition Mean Reward (↑)"
    if metric == "noncoalition_mean_reward":
        return "Non-coalition Mean Reward (↑)"
    return metric.replace("_", " ")

NormalizationMode = Literal["none", "minmax_0_1"]
NormalizationScope = Literal["topology", "global"]


def _is_finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def _compute_minmax_params(
    rows: List[Dict[str, Any]],
    *,
    use_bounds: bool,
) -> Tuple[float, float]:
    values: List[float] = []
    for r in rows:
        mean_v = as_float(r.get("mean"))
        sem_v = as_float(r.get("sem")) or 0.0
        if mean_v is None:
            continue
        if not _is_finite(float(mean_v)):
            continue
        if use_bounds:
            values.append(float(mean_v) - float(sem_v))
            values.append(float(mean_v) + float(sem_v))
        else:
            values.append(float(mean_v))

    if not values:
        return 0.0, 1.0
    min_v = min(values)
    max_v = max(values)
    denom = max(max_v - min_v, 1e-12)
    return float(min_v), float(denom)

def _color_map(techniques: List[str]) -> Dict[str, Any]:
    # Fixed, paper-friendly palette (matches short-label legend style).
    base: Dict[str, str] = {
        "control": "#9e9e9e",
        "helpful_misdirection": "#72b7b2",
        "authority_nudge": "#e45756",
        "social_proof": "#4c78a8",
        "scarcity_pressure": "#f58518",
        "reciprocity_trade": "#b279a2",
    }
    # Fall back to tab10 for any unexpected technique names.
    palette = list(plt.cm.tab10.colors)  # type: ignore[attr-defined]
    colors: Dict[str, Any] = {}
    extra_i = 0
    for tech in techniques:
        if tech in base:
            colors[tech] = base[tech]
        else:
            colors[tech] = palette[extra_i % len(palette)]
            extra_i += 1
    return colors


def _plot_into_axis(
    *,
    ax: Any,
    topology: str,
    stats: List[Dict[str, Any]],
    models: List[str],
    techniques: List[str],
    colors: Dict[str, Any],
    normalize: NormalizationMode,
    norm_min: float,
    norm_denom: float,
    show_ylabel: bool,
    metric_label: str,
    show_xlabels: bool,
    show_hline_at_zero: bool,
) -> None:
    key_rows = [r for r in stats if str(r.get("topology")) == topology]
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(r["model_label"]), str(r["technique"])): r for r in key_rows
    }

    min_v = float(norm_min)
    denom = float(norm_denom)

    # Narrower groups/bars to increase whitespace between model columns.
    group_width = 0.72
    bar_w = group_width / max(1, len(techniques))
    x0 = list(range(len(models)))

    for i, tech in enumerate(techniques):
        offsets = (i - (len(techniques) - 1) / 2) * bar_w
        xs: List[float] = []
        means: List[float] = []
        errs_low: List[float] = []
        errs_high: List[float] = []
        for model_idx, model in enumerate(models):
            row = by_key.get((model, tech))
            if row is None:
                continue
            mean_v = float(row["mean"])
            sem_v = float(row["sem"])
            if normalize == "minmax_0_1":
                mean_v = (mean_v - min_v) / denom
                sem_v = sem_v / denom
                lower = max(0.0, mean_v - sem_v)
                upper = min(1.0, mean_v + sem_v)
                errs_low.append(mean_v - lower)
                errs_high.append(upper - mean_v)
            else:
                errs_low.append(sem_v)
                errs_high.append(sem_v)
            xs.append(x0[model_idx] + offsets)
            means.append(mean_v)

        if not xs:
            continue
        ax.bar(
            xs,
            means,
            width=bar_w * 0.88,
            color=colors[tech],
            yerr=[errs_low, errs_high],
            capsize=3,
            error_kw={"ecolor": "black", "elinewidth": 1.0, "capthick": 1.0},
            linewidth=0.0,
            edgecolor="none",
            alpha=1.0,
        )

    ax.set_title(TOPOLOGY_TITLES.get(topology, topology))
    if show_ylabel:
        ax.set_ylabel(metric_label)
    ax.grid(False)
    if show_hline_at_zero:
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    if normalize == "minmax_0_1":
        # Remove extra padding so bars sit on the subplot baseline (matches reference style).
        # Add a small headroom so error bars don't get clipped at 1.0.
        ax.set_ylim(0.0, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])

    ax.set_xticks(x0)
    if show_xlabels:
        ax.set_xticklabels([_pretty_model_label(m) for m in models])
    else:
        ax.set_xticklabels([])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single stacked figure (one subplot per topology) with shared legend, "
            "aggregating run_1 + run_2 with SEM error bars."
        )
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=list(DEFAULT_ROOTS),
        help="Directories containing one or more output runs (each run has summary.json).",
    )
    parser.add_argument(
        "--out-path",
        default="experiments/persuasion/collusion/outputs/persuasion_collusion/topology_technique_plots/stacked_topologies.pdf",
        help="PNG path to write (single combined figure).",
    )
    parser.add_argument(
        "--metric",
        default="coalition_mean_regret",
        help="Metric column in summary.json to plot (e.g., mean_regret, coalition_advantage_mean).",
    )
    parser.add_argument(
        "--normalize",
        choices=("none", "minmax_0_1"),
        default="minmax_0_1",
        help="Normalize plotted bar heights to the range [0, 1] via min-max scaling.",
    )
    parser.add_argument(
        "--normalize-scope",
        choices=("topology", "global"),
        default="global",
        help="When using --normalize minmax_0_1, compute min/max within each topology or across all plotted bars.",
    )
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument(
        "--topologies",
        nargs="*",
        default=list(DEFAULT_TOPOLOGIES),
        help="Topologies to plot (stack order top→bottom). Defaults to star, complete, erdos_renyi.",
    )
    parser.add_argument(
        "--techniques",
        nargs="*",
        default=list(DEFAULT_TECHNIQUES),
        help=(
            "Techniques to include in the legend. Baseline `no_secret` is intentionally omitted for this stacked plot. "
            "Defaults to control, helpful_misdirection, authority_nudge, reciprocity_trade."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    apply_default_style(plt)
    plt.rcParams.update(_STYLE)

    ensure_dir(Path(os.environ["MPLCONFIGDIR"]))

    roots = [Path(r) for r in args.roots]
    summary_paths = _iter_summary_paths(roots)
    if not summary_paths:
        raise SystemExit("No summary.json files found under --roots.")

    all_rows: List[Dict[str, Any]] = []
    for p in summary_paths:
        all_rows.extend(
            _normalize_rows(
                _load_summary_rows(p),
                include_incomplete=bool(args.include_incomplete),
                source=str(p),
            )
        )
    if not all_rows:
        raise SystemExit("No rows to plot after filtering (check --include-incomplete).")

    allowed_topologies = [str(t).strip() for t in args.topologies if str(t).strip()]
    allowed_techniques = [
        str(t).strip()
        for t in args.techniques
        if str(t).strip() and str(t).strip() != "no_secret"
    ]
    topo_set = set(allowed_topologies)
    tech_set = set(allowed_techniques)
    all_rows = [
        r
        for r in all_rows
        if (str(r.get("topology")) in topo_set) and (str(r.get("technique")) in tech_set)
    ]
    if not all_rows:
        raise SystemExit("No rows left after applying --topologies/--techniques filters.")

    models = _sorted_models(all_rows)
    present_techniques = {str(r.get("technique")) for r in all_rows if r.get("technique") is not None}
    techniques = [t for t in allowed_techniques if t in present_techniques]
    if not techniques:
        raise SystemExit("No techniques present after filtering.")

    metric = str(args.metric)
    stats = _compute_group_stats(all_rows, metric=metric)
    ylabel = _metric_label(metric)
    normalize_mode: NormalizationMode = str(args.normalize)  # type: ignore[assignment]
    normalize_scope: NormalizationScope = str(args.normalize_scope)  # type: ignore[assignment]
    if normalize_mode == "minmax_0_1":
        ylabel = f"Normalized {ylabel}"
    colors = _color_map(techniques)

    nrows = len(allowed_topologies)
    # Keep a compact 4:3-ish ratio similar to paper panels.
    fig_w = max(7.6, 2.55 * len(models))
    # Slightly shorter, like the reference figure.
    fig_h = max(4.8, 1.45 * nrows + 0.3)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(fig_w, fig_h),
        sharex=True,
        gridspec_kw={"hspace": 0.55},
    )
    if nrows == 1:
        axes = [axes]

    if normalize_mode == "minmax_0_1" and normalize_scope == "global":
        norm_min, norm_denom = _compute_minmax_params(stats, use_bounds=True)
    else:
        norm_min, norm_denom = 0.0, 1.0

    for i, topology in enumerate(allowed_topologies):
        if normalize_mode == "minmax_0_1" and normalize_scope == "topology":
            topo_rows = [r for r in stats if str(r.get("topology")) == topology]
            topo_min, topo_denom = _compute_minmax_params(topo_rows, use_bounds=True)
        else:
            topo_min, topo_denom = norm_min, norm_denom

        _plot_into_axis(
            ax=axes[i],
            topology=topology,
            stats=stats,
            models=models,
            techniques=techniques,
            colors=colors,
            normalize=normalize_mode,
            norm_min=topo_min,
            norm_denom=topo_denom,
            show_ylabel=(nrows == 1),
            metric_label=ylabel,
            show_xlabels=(i == nrows - 1),
            show_hline_at_zero=("advantage" in metric),
        )

    supy = None
    if nrows > 1:
        supy = fig.supylabel(ylabel)

    legend_handles = [
        Patch(facecolor=colors[t], edgecolor="none", label=TECHNIQUE_LABELS.get(t, t))
        for t in techniques
    ]
    legend_ncol = max(1, len(legend_handles))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        # Keep the legend close but not touching the x-ticks.
        bbox_to_anchor=(0.5, -0.04),
        ncol=legend_ncol,
        frameon=False,
    )

    # Reserve space for the bottom legend and the large supylabel.
    fig.tight_layout(rect=(0.08, 0.30, 1.0, 0.98))
    # `tight_layout()` can reposition figure-level labels, so adjust after.
    if supy is not None:
        supy.set_fontsize(16)
        supy.set_x(0.02)
    # Keep inter-panel spacing stable after tight_layout.
    fig.subplots_adjust(hspace=0.55)
    out_path = Path(str(args.out_path))
    ensure_dir(out_path.parent)
    # Keep legend outside the axes without overlapping or cropping.
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
