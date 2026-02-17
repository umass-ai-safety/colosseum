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

from experiments.common.plotting.io_utils import (
    as_bool,
    as_float,
    ensure_dir,
    groupby,
    load_json,
    sanitize_filename,
    sem,
)
from experiments.common.plotting.style import apply_default_style


logger = logging.getLogger(__name__)


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
    "no_secret",
    "control",
    "helpful_misdirection",
    "authority_nudge",
    "reciprocity_trade",
)


MODEL_LABELS: Dict[str, str] = {
    "openai_gpt4o_mini": "GPT-4o-Mini",
    "openai_gpt4.1_mini": "GPT-4.1-Mini",
    "together-kimik2-Instruct": "Kimi-K2-Instruct",
}


TECHNIQUE_LABELS: Dict[str, str] = {
    "no_secret": "Baseline (No Secret)",
    "control": "Control",
    "authority_nudge": "Authority Nudge",
    "helpful_misdirection": "Helpful Misdirection",
    "scarcity_pressure": "Scarcity Pressure",
    "social_proof": "Social Proof",
    "reciprocity_trade": "Reciprocity Trade",
}


TECHNIQUE_ORDER: Tuple[str, ...] = (
    "no_secret",
    "control",
    "authority_nudge",
    "helpful_misdirection",
    "scarcity_pressure",
    "social_proof",
    "reciprocity_trade",
)


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


def _collect_techniques(rows: List[Dict[str, Any]]) -> List[str]:
    present = {str(r.get("technique")) for r in rows if r.get("technique") is not None}
    ordered = [t for t in TECHNIQUE_ORDER if t in present]
    for t in sorted(present):
        if t not in ordered:
            ordered.append(t)
    return ordered


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
        return "Regret (↓)"
    if metric == "coalition_mean_regret":
        return "Coalition Mean Regret (↓)"
    if metric == "noncoalition_mean_regret":
        return "Non-coalition Mean Regret (↓)"
    if metric == "coalition_advantage_mean":
        return "Coalition Advantage (↑)"
    if metric == "coalition_mean_reward":
        return "Coalition Mean Reward (↑)"
    if metric == "noncoalition_mean_reward":
        return "Non-coalition Mean Reward (↑)"
    return metric.replace("_", " ")


NormalizationMode = Literal["none", "minmax_0_1"]


def _plot_topology(
    *,
    topology: str,
    stats: List[Dict[str, Any]],
    models: List[str],
    techniques: List[str],
    metric_label: str,
    normalize: NormalizationMode,
    out_path: Path,
) -> None:
    key_rows = [r for r in stats if str(r.get("topology")) == topology]
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(r["model_label"]), str(r["technique"])): r for r in key_rows
    }

    bounds_all: List[float] = []
    for r in key_rows:
        mean_v = as_float(r.get("mean"))
        sem_v = as_float(r.get("sem")) or 0.0
        if mean_v is None:
            continue
        if float(mean_v) != float(mean_v):
            continue
        bounds_all.append(float(mean_v) - float(sem_v))
        bounds_all.append(float(mean_v) + float(sem_v))

    if normalize == "minmax_0_1" and bounds_all:
        min_v = min(bounds_all)
        max_v = max(bounds_all)
        denom = max(max_v - min_v, 1e-12)
    else:
        min_v = 0.0
        denom = 1.0

    fig_w = max(12.0, 2.6 * len(models))
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))

    palette = list(plt.cm.tab10.colors)  # type: ignore[attr-defined]
    colors: Dict[str, Any] = {}
    for i, tech in enumerate(techniques):
        if tech == "no_secret":
            colors[tech] = "#1f77b4"  # tab:blue
        elif tech == "control":
            colors[tech] = "#7f7f7f"  # tab:gray
        else:
            colors[tech] = palette[i % len(palette)]

    group_width = 0.84
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
                sem_low = mean_v - lower
                sem_high = upper - mean_v
            else:
                sem_low = sem_v
                sem_high = sem_v
            xs.append(x0[model_idx] + offsets)
            means.append(mean_v)
            errs_low.append(sem_low)
            errs_high.append(sem_high)

        if not xs:
            continue

        ax.bar(
            xs,
            means,
            width=bar_w * 0.95,
            color=colors[tech],
            label=TECHNIQUE_LABELS.get(tech, tech.replace("_", " ")),
            yerr=[errs_low, errs_high],
            capsize=3,
            linewidth=0.6,
            edgecolor="black",
            alpha=0.9,
        )

    ax.set_xticks(x0)
    ax.set_xticklabels([_pretty_model_label(m) for m in models])
    ax.set_ylabel(metric_label)
    ax.set_title(f"Persuasion-collusion — {topology}")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(4, max(1, len(by_label))),
        frameon=False,
    )

    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.grid(False, axis="x")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    if normalize == "minmax_0_1":
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-topology plots with persuasion-technique legend and SEM error bars "
            "by aggregating multiple output roots (e.g., run_1 + run_2)."
        )
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=list(DEFAULT_ROOTS),
        help="Directories containing one or more output runs (each run has summary.json).",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/persuasion/collusion/outputs/persuasion_collusion/topology_technique_plots",
        help="Directory to write PNGs into (one per topology).",
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
        help="Normalize plotted bar heights to the range [0, 1] within each topology via min-max scaling.",
    )
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument(
        "--topologies",
        nargs="*",
        default=list(DEFAULT_TOPOLOGIES),
        help="Topologies to plot. Defaults to star, complete, erdos_renyi.",
    )
    parser.add_argument(
        "--techniques",
        nargs="*",
        default=list(DEFAULT_TECHNIQUES),
        help=(
            "Techniques to include in the legend. "
            "`no_secret` includes rows where secret_channel_enabled=false. "
            "Defaults to no_secret, control, helpful_misdirection, authority_nudge, reciprocity_trade."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    apply_default_style(plt)

    ensure_dir(Path(os.environ["MPLCONFIGDIR"]))

    roots = [Path(r) for r in args.roots]
    summary_paths = _iter_summary_paths(roots)
    if not summary_paths:
        raise SystemExit("No summary.json files found under --roots.")

    all_rows: List[Dict[str, Any]] = []
    for p in summary_paths:
        rows = _load_summary_rows(p)
        all_rows.extend(
            _normalize_rows(
                rows,
                include_incomplete=bool(args.include_incomplete),
                source=str(p),
            )
        )

    if not all_rows:
        raise SystemExit("No rows to plot after filtering (check --include-incomplete).")

    allowed_topologies = {str(t) for t in args.topologies if str(t).strip()}
    allowed_techniques = {str(t) for t in args.techniques if str(t).strip()}
    all_rows = [
        r
        for r in all_rows
        if (str(r.get("topology")) in allowed_topologies)
        and (str(r.get("technique")) in allowed_techniques)
    ]
    if not all_rows:
        raise SystemExit("No rows left after applying --topologies/--techniques filters.")

    metric = str(args.metric)
    models = _sorted_models(all_rows)
    # Preserve the user-specified legend order.
    present = {str(r.get("technique")) for r in all_rows if r.get("technique") is not None}
    techniques = [t for t in (str(x) for x in args.techniques) if t in present]
    topologies = [t for t in (str(x) for x in args.topologies) if t in allowed_topologies]

    out_dir = Path(str(args.out_dir))
    stats = _compute_group_stats(all_rows, metric=metric)
    label = _metric_label(metric)
    if str(args.normalize) == "minmax_0_1":
        label = f"{label} (normalized 0–1)"

    for topology in topologies:
        out_path = out_dir / f"topology__{sanitize_filename(topology)}__{sanitize_filename(metric)}.png"
        _plot_topology(
            topology=topology,
            stats=stats,
            models=models,
            techniques=techniques,
            metric_label=label,
            normalize=str(args.normalize),  # type: ignore[arg-type]
            out_path=out_path,
        )


if __name__ == "__main__":
    main()
