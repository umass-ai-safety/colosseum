from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    "experiments/persuasion/collusion/outputs/persuasion_collusion/20260123-142821",
    "experiments/persuasion/collusion/outputs/persuasion_collusion/20260123-170703",
    "experiments/persuasion/collusion/outputs/persuasion_collusion/20260123-220456",
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


def _load_summary_rows(root: Path) -> List[Dict[str, Any]]:
    summary_path = root / "summary.json"
    payload = load_json(summary_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {summary_path}")
    return [r for r in payload if isinstance(r, dict)]


def _normalize_rows(
    rows: List[Dict[str, Any]], *, include_incomplete: bool
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


def _plot_topology(
    *,
    topology: str,
    stats: List[Dict[str, Any]],
    models: List[str],
    techniques: List[str],
    out_path: Path,
    metric_label: str,
) -> None:
    key_rows = [r for r in stats if str(r.get("topology")) == topology]
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(r["model_label"]), str(r["technique"])): r for r in key_rows
    }

    fig_w = max(10.0, 2.4 * len(models))
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
        xs = [x + offsets for x in x0]
        means: List[float] = []
        errs: List[float] = []
        for model in models:
            row = by_key.get((model, tech))
            if row is None:
                means.append(float("nan"))
                errs.append(0.0)
            else:
                means.append(float(row["mean"]))
                errs.append(float(row["sem"]))
        ax.bar(
            xs,
            means,
            width=bar_w * 0.95,
            color=colors.get(tech, "#333333"),
            yerr=errs,
            capsize=3,
            linewidth=0.6,
            edgecolor="black",
            alpha=0.9,
            label=TECHNIQUE_LABELS.get(tech, tech.replace("_", " ")),
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
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-topology bar charts (example-plot style) from persuasion_collusion summaries."
        )
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=list(DEFAULT_ROOTS),
        help="Output roots (each must contain summary.json). Defaults to the three 20260123 runs.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/persuasion/collusion/outputs/persuasion_collusion/20260123-example-style-plots",
    )
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument(
        "--metric",
        "--regret-metric",
        dest="metric",
        default="mean_regret",
        help="Metric column in summary.json to plot.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    apply_default_style(plt)

    roots = [Path(r) for r in args.roots]
    all_rows: List[Dict[str, Any]] = []
    for root in roots:
        all_rows.extend(_load_summary_rows(root))
    rows = _normalize_rows(all_rows, include_incomplete=bool(args.include_incomplete))

    if not rows:
        raise SystemExit("No rows to plot (check roots or --include-incomplete).")

    models = _sorted_models(rows)
    techniques = _collect_techniques(rows)
    metric = str(args.metric)
    stats = _compute_group_stats(rows, metric=metric)

    topologies = sorted({str(r.get("topology")) for r in rows if r.get("topology")})
    out_dir = Path(str(args.out_dir))

    metric_label = metric.replace("_", " ")
    if metric == "mean_regret":
        metric_label = "Regret (↓)"
    elif metric == "coalition_mean_regret":
        metric_label = "Coalition Mean Regret (↓)"
    elif metric == "noncoalition_mean_regret":
        metric_label = "Non-coalition Mean Regret (↓)"

    for topology in topologies:
        out_path = out_dir / f"example_style__{sanitize_filename(topology)}.png"
        _plot_topology(
            topology=topology,
            stats=stats,
            models=models,
            techniques=techniques,
            out_path=out_path,
            metric_label=metric_label,
        )


if __name__ == "__main__":
    main()
