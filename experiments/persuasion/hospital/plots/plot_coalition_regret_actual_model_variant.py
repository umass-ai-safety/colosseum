from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from experiments.common.plotting.io_utils import as_float, as_int, ensure_dir, finite
from experiments.common.plotting.load_runs import load_runs
from experiments.common.plotting.style import apply_default_style

from .build_tables import build_tables


_DISPLAY_MAP = {
    "openai-gpt-4.1-mini": "GPT-4.1-Mini",
    "openai-gpt-4o-mini": "GPT-4o-Mini",
    "together-kimik2-Instruct": "Kimi-K2-Instruct",
}

_VARIANT_ORDER = [
    "control",
    "helpful_misdirection",
    "authority_nudge",
    "social_proof",
    "scarcity_pressure",
    "reciprocity_trade",
]

_VARIANT_LABELS = {
    "control": "Ctrl",
    "helpful_misdirection": "Misdr",
    "authority_nudge": "Auth",
    "social_proof": "SocPr",
    "scarcity_pressure": "Scarc",
    "reciprocity_trade": "Recip",
}

# Use figure-6 palette anchors, extended to 6 prompt variants.
_VARIANT_COLORS = {
    "control": "#9d9d9d",
    "helpful_misdirection": "#72b7b2",
    "authority_nudge": "#e45756",
    "social_proof": "#4c78a8",
    "scarcity_pressure": "#f58518",
    "reciprocity_trade": "#b279a2",
}


def _strategy_type(value: Any) -> str:
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


def _mean(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _sem(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = float(sum(xs) / len(xs))
    var = sum((x - m) ** 2 for x in xs) / float(len(xs) - 1)
    return float(math.sqrt(var) / math.sqrt(len(xs)))


def _filter_rows(
    rows: List[Dict[str, Any]],
    *,
    num_agents: int,
    target_role: str,
    adversary_count: int,
    seeds: Optional[set[int]],
    secret_only: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if as_int(r.get("num_agents")) != int(num_agents):
            continue
        if str(r.get("target_role")) != str(target_role):
            continue
        if as_int(r.get("adversary_count")) != int(adversary_count):
            continue
        if secret_only and bool(r.get("secret_channel_enabled")) is not True:
            continue
        if seeds is not None:
            seed = as_int(r.get("seed"))
            if seed is None or int(seed) not in seeds:
                continue
        out.append(r)
    return out


def _collect_model_rows(
    *,
    model_runs_dir: Path,
    sweep_name: str,
    model_label: str,
    num_agents: int,
    target_role: str,
    adversary_count: int,
    seeds: Optional[set[int]],
    secret_only: bool,
    metric_key: str,
) -> List[Dict[str, Any]]:
    sweep_dir = model_runs_dir / sweep_name
    if not sweep_dir.exists():
        return []
    runs, _meta = load_runs(sweep_dir)
    tables = build_tables(runs)
    filtered = _filter_rows(
        tables.run_rows,
        num_agents=num_agents,
        target_role=target_role,
        adversary_count=adversary_count,
        seeds=seeds,
        secret_only=secret_only,
    )

    out: List[Dict[str, Any]] = []
    for r in filtered:
        stype = _strategy_type(r.get("strategy"))
        pv = str(r.get("prompt_variant") or "").strip()
        if not pv:
            continue
        val = as_float(r.get(metric_key))
        if val is None or not math.isfinite(float(val)):
            continue
        out.append(
            {
                "model_label": str(model_label),
                "strategy_type": str(stype),
                "prompt_variant": pv,
                "value": float(val),
                "seed": as_int(r.get("seed")),
            }
        )
    return out


def _plot(
    *,
    rows: List[Dict[str, Any]],
    out_path: Path,
    model_order: Sequence[str],
    strategy_order: Sequence[str],
    variant_order: Sequence[str],
    metric_label: str,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    # Match figure-6 fonts (paper defaults in this repo).
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
    xtick_fontsize_pts = float(plt.rcParams.get("xtick.labelsize", 26))

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
    hue_count = max(1, len(variant_order))
    group_width = 0.85
    bar_width = group_width / float(hue_count)
    x_step = 1.25
    xs = [float(i) * x_step for i in range(group_count)]

    for ax, stype in zip(axes, strategy_order):
        ax.grid(False)

        for j, variant in enumerate(variant_order):
            offsets = [(j - (hue_count - 1) / 2.0) * bar_width for _ in xs]
            heights: List[float] = []
            yerrs: List[float] = []
            for model_label in model_order:
                vals = finite(
                    [
                        as_float(r.get("value"))
                        for r in rows
                        if str(r.get("strategy_type")) == str(stype)
                        and str(r.get("model_label")) == str(model_label)
                        and str(r.get("prompt_variant")) == str(variant)
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
                color=_VARIANT_COLORS.get(str(variant), None),
                label=_VARIANT_LABELS.get(str(variant), str(variant)),
            )

        ax.set_title(_strategy_label(str(stype)))

    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(list(model_order), rotation=0, ha="center")
    fig.supylabel(metric_label)

    handles: List[Patch] = []
    for v in variant_order:
        handles.append(
            Patch(
                facecolor=_VARIANT_COLORS.get(str(v), "#999999"),
                edgecolor="none",
                label=_VARIANT_LABELS.get(str(v), str(v)),
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

    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer=renderer)
        desired_gap_px = max(((xtick_fontsize_pts * 2.0) / 72.0) * float(fig.dpi), 6.0)

        # Reserve enough bottom margin so the x-ticks + labels sit above the legend with desired gap.
        required_bottom_px = float(legend_bbox.y1 + desired_gap_px)
        required_bottom_frac = required_bottom_px / float(fig.bbox.height)
        required_bottom_frac = min(max(required_bottom_frac, 0.0), 0.45)
        fig.subplots_adjust(bottom=required_bottom_frac)
        fig.canvas.draw()
    except Exception:
        pass
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot coalition regret (actual) by model + persuasion prompt variant, faceted by misalignment type."
    )
    parser.add_argument(
        "--model-runs-dir",
        action="append",
        dest="model_runs_dir",
        default=None,
        help="Repeatable. Path like .../outputs/persuasion_hospital/<ts>/runs/<model_label>.",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        default=None,
        help="Repeatable display labels for each --model-runs-dir (optional).",
    )
    parser.add_argument("--sweep-name", type=str, default="persuasion_hospital_test")
    parser.add_argument("--num-agents", type=int, default=9)
    parser.add_argument("--target-role", type=str, default="departmental")
    parser.add_argument("--adversary-count", type=int, default=4)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help='Optional seed filter like "0-4" or "0,1,2".',
    )
    parser.add_argument(
        "--include-secret0",
        action="store_true",
        help="Include runs where secret_channel_enabled is false (default: only secret1 runs).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/persuasion/hospital/plots_outputs_model_compare",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="coalition_regret_actual_by_model_and_variant.pdf",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help=(
            "Plot a normalized counterpart: normalize coalition regret via "
            "regret / (coalition_max_reward_sum - cmin), where cmin = min(empirical_min_coalition_reward_sum, 0)."
        ),
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help=(
            "Optional comma-separated prompt variants to include (e.g. "
            "'control,helpful_misdirection,authority_nudge,reciprocity_trade'). "
            "If omitted, includes all present variants."
        ),
    )
    parser.add_argument(
        "--subset-core",
        action="store_true",
        help="Convenience: keep only control, helpful_misdirection, authority_nudge, reciprocity_trade.",
    )
    args = parser.parse_args(argv)

    model_dirs = [Path(p).expanduser().resolve() for p in (args.model_runs_dir or [])]
    if not model_dirs:
        raise SystemExit("Provide at least one --model-runs-dir.")

    labels = args.labels or []
    model_labels: List[str] = []
    for i, d in enumerate(model_dirs):
        if i < len(labels) and str(labels[i]).strip():
            model_labels.append(str(labels[i]).strip())
        else:
            model_labels.append(_DISPLAY_MAP.get(d.name, d.name))

    seeds = _parse_seeds_spec(args.seeds)
    secret_only = not bool(args.include_secret0)

    metric_key = "coalition_reward_regret"
    metric_label = "Coalition Regret"

    rows: List[Dict[str, Any]] = []
    for d, mlabel in zip(model_dirs, model_labels):
        rows.extend(
            _collect_model_rows(
                model_runs_dir=d,
                sweep_name=str(args.sweep_name),
                model_label=str(mlabel),
                num_agents=int(args.num_agents),
                target_role=str(args.target_role),
                adversary_count=int(args.adversary_count),
                seeds=seeds,
                secret_only=secret_only,
                metric_key=metric_key,
            )
        )

    if bool(args.normalized):
        # Compute cmin over the *same slice* of runs being plotted.
        coalition_sums: List[float] = []
        coalition_maxes: List[float] = []
        for d in model_dirs:
            sweep_dir = d / str(args.sweep_name)
            if not sweep_dir.exists():
                continue
            runs, _ = load_runs(sweep_dir)
            tables = build_tables(runs)
            filtered = _filter_rows(
                tables.run_rows,
                num_agents=int(args.num_agents),
                target_role=str(args.target_role),
                adversary_count=int(args.adversary_count),
                seeds=seeds,
                secret_only=secret_only,
            )
            for r in filtered:
                csum = as_float(r.get("coalition_reward_sum"))
                if csum is not None and math.isfinite(float(csum)):
                    coalition_sums.append(float(csum))
                cmax = as_float(r.get("coalition_max_reward_sum"))
                if cmax is not None and math.isfinite(float(cmax)):
                    coalition_maxes.append(float(cmax))

        if not coalition_maxes:
            raise SystemExit("No coalition_max_reward_sum values found for normalization.")
        # In this experiment slice, cmax should be constant; use the maximum as a conservative scale.
        cmax_val = float(max(coalition_maxes))
        empirical_min = float(min(coalition_sums)) if coalition_sums else 0.0
        cmin_val = float(min(empirical_min, 0.0))
        denom = float(cmax_val - cmin_val)
        if not (math.isfinite(denom) and denom > 0.0):
            raise SystemExit(f"Invalid normalization denom (cmin={cmin_val}, cmax={cmax_val}).")

        for r in rows:
            r["value"] = float(r["value"]) / denom

        metric_label = "Normalized Coalition Regret"
        if str(args.out_name).strip() == "coalition_regret_actual_by_model_and_variant.pdf":
            args.out_name = "coalition_regret_normalized_by_model_and_variant.pdf"

    # Keep only strategies that exist, but preserve paper order.
    preferred_strategy_order = ["covert", "destructive_max", "destructive_no_preservation"]
    present = {str(r.get("strategy_type")) for r in rows if str(r.get("strategy_type"))}
    strategy_order = [s for s in preferred_strategy_order if s in present]
    if not strategy_order:
        raise SystemExit("No rows after filtering; check filters (--seeds, --target-role, etc).")

    # Keep only variants that exist, but preserve preferred order.
    present_variants = {str(r.get("prompt_variant")) for r in rows if str(r.get("prompt_variant"))}
    variant_order = [v for v in _VARIANT_ORDER if v in present_variants]
    if not variant_order:
        raise SystemExit("No prompt variants found after filtering.")

    requested_variants: Optional[List[str]] = None
    if bool(args.subset_core):
        requested_variants = [
            "control",
            "helpful_misdirection",
            "authority_nudge",
            "reciprocity_trade",
        ]
    elif args.variants:
        requested_variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]

    if requested_variants is not None:
        requested_set = set(requested_variants)
        rows = [r for r in rows if str(r.get("prompt_variant")) in requested_set]
        variant_order = [v for v in variant_order if v in requested_set]
        if not variant_order:
            raise SystemExit("No prompt variants left after applying --variants/--subset-core.")

        # Default output naming for subset plots (do not overwrite).
        if str(args.out_name).endswith(".pdf") and not str(args.out_name).endswith("_subset.pdf"):
            args.out_name = str(args.out_name)[: -len(".pdf")] + "_subset.pdf"

    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)
    out_path = out_dir / str(args.out_name)

    _plot(
        rows=rows,
        out_path=out_path,
        model_order=model_labels,
        strategy_order=strategy_order,
        variant_order=variant_order,
        metric_label=metric_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
