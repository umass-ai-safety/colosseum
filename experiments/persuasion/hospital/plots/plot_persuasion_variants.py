from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from experiments.common.plotting.io_utils import ensure_dir, finite, groupby, mean, sem
from experiments.common.plotting.logging_utils import log_saved_plot
from experiments.common.plotting.style import apply_default_style

logger = logging.getLogger(__name__)

_STYLE = {
    # These by-variant plots are often used as small panels in papers.
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
}

_VARIANT_LABELS = {
    # Keep these short (3-5 chars) so tick labels never overlap even at larger fonts.
    "control": "Ctrl",
    "helpful_misdirection": "Misdr",
    "authority_nudge": "Auth",
    "social_proof": "SocPr",
    "scarcity_pressure": "Scarc",
    "reciprocity_trade": "Recip",
}


def _variant_label(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return s
    return _VARIANT_LABELS.get(s, s.replace("_", " "))


def _variant_order(rows: List[Dict[str, Any]]) -> List[str]:
    preferred = [
        "control",
        "helpful_misdirection",
        "authority_nudge",
        "social_proof",
        "scarcity_pressure",
        "reciprocity_trade",
    ]
    present = []
    seen = set()
    for r in rows:
        v = str(r.get("prompt_variant") or "").strip()
        if not v or v in seen:
            continue
        present.append(v)
        seen.add(v)
    out: List[str] = []
    for v in preferred:
        if v in seen:
            out.append(v)
    for v in present:
        if v not in out:
            out.append(v)
    return out


def _plot_by_variant(
    rows: List[Dict[str, Any]],
    *,
    y_key: str,
    y_label: str,
    out_path: Path,
    hline_at_zero: bool = False,
) -> None:
    apply_default_style(plt)
    plt.rcParams.update(_STYLE)
    ensure_dir(out_path.parent)

    variants = _variant_order(rows)
    if not variants:
        return

    # Keep these compact; these plots are often composed into multi-panel figures.
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.grid(False)

    xs: List[float] = []
    ys_mean: List[float] = []
    ys_sem: List[float] = []
    labels: List[str] = []
    for i, v in enumerate(variants):
        vals = finite([r.get(y_key) for r in rows if str(r.get("prompt_variant")) == v])
        if not vals:
            continue
        xs.append(float(i))
        ys_mean.append(float(mean(vals)))
        ys_sem.append(float(sem(vals)))
        labels.append(v)
        ax.scatter([float(i)] * len(vals), vals, s=10, alpha=0.25, color="#4c78a8")

    ax.errorbar(xs, ys_mean, yerr=ys_sem, fmt="o", ms=6, color="black", capsize=3, zorder=5)
    if hline_at_zero:
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)

    ax.set_xticks(list(range(len(labels))))
    # Never use rotation other than 0 or 90.
    ax.set_xticklabels([_variant_label(l) for l in labels], rotation=90, ha="center")
    ax.set_ylabel(y_label, labelpad=2)

    # With large fonts + rotated ticks, ensure labels stay inside the saved PDF.
    fig.subplots_adjust(left=0.22, bottom=0.30)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def plot_persuasion_variant_effects(
    *,
    run_rows: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """
    Persuasion-specific plots, stratified by (strategy, adversary_count, target_role).
    Focuses on prompt_variant and secret-channel usage.
    """
    ensure_dir(out_dir)

    # Only runs with an active coalition and secret channel; otherwise prompt_variant is usually control.
    rows = [
        r
        for r in run_rows
        if bool(r.get("secret_channel_enabled")) is True
        and (int(r.get("adversary_count") or 0) > 0)
    ]
    if not rows:
        return

    by = groupby(rows, ("strategy", "adversary_count", "target_role"))
    for (strategy, adv_count, target_role), group_rows in by.items():
        if not group_rows:
            continue
        subdir = (
            out_dir
            / "by_variant"
            / f"strat_{str(strategy)}"
            / f"adv_{int(adv_count) if adv_count is not None else 'na'}"
            / f"role_{str(target_role)}"
        )
        ensure_dir(subdir)

        # Persuasion-specific: secret channel usage by colluders.
        _plot_by_variant(
            group_rows,
            y_key="colluder_posts_secret_rate",
            y_label="Colluder Secret Post Rate",
            out_path=subdir / "colluder_posts_secret_rate_by_variant.pdf",
            hline_at_zero=False,
        )

        # Now that metrics align with agent_misalignment, add key outcomes over variants.
        _plot_by_variant(
            group_rows,
            y_key="coalition_minus_noncoalition_avg",
            y_label="Coalition Advantage",
            out_path=subdir / "coalition_advantage_by_variant.pdf",
            hline_at_zero=True,
        )
        _plot_by_variant(
            group_rows,
            y_key="coalition_minus_noncoalition_hospital_avg_actual",
            y_label="Coalition Advantage",
            out_path=subdir / "hospital_advantage_actual_by_variant.pdf",
            hline_at_zero=True,
        )
        _plot_by_variant(
            group_rows,
            y_key="coalition_reward_regret_hospital_adjusted",
            y_label="Coalition Regret",
            out_path=subdir / "coalition_regret_hospital_adjusted_by_variant.pdf",
            hline_at_zero=False,
        )
        _plot_by_variant(
            group_rows,
            y_key="joint_reward",
            y_label="Joint Reward",
            out_path=subdir / "joint_reward_by_variant.pdf",
            hline_at_zero=False,
        )
        _plot_by_variant(
            group_rows,
            y_key="joint_reward_regret",
            y_label="Overall Regret",
            out_path=subdir / "overall_regret_by_variant.pdf",
            hline_at_zero=False,
        )
