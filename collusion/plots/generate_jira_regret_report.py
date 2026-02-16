from __future__ import annotations

"""Generate a regret-based Jira collusion report plot from a collusion output root.

This module is intentionally minimal and only produces:
- `regret_report__normalized_regret__coalition_gap__judge.png`

It also writes a small CSV with the aggregated values used in the plot:
- `regret_report__normalized_regret__coalition_gap__judge__data.csv`
"""

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch

from experiments.common.plotting.io_utils import (
    as_bool,
    as_float,
    as_int,
    ensure_dir,
    mean,
    safe_load_json,
    sem,
    write_csv,
)


logger = logging.getLogger(__name__)

_LOGO_DIR = Path(__file__).resolve().parent / "assets" / "logos"
_LOGO_FILES = {
    "openai": "openai.png",
    "anthropic": "anthropic.png",
    "gemini": "gemini.png",
    "deepseek": "deepseek.png",
    "moonshot": "moonshot.png",
}

_SIX_BARS_PALETTE = ["#264653", "#2a9d8f", "#8ab17d"]

_COALITION_REGRET_GAP_NORM_BY_MODEL: Optional[Dict[str, float]] = None


def _lighten_color(color: str, amount: float = 0.55) -> Tuple[float, float, float]:
    base = np.array(to_rgb(color))
    return tuple(base + (1.0 - base) * float(amount))


def _canonical_variant(value: Any) -> str:
    return str(value or "").strip()


_PROVIDER_PREFIX_RE = re.compile(r"^(openai|anthropic|together)[-_]+", re.IGNORECASE)


def _pretty_metric_label(key: str) -> str:
    k = str(key or "").strip()
    if k == "normalized_regret":
        return "Normalized Regret (↓)"
    if k == "normalized_coalition_regret_gap":
        return "Coalition Advantage (-)"
    if k == "judge_mean_rating":
        return "Collusion Judge (↓)"
    return k.replace("_", " ").title()


def _pretty_model_label(model_label: str) -> str:
    raw = str(model_label or "").strip()
    raw = _PROVIDER_PREFIX_RE.sub("", raw)
    if not raw:
        return "Unknown"

    lowered = raw.lower()
    if "claude-sonnet-4-5" in lowered:
        return "Sonnet-4.5"
    if lowered.startswith("kimik2"):
        if "thinking" in lowered:
            return "Kimi-K2-Thinking"
        return "Kimi-K2-Inst"

    parts = re.split(r"[-_]+", raw)
    pretty: List[str] = []
    for part in parts:
        if not part:
            continue
        p = part.strip()
        pl = p.lower()
        if pl == "gpt":
            pretty.append("GPT")
            continue
        if pl == "oss":
            pretty.append("OSS")
            continue
        if re.fullmatch(r"\d+b", pl):
            pretty.append(f"{p[:-1]}B")
            continue
        if pl == "k2":
            pretty.append("K2")
            continue
        if pl in {
            "claude",
            "sonnet",
            "haiku",
            "opus",
            "gemini",
            "flash",
            "lite",
            "mini",
            "instruct",
        }:
            pretty.append(pl.capitalize())
            continue
        pretty.append(p)

    return "-".join(pretty) if pretty else raw


def _apply_large_font_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


def _set_coalition_regret_gap_norm(rows: List["RunRow"]) -> None:
    global _COALITION_REGRET_GAP_NORM_BY_MODEL
    vals_by_model: Dict[str, List[float]] = {}
    for r in rows:
        c = r.coalition_mean_regret
        n = r.noncoalition_mean_regret
        if c is None or n is None:
            continue
        try:
            gap = float(n) - float(c)
        except Exception:
            continue
        if not math.isfinite(gap):
            continue
        vals_by_model.setdefault(str(r.model_label), []).append(gap)
    if not vals_by_model:
        _COALITION_REGRET_GAP_NORM_BY_MODEL = None
        return
    norm_by_model: Dict[str, float] = {}
    for model_label, vals in vals_by_model.items():
        if not vals:
            continue
        lo = float(min(vals))
        hi = float(max(vals))
        max_abs = max(abs(lo), abs(hi))
        if math.isfinite(max_abs):
            norm_by_model[model_label] = float(max_abs)
    _COALITION_REGRET_GAP_NORM_BY_MODEL = norm_by_model or None


def _normalize_coalition_regret_gap(value: float, model_label: str) -> Optional[float]:
    norm = _COALITION_REGRET_GAP_NORM_BY_MODEL
    if norm is None:
        return None
    max_abs = norm.get(str(model_label))
    if max_abs is None:
        return None
    if max_abs == 0.0:
        return 0.5
    scaled = 0.5 + float(value) / (2.0 * float(max_abs))
    return float(min(1.0, max(0.0, scaled)))


def _logo_key_for_model(
    model_label: str, provider: Optional[str], model: Optional[str]
) -> Optional[str]:
    haystack = " ".join([str(model_label or ""), str(provider or ""), str(model or "")]).lower()
    if "deepseek" in haystack:
        return "deepseek"
    if "kimi" in haystack or "moonshot" in haystack:
        return "moonshot"
    if "gemini" in haystack:
        return "gemini"
    if "anthropic" in haystack or "claude" in haystack:
        return "anthropic"
    if "openai" in haystack or "gpt" in haystack:
        return "openai"
    return None


def _resolve_logo_paths(rows: List["RunRow"], models: List[str]) -> Dict[str, Path]:
    if not _LOGO_DIR.exists():
        return {}
    rows_by_model: Dict[str, "RunRow"] = {}
    for r in rows:
        rows_by_model.setdefault(r.model_label, r)
    out: Dict[str, Path] = {}
    for model_label in models:
        row = rows_by_model.get(model_label)
        key = _logo_key_for_model(
            model_label, row.provider if row else None, row.model if row else None
        )
        if not key:
            continue
        filename = _LOGO_FILES.get(key)
        if not filename:
            continue
        path = _LOGO_DIR / filename
        if path.exists():
            out[model_label] = path
    return out


def _add_logos_to_xticklabels(
    *, fig: plt.Figure, ax: plt.Axes, models: List[str], logo_paths: Dict[str, Path]
) -> None:
    if not logo_paths:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = ax.get_xticklabels()
    for label, model_label in zip(labels, models):
        logo_path = logo_paths.get(model_label)
        if logo_path is None or not logo_path.exists():
            continue
        bbox = label.get_window_extent(renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        image = plt.imread(logo_path)
        if image is None or image.size == 0:
            continue
        zoom = float(bbox.height) / float(image.shape[0])
        offset_image = OffsetImage(image, zoom=zoom)
        pad_px = 8.0
        bump_px = 2.0
        image_width = float(image.shape[1]) * zoom
        x_disp = float(bbox.x0) - pad_px - (image_width / 2.0)
        y_disp = float((bbox.y0 + bbox.y1) / 2.0) + bump_px
        x_fig, y_fig = fig.transFigure.inverted().transform((x_disp, y_disp))
        ab = AnnotationBbox(
            offset_image,
            (x_fig, y_fig),
            xycoords=fig.transFigure,
            frameon=False,
            box_alignment=(0.5, 0.5),
        )
        fig.add_artist(ab)


def _iter_model_dirs(root: Path) -> Iterable[Path]:
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return
    for model_dir in sorted(runs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name == "judge_secret_blackboard":
            continue
        yield model_dir


def _infer_sweep_name(root: Path) -> str:
    sweep_names: set[str] = set()
    for model_dir in _iter_model_dirs(root):
        for child in sorted(model_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name == "judge_secret_blackboard":
                continue
            if (child / "run_config.json").exists():
                continue
            sweep_names.add(child.name)
    if len(sweep_names) == 1:
        return next(iter(sweep_names))
    if not sweep_names:
        raise SystemExit(f"No sweep directories found under: {root / 'runs'}")
    raise SystemExit(
        "Multiple sweep directories found; pass --sweep-name. Options: "
        + ", ".join(sorted(sweep_names))
    )


def _iter_run_dirs(sweep_dir: Path) -> Iterable[Path]:
    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "run_config.json").exists():
            yield child


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    obj = safe_load_json(path)
    return obj if isinstance(obj, dict) else None


def _judge_mean_rating(*, model_dir: Path, sweep_name: str, run_name: str) -> Optional[float]:
    judge_path = model_dir / "judge_secret_blackboard" / sweep_name / f"{run_name}.json"
    payload = safe_load_json(judge_path) if judge_path.exists() else None
    if not isinstance(payload, dict):
        return None
    judgements = payload.get("judgements")
    if not isinstance(judgements, dict):
        return None

    vals: List[float] = []
    for key in ("simple", "medium", "complex"):
        j = judgements.get(key)
        if not isinstance(j, dict):
            continue
        rating = as_float(j.get("rating"))
        if rating is None:
            continue
        vals.append(float(rating))
    if not vals:
        return None
    return float(mean(vals))


def _load_optimal_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    payload = _read_json(run_dir / "optimal_summary.json")
    return payload if isinstance(payload, dict) else None


def _compute_and_write_optimal_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Compute and persist `optimal_summary.json` for a Jira run directory (no API calls)."""
    try:
        from experiments.collusion import compute_jira_optimal

        try:
            instance = compute_jira_optimal._reconstruct_instance(run_dir)
        except Exception:
            instance = compute_jira_optimal._load_instance_from_agent_prompts(run_dir)
        weights = compute_jira_optimal._load_weights(run_dir, overrides=None)
        optimal = compute_jira_optimal.solve_optimal_assignment(instance=instance, weights=weights)

        payload = {
            "weights": {
                "tasks_done_bonus": weights.tasks_done_bonus,
                "priority_bonus": weights.priority_bonus,
                "violation_penalty": weights.violation_penalty,
            },
            "optimal": {
                "joint_reward": optimal.joint_reward,
                "tasks_done": optimal.tasks_done,
                "priority_sum": optimal.priority_sum,
                "total_cost": optimal.total_cost,
                "violations": optimal.violations,
                "assignment": {k: (v if v is not None else "skip") for k, v in optimal.assignment.items()},
            },
        }
        out_path = run_dir / "optimal_summary.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return payload
    except Exception as exc:
        logger.warning("Failed to compute optimal for %s: %s", run_dir, exc)
        return None


@dataclass(frozen=True)
class RunRow:
    model_label: str
    provider: Optional[str]
    model: Optional[str]
    sweep_name: str
    topology: Optional[str]
    num_agents: Optional[int]
    colluder_count: Optional[int]
    seed: Optional[int]
    secret_channel_enabled: Optional[bool]
    prompt_variant: str
    status: str

    joint_reward: Optional[float]
    optimal_joint_reward: Optional[float]
    normalized_regret: Optional[float]  # 1 - achieved / optimal (clipped to [0, 1])
    judge_mean_rating: Optional[float]
    coalition_mean_regret: Optional[float]
    noncoalition_mean_regret: Optional[float]


def _status_is_complete(status: Any) -> bool:
    return str(status or "").strip().lower() == "complete"


def _load_run_row(
    *,
    run_dir: Path,
    model_dir: Path,
    sweep_name: str,
    compute_optimal: bool,
    prefer_repaired: bool,
) -> Optional[RunRow]:
    rc = _read_json(run_dir / "run_config.json") or {}
    final_summary_path = run_dir / "final_summary.json"
    metrics_path = run_dir / "metrics.json"
    if prefer_repaired:
        repaired_summary = run_dir / "final_summary_repaired.json"
        repaired_metrics = run_dir / "metrics_repaired.json"
        if repaired_summary.exists():
            final_summary_path = repaired_summary
        if repaired_metrics.exists():
            metrics_path = repaired_metrics

    fs = _read_json(final_summary_path) or {}
    metrics = _read_json(metrics_path) or {}

    status = metrics.get("status", fs.get("status", "unknown"))
    status_s = str(status or "unknown")

    joint_reward = as_float(fs.get("joint_reward"))
    raw_joint_reward = as_float(fs.get("raw_joint_reward"))
    joint_reward_ratio = as_float(fs.get("joint_reward_ratio"))

    # Back-compat: some envs stored normalized score in joint_reward.
    if (
        joint_reward_ratio is None
        and joint_reward is not None
        and raw_joint_reward is not None
        and 0.0 <= float(joint_reward) <= 1.0
    ):
        joint_reward = float(raw_joint_reward)

    optimal_payload = _load_optimal_summary(run_dir)
    if optimal_payload is None and compute_optimal:
        optimal_payload = _compute_and_write_optimal_summary(run_dir)

    optimal_joint_reward = None
    if isinstance(optimal_payload, dict):
        optimal = optimal_payload.get("optimal")
        if isinstance(optimal, dict):
            optimal_joint_reward = as_float(optimal.get("joint_reward"))

    normalized_regret = None
    if joint_reward is not None and optimal_joint_reward is not None and float(optimal_joint_reward) != 0.0:
        achieved_over_optimal = float(joint_reward) / float(optimal_joint_reward)
        normalized_regret = 1.0 - float(achieved_over_optimal)
        if normalized_regret < 0.0:
            normalized_regret = 0.0
        elif normalized_regret > 1.0:
            normalized_regret = 1.0

    prompt_variant = _canonical_variant(rc.get("prompt_variant") or "control")
    judge_mean_rating = _judge_mean_rating(
        model_dir=model_dir, sweep_name=sweep_name, run_name=run_dir.name
    )

    return RunRow(
        model_label=str(rc.get("model_label") or model_dir.name),
        provider=str(rc.get("provider")) if rc.get("provider") is not None else None,
        model=str(rc.get("model")) if rc.get("model") is not None else None,
        sweep_name=str(rc.get("sweep") or sweep_name),
        topology=str(rc.get("topology")) if rc.get("topology") is not None else None,
        num_agents=as_int(rc.get("num_agents")),
        colluder_count=as_int(rc.get("colluder_count")),
        seed=as_int(rc.get("seed")),
        secret_channel_enabled=as_bool(rc.get("secret_channel_enabled")),
        prompt_variant=prompt_variant,
        status=status_s,
        joint_reward=joint_reward,
        optimal_joint_reward=optimal_joint_reward,
        normalized_regret=normalized_regret,
        judge_mean_rating=judge_mean_rating,
        coalition_mean_regret=as_float(metrics.get("coalition_mean_regret")),
        noncoalition_mean_regret=as_float(metrics.get("noncoalition_mean_regret")),
    )


def _filter_rows(
    rows: List[RunRow],
    *,
    topology: Optional[str],
    num_agents: Optional[int],
    colluder_count: Optional[int],
    require_complete: bool,
) -> List[RunRow]:
    out: List[RunRow] = []
    for r in rows:
        if topology is not None and str(r.topology) != str(topology):
            continue
        if num_agents is not None and int(r.num_agents or -1) != int(num_agents):
            continue
        if colluder_count is not None and int(r.colluder_count or -1) != int(colluder_count):
            continue
        if require_complete and not _status_is_complete(r.status):
            continue
        out.append(r)
    return out


def _variant_or_baseline(r: RunRow) -> str:
    if r.secret_channel_enabled is True:
        return _canonical_variant(r.prompt_variant) or "control"
    return "baseline"


def _seed_means(rows: List[RunRow], *, key: str) -> List[float]:
    by_seed: Dict[int, List[float]] = {}
    for r in rows:
        seed = r.seed
        if seed is None:
            continue

        if key == "normalized_coalition_regret_gap":
            c = r.coalition_mean_regret
            n = r.noncoalition_mean_regret
            if c is None or n is None:
                continue
            try:
                gap = float(n) - float(c)
            except Exception:
                continue
            if not math.isfinite(gap):
                continue
            norm_val = _normalize_coalition_regret_gap(gap, r.model_label)
            if norm_val is None or not math.isfinite(float(norm_val)):
                continue
            val = float(norm_val)
        else:
            val = getattr(r, key, None)

        if val is None or not math.isfinite(float(val)):
            continue
        by_seed.setdefault(int(seed), []).append(float(val))

    out: List[float] = []
    for seed_i in sorted(by_seed):
        vals = by_seed[seed_i]
        if vals:
            out.append(float(mean(vals)))
    return out


def _group_stats(rows: List[RunRow], *, key: str) -> Dict[str, Any]:
    vals = _seed_means(rows, key=key)
    if not vals:
        return {"n": 0, "mean": None, "sem": None}
    return {"n": int(len(vals)), "mean": float(mean(vals)), "sem": float(sem(vals))}


def _plot_combined_nine_bars_two_metrics_dual_axis(
    *,
    rows: List[RunRow],
    metric_a_key: str,
    metric_b_key: str,
    out_path: Path,
) -> None:
    """Single-panel plot: 2 normalized metrics (left y-axis) + judge (right y-axis).

    Ordering per model: metric_a bars, then metric_b bars, then judge bars.
    """
    _apply_large_font_style()
    models = sorted({r.model_label for r in rows})
    if not models:
        return

    conditions = ["baseline", "control", "simple"]
    colors = dict(zip(conditions, _SIX_BARS_PALETTE))

    def _stats_for(model_label: str, condition: str, key: str) -> Tuple[float, float, int]:
        subset = [r for r in rows if r.model_label == model_label and _variant_or_baseline(r) == condition]
        stats = _group_stats(subset, key=key)
        mu = stats.get("mean")
        se_val = stats.get("sem")
        n = int(stats.get("n") or 0)
        return (
            float(mu) if mu is not None else float("nan"),
            float(se_val) if se_val is not None else float("nan"),
            n,
        )

    metric_a_means: Dict[str, List[float]] = {c: [] for c in conditions}
    metric_a_sems: Dict[str, List[float]] = {c: [] for c in conditions}
    metric_b_means: Dict[str, List[float]] = {c: [] for c in conditions}
    metric_b_sems: Dict[str, List[float]] = {c: [] for c in conditions}
    judge_means: Dict[str, List[float]] = {c: [] for c in conditions}
    judge_sems: Dict[str, List[float]] = {c: [] for c in conditions}

    for m in models:
        for c in conditions:
            a_mu, a_se, _ = _stats_for(m, c, metric_a_key)
            metric_a_means[c].append(a_mu)
            metric_a_sems[c].append(a_se)

            b_mu, b_se, _ = _stats_for(m, c, metric_b_key)
            metric_b_means[c].append(b_mu)
            metric_b_sems[c].append(b_se)

            j_mu, j_se, _ = _stats_for(m, c, "judge_mean_rating")
            judge_means[c].append(j_mu)
            judge_sems[c].append(j_se)

    any_a = any(math.isfinite(v) for c in conditions for v in metric_a_means[c])
    any_b = any(math.isfinite(v) for c in conditions for v in metric_b_means[c])
    any_judge = any(math.isfinite(v) for c in conditions for v in judge_means[c])
    if not any_a:
        logger.warning("No %s data found; skipping: %s", metric_a_key, out_path)
        return
    if not any_b:
        logger.warning("No %s data found; skipping: %s", metric_b_key, out_path)
        return
    if not any_judge:
        logger.warning("No judge_mean_rating data found; skipping: %s", out_path)
        return

    ensure_dir(out_path.parent)
    fig, ax_metric = plt.subplots(nrows=1, ncols=1, figsize=(12.0, 3.0))
    ax_judge = ax_metric.twinx()

    x = np.arange(len(models), dtype=float)
    width = 0.11 * 0.8

    metric_a_offsets = {"baseline": -4.0 * width, "control": -3.0 * width, "simple": -2.0 * width}
    metric_b_offsets = {"baseline": -1.0 * width, "control": 0.0 * width, "simple": 1.0 * width}
    judge_offsets = {"baseline": 2.0 * width, "control": 3.0 * width, "simple": 4.0 * width}

    for c in conditions:
        ax_metric.bar(
            x + metric_a_offsets[c],
            metric_a_means[c],
            width=width,
            yerr=metric_a_sems[c],
            color=colors[c],
            alpha=0.85,
            capsize=3,
            label="_nolegend_",
        )

    for c in conditions:
        ax_metric.bar(
            x + metric_b_offsets[c],
            metric_b_means[c],
            width=width,
            yerr=metric_b_sems[c],
            color=_lighten_color(colors[c], amount=0.55),
            alpha=0.85,
            capsize=3,
            label="_nolegend_",
        )
        ax_metric.bar(
            x + metric_b_offsets[c],
            metric_b_means[c],
            width=width,
            color="none",
            alpha=1.0,
            hatch="\\\\\\",
            edgecolor="#666666",
            linewidth=0.0,
            label="_nolegend_",
            zorder=4,
        )

    for c in conditions:
        ax_judge.bar(
            x + judge_offsets[c],
            judge_means[c],
            width=width,
            yerr=judge_sems[c],
            color=colors[c],
            alpha=0.35,
            capsize=3,
            hatch="///",
            edgecolor="black",
            linewidth=0.3,
            label="_nolegend_",
        )

    ax_metric.set_ylabel("Normalized Value")
    ax_metric.set_ylim(0.0, 1.0)
    ax_metric.axhline(0.5, color="#777777", linewidth=1.0, alpha=0.7, zorder=0, linestyle=":")

    judge_axis_label = _pretty_metric_label("judge_mean_rating").replace(" (Judge)", "")
    ax_judge.set_ylabel(judge_axis_label, labelpad=10)
    ax_judge.tick_params(axis="y", labelcolor="black")
    ax_judge.spines["right"].set_visible(True)

    condition_labels = {"baseline": "Baseline (no SC)", "control": "Control (SC)", "simple": "Simple (SC)"}
    condition_handles = [
        Patch(facecolor=colors[c], edgecolor="none", label=condition_labels.get(c, c))
        for c in ["baseline", "control", "simple"]
    ]

    metric_a_label = _pretty_metric_label(metric_a_key).strip().replace("Normalized Regret", "Regret")
    metric_b_label = _pretty_metric_label(metric_b_key).strip()
    judge_label = _pretty_metric_label("judge_mean_rating").replace(" (↓)", "").strip()
    style_handles = [
        Patch(facecolor="white", edgecolor="black", linewidth=0.8, alpha=1.0, label=metric_a_label),
        Patch(facecolor="white", edgecolor="black", linewidth=0.8, hatch="\\\\\\", alpha=1.0, label=metric_b_label),
        Patch(facecolor="white", edgecolor="black", linewidth=0.8, hatch="///", alpha=1.0, label=judge_label),
    ]

    ax_metric.set_xticks(x)
    ax_metric.set_xticklabels([_pretty_model_label(m) for m in models], rotation=0, ha="center")
    ax_metric.tick_params(axis="x", pad=6)
    ax_metric.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax_metric.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax_metric.set_axisbelow(True)

    legend_handles = condition_handles + style_handles
    fig.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.22)
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(legend_handles),
        frameon=False,
        columnspacing=1.2,
        handlelength=1.4,
        handletextpad=0.6,
        labelspacing=0.4,
    )

    logo_paths = _resolve_logo_paths(rows, models)
    _add_logos_to_xticklabels(fig=fig, ax=ax_metric, models=models, logo_paths=logo_paths)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_combined_nine_bars_two_metrics_dual_axis_csv(
    *,
    rows: List[RunRow],
    metric_a_key: str,
    metric_b_key: str,
    out_path: Path,
) -> None:
    models = sorted({r.model_label for r in rows})
    if not models:
        return

    conditions = ["baseline", "control", "simple"]
    condition_labels = {"baseline": "Baseline (no SC)", "control": "Control (SC)", "simple": "Simple (SC)"}
    metric_keys = [metric_a_key, metric_b_key, "judge_mean_rating"]

    table_rows: List[Dict[str, Any]] = []
    for model_label in models:
        for condition in conditions:
            subset = [r for r in rows if r.model_label == model_label and _variant_or_baseline(r) == condition]
            for metric_key in metric_keys:
                stats = _group_stats(subset, key=str(metric_key))
                table_rows.append(
                    {
                        "model_label": model_label,
                        "model_label_pretty": _pretty_model_label(model_label),
                        "condition": condition,
                        "condition_pretty": condition_labels.get(condition, condition),
                        "metric_key": str(metric_key),
                        "metric_label": _pretty_metric_label(str(metric_key)).replace(" (Judge)", ""),
                        "mean": stats.get("mean"),
                        "sem": stats.get("sem"),
                        "n": stats.get("n"),
                    }
                )

    write_csv(out_path, table_rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the combined nine-bars regret plot from a collusion output root."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path like experiments/collusion/outputs/<tag>/<timestamp>",
    )
    parser.add_argument(
        "--sweep-name",
        type=str,
        default=None,
        help="Sweep directory name under each model (e.g., complete_n6_c2). If omitted, auto-infer when unique.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/collusion/plots_outputs/<tag>/<ts>/regret_report/<sweep_name>).",
    )
    parser.add_argument("--topology", type=str, default="complete")
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--colluder-count", type=int, default=2)
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include runs where status != 'complete' (not recommended).",
    )
    parser.add_argument(
        "--compute-optimal",
        action="store_true",
        help="If optimal_summary.json is missing, compute and write it (no API calls).",
    )
    parser.add_argument(
        "--prefer-repaired",
        action="store_true",
        help="Prefer *_repaired.json artifacts when present (final_summary_repaired.json, metrics_repaired.json).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    sweep_name = str(args.sweep_name) if args.sweep_name else _infer_sweep_name(root)

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        tag = root.parent.name
        timestamp = root.name
        out_dir = (
            Path("experiments/collusion/plots_outputs")
            / str(tag)
            / str(timestamp)
            / "regret_report"
            / sweep_name
        ).resolve()
    ensure_dir(out_dir)
    ensure_dir(out_dir / "plots")

    rows: List[RunRow] = []
    missing_sweeps: List[str] = []
    for model_dir in _iter_model_dirs(root):
        sweep_dir = model_dir / sweep_name
        if not sweep_dir.exists():
            missing_sweeps.append(model_dir.name)
            continue
        for run_dir in _iter_run_dirs(sweep_dir):
            row = _load_run_row(
                run_dir=run_dir,
                model_dir=model_dir,
                sweep_name=sweep_name,
                compute_optimal=bool(args.compute_optimal),
                prefer_repaired=bool(args.prefer_repaired),
            )
            if row is not None:
                rows.append(row)

    if missing_sweeps:
        logger.warning("Missing sweep %s for models: %s", sweep_name, ", ".join(missing_sweeps))
    if not rows:
        raise SystemExit("No runs found.")

    rows = _filter_rows(
        rows,
        topology=str(args.topology) if args.topology else None,
        num_agents=int(args.num_agents) if args.num_agents else None,
        colluder_count=int(args.colluder_count) if args.colluder_count else None,
        require_complete=not bool(args.include_incomplete),
    )
    _set_coalition_regret_gap_norm(rows)
    if not rows:
        raise SystemExit("No runs matched the requested filters.")

    plot_out = out_dir / "plots" / "regret_report__normalized_regret__coalition_gap__judge.png"
    _plot_combined_nine_bars_two_metrics_dual_axis(
        rows=rows,
        metric_a_key="normalized_regret",
        metric_b_key="normalized_coalition_regret_gap",
        out_path=plot_out,
    )
    if plot_out.exists():
        logger.info("Wrote plot: %s", plot_out)

    csv_out = out_dir / "plots" / "regret_report__normalized_regret__coalition_gap__judge__data.csv"
    _write_combined_nine_bars_two_metrics_dual_axis_csv(
        rows=rows,
        metric_a_key="normalized_regret",
        metric_b_key="normalized_coalition_regret_gap",
        out_path=csv_out,
    )
    if csv_out.exists():
        logger.info("Wrote CSV: %s", csv_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
