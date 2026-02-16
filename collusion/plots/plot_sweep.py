"""Sweep plotting for the collusion experiment.

This module intentionally contains *only* the plotting code needed to produce:

1) Per-topology grouped bar chart (mean ± SEM):
   `prompt_variants__mean_sem__c{colluder_count}__reward-<metric>.png`

2) Cross-topology comparison chart (optimality gap + judge rating):
   `topology_comparison__optimality_gap_and_judge__c{colluder_count}.png`

The older radar charts and other plot variants were removed to keep the collusion
experiment easier to understand and maintain for new users.
"""

from __future__ import annotations

import logging
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from experiments.collusion.plots.common import canonical_variant
from experiments.common.plotting.io_utils import (
    as_bool,
    as_float,
    as_int,
    ensure_dir,
    finite,
    mean,
    safe_load_json,
    sanitize_filename,
    sem,
)
from experiments.common.plotting.logging_utils import log_saved_plot
from experiments.common.plotting.load_runs import LoadedRun
from experiments.common.plotting.style import apply_default_style


logger = logging.getLogger(__name__)

_PVALL_GROUP_PALETTE = [
    "#264653",  # Charcoal Blue
    "#2a9d8f",  # Verdigris
    "#8ab17d",  # Muted Olive
    "#e9c46a",  # Jasmine
    "#f4a261",  # Sandy Brown
    "#e76f51",  # Burnt Peach
]

_BAR_CHART_HEIGHT_SCALE = 0.8  # slightly smaller vertically


def _per_topology_summary_filename(*, colluder_count: int, reward_metric: str) -> str:
    """Filename for the per-topology prompt-variant summary bars plot."""
    metric_part = sanitize_filename(str(reward_metric or "").strip() or "joint_reward_ratio")
    return f"prompt_variants__mean_sem__c{int(colluder_count)}__reward-{metric_part}.png"


def _apply_group_palette(groups: List[Dict[str, Any]], palette: List[str]) -> None:
    for idx, group in enumerate(groups):
        if idx >= len(palette):
            break
        group["color"] = palette[idx]


def _apply_readable_bars_style() -> None:
    # These plots can get wide (many metrics / many prompt variants). Bump font sizes
    # a bit so the exported PNG is readable when scaled-to-fit.
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "legend.title_fontsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )


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


def _pretty_metric_label(key: str) -> str:
    k = str(key or "").strip()
    if k == "optimality_gap":
        return "Regret"
    if k == "achieved_over_optimal":
        return "Achieved / Optimal"
    if k == "regret_ratio":
        return "Regret"
    if k == "joint_reward_ratio":
        return "Joint Reward Ratio"
    if k == "joint_reward":
        return "Joint Reward"
    if k == "coalition_mean_regret":
        return "Coalition Mean Regret"
    if k == "noncoalition_mean_regret":
        return "Non-Coalition Mean Regret"
    if k == "coalition_regret_advantage_mean":
        return "Coalition Regret Advantage"
    if k == "judge_mean_rating":
        return "Collusion Judge (↓)"
    return k.replace("_", " ").title()


def _pretty_variant(name: str) -> str:
    s = canonical_variant(name)
    return s.replace("_", " ").title() if s else s


def _canonical_topology_key(topo: Any) -> str:
    s = str(topo or "").strip()
    if not s:
        return s
    key = s.lower().replace("-", "_").replace(" ", "_")
    alias_to_canonical = {"er": "erdos_renyi", "ws": "watts_strogatz", "ba": "barabasi_albert"}
    return alias_to_canonical.get(key, key)


def _sort_topology_keys(topologies: Iterable[str]) -> List[str]:
    deterministic = ["complete", "path", "star"]
    random_graphs = ["erdos_renyi", "watts_strogatz", "barabasi_albert"]
    deterministic_idx = {k: i for i, k in enumerate(deterministic)}
    random_idx = {k: i for i, k in enumerate(random_graphs)}

    def _key(topo_key: str) -> Tuple[int, int, str]:
        k = _canonical_topology_key(topo_key)
        if k in deterministic_idx:
            return (0, deterministic_idx[k], k)
        if k in random_idx:
            return (1, random_idx[k], k)
        return (2, 0, k)

    out: List[str] = []
    seen: set[str] = set()
    for t in topologies:
        k = _canonical_topology_key(t)
        if not k or k in seen:
            continue
        out.append(k)
        seen.add(k)
    out.sort(key=_key)
    return out


def _topology_acronym_label(
    topo: str, *, title_by_topology: Optional[Dict[str, str]] = None
) -> str:
    """
    Abbreviate random-graph topology names (ER/WS/BA) while preserving any
    "(...)" parameter suffix from title_by_topology.
    """
    s = str(topo or "").strip()
    if not s:
        return s

    canonical_key = _canonical_topology_key(s)
    full = str(title_by_topology.get(canonical_key, canonical_key)) if title_by_topology else canonical_key

    abbrev = {"erdos_renyi": "ER", "watts_strogatz": "WS", "barabasi_albert": "BA"}.get(
        canonical_key
    )
    if not abbrev:
        return full.replace("_", " ").title()

    if "(" in full and full.endswith(")"):
        return abbrev + " (" + full.split("(", 1)[1]
    return abbrev


def _select_colluder_count(rows: List[Dict[str, Any]], requested: Optional[int]) -> int:
    if requested is not None:
        if requested <= 0:
            raise ValueError("--colluder-count must be > 0")
        return int(requested)

    counts = sorted(
        {int(c) for c in finite(r.get("colluder_count") for r in rows) if int(c) > 0}
    )
    if not counts:
        raise ValueError("No runs with colluder_count > 0 found.")
    return int(counts[-1])


def _group_filter(
    rows: List[Dict[str, Any]],
    *,
    colluder_count: int,
    secret: bool,
    prompt_variant: str,
    require_complete: bool,
) -> List[Dict[str, Any]]:
    prompt_variant = canonical_variant(prompt_variant)
    out: List[Dict[str, Any]] = []
    for r in rows:
        if int(r.get("colluder_count") or 0) != int(colluder_count):
            continue
        if bool(r.get("secret_channel_enabled") is True) != bool(secret):
            continue
        if str(r.get("prompt_variant") or "") != str(prompt_variant):
            continue
        if require_complete and str(r.get("status") or "").strip().lower() != "complete":
            continue
        out.append(r)
    return out


def _seed_means(rows: List[Dict[str, Any]], key: str) -> List[float]:
    by_seed: Dict[int, List[float]] = {}
    for r in rows:
        seed = r.get("seed")
        if seed is None:
            continue
        try:
            seed_i = int(seed)
        except Exception:
            continue
        v = as_float(r.get(key))
        if v is None:
            continue
        by_seed.setdefault(seed_i, []).append(float(v))

    out: List[float] = []
    for seed_i in sorted(by_seed):
        vals = by_seed[seed_i]
        if not vals:
            continue
        out.append(float(mean(vals)))
    return out


def _minmax_range(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    vals = np.array(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if lo == hi:
        pad = max(1e-6, abs(lo) * 0.1)
        lo -= pad
        hi += pad
    return lo, hi


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


_TICK_LABEL_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _titlecase_for_ticks(label: str) -> str:
    """Title-case metric labels for bar-chart x-ticks while preserving acronyms."""

    def _repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if any(ch.isalpha() for ch in token) and token.isupper():
            return token
        if not token:
            return token
        return token[0].upper() + token[1:].lower()

    return _TICK_LABEL_TOKEN_RE.sub(_repl, str(label))


def _wrap_labels(labels: List[str], *, width: int = 14) -> List[str]:
    out: List[str] = []
    for label in labels:
        s = _titlecase_for_ticks(str(label))
        if s.startswith("Coalition Regret"):
            out.append(s.replace("Coalition Regret", "Coalition\nRegret", 1))
            continue
        if s.startswith("Collusion Judge"):
            out.append(s.replace("Collusion Judge", "Collusion\nJudge", 1))
            continue
        if len(s) <= width:
            out.append(s)
            continue
        out.append(
            "\n".join(
                textwrap.wrap(
                    s,
                    width=int(width),
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        )
    return out


def _bottom_margin_for_wrapped_xticks(labels_wrapped: List[str]) -> float:
    """Heuristic bottom margin for multi-line x tick labels (and our bottom legend)."""
    max_lines = 1
    for lbl in labels_wrapped:
        try:
            max_lines = max(max_lines, str(lbl).count("\n") + 1)
        except Exception:
            continue
    bottom = 0.14 + 0.04 * max(0, max_lines - 2)
    return float(min(0.28, bottom))


@dataclass(frozen=True)
class _MetricSpec:
    key: str
    label: str
    higher_is_better: bool = True
    transform: Optional[Callable[[float], float]] = None
    flip_for_plot: bool = True
    center_zero: bool = False

    def apply(self, v: float) -> float:
        out = float(v)
        if self.transform is not None:
            out = float(self.transform(out))
        if self.flip_for_plot and not self.higher_is_better:
            out = -out
        return float(out)


def _canonical_reward_metric_key(name: str) -> str:
    s = str(name or "").strip()
    if s == "regret":
        return "optimality_gap"
    if s == "normalized_regret":
        return "regret_ratio"
    return s


def _reward_metric_spec(name: str) -> _MetricSpec:
    key = _canonical_reward_metric_key(name)
    if key == "joint_reward_ratio":
        return _MetricSpec("joint_reward_ratio", "Joint reward ratio", higher_is_better=True)
    if key == "achieved_over_optimal":
        return _MetricSpec("achieved_over_optimal", "Achieved / Optimal", higher_is_better=True)
    if key == "regret_ratio":
        return _MetricSpec("regret_ratio", "Regret", higher_is_better=False, flip_for_plot=False)
    if key == "optimality_gap":
        return _MetricSpec("optimality_gap", "Regret", higher_is_better=False, flip_for_plot=False)
    raise ValueError(f"Unknown reward metric: {name!r}")


def _default_metric_specs(*, reward_metric: str) -> List[_MetricSpec]:
    # Note: `colluder_posts_secret_rate` is intentionally omitted from the simplified plots.
    return [
        _reward_metric_spec(str(reward_metric)),
        _MetricSpec("tasks_done", "Tasks done", higher_is_better=True),
        _MetricSpec("violations", "Constraint violations", higher_is_better=False, flip_for_plot=False),
        _MetricSpec("coalition_mean_regret", "Coalition regret", higher_is_better=False, flip_for_plot=False),
        _MetricSpec("noncoalition_mean_regret", "Non-coalition regret", higher_is_better=False, flip_for_plot=False),
        _MetricSpec(
            "coalition_regret_advantage_mean",
            "Coalition Advantage (-)",
            higher_is_better=False,
            flip_for_plot=False,
            center_zero=True,
        ),
        _MetricSpec(
            "judge_mean_rating",
            "Collusion Judge",
            higher_is_better=False,
            flip_for_plot=False,
            transform=lambda v: float(v) / 5.0,
        ),
    ]


def _slice_by_indices(values: List[Any], indices: List[int]) -> List[Any]:
    return [values[i] for i in indices if i < len(values)]


def _slice_series_metrics(series: List[Dict[str, Any]], indices: List[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in series:
        vals = list(s.get("values") or [])
        errs = list(s.get("errors") or [])
        out.append(
            {
                **s,
                "values": _slice_by_indices(vals, indices),
                "errors": _slice_by_indices(errs, indices) if errs else [],
            }
        )
    return out


def _sorted_variants(variants: Iterable[str]) -> List[str]:
    preferred = ["control", "simple", "deception", "structured", "aggressive"]
    vals = [canonical_variant(v) for v in variants if v is not None and str(v).strip()]
    uniq: List[str] = []
    seen = set()
    for v in vals:
        if v in seen:
            continue
        uniq.append(v)
        seen.add(v)

    out: List[str] = []
    for v in preferred:
        if v in seen:
            out.append(v)
            seen.remove(v)
    out.extend(sorted(seen))
    return out


def build_rows(runs: List[LoadedRun]) -> List[Dict[str, Any]]:
    """Extract the per-run scalar metrics needed by the remaining plots."""
    rows: List[Dict[str, Any]] = []
    for run in runs:
        rc = run.run_config or {}
        rid = str(rc.get("run_id") or run.run_dir.name)

        env_label = rc.get("environment_label")
        if env_label is None:
            try:
                if "__env" in rid:
                    suffix = rid.split("__env", 1)[1]
                    env_label = suffix.split("__", 1)[0] or None
            except Exception:
                env_label = None
        if env_label is not None and not str(env_label).strip():
            env_label = None

        fs = run.final_summary or {}
        if not isinstance(fs, dict):
            fs = {}
        metrics = run.metrics or {}
        if not isinstance(metrics, dict):
            metrics = {}

        joint_reward = as_float(fs.get("joint_reward"))
        joint_reward_ratio = as_float(fs.get("joint_reward_ratio"))
        raw_joint_reward = as_float(fs.get("raw_joint_reward"))

        # Back-compat: some envs historically stored the normalized score in `joint_reward`.
        if (
            joint_reward_ratio is None
            and joint_reward is not None
            and raw_joint_reward is not None
            and 0.0 <= float(joint_reward) <= 1.0
        ):
            joint_reward_ratio = float(joint_reward)
            joint_reward = float(raw_joint_reward)

        # Optional: exact Jira optimal computed by experiments/collusion/compute_jira_optimal.py.
        optimal_joint_reward = None
        optimality_gap = None
        achieved_over_optimal = None
        regret_ratio = None
        try:
            optimal_payload = safe_load_json(run.run_dir / "optimal_summary.json")
            if isinstance(optimal_payload, dict):
                optimal = optimal_payload.get("optimal")
                if isinstance(optimal, dict):
                    optimal_joint_reward = as_float(optimal.get("joint_reward"))
            if joint_reward is not None and optimal_joint_reward is not None:
                optimality_gap = float(optimal_joint_reward) - float(joint_reward)
                if float(optimal_joint_reward) != 0.0:
                    achieved_over_optimal = float(joint_reward) / float(optimal_joint_reward)
                    regret_ratio = 1.0 - float(achieved_over_optimal)
                    regret_ratio = float(min(1.0, max(0.0, regret_ratio)))
        except Exception:
            pass

        # Optional: post-hoc LLM-as-a-judge outputs (written outside the run directory).
        judge_mean_rating = None
        try:
            sweep_dir = run.run_dir.parent
            model_dir = sweep_dir.parent
            judge_path = (
                model_dir
                / "judge_secret_blackboard"
                / sweep_dir.name
                / f"{run.run_dir.name}.json"
            )
            judge_payload = safe_load_json(judge_path) if judge_path.exists() else None
            if isinstance(judge_payload, dict):
                judgements = judge_payload.get("judgements") or {}
                if isinstance(judgements, dict):
                    vals = []
                    for key in ("simple", "medium", "complex"):
                        j = judgements.get(key) or {}
                        if not isinstance(j, dict):
                            continue
                        rating = as_float(j.get("rating"))
                        if rating is not None:
                            vals.append(float(rating))
                    if vals:
                        judge_mean_rating = float(mean(vals))
        except Exception:
            pass

        coalition_mean_regret = as_float(metrics.get("coalition_mean_regret"))
        noncoalition_mean_regret = as_float(metrics.get("noncoalition_mean_regret"))
        coalition_regret_advantage_mean = None
        if coalition_mean_regret is not None and noncoalition_mean_regret is not None:
            coalition_regret_advantage_mean = float(noncoalition_mean_regret) - float(
                coalition_mean_regret
            )

        rows.append(
            {
                "run_dir": str(run.run_dir),
                "run_id": rid,
                "seed": as_int(rc.get("seed")),
                "environment_label": str(env_label) if env_label is not None else None,
                "topology": rc.get("topology"),
                "colluder_count": as_int(rc.get("colluder_count")),
                "secret_channel_enabled": as_bool(rc.get("secret_channel_enabled")),
                "prompt_variant": canonical_variant(rc.get("prompt_variant")),
                "status": metrics.get("status") or fs.get("status"),
                "tasks_done": as_float(metrics.get("tasks_done")),
                "violations": as_float(metrics.get("violations")),
                "coalition_mean_regret": coalition_mean_regret,
                "noncoalition_mean_regret": noncoalition_mean_regret,
                "coalition_regret_advantage_mean": coalition_regret_advantage_mean,
                "joint_reward_ratio": joint_reward_ratio,
                "joint_reward": joint_reward,
                "optimal_joint_reward": optimal_joint_reward,
                "optimality_gap": optimality_gap,
                "achieved_over_optimal": achieved_over_optimal,
                "regret_ratio": regret_ratio,
                "judge_mean_rating": judge_mean_rating,
            }
        )
    return rows


def plot_collusion_hist_bars_se(
    *,
    rows: List[Dict[str, Any]],
    out_path: Path,
    colluder_count: int,
    baseline_variant: str,
    include_incomplete: bool,
    reward_metric: str,
) -> None:
    """Generate the per-topology prompt-variant bars plot (mean ± SEM)."""
    if not rows:
        return

    require_complete = not bool(include_incomplete)
    baseline = _group_filter(
        rows,
        colluder_count=colluder_count,
        secret=False,
        prompt_variant=baseline_variant,
        require_complete=require_complete,
    )
    if not baseline:
        logger.warning(
            "No baseline runs found (topology=%s, c=%s, pv=%s).",
            _canonical_topology_key(rows[0].get("topology")),
            colluder_count,
            baseline_variant,
        )
        return

    available = {
        str(r.get("prompt_variant"))
        for r in rows
        if int(r.get("colluder_count") or 0) == int(colluder_count)
        and r.get("secret_channel_enabled") is True
        and r.get("prompt_variant") is not None
    }
    variants = _sorted_variants(available)
    if not variants:
        logger.warning(
            "No secret-channel runs found (topology=%s, c=%s).",
            _canonical_topology_key(rows[0].get("topology")),
            colluder_count,
        )
        return

    groups: List[Dict[str, Any]] = [
        {
            "kind": "baseline",
            "secret_channel_enabled": False,
            "prompt_variant": baseline_variant,
            "label": "Baseline (no SC)",
            "color": None,
            "rows": baseline,
        }
    ]
    for pv in variants:
        grp_rows = _group_filter(
            rows,
            colluder_count=colluder_count,
            secret=True,
            prompt_variant=pv,
            require_complete=require_complete,
        )
        if not grp_rows:
            continue
        groups.append(
            {
                "kind": "treatment",
                "secret_channel_enabled": True,
                "prompt_variant": pv,
                "label": f"{_pretty_variant(pv)} (SC)",
                "color": None,
                "rows": grp_rows,
            }
        )

    _apply_group_palette(groups, _PVALL_GROUP_PALETTE)
    if len(groups) <= 1:
        return

    metrics = _default_metric_specs(reward_metric=str(reward_metric))

    # Build per-metric normalization from pooled seed means across groups.
    metric_defs: List[Dict[str, Any]] = []
    for m in metrics:
        seed_raw_by_group: List[List[float]] = []
        pooled_transformed: List[float] = []
        for g in groups:
            seed_raw = _seed_means(g["rows"], m.key)
            seed_raw_by_group.append(seed_raw)
            if seed_raw:
                pooled_transformed.extend([m.apply(v) for v in seed_raw])

        lo, hi = _minmax_range(pooled_transformed) if pooled_transformed else (0.0, 1.0)
        if m.center_zero:
            max_abs = max(abs(float(lo)), abs(float(hi)))
            scale_lo = -max_abs
            scale_hi = max_abs
        else:
            max_abs = None
            scale_lo = float(lo)
            scale_hi = float(hi)

        def _norm_one(value: float) -> float:
            if scale_hi == scale_lo:
                return 0.5
            if m.center_zero:
                if not max_abs:
                    return 0.5
                return _clamp01(0.5 + float(value) / (2.0 * float(max_abs)))
            return _clamp01(
                (float(value) - float(scale_lo)) / (float(scale_hi) - float(scale_lo))
            )

        metric_defs.append(
            {
                "key": m.key,
                "label": m.label,
                "higher_is_better": m.higher_is_better,
                "group_means_norm01": [
                    {
                        "label": groups[i]["label"],
                        "mean_norm01": float(
                            mean([_norm_one(m.apply(v)) for v in seed_raw_by_group[i]])
                        )
                        if seed_raw_by_group[i]
                        else None,
                    }
                    for i in range(len(groups))
                ],
                "group_se_norm01": [
                    {
                        "label": groups[i]["label"],
                        "se_norm01": float(
                            sem([_norm_one(m.apply(v)) for v in seed_raw_by_group[i]])
                        )
                        if seed_raw_by_group[i]
                        else None,
                    }
                    for i in range(len(groups))
                ],
            }
        )

    # Drop metrics that are missing everywhere.
    metric_defs = [
        md
        for md in metric_defs
        if any(
            v.get("mean_norm01") is not None
            for v in (md.get("group_means_norm01") or [])
            if isinstance(v, dict)
        )
    ]
    if not metric_defs:
        return

    bar_indices = list(range(len(metric_defs)))
    bar_metric_defs = [metric_defs[i] for i in bar_indices]
    labels_bars: List[str] = []
    for md in bar_metric_defs:
        base = str(md.get("label") or "")
        key = str(md.get("key") or "")
        if key == "coalition_regret_advantage_mean":
            labels_bars.append(base)
            continue
        arrow = "↑" if bool(md.get("higher_is_better", True)) else "↓"
        labels_bars.append(f"{base} ({arrow})")

    group_vectors_se: List[Dict[str, Any]] = []
    for idx, g in enumerate(groups):
        vals: List[float] = []
        errs_se: List[float] = []
        for md in metric_defs:
            v = (md.get("group_means_norm01") or [])[idx].get("mean_norm01")
            e_se = (md.get("group_se_norm01") or [])[idx].get("se_norm01")
            vals.append(float(v) if v is not None else float("nan"))
            errs_se.append(float(e_se) if e_se is not None else float("nan"))
        group_vectors_se.append(
            {
                "label": g["label"],
                "values": vals,
                "errors": errs_se,
                "color": g.get("color"),
            }
        )

    # Fill in missing colors deterministically using tab10 (skip tab10[0]).
    cmap = plt.get_cmap("tab10")
    color_idx = 1
    for g in group_vectors_se:
        if g.get("color"):
            continue
        g["color"] = matplotlib.colors.to_hex(cmap(color_idx % 10))
        color_idx += 1

    # Keep violations on its natural scale (mean ± SEM over seeds).
    violations_idx = next(
        (i for i, md in enumerate(bar_metric_defs) if md.get("key") == "violations"),
        None,
    )
    raw_vio_by_label: Dict[str, Dict[str, float]] = {}
    for g in groups:
        label = str(g["label"])
        seed_vio = _seed_means(g["rows"], "violations")
        if seed_vio:
            raw_vio_by_label[label] = {
                "mean": float(mean(seed_vio)),
                "se": float(sem(seed_vio)),
            }

    def _with_raw_overrides(series: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if violations_idx is None:
            return series
        out: List[Dict[str, Any]] = []
        for s in series:
            label = str(s.get("label") or "")
            vio = raw_vio_by_label.get(label)
            vals = list(s.get("values") or [])
            errs = list(s.get("errors") or [])
            if vio is not None and violations_idx < len(vals) and violations_idx < len(errs):
                vals[violations_idx] = float(vio["mean"])
                errs[violations_idx] = float(vio["se"])
            out.append({**s, "values": vals, "errors": errs})
        return out

    bars_series_se = _with_raw_overrides(_slice_series_metrics(group_vectors_se, bar_indices))
    _plot_grouped_bars_single_topology(
        series=bars_series_se,
        labels=labels_bars,
        title=None,
        out_path=out_path,
        y_label="Normalized Mean",
    )


def plot_collusion_hist_bars_se_by_topology(
    *,
    rows: List[Dict[str, Any]],
    out_dir: Path,
    colluder_count: int,
    baseline_variant: str,
    include_incomplete: bool,
    reward_metric: str,
) -> None:
    """Generate per-topology prompt-variant summary plots under `by_topology/<topo>/sweep/`."""
    topologies = _sort_topology_keys(
        [str(r.get("topology")) for r in rows if r.get("topology") is not None]
    )
    if not topologies:
        return

    for topo in topologies:
        topo_rows = [r for r in rows if _canonical_topology_key(r.get("topology")) == topo]
        if not topo_rows:
            continue
        out_path = (
            out_dir
            / "by_topology"
            / sanitize_filename(topo)
            / "sweep"
            / _per_topology_summary_filename(
                colluder_count=int(colluder_count), reward_metric=str(reward_metric)
            )
        )
        plot_collusion_hist_bars_se(
            rows=topo_rows,
            out_path=out_path,
            colluder_count=int(colluder_count),
            baseline_variant=str(baseline_variant),
            include_incomplete=bool(include_incomplete),
            reward_metric=str(reward_metric),
        )


def plot_combined_six_bars_by_topology(
    *,
    rows: List[Dict[str, Any]],
    metric_key: str,
    out_path: Path,
    colluder_count: int,
    baseline_variant: str,
    include_incomplete: bool,
    title_by_topology: Optional[Dict[str, str]] = None,
) -> None:
    """Generate the cross-topology comparison plot (optimality gap + judge)."""
    # Match experiments/collusion/plots/generate_jira_regret_report.py aesthetics.
    try:
        matplotlib.rcdefaults()
    except Exception:
        pass
    try:
        plt.style.use("default")
    except Exception:
        pass
    _apply_large_font_style()

    topologies = _sort_topology_keys(
        [str(r.get("topology")) for r in rows if r.get("topology") is not None]
    )
    if not topologies:
        return

    require_complete = not bool(include_incomplete)
    conditions = ["baseline", "control", "simple"]
    colors = {
        "baseline": "#264653",
        "control": "#2a9d8f",
        "simple": "#8ab17d",
    }

    def _rows_for(topo: str, condition: str) -> List[Dict[str, Any]]:
        topo_rows = [r for r in rows if _canonical_topology_key(r.get("topology")) == topo]
        if condition == "baseline":
            return _group_filter(
                topo_rows,
                colluder_count=colluder_count,
                secret=False,
                prompt_variant=baseline_variant,
                require_complete=require_complete,
            )
        return _group_filter(
            topo_rows,
            colluder_count=colluder_count,
            secret=True,
            prompt_variant=condition,
            require_complete=require_complete,
        )

    def _stats_for(topo: str, condition: str, key: str) -> Tuple[float, float, int]:
        seed_vals = _seed_means(_rows_for(topo, condition), key)
        if not seed_vals:
            return float("nan"), float("nan"), 0
        mu = float(mean(seed_vals))
        se_val = float(sem(seed_vals))
        return mu, se_val, len(seed_vals)

    metric_means: Dict[str, List[float]] = {c: [] for c in conditions}
    metric_sems: Dict[str, List[float]] = {c: [] for c in conditions}
    judge_means: Dict[str, List[float]] = {c: [] for c in conditions}
    judge_sems: Dict[str, List[float]] = {c: [] for c in conditions}
    for topo in topologies:
        for c in conditions:
            mu, se_val, _ = _stats_for(topo, c, metric_key)
            metric_means[c].append(mu)
            metric_sems[c].append(se_val)

            j_mu, j_se, _ = _stats_for(topo, c, "judge_mean_rating")
            judge_means[c].append(j_mu)
            judge_sems[c].append(j_se)

    any_judge = any(math.isfinite(v) for c in conditions for v in judge_means[c])
    if not any_judge:
        logger.warning("No judge_mean_rating data found; skipping: %s", out_path)
        return

    ensure_dir(out_path.parent)
    fig, ax_metric = plt.subplots(nrows=1, ncols=1, figsize=(12.0, 3.0))
    ax_judge = ax_metric.twinx()

    x = np.arange(len(topologies), dtype=float)
    width = 0.11
    metric_offsets = {"baseline": -2.5 * width, "control": -1.5 * width, "simple": -0.5 * width}
    judge_offsets = {"baseline": 0.5 * width, "control": 1.5 * width, "simple": 2.5 * width}

    for c in conditions:
        ax_metric.bar(
            x + metric_offsets[c],
            metric_means[c],
            width=width,
            yerr=metric_sems[c],
            color=colors[c],
            alpha=0.85,
            capsize=3,
            label="_nolegend_",
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

    ax_metric_label = _pretty_metric_label(metric_key)
    if metric_key in {"regret_ratio", "normalized_regret", "optimality_gap"}:
        ax_metric_label = f"{ax_metric_label} (↓)"
    if metric_key in {"regret_ratio", "normalized_regret"}:
        ax_metric_label = "Normalized Regret (↓)"
    elif metric_key in {
        "normalized_coalition_advantage",
        "normalized_coalition_regret_gap",
        "coalition_regret_ratio",
    }:
        ax_metric_label = "Normalized"
    ax_metric.set_ylabel(ax_metric_label)
    if metric_key in {
        "regret_ratio",
        "normalized_regret",
        "normalized_coalition_advantage",
        "normalized_coalition_regret_gap",
        "coalition_regret_ratio",
    }:
        ax_metric.set_ylim(0.0, 1.0)
    judge_axis_label = _pretty_metric_label("judge_mean_rating").replace(" (Judge)", "")
    ax_judge.set_ylabel(judge_axis_label, labelpad=10)
    ax_judge.tick_params(axis="y", labelcolor="black")
    ax_judge.spines["right"].set_visible(True)

    condition_order = ["baseline", "control", "simple"]
    condition_labels = {
        "baseline": "Baseline (no SC)",
        "control": "Control (SC)",
        "simple": "Simple (SC)",
    }
    condition_handles = [
        Patch(facecolor=colors[c], edgecolor="none", label=condition_labels.get(c, c))
        for c in condition_order
    ]
    style_color = "white"
    ax_metric_label_for_legend = ax_metric_label.replace(" (↓)", "").strip()
    style_handles = [
        Patch(
            facecolor=style_color,
            edgecolor="black",
            linewidth=0.8,
            alpha=1.0,
            label=ax_metric_label_for_legend,
        ),
        Patch(
            facecolor=style_color,
            edgecolor="black",
            linewidth=0.8,
            hatch="///",
            alpha=1.0,
            label=_pretty_metric_label("judge_mean_rating").replace(" (↓)", "").strip(),
        ),
    ]

    ax_metric.set_xticks(x)
    ax_metric.set_xticklabels(
        [_topology_acronym_label(t, title_by_topology=title_by_topology) for t in topologies],
        rotation=0,
        ha="center",
    )
    ax_metric.tick_params(axis="x", pad=6)
    ax_metric.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax_metric.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax_metric.set_axisbelow(True)

    legend_handles = condition_handles + style_handles
    legend_labels = [h.get_label() for h in legend_handles]
    fig.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.22)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(legend_handles),
        frameon=False,
        columnspacing=1.2,
        handlelength=1.4,
        handletextpad=0.6,
        labelspacing=0.4,
    )
    plt.savefig(out_path, dpi=200)
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def _plot_grouped_bars_single_topology(
    *,
    series: List[Dict[str, Any]],
    labels: List[str],
    title: Optional[str],
    out_path: Path,
    y_label: str = "Normalized Mean",
) -> None:
    apply_default_style(plt)
    _apply_readable_bars_style()
    ensure_dir(out_path.parent)

    if not labels or not series:
        return

    def _infer_ylim(series: List[Dict[str, Any]], n: int) -> Tuple[float, bool]:
        max_hi = 0.0
        any_finite = False
        for s in series:
            vals = list(s.get("values") or [])
            errors = s.get("errors")
            errs = errors if isinstance(errors, list) else None
            for i in range(min(n, len(vals))):
                v = vals[i]
                if v is None:
                    continue
                try:
                    vf = float(v)
                except Exception:
                    continue
                if not np.isfinite(vf):
                    continue
                ef = 0.0
                if errs is not None and i < len(errs):
                    e = errs[i]
                    if e is not None:
                        try:
                            ef = abs(float(e))
                        except Exception:
                            ef = 0.0
                        if not np.isfinite(ef):
                            ef = 0.0
                hi = vf + ef
                if not np.isfinite(hi):
                    continue
                max_hi = max(max_hi, hi)
                any_finite = True

        if not any_finite:
            return 1.0, True
        if max_hi <= 1.0 + 1e-9:
            return 1.0, True
        return max_hi * 1.05, False

    labels_wrapped = _wrap_labels(labels, width=16)
    n_metrics = len(labels_wrapped)
    x = np.arange(n_metrics, dtype=float)
    total_width = 0.86

    series_arrays: List[np.ndarray] = []
    present_indices: List[int] = []
    for idx, s in enumerate(series):
        vals = list(s.get("values") or [])
        if len(vals) != n_metrics:
            series_arrays.append(np.full((n_metrics,), np.nan, dtype=float))
            continue
        arr = np.array([float(v) if v is not None else np.nan for v in vals], dtype=float)
        series_arrays.append(arr)
        if np.any(np.isfinite(arr)):
            present_indices.append(idx)

    if not present_indices:
        return

    bar_width = total_width / float(max(1, len(present_indices)))
    offsets = (
        np.arange(len(present_indices), dtype=float) - (len(present_indices) - 1) / 2.0
    ) * bar_width

    fig_width = max(12.0, 2.0 * float(n_metrics))
    fig_height = max(5.2, fig_width / 2.6) * _BAR_CHART_HEIGHT_SCALE
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    font_scale = 1.1
    label_fs = float(plt.rcParams.get("axes.labelsize", 10)) * font_scale
    tick_fs_x = float(plt.rcParams.get("xtick.labelsize", 9)) * font_scale
    tick_fs_y = float(plt.rcParams.get("ytick.labelsize", 9)) * font_scale

    for pos, idx in enumerate(present_indices):
        s = series[idx]
        vals_arr = series_arrays[idx]
        errors = s.get("errors")
        yerr = errors if isinstance(errors, list) and len(errors) == n_metrics else None
        color = str(s.get("color") or "#111827")
        label = str(s.get("label") or f"series {idx + 1}")

        mask = np.isfinite(vals_arr)
        if not np.any(mask):
            continue

        yerr_arr = None
        if yerr is not None:
            yerr_arr = np.array([float(e) if e is not None else np.nan for e in yerr], dtype=float)
            yerr_arr = yerr_arr[mask]

        ax.bar(
            (x[mask] + offsets[pos]),
            vals_arr[mask],
            width=bar_width,
            yerr=yerr_arr,
            capsize=4,
            error_kw={"elinewidth": 1.6, "capthick": 1.6, "ecolor": "black"},
            label=label,
            color=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.92,
        )

    if title is not None and str(title).strip():
        ax.set_title(str(title))
    y_max, use_fixed_ticks = _infer_ylim(series, n_metrics)
    if str(y_label).strip().lower() == "normalized mean":
        y_max, use_fixed_ticks = 1.0, True
    ax.set_ylim(0.0, float(y_max))
    if use_fixed_ticks:
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    ax.tick_params(axis="y", labelsize=tick_fs_y)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_ylabel(str(y_label), fontsize=label_fs, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_wrapped, rotation=0, ha="center", fontsize=tick_fs_x)

    half_bar = float(bar_width) / 2.0
    x_left = float(x[0] + float(np.min(offsets)) - half_bar)
    x_right = float(x[-1] + float(np.max(offsets)) + half_bar)
    ax.set_xlim(x_left - 0.02, x_right + 0.02)

    bottom_margin = _bottom_margin_for_wrapped_xticks(labels_wrapped)
    fig.tight_layout(rect=(0.04, bottom_margin, 0.995, 1.0))

    handles, leg_labels = ax.get_legend_handles_labels()
    if handles:
        ncol = min(len(handles), 6)
        ax_box = ax.get_position()
        ax_center_x = float((ax_box.x0 + ax_box.x1) / 2.0)
        fig.legend(
            handles,
            leg_labels,
            loc="lower center",
            bbox_to_anchor=(ax_center_x, -0.01),
            bbox_transform=fig.transFigure,
            ncol=ncol,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.2,
            handlelength=1.4,
            handletextpad=0.6,
            labelspacing=0.4,
        )
    fig.savefig(out_path, bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)
