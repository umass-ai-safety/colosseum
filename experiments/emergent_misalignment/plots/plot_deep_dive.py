from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from experiments.common.plotting.io_utils import ensure_dir
from experiments.common.plotting.logging_utils import log_saved_plot
from experiments.common.plotting.style import apply_default_style


logger = logging.getLogger(__name__)


def build_graph_from_blackboards(blackboards: Optional[List[Any]]) -> nx.Graph:
    """
    Reconstruct a communication graph from consolidated blackboards.
    - Pairwise channels: participants=[a,b] => edge(a,b)
    - Group channel: participants=[...] => add clique over participants
    """
    g = nx.Graph()
    if not blackboards:
        return g
    for bb in blackboards:
        if not isinstance(bb, dict):
            continue
        participants = bb.get("participants")
        if not isinstance(participants, list) or not participants:
            continue
        parts = [str(p) for p in participants]
        for p in parts:
            g.add_node(p)
        if len(parts) == 1:
            continue
        # Add clique edges within channel participants.
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                g.add_edge(parts[i], parts[j])
    return g


def _layout_for_topology(g: nx.Graph, topology: Optional[str]) -> Dict[str, np.ndarray]:
    if g.number_of_nodes() == 0:
        return {}
    topo = (topology or "").lower()

    if topo == "path":
        # Place nodes in path order if it looks like a path.
        deg1 = [n for n in g.nodes if g.degree[n] == 1]
        if len(deg1) >= 2:
            # Choose the farthest pair among endpoints.
            best = None
            best_len = -1
            for i in range(len(deg1)):
                for j in range(i + 1, len(deg1)):
                    try:
                        p = nx.shortest_path(g, deg1[i], deg1[j])
                    except Exception:
                        continue
                    if len(p) > best_len:
                        best = p
                        best_len = len(p)
            if best and len(best) == g.number_of_nodes():
                return {n: np.array([float(i), 0.0]) for i, n in enumerate(best)}

    if topo == "star":
        # Center at origin, leaves on circle.
        center = max(g.nodes, key=lambda n: g.degree[n])
        leaves = [n for n in g.nodes if n != center]
        pos: Dict[str, np.ndarray] = {center: np.array([0.0, 0.0])}
        if leaves:
            angles = np.linspace(0, 2 * np.pi, num=len(leaves), endpoint=False)
            for n, ang in zip(leaves, angles):
                pos[n] = np.array([np.cos(ang), np.sin(ang)])
        return pos

    if topo == "complete":
        return nx.circular_layout(g)

    # Fallback: spring layout (stable seed).
    return nx.spring_layout(g, seed=0)


def plot_belief_network(
    *,
    graph: nx.Graph,
    topology: Optional[str],
    agent_rows: List[Dict[str, Any]],
    run_title: str,
    out_path: Path,
    target_agent: Optional[str] = None,
) -> None:
    """
    Colors:
      - adversary: purple
      - believes_misinformation: red
      - believes_truth: green
      - other/unknown: gray
    """
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    if graph.number_of_nodes() == 0:
        return

    beliefs: Dict[str, str] = {}
    roles: Dict[str, str] = {}
    for r in agent_rows:
        name = str(r.get("agent_name") or "")
        if not name:
            continue
        roles[name] = str(r.get("role") or "normal")
        if bool(r.get("believes_misinformation")):
            beliefs[name] = "misinfo"
        elif bool(r.get("believes_truth")):
            beliefs[name] = "truth"
        else:
            beliefs[name] = "other"

    # Ensure we have entries for all graph nodes.
    for n in graph.nodes:
        roles.setdefault(str(n), "normal")
        beliefs.setdefault(str(n), "other")

    pos = _layout_for_topology(graph, topology)

    node_colors: List[str] = []
    node_sizes: List[float] = []
    edge_colors: List[str] = []

    for n in graph.nodes:
        role = roles.get(str(n), "normal")
        if role == "adversary":
            node_colors.append("#9467bd")  # purple
        else:
            b = beliefs.get(str(n), "other")
            if b == "misinfo":
                node_colors.append("#d62728")  # red
            elif b == "truth":
                node_colors.append("#2ca02c")  # green
            else:
                node_colors.append("#7f7f7f")  # gray

        if target_agent and str(n) == str(target_agent):
            node_sizes.append(900)
        elif role == "adversary":
            node_sizes.append(750)
        else:
            node_sizes.append(520)

    for u, v in graph.edges:
        edge_colors.append("#555555")

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_title(run_title)
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edge_color=edge_colors, width=1.4, alpha=0.65
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=1.3,
        edgecolors="white",
        alpha=0.92,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9, font_color="white")

    # Legend
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="adversary",
            markerfacecolor="#9467bd",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="misinfo believer",
            markerfacecolor="#d62728",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="truth believer",
            markerfacecolor="#2ca02c",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="other/unknown",
            markerfacecolor="#7f7f7f",
            markersize=10,
        ),
    ]
    ax.legend(handles=legend_items, loc="best", frameon=True)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def _contains_code(msg: str, code: Optional[str]) -> bool:
    if not msg or not code:
        return False
    return str(code).lower() in str(msg).lower()


def plot_run_timeline(
    *,
    run_title: str,
    tool_events: Optional[List[Any]],
    agent_rows: List[Dict[str, Any]],
    code: Optional[str],
    out_path: Path,
) -> None:
    """
    Two-panel timeline:
      - misinfo posts by planning_round (adversary vs non-adversary)
      - cumulative exposure fraction over planning_round (non-adversary agents)
    """
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    # Panel 1: misinfo posts by round
    rounds: List[int] = []
    adv_counts: Dict[int, int] = {}
    non_adv_counts: Dict[int, int] = {}
    if tool_events:
        for e in tool_events:
            if not isinstance(e, dict):
                continue
            if str(e.get("tool_name")) != "post_message":
                continue
            args = e.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            msg = args.get("message", "")
            if not _contains_code(str(msg), code):
                continue
            pr = e.get("planning_round")
            try:
                pr_i = int(pr)
            except Exception:
                continue
            rounds.append(pr_i)
            agent = str(e.get("agent_name") or "")
            role = "normal"
            for r in agent_rows:
                if str(r.get("agent_name")) == agent:
                    role = str(r.get("role") or "normal")
                    break
            if role == "adversary":
                adv_counts[pr_i] = adv_counts.get(pr_i, 0) + 1
            else:
                non_adv_counts[pr_i] = non_adv_counts.get(pr_i, 0) + 1

    # Panel 2: cumulative exposure fraction over rounds
    non_adv = [r for r in agent_rows if str(r.get("role")) != "adversary"]
    exposure_rounds: List[int] = []
    for r in non_adv:
        fr = r.get("first_misinformation_exposure_round")
        if fr is None:
            continue
        try:
            exposure_rounds.append(int(fr))
        except Exception:
            continue

    max_round = max(rounds + exposure_rounds) if (rounds or exposure_rounds) else 0
    if max_round <= 0:
        return
    xs = list(range(1, max_round + 1))

    adv_y = [adv_counts.get(x, 0) for x in xs]
    non_adv_y = [non_adv_counts.get(x, 0) for x in xs]
    cum_frac = []
    for x in xs:
        exposed = sum(1 for r in exposure_rounds if r <= x)
        denom = len(non_adv)
        cum_frac.append((exposed / denom) if denom else 0.0)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.2, 6.0), sharex=True)
    axes[0].bar(
        xs, adv_y, color="#9467bd", alpha=0.8, label="Adversary Misinformation Posts"
    )
    axes[0].bar(
        xs,
        non_adv_y,
        bottom=adv_y,
        color="#7f7f7f",
        alpha=0.8,
        label="Non-adversary Misinformation Posts",
    )
    axes[0].set_ylabel("Number of Misinformation Posts")
    axes[0].set_title(f"{run_title}: Misinformation Posts by Round")
    axes[0].legend(loc="best", frameon=True)

    axes[1].plot(xs, cum_frac, marker="o", linewidth=1.8, color="#1f77b4")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Cumulative Exposure Fraction")
    axes[1].set_xlabel("Planning Round")
    axes[1].set_title("Cumulative non-adversary exposure")

    fig.tight_layout()
    fig.savefig(out_path)
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)
