from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from experiments.common.plotting.io_utils import as_float, as_int, flatten_dict
from experiments.common.plotting.load_runs import LoadedRun


@dataclass(frozen=True)
class Tables:
    run_rows: List[Dict[str, Any]]
    target_rows: List[Dict[str, Any]]
    agent_rows: List[Dict[str, Any]]


def build_tables(runs: List[LoadedRun]) -> Tables:
    run_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []

    for run in runs:
        rc = run.run_config or {}
        rid = str(rc.get("run_id") or run.run_dir.name)
        topology = rc.get("topology")
        num_agents = as_int(rc.get("num_agents"))
        adversary_count = as_int(rc.get("adversary_count"))
        seed = as_int(rc.get("seed"))

        cn = rc.get("communication_network") or {}
        if not isinstance(cn, dict):
            cn = {}
        cn_edge_prob = as_float(cn.get("edge_prob", rc.get("edge_prob")))
        cn_k = as_int(
            cn.get(
                "k",
                cn.get("nearest_neighbors", cn.get("num_neighbors", rc.get("k"))),
            )
        )
        cn_rewire_prob = as_float(
            cn.get(
                "rewire_prob",
                cn.get(
                    "rewiring_prob",
                    cn.get("beta", rc.get("rewire_prob", rc.get("rewiring_prob"))),
                ),
            )
        )
        cn_m = as_int(
            cn.get(
                "m",
                cn.get(
                    "edges_per_node",
                    cn.get("num_edges_to_attach", rc.get("m")),
                ),
            )
        )

        fs = run.final_summary or {}
        run_row: Dict[str, Any] = {
            "run_dir": str(run.run_dir),
            "run_id": rid,
            "model_label": rc.get("model_label"),
            "provider": rc.get("provider"),
            "model": rc.get("model"),
            "sweep_name": rc.get("sweep_name") or rc.get("sweep") or None,
            "topology": topology,
            "num_agents": num_agents,
            "adversary_count": adversary_count,
            "seed": seed,
            "cn_edge_prob": cn_edge_prob,
            "cn_k": cn_k,
            "cn_rewire_prob": cn_rewire_prob,
            "cn_m": cn_m,
            "status": fs.get("status"),
            "joint_reward": as_float(fs.get("joint_reward")),
            "joint_reward_ratio": as_float(fs.get("joint_reward_ratio")),
            "average_agent_reward": as_float(fs.get("average_agent_reward")),
            "variables_assigned": as_int(fs.get("variables_assigned")),
            "total_variables": as_int(fs.get("total_variables")),
        }
        run_rows.append(run_row)

        metrics = run.metrics or {}
        by_target = metrics.get("by_target") if isinstance(metrics, dict) else None
        if isinstance(by_target, dict):
            for target_agent, tmetrics in by_target.items():
                if not isinstance(tmetrics, dict):
                    continue
                flat = flatten_dict(tmetrics, prefix="")
                # Avoid embedding large nested structures in the target-level row.
                flat.pop("agents", None)
                # Make sure key identifiers are always present
                target_row: Dict[str, Any] = {
                    "run_dir": str(run.run_dir),
                    "run_id": rid,
                    "topology": topology,
                    "num_agents": num_agents,
                    "adversary_count": adversary_count,
                    "seed": seed,
                    "target_agent": target_agent,
                }
                target_row.update(flat)
                # Normalize a few common numeric fields
                for k in [
                    "propagation_rate_misinfo_non_adversary",
                    "propagation_rate_truth_non_adversary",
                    "avg_distance_to_adversary_misinfo_believers",
                    "meeting_outcome.joint_reward",
                    "meeting_outcome.joint_reward_ratio",
                    "meeting_outcome.average_agent_reward",
                    "graph.density",
                    "graph.avg_degree",
                    "graph.avg_clustering",
                    "graph.diameter",
                    "graph.avg_shortest_path_length",
                ]:
                    if k in target_row:
                        if k.endswith("diameter"):
                            target_row[k] = as_int(target_row[k])
                        else:
                            target_row[k] = as_float(target_row[k])
                for k in [
                    "total_agents",
                    "adversary_count",
                    "total_messages",
                    "misinformation_messages",
                    "believes_misinformation_total",
                    "believes_misinformation_non_adversary",
                    "believes_truth_total",
                    "believes_truth_non_adversary",
                ]:
                    if k in target_row:
                        target_row[k] = as_int(target_row[k])
                target_rows.append(target_row)

                # Agent rows (optional)
                agents = tmetrics.get("agents") if isinstance(tmetrics, dict) else None
                if isinstance(agents, list):
                    for a in agents:
                        if not isinstance(a, dict):
                            continue
                        arow: Dict[str, Any] = {
                            "run_dir": str(run.run_dir),
                            "run_id": rid,
                            "topology": topology,
                            "num_agents": num_agents,
                            "adversary_count": adversary_count,
                            "seed": seed,
                            "target_agent": target_agent,
                        }
                        arow.update(a)
                        # Normalize
                        for k in [
                            "degree",
                            "distance_to_nearest_adversary",
                            "total_posts",
                            "misinformation_posts",
                            "misinformation_exposures",
                            "first_misinformation_exposure_round",
                            "first_misinformation_post_round",
                        ]:
                            if k in arow:
                                arow[k] = as_int(arow[k])
                        for k in ["judge_confidence"]:
                            if k in arow:
                                arow[k] = as_float(arow[k])
                        agent_rows.append(arow)

    return Tables(run_rows=run_rows, target_rows=target_rows, agent_rows=agent_rows)
