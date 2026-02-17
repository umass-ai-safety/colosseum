from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from experiments.common.plotting.io_utils import as_float, as_int
from experiments.common.plotting.load_runs import LoadedRun


@dataclass(frozen=True)
class Tables:
    run_rows: List[Dict[str, Any]]
    target_rows: List[Dict[str, Any]]
    agent_rows: List[Dict[str, Any]]


_RUN_ID_RE = re.compile(
    r"""
    ^(?P<model>.+?)__                           # model label
    (?P<sweep>.+?)__                            # sweep name
    agents(?P<num_agents>\d+)_                  # agentsN
    adv(?P<adversary_count>\d+)_                # advK
    (?P<target_role>.+?)_                       # target role (can include underscores)
    seed(?P<seed>\d+)$                          # seedS
    """,
    re.VERBOSE,
)


def _infer_run_fields(run_dir: Path, run_config: Dict[str, Any]) -> Dict[str, Any]:
    rid = str(run_config.get("run_id") or run_dir.name)
    out: Dict[str, Any] = {"run_id": rid}

    m = _RUN_ID_RE.match(rid)
    if m:
        out.update(
            {
                "model_label": m.group("model"),
                "sweep_name": m.group("sweep"),
                "num_agents": as_int(m.group("num_agents")),
                "adversary_count": as_int(m.group("adversary_count")),
                "target_role": m.group("target_role"),
                "seed": as_int(m.group("seed")),
            }
        )
    else:
        # Fallback: infer sweep/model from directory structure.
        # run_dir: .../runs/<model>/<sweep>/<run_id>
        try:
            out["sweep_name"] = run_dir.parent.name
            out["model_label"] = run_dir.parent.parent.name
        except Exception:
            pass

    adversaries = run_config.get("adversaries")
    if isinstance(adversaries, list):
        out["adversary_count"] = int(len(adversaries))
    roles = run_config.get("roles")
    if isinstance(roles, dict):
        out["num_agents"] = int(len(roles))

    return out


def _sum_resource_failures(final_summary: Dict[str, Any]) -> int | None:
    convergence = final_summary.get("convergence_report")
    if not isinstance(convergence, dict):
        return None
    failures = convergence.get("resource_failures")
    if not isinstance(failures, dict):
        return None
    try:
        return int(sum(int(v or 0) for v in failures.values()))
    except Exception:
        return None


def build_tables(runs: List[LoadedRun]) -> Tables:
    run_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []

    for run in runs:
        rc = run.run_config or {}
        inferred = _infer_run_fields(run.run_dir, rc)
        rid = str(inferred.get("run_id") or run.run_dir.name)

        fs = run.final_summary or {}
        if not isinstance(fs, dict):
            fs = {}
        convergence = fs.get("convergence_report")
        if not isinstance(convergence, dict):
            convergence = {}

        metrics = run.metrics or {}
        if not isinstance(metrics, dict):
            metrics = {}

        meeting = metrics.get("meeting_outcome")
        if not isinstance(meeting, dict):
            meeting = {}
        graph = metrics.get("graph")
        if not isinstance(graph, dict):
            graph = {}

        run_rows.append(
            {
                "run_dir": str(run.run_dir),
                "run_id": rid,
                "model_label": inferred.get("model_label"),
                "sweep_name": inferred.get("sweep_name"),
                "num_agents": inferred.get("num_agents"),
                "adversary_count": inferred.get("adversary_count"),
                "target_role": inferred.get("target_role"),
                "seed": inferred.get("seed"),
                "status": fs.get("status"),
                "attack_type": metrics.get("attack_type"),
                "joint_reward": as_float(fs.get("joint_reward")),
                "converged_patients": as_int(convergence.get("converged_count")),
                "total_patients": as_int(convergence.get("total_patients")),
                "resource_failures_total": _sum_resource_failures(fs),
                "failed_patients_count": as_int(
                    len(convergence.get("failed_patients") or [])
                ),
                "suspicion_rate_non_adversary": as_float(
                    metrics.get("suspicion_rate_non_adversary")
                ),
                "correct_attribution_rate_non_adversary": as_float(
                    metrics.get("correct_attribution_rate_non_adversary")
                ),
                "total_messages": as_int(metrics.get("total_messages")),
                "complaint_messages": as_int(metrics.get("complaint_messages")),
                "avg_distance_to_adversary_suspicious_agents": as_float(
                    metrics.get("avg_distance_to_adversary_suspicious_agents")
                ),
                "coalition_reward_avg": as_float(metrics.get("coalition_reward_avg")),
                "noncoalition_reward_avg": as_float(
                    metrics.get("noncoalition_reward_avg")
                ),
                "coalition_minus_noncoalition_avg": as_float(
                    metrics.get("coalition_minus_noncoalition_avg")
                ),
                # Convenience flattened nested values
                "meeting_outcome_joint_reward": as_float(meeting.get("joint_reward")),
                "meeting_outcome_variables_assigned": as_int(
                    meeting.get("variables_assigned")
                ),
                "meeting_outcome_total_variables": as_int(meeting.get("total_variables")),
                "graph_density": as_float(graph.get("density")),
                "graph_avg_degree": as_float(graph.get("avg_degree")),
                "graph_avg_clustering": as_float(graph.get("avg_clustering")),
            }
        )

        # Agent rows from metrics.json["agents"]
        agents = metrics.get("agents")
        if isinstance(agents, list):
            for a in agents:
                if not isinstance(a, dict):
                    continue
                arow: Dict[str, Any] = {
                    "run_dir": str(run.run_dir),
                    "run_id": rid,
                    "model_label": inferred.get("model_label"),
                    "sweep_name": inferred.get("sweep_name"),
                    "num_agents": inferred.get("num_agents"),
                    "adversary_count": inferred.get("adversary_count"),
                    "target_role": inferred.get("target_role"),
                    "seed": inferred.get("seed"),
                }
                arow.update(a)
                for k in [
                    "degree",
                    "distance_to_nearest_adversary",
                    "total_posts",
                    "complaint_posts",
                    "complaint_exposures",
                    "first_complaint_exposure_round",
                    "first_complaint_post_round",
                ]:
                    if k in arow:
                        arow[k] = as_int(arow[k])
                if "judge_confidence" in arow:
                    arow["judge_confidence"] = as_float(arow.get("judge_confidence"))
                agent_rows.append(arow)

    # target_rows unused for this experiment (kept for compatibility with the CLI output tables)
    return Tables(run_rows=run_rows, target_rows=target_rows, agent_rows=agent_rows)

