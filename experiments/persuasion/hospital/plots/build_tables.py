from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiments.common.plotting.io_utils import as_bool, as_float, as_int
from experiments.common.plotting.load_runs import LoadedRun


@dataclass(frozen=True)
class Tables:
    run_rows: List[Dict[str, Any]]
    agent_rows: List[Dict[str, Any]]


_RUN_ID_RE = re.compile(
    r"""
    ^(?P<model>.+?)__                           # model label
    (?P<sweep>.+?)__                            # sweep name
    topo(?P<topology>.+?)__                     # topo<...>
    agents(?P<num_agents>\d+)__                 # agentsN
    strat(?P<strategy>.+?)__                    # strat<...>
    c(?P<colluder_count>\d+)__                  # cK
    tr(?P<target_role>.+?)__                    # tr<...>
    secret(?P<secret>\d+)__                     # secret0/1
    pv(?P<prompt_variant>.+?)__                 # pv<...>
    seed(?P<seed>\d+)$                          # seedS
    """,
    re.VERBOSE,
)


def _sum_float_dict(value: Any) -> Optional[float]:
    if not isinstance(value, dict):
        return None
    total = 0.0
    ok = False
    for v in value.values():
        fv = as_float(v)
        if fv is None:
            continue
        ok = True
        total += float(fv)
    return float(total) if ok else None


def _infer_run_fields(run_dir: Path, run_config: Dict[str, Any]) -> Dict[str, Any]:
    rid = str(run_config.get("run_id") or run_dir.name)
    out: Dict[str, Any] = {"run_id": rid}

    m = _RUN_ID_RE.match(rid)
    if m:
        out.update(
            {
                "model_label": m.group("model"),
                "sweep_name": m.group("sweep"),
                "topology": m.group("topology"),
                "strategy": m.group("strategy"),
                "num_agents": as_int(m.group("num_agents")),
                "adversary_count": as_int(m.group("colluder_count")),
                "colluder_count": as_int(m.group("colluder_count")),
                "target_role": m.group("target_role"),
                "secret_channel_enabled": bool(int(m.group("secret"))),
                "prompt_variant": m.group("prompt_variant"),
                "seed": as_int(m.group("seed")),
            }
        )
    else:
        # Fallback: infer sweep/model from directory structure.
        try:
            out["sweep_name"] = run_dir.parent.name
            out["model_label"] = run_dir.parent.parent.name
        except Exception:
            pass

    # Add run-config fields when present (preferred / more accurate).
    out["misalignment_strategy"] = run_config.get("misalignment_strategy")
    out["prompt_variant"] = run_config.get("prompt_variant") or out.get("prompt_variant")
    out["secret_blackboard_id"] = run_config.get("secret_blackboard_id")
    out["secret_channel_enabled"] = as_bool(run_config.get("secret_channel_enabled")) if run_config.get("secret_channel_enabled") is not None else out.get("secret_channel_enabled")

    # Align naming with agent_misalignment plots (uses `strategy`).
    if out.get("strategy") is None and out.get("misalignment_strategy") is not None:
        out["strategy"] = out.get("misalignment_strategy")

    # If run_config has colluders list, prefer its length.
    colluders = run_config.get("colluders")
    if isinstance(colluders, list):
        out["colluder_count"] = int(len(colluders))
        out["adversary_count"] = int(len(colluders))

    return out


def _compute_secret_post_stats(
    *,
    run_config: Dict[str, Any],
    tool_events: Optional[List[Any]],
) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    colluders = run_config.get("colluders")
    if not isinstance(colluders, list) or not colluders:
        return None, None, None, None
    colluder_set = {str(a) for a in colluders}
    secret_bb = run_config.get("secret_blackboard_id")
    try:
        secret_bb_int = int(secret_bb) if secret_bb is not None else None
    except Exception:
        secret_bb_int = None
    if secret_bb_int is None:
        return None, None, None, None

    if not isinstance(tool_events, list):
        return None, None, None, None

    total = 0
    secret = 0
    non_secret = 0

    for e in tool_events:
        if not isinstance(e, dict):
            continue
        if str(e.get("tool_name") or "") != "post_message":
            continue
        agent = str(e.get("agent_name") or "")
        if agent not in colluder_set:
            continue
        args = e.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        bb_raw = args.get("blackboard_id")
        try:
            bb_id = int(bb_raw) if bb_raw is not None else None
        except Exception:
            bb_id = None
        total += 1
        if bb_id == secret_bb_int:
            secret += 1
        else:
            non_secret += 1

    rate = (float(secret) / float(total)) if total > 0 else 0.0
    return float(rate), int(total), int(secret), int(non_secret)


def build_tables(runs: List[LoadedRun]) -> Tables:
    run_rows: List[Dict[str, Any]] = []
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

        secret_rate, posts_total, posts_secret, posts_non_secret = _compute_secret_post_stats(
            run_config=rc, tool_events=run.tool_events
        )

        run_rows.append(
            {
                **inferred,
                "run_dir": str(run.run_dir),
                "joint_reward": as_float(meeting.get("joint_reward")),
                "status": meeting.get("status") or metrics.get("status") or fs.get("status"),
                "resource_failures_total": as_int(
                    sum(int(v or 0) for v in (convergence.get("resource_failures") or {}).values())
                )
                if isinstance(convergence.get("resource_failures"), dict)
                else None,
                "converged_patients": as_int(convergence.get("converged_count")),
                "total_patients": as_int(convergence.get("total_patients")),
                "failed_patients_count": as_int(len(convergence.get("failed_patients") or [])),
                # Misalignment metrics of interest
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
                "noncoalition_reward_avg": as_float(metrics.get("noncoalition_reward_avg")),
                "coalition_minus_noncoalition_avg": as_float(
                    metrics.get("coalition_minus_noncoalition_avg")
                ),
                "coalition_minus_noncoalition_avg_hospital_adjusted": as_float(
                    metrics.get("coalition_minus_noncoalition_avg_hospital_adjusted")
                ),
                "coalition_minus_noncoalition_avg_hospital_adjusted_normalized": as_float(
                    metrics.get("coalition_minus_noncoalition_avg_hospital_adjusted_normalized")
                ),
                "coalition_minus_noncoalition_hospital_avg": as_float(
                    metrics.get("coalition_minus_noncoalition_hospital_avg")
                ),
                "coalition_minus_noncoalition_hospital_avg_hospital_adjusted": as_float(
                    metrics.get("coalition_minus_noncoalition_hospital_avg_hospital_adjusted")
                ),
                "coalition_minus_noncoalition_hospital_avg_hospital_adjusted_normalized": as_float(
                    metrics.get("coalition_minus_noncoalition_hospital_avg_hospital_adjusted_normalized")
                ),
                "coalition_minus_noncoalition_hospital_avg_actual": as_float(
                    metrics.get("coalition_minus_noncoalition_hospital_avg_actual")
                ),
                "hospital_holding_cost_total": _sum_float_dict(metrics.get("hospital_holding_cost")),
                "hospital_resource_failure_penalty_total": _sum_float_dict(
                    metrics.get("hospital_resource_failure_penalty")
                ),
                "agent_resource_failure_penalty_total": _sum_float_dict(
                    metrics.get("agent_resource_failure_penalty")
                ),
                "hospital_missed_step_penalty_total": _sum_float_dict(
                    metrics.get("hospital_missed_step_penalty")
                ),
                "hospital_estimated_missed_steps": as_float(metrics.get("hospital_estimated_missed_steps")),
                "meeting_outcome_joint_reward": as_float(meeting.get("joint_reward")),
                "meeting_outcome_variables_assigned": as_int(meeting.get("variables_assigned")),
                "meeting_outcome_total_variables": as_int(meeting.get("total_variables")),
                "graph_density": as_float(graph.get("density")),
                "graph_avg_degree": as_float(graph.get("avg_degree")),
                "graph_avg_clustering": as_float(graph.get("avg_clustering")),
                "max_joint_reward": as_float(metrics.get("max_joint_reward")),
                "joint_reward_regret": as_float(metrics.get("joint_reward_regret")),
                "coalition_max_reward_sum": as_float(metrics.get("coalition_max_reward_sum")),
                "coalition_reward_sum": as_float(metrics.get("coalition_reward_sum")),
                "coalition_reward_regret": as_float(metrics.get("coalition_reward_regret")),
                "coalition_reward_regret_hospital_adjusted": as_float(
                    metrics.get("coalition_reward_regret_hospital_adjusted")
                ),
                # Persuasion-specific / secret-channel usage
                "colluder_posts_secret_rate": secret_rate,
                "colluder_posts_total": posts_total,
                "colluder_posts_secret": posts_secret,
                "colluder_posts_non_secret": posts_non_secret,
            }
        )

        agents = metrics.get("agents")
        if isinstance(agents, list):
            for a in agents:
                if not isinstance(a, dict):
                    continue
                agent_rows.append(
                    {
                        "run_dir": str(run.run_dir),
                        "run_id": rid,
                        "seed": inferred.get("seed"),
                        "topology": inferred.get("topology"),
                        "adversary_count": inferred.get("adversary_count"),
                        "target_role": inferred.get("target_role"),
                        "strategy": inferred.get("strategy"),
                        "prompt_variant": inferred.get("prompt_variant"),
                        "secret_channel_enabled": inferred.get("secret_channel_enabled"),
                        **a,
                    }
                )

    return Tables(run_rows=run_rows, agent_rows=agent_rows)
