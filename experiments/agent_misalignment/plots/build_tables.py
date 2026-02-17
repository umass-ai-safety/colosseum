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


_RUN_ID_RE_V2 = re.compile(
    r"""
    ^(?P<model>.+?)__                           # model label
    (?P<sweep>.+?)__                            # sweep name
    topo(?P<topology>.+?)__                     # topo<...>
    strat(?P<strategy>.+?)__                    # strat<...>
    agents(?P<num_agents>\d+)_                  # agentsN
    patients(?P<num_patients>\d+)_              # patientsP
    adv(?P<adversary_count>\d+)_                # advK
    (?P<target_role>.+?)_                       # target role (can include underscores)
    seed(?P<seed>\d+)$                          # seedS
    """,
    re.VERBOSE,
)

_RUN_ID_RE_V1 = re.compile(
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


def _sum_float_dict(value: Any) -> float | None:
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

    m = _RUN_ID_RE_V2.match(rid) or _RUN_ID_RE_V1.match(rid)
    if m:
        requested_num_agents = as_int(m.group("num_agents"))
        requested_adversary_count = as_int(m.group("adversary_count"))
        out.update(
            {
                "model_label": m.group("model"),
                "sweep_name": m.group("sweep"),
                "topology": m.groupdict().get("topology"),
                "strategy": m.groupdict().get("strategy"),
                "requested_num_agents": requested_num_agents,
                "num_patients": as_int(m.groupdict().get("num_patients")),
                "requested_adversary_count": requested_adversary_count,
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

    # Record actual realized values from the run_config (may differ from the
    # requested sweep values embedded in run_id).
    adversaries = run_config.get("adversaries")
    if isinstance(adversaries, list):
        out["actual_adversary_count"] = int(len(adversaries))
    roles = run_config.get("roles")
    if isinstance(roles, dict):
        out["actual_num_agents"] = int(len(roles))

    # For sweep plots, use the requested values from run_id (what the config
    # intended). Keep actual_* fields for diagnostics and debugging.
    out["num_agents"] = out.get("requested_num_agents")
    out["adversary_count"] = out.get("requested_adversary_count")

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
                "requested_num_agents": inferred.get("requested_num_agents"),
                "actual_num_agents": inferred.get("actual_num_agents"),
                "adversary_count": inferred.get("adversary_count"),
                "requested_adversary_count": inferred.get("requested_adversary_count"),
                "actual_adversary_count": inferred.get("actual_adversary_count"),
                "target_role": inferred.get("target_role"),
                "strategy": inferred.get("strategy"),
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
                "hospital_holding_cost_total": _sum_float_dict(
                    metrics.get("hospital_holding_cost")
                ),
                "hospital_resource_failure_penalty_total": _sum_float_dict(
                    metrics.get("hospital_resource_failure_penalty")
                ),
                "agent_resource_failure_penalty_total": _sum_float_dict(
                    metrics.get("agent_resource_failure_penalty")
                ),
                "hospital_missed_step_penalty_total": _sum_float_dict(
                    metrics.get("hospital_missed_step_penalty")
                ),
                "hospital_estimated_missed_steps": as_float(
                    metrics.get("hospital_estimated_missed_steps")
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

                # Upper bound "global optimum" (no solver) + regret values.
                "max_joint_reward": as_float(metrics.get("max_joint_reward"))
                if metrics.get("max_joint_reward") is not None
                else (
                    float(as_int(convergence.get("total_patients")) or 0) * 1000.0
                    if (as_int(convergence.get("total_patients")) or 0) > 0
                    else None
                ),
                "joint_reward_regret": as_float(metrics.get("joint_reward_regret")),
                "coalition_max_reward_sum": as_float(metrics.get("coalition_max_reward_sum")),
                "coalition_reward_sum": as_float(metrics.get("coalition_reward_sum")),
                "noncoalition_reward_sum": as_float(metrics.get("noncoalition_reward_sum")),
                "coalition_reward_regret": as_float(metrics.get("coalition_reward_regret")),
                "coalition_reward_regret_hospital_adjusted": as_float(
                    metrics.get("coalition_reward_regret_hospital_adjusted")
                ),
            }
        )

        # Backwards-compatible regret derivation for older logs.
        row = run_rows[-1]
        if row.get("joint_reward_regret") is None:
            try:
                mjr = row.get("max_joint_reward")
                jr = row.get("joint_reward")
                if mjr is not None and jr is not None:
                    row["joint_reward_regret"] = max(0.0, float(mjr) - float(jr))
            except Exception:
                pass

        # Convenience normalized regret (per-run):
        #   normalized_regret = 1 - (achieved_joint_reward / optimal_joint_reward)
        # We use max_joint_reward as the "optimal" scale when no explicit optimum is available.
        # This matches the desired definition used in the paper plots for misalignment.
        mjr = row.get("max_joint_reward")
        jr = row.get("joint_reward")
        if mjr not in (None, 0, 0.0):
            try:
                if jr is not None:
                    row["joint_reward_regret_normalized"] = max(0.0, 1.0 - (float(jr) / float(mjr)))
                elif row.get("joint_reward_regret") is not None:
                    row["joint_reward_regret_normalized"] = float(row["joint_reward_regret"]) / float(mjr)
            except Exception:
                pass

        if row.get("coalition_max_reward_sum") is None:
            try:
                mjr = row.get("max_joint_reward")
                n_agents = row.get("actual_num_agents") or row.get("num_agents")
                adv = row.get("actual_adversary_count") or row.get("adversary_count")
                if mjr is not None and n_agents and adv is not None:
                    row["coalition_max_reward_sum"] = (float(mjr) / float(int(n_agents))) * float(int(adv))
            except Exception:
                pass

        if row.get("coalition_reward_regret") is None:
            try:
                cmax = row.get("coalition_max_reward_sum")
                csum = as_float(metrics.get("coalition_reward_sum"))
                if csum is None:
                    csum = as_float(metrics.get("coalition_reward_sum"))
                if cmax is not None and csum is not None:
                    row["coalition_reward_regret"] = max(0.0, float(cmax) - float(csum))
            except Exception:
                pass

        # Convenience normalized coalition regret (per-run):
        #   normalized_coalition_regret = 1 - (achieved_coalition_reward / optimal_coalition_reward)
        # where optimal_coalition_reward is approximated by coalition_max_reward_sum.
        cmax = row.get("coalition_max_reward_sum")
        csum = as_float(metrics.get("coalition_reward_sum"))
        if cmax not in (None, 0, 0.0):
            try:
                if csum is not None:
                    row["coalition_reward_regret_normalized"] = max(0.0, 1.0 - (float(csum) / float(cmax)))
                elif row.get("coalition_reward_regret") is not None:
                    row["coalition_reward_regret_normalized"] = float(row["coalition_reward_regret"]) / float(cmax)
            except Exception:
                pass

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
                    "requested_num_agents": inferred.get("requested_num_agents"),
                    "actual_num_agents": inferred.get("actual_num_agents"),
                    "adversary_count": inferred.get("adversary_count"),
                    "requested_adversary_count": inferred.get("requested_adversary_count"),
                    "actual_adversary_count": inferred.get("actual_adversary_count"),
                    "target_role": inferred.get("target_role"),
                    "strategy": inferred.get("strategy"),
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
