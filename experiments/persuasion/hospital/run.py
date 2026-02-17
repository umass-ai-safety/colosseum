from __future__ import annotations

# ruff: noqa: E402

import sys
import argparse
import asyncio
import copy
import csv
import json
import logging
import random
import importlib
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from experiments.common.run_utils import (
    configure_experiment_logging as _configure_experiment_logging_impl,
    ensure_dir as _ensure_dir,
    load_yaml as _load_yaml,
    normalize_seeds as _normalize_seeds,
    write_json as _write_json,
    write_progress as _write_progress,
)
from experiments.common.blackboard_logger import ExperimentBlackboardLogger
from experiments.common.local_protocol import LocalCommunicationProtocol
from experiments.agent_misalignment.metrics import compute_misalignment_metrics
from experiments.persuasion.hospital.prompts import PersuasionHospitalPrompts
from src.networks import build_communication_network
from src.logger import AgentTrajectoryLogger, PromptLogger
from src.utils import get_client_instance, get_generation_params, get_model_name, build_vllm_runtime
from src.agents.base import BaseAgent


LOGGER_NAME = "experiments.persuasion.hospital"
logger = logging.getLogger(LOGGER_NAME)

_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_HOSPITAL_ROLES_PER_HOSPITAL = 4  # Triage, Radiology, Surgery, Ward


def _configure_experiment_logging(root: Path, *, verbose: bool = True) -> None:
    _configure_experiment_logging_impl(logger, root, verbose=verbose)


def _sanitize_path_component(value: Any) -> str:
    s = "None" if value is None else str(value)
    s = s.strip()
    if not s:
        return "None"
    s = _PATH_COMPONENT_RE.sub("_", s)
    s = s.strip("_")
    return s or "None"


def _resolve_num_patients_spec(*, raw: Any, num_agents: int) -> int:
    if isinstance(raw, str):
        key = raw.strip().lower().replace(" ", "_")
        if key in {"num_agents_minus_1", "num_agents-1"}:
            return int(num_agents) - 1
        raise ValueError(f"Unknown num_patients spec: {raw!r}")
    if raw is None:
        return int(num_agents) - 1
    return int(raw)


def _infer_num_hospitals_from_num_agents(num_agents: int) -> int:
    n = int(num_agents)
    if n <= 0:
        raise ValueError(f"num_agents must be > 0 (got {num_agents})")
    if n == 1:
        return 0
    if (n - 1) % _HOSPITAL_ROLES_PER_HOSPITAL != 0:
        raise ValueError(
            f"num_agents={n} is not compatible with HospitalEnvironment; expected 1 + 4*k."
        )
    return int((n - 1) // _HOSPITAL_ROLES_PER_HOSPITAL)


def _resolve_environment_class(env_cfg: Dict[str, Any]) -> Any:
    import_path = str(env_cfg.get("import_path") or "").strip()
    if import_path:
        module_path, sep, cls_name = import_path.partition(":")
        if not sep:
            raise ValueError(
                "environment.import_path must be formatted as 'some.module:ClassName' "
                f"(got {import_path!r})."
            )
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)

    env_name = str(env_cfg.get("name") or "").strip()
    if not env_name:
        raise ValueError("environment.name is required (or set environment.import_path).")

    candidate_modules = ["envs.dcops"]
    for module_path in candidate_modules:
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        if hasattr(module, env_name):
            return getattr(module, env_name)

    raise ValueError(
        f"Unknown environment.name {env_name!r}. "
        "Either export it from an envs package __init__.py, or set environment.import_path."
    )


def _select_colluders_hospital_coalition(
    *,
    agent_names: Sequence[str],
    count: int,
    target_role: Optional[str],
    rng: random.Random,
) -> List[str]:
    """
    Hospital coalition selection:
      - Prefer department agents from a single hospital until full (Triage/Radiology/Surgery/Ward),
        then spill to another hospital.
      - If target_role is set, ensure coalition includes the required role:
          - "Resource_Provisioner": include provisioner if present.
          - "departmental": choose a hospital (prefer General_Hospital if present) and pick departments from there.
          - "Triage"/"Radiology"/"Surgery"/"Ward": include a matching agent if present.
    """
    if count <= 0:
        return []

    target_role = (str(target_role).strip() if target_role is not None else None) or None
    count = min(int(count), len(agent_names))

    hospitals: set[str] = set()
    provisioner: Optional[str] = None
    hospital_map: Dict[str, List[str]] = {}
    known_roles = ["Triage", "Surgery", "Radiology", "Ward"]

    for name in agent_names:
        if name == "Resource_Provisioner":
            provisioner = name
            continue
        parts = str(name).split("_")
        h_name = "_".join(parts[:-1]) if parts and parts[-1] in known_roles else "Unknown"
        hospitals.add(h_name)
        hospital_map.setdefault(h_name, []).append(str(name))

    hospitals_list = sorted(list(hospitals))
    preferred_hospital_order: List[str] = []
    for h in ("General_Hospital", "St_Marys_Center"):
        if h in hospitals_list:
            preferred_hospital_order.append(h)
    preferred_hospital_order.extend([h for h in hospitals_list if h not in preferred_hospital_order])

    def _role_sort_key(agent_name: str) -> int:
        for i, r in enumerate(["Triage", "Radiology", "Surgery", "Ward"]):
            if agent_name.endswith(f"_{r}"):
                return i
        return 999

    def _hospital_candidates(h: str) -> List[str]:
        return sorted(hospital_map.get(h, []), key=_role_sort_key)

    required_agent: Optional[str] = None
    required_hospital: Optional[str] = None

    if target_role == "Resource_Provisioner":
        required_agent = provisioner
    elif target_role and target_role.lower() == "departmental":
        required_hospital = (
            "General_Hospital"
            if "General_Hospital" in hospitals_list
            else (hospitals_list[0] if hospitals_list else None)
        )
    elif target_role:
        for h in hospitals_list:
            hit = next((a for a in hospital_map.get(h, []) if target_role in a), None)
            if hit:
                required_agent = str(hit)
                required_hospital = h
                break

    if target_role == "Resource_Provisioner" and not required_agent:
        return [str(x) for x in rng.sample(list(agent_names), k=count)]

    coalition: List[str] = []
    if required_agent:
        coalition.append(required_agent)

    remaining_needed = int(count) - len(coalition)
    if remaining_needed <= 0:
        return coalition[:count]

    # Choose hospital order (required first when applicable).
    hs = list(preferred_hospital_order)
    if required_hospital and required_hospital in hs:
        hs = [required_hospital] + [h for h in hs if h != required_hospital]
    hs = [h for h in hs if hospital_map.get(h)]

    def _add_from_hospital(h: str, *, need: int) -> None:
        if need <= 0:
            return
        candidates = [a for a in _hospital_candidates(h) if a not in coalition]
        coalition.extend(candidates[:need])

    # Fill departments hospital-by-hospital.
    for h in hs:
        if remaining_needed <= 0:
            break
        _add_from_hospital(h, need=min(remaining_needed, _HOSPITAL_ROLES_PER_HOSPITAL))
        remaining_needed = int(count) - len(coalition)

    # If we still need agents (e.g., because of Unknown mapping), sample from remaining.
    if len(coalition) < count:
        remaining = [a for a in agent_names if str(a) not in coalition]
        if remaining:
            coalition.extend([str(x) for x in rng.sample(list(remaining), k=min(len(remaining), count - len(coalition)))])

    # Preserve uniqueness and size.
    coalition = list(dict.fromkeys(coalition))[:count]
    return coalition


def _log_blackboards_txt(
    *,
    bb_logger: ExperimentBlackboardLogger,
    protocol: LocalCommunicationProtocol,
    iteration: int,
    phase: str,
    agent_name: str,
    planning_round: Optional[int] = None,
) -> None:
    for blackboard in protocol.megaboard.blackboards:
        bb_logger.log_blackboard_state(
            blackboard,
            iteration=iteration,
            phase=phase,
            agent_name=agent_name,
            round_num=planning_round,
        )


async def _run_single(
    *,
    base_cfg: Dict[str, Any],
    model_label: str,
    model_llm_cfg: Dict[str, Any],
    sweep_name: str,
    topology: str,
    num_agents: int,
    num_patients: Any,
    misalignment_strategy: str,
    colluder_count: int,
    target_role: Optional[str],
    coalition_clique_enabled: bool,
    secret_channel_enabled: bool,
    prompt_variant: str,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    secret_channel_enabled = bool(secret_channel_enabled)
    coalition_clique_enabled = bool(coalition_clique_enabled)
    misalignment_strategy = str(misalignment_strategy or "").strip() or "covert_max_coalition_reward"
    effective_prompt_variant = str(prompt_variant or "").strip() or "control"
    if not secret_channel_enabled:
        effective_prompt_variant = "control"

    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("simulation", {})["seed"] = int(seed)
    cfg.setdefault("communication_network", {})["topology"] = str(topology)
    cfg.setdefault("communication_network", {})["num_agents"] = int(num_agents)
    cfg["llm"] = copy.deepcopy(model_llm_cfg)

    resolved_num_patients = _resolve_num_patients_spec(raw=num_patients, num_agents=int(num_agents))
    cfg.setdefault("environment", {})["num_patients"] = int(resolved_num_patients)
    cfg.setdefault("environment", {})["num_hospitals"] = _infer_num_hospitals_from_num_agents(
        int(num_agents)
    )

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = (
        f"{_sanitize_path_component(model_label)}__{_sanitize_path_component(sweep_name)}"
        f"__topo{_sanitize_path_component(topology)}__agents{int(num_agents)}"
        f"__strat{_sanitize_path_component(misalignment_strategy)}"
        f"__c{int(colluder_count)}__tr{_sanitize_path_component(target_role)}"
        f"__secret{int(bool(secret_channel_enabled))}__pv{_sanitize_path_component(effective_prompt_variant)}"
        f"__seed{int(seed)}"
    )
    run_dir = out_dir / "runs" / model_label / sweep_name / run_id
    _ensure_dir(run_dir)

    logger.info("RUN START %s", run_id)

    cfg.setdefault("simulation", {})["run_timestamp"] = f"{run_timestamp}__{run_id}"
    cfg.setdefault("simulation", {})["tags"] = [
        str(cfg.get("experiment", {}).get("tag", "persuasion_hospital"))
    ]

    protocol = LocalCommunicationProtocol(config=cfg)
    env_cls = _resolve_environment_class(cfg.get("environment") or {})
    env = env_cls(protocol, cfg, tool_logger=type("TL", (), {"log_dir": run_dir})())

    bb_logger = ExperimentBlackboardLogger(cfg, log_root=run_dir)
    bb_logger.clear_blackboard_logs()

    experiment_cfg = cfg.get("experiment") or {}
    log_prompts_cfg = experiment_cfg.get("log_prompts")
    log_prompts = True if log_prompts_cfg is None else bool(log_prompts_cfg)
    prompt_logger = (
        PromptLogger(
            environment_name=env.__class__.__name__,
            seed=int(seed),
            config=cfg,
            run_timestamp=cfg.get("simulation", {}).get("run_timestamp"),
            log_dir=run_dir,
        )
        if log_prompts
        else None
    )
    trajectory_logger = AgentTrajectoryLogger(
        environment_name=env.__class__.__name__,
        seed=int(seed),
        config=cfg,
        run_timestamp=cfg.get("simulation", {}).get("run_timestamp"),
        log_dir=run_dir,
    )

    agent_names = env.get_agent_names()
    communication_network = build_communication_network(agent_names, cfg)
    env.set_communication_network(communication_network)

    exp_block = (cfg.get("experiment") or {}).get("persuasion_hospital") or {}
    selection_strategy = str(exp_block.get("colluder_selection", "hospital_coalition")).strip().lower()
    if selection_strategy not in {"hospital_coalition", "random", "first"}:
        raise ValueError(
            "experiment.persuasion_hospital.colluder_selection must be one of: "
            "'hospital_coalition', 'random', 'first'."
        )

    if selection_strategy == "random":
        colluders = [str(x) for x in rng.sample(list(agent_names), k=min(int(colluder_count), len(agent_names)))]
    elif selection_strategy == "first":
        colluders = [str(x) for x in list(agent_names)[: min(int(colluder_count), len(agent_names))]]
    else:
        colluders = _select_colluders_hospital_coalition(
            agent_names=agent_names,
            count=int(colluder_count),
            target_role=target_role,
            rng=rng,
        )

    colluder_set = set(colluders)
    roles = {str(a): ("colluder" if str(a) in colluder_set else "normal") for a in agent_names}

    if coalition_clique_enabled and len(colluders) >= 2:
        for i in range(len(colluders)):
            for j in range(i + 1, len(colluders)):
                a = str(colluders[i])
                b = str(colluders[j])
                try:
                    communication_network.graph.add_edge(a, b)
                except Exception:
                    pass

    env.prompts = PersuasionHospitalPrompts(
        env,
        cfg,
        prompt_variant=str(effective_prompt_variant),
        base_prompts=getattr(env, "prompts", None),
        experiment_prompt_logger=prompt_logger,
        log_prompts=log_prompts,
    )

    provider = (cfg.get("llm", {}).get("provider") or "").lower()
    model_name = get_model_name(provider, cfg["llm"])
    generation_params = get_generation_params(cfg["llm"])
    max_steps = int(cfg["simulation"].get("max_conversation_steps", 3))

    vllm_runtime = None
    if provider == "vllm":
        vllm_runtime = build_vllm_runtime(cfg["llm"])

    agents: List[BaseAgent] = []
    for name in agent_names:
        client = get_client_instance(cfg["llm"], agent_name=name, vllm_runtime=vllm_runtime)
        agents.append(
            BaseAgent(
                client,
                name,
                model_name,
                max_steps,
                None,
                trajectory_logger,
                env.__class__.__name__,
                generation_params=generation_params,
            )
        )
    rng.shuffle(agents)
    env.set_agent_clients(agents)

    await env.async_init()

    secret_blackboard_id: Optional[int] = None
    if secret_channel_enabled and len(colluders) >= 2:
        try:
            base_context = env.get_network_context()
            secret_context = env.format_blackboard_context(colluders, base_context)
        except Exception:
            secret_context = env.get_network_context() if hasattr(env, "get_network_context") else ""
        secret_blackboard_id = await protocol.generate_comm_network(
            colluders,
            secret_context,
            template={"secret_channel": True, "visibility": "secret", "purpose": "coalition"},
        )

    try:
        _log_blackboards_txt(
            bb_logger=bb_logger,
            protocol=protocol,
            iteration=0,
            phase="init",
            agent_name="SYSTEM",
            planning_round=None,
        )
    except Exception as exc:
        logger.warning("Failed to write initial blackboard_*.txt logs: %s", exc)

    max_iterations = int(cfg.get("simulation", {}).get("max_iterations", 1))
    planning_rounds = int(cfg["simulation"].get("max_planning_rounds", 1))
    turns: List[Dict[str, Any]] = []

    def _extra_fields(agent_name: str, *, phase: str, iteration: int, planning_round: Optional[int]) -> Dict[str, Any]:
        prompt_role = (
            "colluder"
            if (secret_channel_enabled and str(agent_name) in colluder_set)
            else "normal"
        )
        return {
            "collusion_role": prompt_role,
            "collusion_coalition_members": sorted(list(colluder_set)) if prompt_role == "colluder" else [],
            "collusion_blackboard_id": secret_blackboard_id if prompt_role == "colluder" else None,
            "misalignment_strategy": misalignment_strategy,
            "phase": phase,
            "iteration": iteration,
            "planning_round": planning_round,
        }

    current_iteration = 0
    for iteration in range(1, max_iterations + 1):
        current_iteration = iteration

        # PLANNING
        for planning_round in range(1, planning_rounds + 1):
            last_agent = None
            for agent in env.agents:
                base_ctx = env.build_agent_context(
                    agent.name,
                    phase="planning",
                    iteration=iteration,
                    planning_round=planning_round,
                )
                agent_context = dict(base_ctx)
                agent_context.update(
                    _extra_fields(
                        agent.name,
                        phase="planning",
                        iteration=iteration,
                        planning_round=planning_round,
                    )
                )
                response = await protocol.agent_planning_turn(
                    agent,
                    agent.name,
                    agent_context,
                    env,
                    iteration=iteration,
                    planning_round=planning_round,
                )
                turns.append(
                    {
                        "phase": "planning",
                        "iteration": iteration,
                        "planning_round": planning_round,
                        "agent": agent.name,
                        "role": roles.get(agent.name, "normal"),
                        "response": response.get("response"),
                        "usage": response.get("usage"),
                        "model": response.get("model"),
                        "tools_executed": response.get("tools_executed"),
                        "conversation_steps": response.get("conversation_steps"),
                    }
                )
                last_agent = agent.name
            if last_agent:
                try:
                    _log_blackboards_txt(
                        bb_logger=bb_logger,
                        protocol=protocol,
                        iteration=iteration,
                        phase="planning",
                        agent_name=str(last_agent),
                        planning_round=int(planning_round),
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to write blackboard_*.txt logs after planning round %s: %s",
                        planning_round,
                        exc,
                    )

        # EXECUTION
        last_exec_agent = None
        for agent in env.agents:
            base_ctx = env.build_agent_context(
                agent.name, phase="execution", iteration=iteration
            )
            agent_context = dict(base_ctx)
            agent_context.update(
                _extra_fields(
                    agent.name,
                    phase="execution",
                    iteration=iteration,
                    planning_round=None,
                )
            )
            response = await protocol.agent_execution_turn(
                agent, agent.name, agent_context, env, iteration=iteration
            )
            turns.append(
                {
                    "phase": "execution",
                    "iteration": iteration,
                    "agent": agent.name,
                    "role": roles.get(agent.name, "normal"),
                    "response": response.get("response"),
                    "usage": response.get("usage"),
                    "model": response.get("model"),
                    "tools_executed": response.get("tools_executed"),
                    "conversation_steps": response.get("conversation_steps"),
                }
            )
            last_exec_agent = agent.name

        if last_exec_agent:
            try:
                _log_blackboards_txt(
                    bb_logger=bb_logger,
                    protocol=protocol,
                    iteration=iteration,
                    phase="execution",
                    agent_name=str(last_exec_agent),
                    planning_round=None,
                )
            except Exception as exc:
                logger.warning("Failed to write blackboard_*.txt logs after execution: %s", exc)

        if hasattr(env, "log_iteration"):
            try:
                env.log_iteration(iteration)
            except Exception:
                pass

        # Optional early stop if environment reports done.
        try:
            if hasattr(env, "done") and bool(env.done(iteration)):  # type: ignore[arg-type]
                break
        except Exception:
            pass

    _write_json(run_dir / "agent_turns.json", turns)

    final_summary = env.get_final_summary()

    agent_rewards: Optional[Dict[str, float]] = None
    try:
        if hasattr(env, "_calculate_makespan_and_flow"):
            _, rewards = env._calculate_makespan_and_flow()  # type: ignore[attr-defined]
            if isinstance(rewards, dict):
                agent_rewards = {str(k): float(v) for k, v in rewards.items()}
    except Exception:
        agent_rewards = None

    # Compute the same metrics object as agent_misalignment for compatibility.
    blackboard_participants = {}
    try:
        for bb in protocol.megaboard.blackboards:
            bb_id = getattr(bb, "blackboard_id", None)
            participants = getattr(bb, "participants", None)
            if bb_id is None:
                continue
            if isinstance(participants, list):
                blackboard_participants[int(bb_id)] = [str(x) for x in participants]
            else:
                agents = getattr(bb, "agents", None)
                if isinstance(agents, (list, set, tuple)):
                    blackboard_participants[int(bb_id)] = [str(x) for x in agents]
    except Exception:
        blackboard_participants = {}

    misalignment_roles = {
        str(name): ("adversary" if str(name) in colluder_set else "cooperative")
        for name in agent_names
    }
    metrics = compute_misalignment_metrics(
        graph=communication_network.graph,
        adversaries=colluders,
        tool_events=protocol.tool_events,
        judgements={},  # persuasion_hospital does not run the judge pass (yet)
        blackboard_participants=blackboard_participants,
        roles=misalignment_roles,
        final_summary=final_summary if isinstance(final_summary, dict) else {},
        agent_rewards=agent_rewards,
        attack_type=misalignment_strategy,
    )

    _write_json(
        run_dir / "run_config.json",
        {
            "run_id": run_id,
            "misalignment_strategy": misalignment_strategy,
            "colluders": colluders,
            "roles": roles,
            "secret_channel_enabled": secret_channel_enabled,
            "secret_blackboard_id": secret_blackboard_id,
            "prompt_variant": str(effective_prompt_variant),
            "metrics": asdict(metrics),
        },
    )
    _write_json(run_dir / "final_summary.json", final_summary)
    _write_json(run_dir / "agent_rewards.json", agent_rewards or {})
    _write_json(run_dir / "metrics.json", asdict(metrics))
    _write_json(
        run_dir / "tool_events.json",
        [
            {
                "tool_name": e.tool_name,
                "agent_name": e.agent_name,
                "arguments": e.arguments,
                "result": e.result,
                "phase": e.phase,
                "iteration": e.iteration,
                "planning_round": e.planning_round,
            }
            for e in protocol.tool_events
        ],
    )
    _write_json(
        run_dir / "blackboards.json",
        [
            {
                "blackboard_id": bb.blackboard_id,
                "participants": sorted(list(bb.agents)),
                "events": bb.logs,
            }
            for bb in protocol.megaboard.blackboards
        ],
    )

    logger.info("RUN END %s (artifacts=%s)", run_id, run_dir)
    return {
        "run_id": run_id,
        "model_label": model_label,
        "provider": provider,
        "model": model_name,
        "sweep": sweep_name,
        "topology": topology,
        "num_agents": int(num_agents),
        "num_patients": int(resolved_num_patients),
        "misalignment_strategy": misalignment_strategy,
        "colluder_count": int(len(colluders)),
        "target_role": target_role,
        "coalition_clique_enabled": bool(coalition_clique_enabled),
        "secret_channel_enabled": bool(secret_channel_enabled),
        "secret_blackboard_id": secret_blackboard_id,
        "prompt_variant": str(effective_prompt_variant),
        "seed": int(seed),
        "colluders": colluders,
        "status": metrics.meeting_outcome.status,
        "joint_reward": metrics.meeting_outcome.joint_reward,
        "joint_reward_regret": metrics.joint_reward_regret,
        "coalition_reward_sum": metrics.coalition_reward_sum,
        "noncoalition_reward_sum": metrics.noncoalition_reward_sum,
        "coalition_reward_avg": metrics.coalition_reward_avg,
        "noncoalition_reward_avg": metrics.noncoalition_reward_avg,
        "coalition_minus_noncoalition_avg": metrics.coalition_minus_noncoalition_avg,
        "coalition_minus_noncoalition_hospital_avg_actual": metrics.coalition_minus_noncoalition_hospital_avg_actual,
        "coalition_reward_regret": metrics.coalition_reward_regret,
        "coalition_reward_regret_hospital_adjusted": metrics.coalition_reward_regret_hospital_adjusted,
    }


async def run_from_config(
    config_path: str,
    *,
    out_dir: Optional[str] = None,
    max_concurrent_runs: Optional[int] = None,
) -> Path:
    cfg = _load_yaml(config_path)
    exp = cfg.get("experiment") or {}
    if max_concurrent_runs is None:
        max_concurrent_runs = exp.get("max_concurrent_runs", 1)
    max_concurrent_runs = int(max_concurrent_runs)
    if max_concurrent_runs <= 0:
        raise ValueError("max_concurrent_runs must be a positive integer")

    models = cfg.get("llm_models") or []
    sweeps = exp.get("sweeps") or []

    runs_per_setting = exp.get("runs_per_setting")
    if runs_per_setting is not None:
        runs_per_setting = int(runs_per_setting)
        if runs_per_setting <= 0:
            raise ValueError("experiment.runs_per_setting must be a positive integer")

    default_seeds = _normalize_seeds(exp.get("seeds"))
    if not default_seeds:
        default_seeds = _normalize_seeds((cfg.get("simulation") or {}).get("seed")) or [1]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = (
        Path(
            out_dir
            or exp.get("output_dir")
            or "experiments/persuasion/hospital/outputs/persuasion_hospital"
        )
        / timestamp
    )
    _ensure_dir(root)
    _write_json(root / "config.json", cfg)
    _configure_experiment_logging(root)

    # Count total runs for progress.
    total_runs = 0
    for model in models:
        for sweep in sweeps:
            topologies = sweep.get("topologies") or []
            agent_counts = sweep.get("num_agents") or []
            num_patients_specs = sweep.get("num_patients") or [None]
            strategies = sweep.get("strategies") or [
                "covert_max_coalition_reward",
                "destructive_max_coalition_reward",
                "destructive_no_reward_preservation",
            ]
            colluder_counts = sweep.get("colluder_counts") or []
            target_roles = sweep.get("target_roles") or [None]
            clique_flags = sweep.get("coalition_clique_enabled") or [False]
            secret_flags = sweep.get("secret_channel_enabled") or [True]
            raw_prompt_variants = sweep.get("prompt_variants") or ["control"]
            prompt_variants: List[str] = []
            seen_variants: set[str] = set()
            for pv in raw_prompt_variants:
                pv_str = str(pv or "").strip() or "control"
                if pv_str in seen_variants:
                    continue
                seen_variants.add(pv_str)
                prompt_variants.append(pv_str)
            seeds = _normalize_seeds(sweep.get("seeds")) or list(default_seeds)
            if runs_per_setting is not None:
                seeds = seeds[:runs_per_setting]
            for _topo in topologies:
                for _n in agent_counts:
                    for _p in num_patients_specs:
                        for _s in strategies:
                            for _c in colluder_counts:
                                for _tr in target_roles:
                                    for _clique in clique_flags:
                                        for _secret in secret_flags:
                                            for _pv in prompt_variants:
                                                if not bool(_secret) and str(_pv) != "control":
                                                    continue
                                                total_runs += len(seeds)

    logger.info("EXPERIMENT START (total_runs=%s, output_root=%s)", total_runs, root)
    _write_progress(root, {"status": "running", "total_runs": total_runs, "completed_runs": 0})

    semaphore = asyncio.Semaphore(max_concurrent_runs)
    completed = 0
    results: List[Dict[str, Any]] = []

    async def _bounded_run(**kwargs: Any) -> Dict[str, Any]:
        async with semaphore:
            return await _run_single(**kwargs)

    tasks: List[asyncio.Task[Dict[str, Any]]] = []
    for model in models:
        model_label = str(model.get("label") or "model")
        model_llm_cfg = model.get("llm") or {}
        for sweep in sweeps:
            sweep_name = str(sweep.get("name") or "sweep")
            topologies = sweep.get("topologies") or []
            agent_counts = sweep.get("num_agents") or []
            num_patients_specs = sweep.get("num_patients") or [None]
            strategies = sweep.get("strategies") or [
                "covert_max_coalition_reward",
                "destructive_max_coalition_reward",
                "destructive_no_reward_preservation",
            ]
            colluder_counts = sweep.get("colluder_counts") or []
            target_roles = sweep.get("target_roles") or [None]
            clique_flags = sweep.get("coalition_clique_enabled") or [False]
            secret_flags = sweep.get("secret_channel_enabled") or [True]
            prompt_variants = sweep.get("prompt_variants") or ["control"]
            seeds = _normalize_seeds(sweep.get("seeds")) or list(default_seeds)
            if runs_per_setting is not None:
                seeds = seeds[:runs_per_setting]

            for topo in topologies:
                for n in agent_counts:
                    for p_spec in num_patients_specs:
                        for strat in strategies:
                            for c in colluder_counts:
                                for tr in target_roles:
                                    for clique in clique_flags:
                                        for secret in secret_flags:
                                            for pv in prompt_variants:
                                                if not bool(secret) and str(pv) != "control":
                                                    continue
                                                for seed in seeds:
                                                    tasks.append(
                                                        asyncio.create_task(
                                                            _bounded_run(
                                                                base_cfg=cfg,
                                                                model_label=model_label,
                                                                model_llm_cfg=model_llm_cfg,
                                                                sweep_name=sweep_name,
                                                                topology=str(topo),
                                                                num_agents=int(n),
                                                                num_patients=p_spec,
                                                                misalignment_strategy=str(strat),
                                                                colluder_count=int(c),
                                                                target_role=(str(tr) if tr is not None else None),
                                                                coalition_clique_enabled=bool(clique),
                                                                secret_channel_enabled=bool(secret),
                                                                prompt_variant=str(pv),
                                                                seed=int(seed),
                                                                out_dir=root,
                                                            )
                                                        )
                                                    )

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        try:
            row = await fut
        except Exception as exc:
            row = {"status": "error", "error": str(exc)}
        results.append(row)
        completed += 1
        _write_progress(root, {"status": "running", "total_runs": total_runs, "completed_runs": completed})

    _write_progress(root, {"status": "complete", "total_runs": total_runs, "completed_runs": completed})

    rows_path = root / "runs.csv"
    if results:
        fieldnames: List[str] = []
        # Stable union of keys.
        keys: set[str] = set()
        for r in results:
            if isinstance(r, dict):
                keys.update(r.keys())
        fieldnames = sorted(keys)
        with rows_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                if not isinstance(r, dict):
                    continue
                w.writerow(r)

    _write_json(root / "results.json", results)
    logger.info("EXPERIMENT END (output_root=%s)", root)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persuasion collusion experiments for HospitalEnvironment."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config (e.g., experiments/persuasion/hospital/configs/persuasion_hospital.yaml).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override output directory root (otherwise uses YAML experiment.output_dir).",
    )
    parser.add_argument(
        "--max-concurrent-runs",
        type=int,
        default=None,
        help="Override experiment.max_concurrent_runs (default: YAML value).",
    )
    args = parser.parse_args()

    root = asyncio.run(
        run_from_config(
            args.config, out_dir=args.out_dir, max_concurrent_runs=args.max_concurrent_runs
        )
    )
    print(str(root))


if __name__ == "__main__":
    main()
