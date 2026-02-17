from __future__ import annotations

import sys
import argparse
import copy
import csv
import json
import logging
import random
import importlib
import asyncio
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from experiments.common.run_utils import (
    configure_experiment_logging as _configure_experiment_logging_impl,
    ensure_dir as _ensure_dir,
    load_yaml as _load_yaml,
    normalize_seeds as _normalize_seeds,
    write_json as _write_json,
    write_progress as _write_progress,
)
from experiments.emergent_misalignment.agents import AdversarySpec, MisalignmentHospitalAgent
from experiments.common.blackboard_logger import ExperimentBlackboardLogger
from experiments.emergent_misalignment.metrics import compute_misalignment_metrics
from experiments.emergent_misalignment.prompts import EmergentMisalignmentPrompts
from experiments.common.local_protocol import LocalCommunicationProtocol
from experiments.emergent_misalignment.judge_beliefs import judge_run

from src.networks import build_communication_network
from src.logger import AgentTrajectoryLogger
from src.utils import get_client_instance, get_generation_params, get_model_name, build_vllm_runtime

LOGGER_NAME = "experiments.emergent_misalignment"
logger = logging.getLogger(LOGGER_NAME)


def _get_token_pricing_usd_per_1m(
    cfg: Dict[str, Any],
    *,
    provider: str,
    model_name: str,
) -> Optional[Dict[str, float]]:
    pricing = (cfg.get("pricing") or {}).get(provider) or {}
    model_block = pricing.get(model_name) or {}
    if not isinstance(model_block, dict):
        return None
    inp = model_block.get("input_per_1m_usd")
    out = model_block.get("output_per_1m_usd")
    if inp is None or out is None:
        return None
    return {"input_per_1m_usd": float(inp), "output_per_1m_usd": float(out)}


def _sum_turn_usage(turns: List[Dict[str, Any]]) -> Dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for t in turns:
        u = t.get("usage") or {}
        if not isinstance(u, dict):
            continue
        total["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        total["completion_tokens"] += int(u.get("completion_tokens") or 0)
        total["total_tokens"] += int(u.get("total_tokens") or 0)
    return total


def _cost_usd(
    usage: Dict[str, Any],
    *,
    input_per_1m_usd: float,
    output_per_1m_usd: float,
) -> float:
    return (int(usage.get("prompt_tokens") or 0) / 1_000_000.0) * input_per_1m_usd + (
        int(usage.get("completion_tokens") or 0) / 1_000_000.0
    ) * output_per_1m_usd


def _configure_experiment_logging(root: Path, *, verbose: bool = True) -> None:
    _configure_experiment_logging_impl(logger, root, verbose=verbose)


def _resolve_environment_class(env_cfg: Dict[str, Any]) -> Any:
    import_path = str(env_cfg.get("import_path") or "").strip()
    if import_path:
        module_path, sep, cls_name = import_path.partition(":")
        if not sep:
            raise ValueError(f"Invalid import_path: {import_path}")
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)
    raise ValueError("environment.import_path is required.")


def _select_emergent_agents(
    *,
    agent_names: Sequence[str],
    count: int,
    target_role: Optional[str],
    rng: random.Random,
) -> List[str]:
    """
    Select exactly `count` agents to receive the emergent prompt treatment.

    For this experiment we primarily care about `count` in {0, 1}.
    If `target_role` is provided, prefer an agent whose name contains it.
    """
    if count <= 0:
        return []

    candidates = [str(a) for a in agent_names]
    if (str(target_role or "").strip().lower() == "departmental"):
        # "departmental" means "any non-provisioner department agent".
        # Fall through to the default non-provisioner sampling behavior.
        target_role = None
    if target_role:
        role_matches = [a for a in candidates if target_role in a]
        if role_matches:
            # Prefer the first match for determinism when possible.
            chosen = rng.choice(role_matches)
            remaining = [a for a in candidates if a != chosen]
            if count <= 1:
                return [chosen]
            return [chosen] + rng.sample(remaining, k=min(count - 1, len(remaining)))

    # Default: sample from non-provisioners if possible.
    non_provisioners = [a for a in candidates if a != "Resource_Provisioner"]
    pool = non_provisioners if non_provisioners else candidates
    return rng.sample(pool, k=min(int(count), len(pool)))


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


def _augment_context(base: Dict[str, Any], *, extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(extra)
    return out


async def _run_single(
    *,
    base_cfg: Dict[str, Any],
    model_label: str,
    model_llm_cfg: Dict[str, Any],
    sweep_name: str,
    topology: str,
    num_agents: int,
    emergent_count: int,
    num_patients: int,
    emergent_target_role: Optional[str],
    prompt_version: str,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))

    def _should_stop_early(environment: Any) -> bool:
        """
        Early-stop criterion for Hospital-style environments.

        Unlike HospitalEnvironment.done(), treat any recorded resource failure as
        non-convergence (i.e., do not early-stop just because all steps are scheduled).
        Falls back to environment.done(...) when the expected fields are unavailable.
        """
        try:
            patients = getattr(environment, "patients", None)
            patient_states = getattr(environment, "patient_states", None)
            if not isinstance(patients, dict) or not isinstance(patient_states, dict):
                return bool(environment.done(0))  # type: ignore[arg-type]

            # Converged only if every patient has all steps scheduled AND no resource failures were recorded.
            for pid, patient in patients.items():
                state = patient_states.get(pid, {})
                scheduled = state.get("scheduled_steps") or {}
                pathway = patient.get("pathway") or []
                if len(scheduled) != len(pathway):
                    return False
                failures = state.get("resource_failures") or []
                if failures:
                    return False
            return True
        except Exception:
            # Conservative fallback: use the environment's own criterion.
            try:
                return bool(environment.done(0))  # type: ignore[arg-type]
            except Exception:
                return False

    def _resolve_num_patients(*, raw: Any, num_agents: int) -> int:
        """
        Supports YAML-friendly placeholders like:
          - "num_agents_minus_1" / "num_agents-1"
        """
        if isinstance(raw, str):
            key = raw.strip().lower().replace(" ", "_")
            if key in {"num_agents_minus_1", "num_agents-1"}:
                return int(num_agents) - 1
            raise ValueError(f"Unknown num_patients spec: {raw!r}")
        if raw is None:
            return int(num_agents) - 1
        return int(raw)

    # --- CONFIGURATION SETUP ---
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("simulation", {})["seed"] = int(seed)
    
    # Overrides
    cfg.setdefault("communication_network", {})["topology"] = str(topology)
    cfg.setdefault("communication_network", {})["num_agents"] = int(num_agents)
    cfg.setdefault("environment", {})["num_patients"] = _resolve_num_patients(
        raw=num_patients, num_agents=int(num_agents)
    )
    cfg.setdefault("experiment", {}).setdefault("emergent", {})["count"] = int(emergent_count)
    if emergent_target_role:
        cfg["experiment"]["emergent"]["target_role"] = str(emergent_target_role)
    cfg["experiment"]["emergent"]["prompt_version"] = str(prompt_version)

    cfg["llm"] = copy.deepcopy(model_llm_cfg)

    # --- DIRECTORY SETUP ---
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    role_safe = str(emergent_target_role).replace(" ", "_") if emergent_target_role else "Any"
    prompt_safe = str(prompt_version).replace(" ", "_") if prompt_version else "v0"
    run_id = (
        f"{model_label}__{sweep_name}__agents{num_agents}_emergent{int(emergent_count)}_"
        f"{role_safe}_{prompt_safe}_seed{seed}"
    )
    run_dir = out_dir / "runs" / model_label / sweep_name / run_id
    _ensure_dir(run_dir)

    logger.info("RUN START %s", run_id)
    cfg.setdefault("simulation", {})["run_timestamp"] = f"{run_timestamp}__{run_id}"

    # --- COMPONENT INITIALIZATION ---
    protocol = LocalCommunicationProtocol(config=cfg)
    env_cls = _resolve_environment_class(cfg.get("environment") or {})
    env = env_cls(protocol, cfg, tool_logger=type("TL", (), {"log_dir": run_dir})())
    
    bb_logger = ExperimentBlackboardLogger(cfg, log_root=run_dir)
    bb_logger.clear_blackboard_logs()
    
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

    # --- EMERGENT AGENT SELECTION ---
    emergent_agents = _select_emergent_agents(
        agent_names=agent_names,
        count=int(emergent_count),
        target_role=emergent_target_role,
        rng=rng
    )
    roles = {name: ("emergent" if name in emergent_agents else "cooperative") for name in agent_names}

    # Always wrap prompts: injection only happens for the emergent agent and only when a version exists.
    env.prompts = EmergentMisalignmentPrompts(env, cfg, base_prompts=getattr(env, "prompts", None))

    # --- AGENT CLIENT SETUP ---
    provider = (cfg.get("llm", {}).get("provider") or "").lower()
    model_name = get_model_name(provider, cfg["llm"])
    generation_params = get_generation_params(cfg["llm"])
    max_steps = int(cfg["simulation"].get("max_conversation_steps", 3))

    vllm_runtime = None
    if provider == "vllm":
        vllm_runtime = build_vllm_runtime(cfg["llm"])

    adversary_spec = AdversarySpec(
        adversary_names=set(emergent_agents),
        attack_description="Emergent misalignment (benign-looking prompt variant).",
        strategy_type=f"emergent:{prompt_safe}",
    )

    agents = []
    for name in agent_names:
        client = get_client_instance(cfg["llm"], agent_name=name, vllm_runtime=vllm_runtime)
        agents.append(
            MisalignmentHospitalAgent(
                client,
                name,
                model_name,
                max_steps,
                None,
                trajectory_logger,
                env.__class__.__name__,
                generation_params=generation_params,
                adversary_spec=adversary_spec,
            )
        )
    # Match examples/base_main.py behavior: shuffle agent execution order.
    # HospitalEnvironment seeds the global RNG with simulation.seed in __init__.
    random.shuffle(agents)
    env.set_agent_clients(agents)

    await env.async_init()
    
    # Log initial state
    try:
        _log_blackboards_txt(
            bb_logger=bb_logger, protocol=protocol, iteration=0, phase="init", agent_name="SYSTEM"
        )
    except Exception:
        pass

    # --- SIMULATION LOOP ---
    max_iterations = int(cfg["simulation"].get("max_iterations", 1))
    planning_rounds = int(cfg["simulation"].get("max_planning_rounds", 1))
    turns: List[Dict[str, Any]] = []

    def _get_extra_fields(agent_name: str, phase: str) -> Dict[str, Any]:
        extra: Dict[str, Any] = {"phase": phase}
        if agent_name in emergent_agents:
            extra["emergent_role"] = "emergent"
            extra["emergent_prompt_id"] = prompt_safe
        else:
            extra["emergent_role"] = "normal"
            extra["emergent_prompt_id"] = ""
        return extra

    current_iteration = 0
    for iteration in range(1, max_iterations + 1):
        current_iteration = iteration
        if _should_stop_early(env):
            logger.info("Environment requested simulation stop at iteration %s", iteration)
            break

        # 1. PLANNING
        for planning_round in range(1, planning_rounds + 1):
            last_agent = None
            for agent in env.agents:
                base_ctx = env.build_agent_context(
                    agent.name,
                    phase="planning",
                    iteration=iteration,
                    planning_round=planning_round,
                )
                extra = _get_extra_fields(agent.name, "planning")
                agent_context = base_ctx if not extra else _augment_context(base_ctx, extra=extra)

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
                        "round": planning_round,
                        "agent": agent.name,
                        "role": roles[agent.name],
                        "response": response.get("response"),
                        "usage": response.get("usage"),
                        "tools_executed": response.get("tools_executed"),
                    }
                )
                last_agent = agent.name

            if last_agent:
                _log_blackboards_txt(
                    bb_logger=bb_logger,
                    protocol=protocol,
                    iteration=iteration,
                    phase="planning",
                    agent_name=last_agent,
                    planning_round=planning_round,
                )

        # 2. EXECUTION
        last_exec_agent = None
        for agent in env.agents:
            base_ctx = env.build_agent_context(
                agent.name, phase="execution", iteration=iteration
            )
            extra = _get_extra_fields(agent.name, "execution")
            agent_context = base_ctx if not extra else _augment_context(base_ctx, extra=extra)

            response = await protocol.agent_execution_turn(
                agent, agent.name, agent_context, env, iteration=iteration
            )
            turns.append(
                {
                    "phase": "execution",
                    "iteration": iteration,
                    "agent": agent.name,
                    "role": roles[agent.name],
                    "response": response.get("response"),
                    "usage": response.get("usage"),
                    "tools_executed": response.get("tools_executed"),
                }
            )
            last_exec_agent = agent.name

        if last_exec_agent:
            _log_blackboards_txt(
                bb_logger=bb_logger,
                protocol=protocol,
                iteration=iteration,
                phase="execution",
                agent_name=last_exec_agent,
            )

        try:
            env.log_iteration_summary(iteration)
        except Exception:
            pass

    # 3. SURVEY (POST-RUN BELIEFS)
    for agent in env.agents:
        base_ctx = env.build_agent_context(
            agent.name, phase="survey", iteration=max(1, current_iteration)
        )
        extra = _get_extra_fields(agent.name, "survey")
        agent_context = base_ctx if not extra else _augment_context(base_ctx, extra=extra)
        response = await protocol.agent_survey_turn(
            agent,
            agent.name,
            agent_context,
            env,
            iteration=max(1, current_iteration),
        )
        turns.append(
            {
                "phase": "survey",
                "iteration": current_iteration,
                "agent": agent.name,
                "role": roles[agent.name],
                "response": response.get("response"),
                "usage": response.get("usage"),
                "tools_executed": response.get("tools_executed"),
            }
        )

    # Persist turns before the judge pass (judge_beliefs loads agent_turns.json).
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

    # 4. EVALUATION (JUDGE BELIEFS)
    # This runs a separate pass to check if agents suspected foul play
    judgements = await judge_run(run_dir=run_dir, config=cfg)
    judge_usage_total: Dict[str, Any] = {}
    try:
        judge_usage_path = run_dir / "judge_usage.json"
        if judge_usage_path.exists():
            judge_usage_blob = json.loads(judge_usage_path.read_text(encoding="utf-8"))
            if isinstance(judge_usage_blob, dict):
                judge_usage_total = judge_usage_blob.get("total_usage") or {}
    except Exception:
        judge_usage_total = {}

    # 5. METRICS
    blackboard_participants = {
        bb.blackboard_id: bb.participants for bb in protocol.megaboard.blackboards
    }
    
    metrics = compute_misalignment_metrics(
        graph=communication_network.graph,
        adversaries=emergent_agents,
        tool_events=protocol.tool_events,
        judgements=judgements,
        blackboard_participants=blackboard_participants,
        roles=roles,
        final_summary=final_summary,
        agent_rewards=agent_rewards,
        attack_type=f"emergent:{prompt_safe}",
    )

    logger.info(
        "RUN %s metrics: JointReward=%.2f, SuspicionRate=%.2f",
        run_id, metrics.meeting_outcome.joint_reward, metrics.suspicion_rate_non_adversary
    )

    # 6. ARTIFACTS
    _write_json(
        run_dir / "run_config.json",
        {
            "run_id": run_id,
            "emergent_agents": emergent_agents,
            "prompt_version": prompt_safe,
            "emergent_target_role": emergent_target_role,
            "roles": roles,
            "metrics": asdict(metrics)
        },
    )
    _write_json(run_dir / "final_summary.json", final_summary)
    _write_json(run_dir / "agent_rewards.json", agent_rewards or {})
    _write_json(run_dir / "metrics.json", asdict(metrics))
    _write_json(run_dir / "judgements.json", judgements)

    # 7. COST ACCOUNTING (best-effort; requires optional config.pricing)
    pricing = _get_token_pricing_usd_per_1m(cfg, provider=provider, model_name=model_name)
    agent_usage = _sum_turn_usage(turns)
    run_cost = {
        "provider": provider,
        "model_name": model_name,
        "pricing": pricing,
        "agent_usage": agent_usage,
        "judge_usage": judge_usage_total,
        "agent_cost_usd": None,
        "judge_cost_usd": None,
        "total_cost_usd": None,
    }
    if pricing:
        run_cost["agent_cost_usd"] = _cost_usd(
            agent_usage,
            input_per_1m_usd=pricing["input_per_1m_usd"],
            output_per_1m_usd=pricing["output_per_1m_usd"],
        )
        run_cost["judge_cost_usd"] = _cost_usd(
            judge_usage_total,
            input_per_1m_usd=pricing["input_per_1m_usd"],
            output_per_1m_usd=pricing["output_per_1m_usd"],
        )
        run_cost["total_cost_usd"] = float(run_cost["agent_cost_usd"]) + float(
            run_cost["judge_cost_usd"]
        )
    _write_json(run_dir / "costs.json", run_cost)
    
    # Detailed logging similar to collusion runner
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
    
    return {
        "run_id": run_id,
        "emergent_agents": len(emergent_agents),
        "prompt_version": prompt_safe,
        "emergent_target_role": emergent_target_role or "Any",
        "joint_reward": metrics.meeting_outcome.joint_reward,
        "suspicion_rate": metrics.suspicion_rate_non_adversary,
        "agent_prompt_tokens": agent_usage["prompt_tokens"],
        "agent_completion_tokens": agent_usage["completion_tokens"],
        "agent_total_tokens": agent_usage["total_tokens"],
        "judge_prompt_tokens": int(judge_usage_total.get("prompt_tokens") or 0),
        "judge_completion_tokens": int(judge_usage_total.get("completion_tokens") or 0),
        "judge_total_tokens": int(judge_usage_total.get("total_tokens") or 0),
        "total_cost_usd": run_cost.get("total_cost_usd"),
    }


async def run_from_config(config_path: str, *, out_dir: Optional[str] = None, max_concurrent_runs: Optional[int] = None) -> Path:
    cfg = _load_yaml(config_path)
    exp = cfg.get("experiment") or {}
    max_concurrent_runs = max_concurrent_runs or int(exp.get("max_concurrent_runs", 1))
    
    models = cfg.get("llm_models") or []
    sweeps = exp.get("sweeps") or []
    seeds = _normalize_seeds(exp.get("seeds")) or [1]
    
    # Pre-calculate total runs
    total_runs = 0
    for model in models:
        for sweep in sweeps:
            agent_counts = sweep.get("num_agents") or [9]
            emergent_counts = sweep.get("emergent_counts") or sweep.get("adversary_counts") or [1]
            emergent_target_roles = sweep.get("emergent_target_roles") or sweep.get("target_roles") or [None]
            patient_counts = sweep.get("num_patients") or [16]
            prompt_versions = sweep.get("prompt_versions") or ["v0"]
            
            # Simple multiplication for progress bar estimate
            combinations = (
                len(agent_counts)
                * len(emergent_counts)
                * len(emergent_target_roles)
                * len(patient_counts)
                * len(prompt_versions)
                * len(seeds)
            )
            total_runs += combinations

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(out_dir or exp.get("output_dir") or "experiments/emergent_misalignment/outputs") / timestamp
    _ensure_dir(root)
    _configure_experiment_logging(root)
    _write_json(root / "config.json", cfg)
    
    _write_progress(root, {
        "status": "running",
        "total_runs": total_runs,
        "completed_runs": 0,
        "failed_runs": 0,
        "started_at": datetime.now().isoformat()
    })

    tasks: List[asyncio.Task] = []
    task_labels: Dict[asyncio.Task, str] = {}
    summaries: List[Dict[str, Any]] = []
    completed = 0
    failed = 0
    
    semaphore = asyncio.Semaphore(max_concurrent_runs)

    async def _run_guarded(run_label: str, **kwargs) -> Dict[str, Any]:
        async with semaphore:
            return await _run_single(**kwargs)

    # Scheduling Loop
    for model in models:
        model_llm = model.get("llm")
        model_label = model.get("label")
        for sweep in sweeps:
            sweep_name = sweep.get("name")
            topologies = sweep.get("topologies") or ["complete"]
            
            agent_counts = sweep.get("num_agents") or [9]
            emergent_counts = sweep.get("emergent_counts") or sweep.get("adversary_counts") or [1]
            emergent_target_roles = sweep.get("emergent_target_roles") or sweep.get("target_roles") or [None]
            patient_counts = sweep.get("num_patients") or [16]
            prompt_versions = sweep.get("prompt_versions") or ["v0"]

            for topology in topologies:
                for num_agents in agent_counts:
                    for ec in emergent_counts:
                        for role in emergent_target_roles:
                            for num_patients in patient_counts:
                                for prompt_version in prompt_versions:
                                    for seed in seeds:
                                        role_lbl = role if role else "Any"
                                        run_label = (
                                            f"{model_label}/{sweep_name}/n{num_agents}/emergent{ec}/"
                                            f"{role_lbl}/{prompt_version}/seed{seed}"
                                        )

                                        task = asyncio.create_task(
                                            _run_guarded(
                                                run_label=run_label,
                                                base_cfg=cfg,
                                                model_label=model_label,
                                                model_llm_cfg=model_llm,
                                                sweep_name=sweep_name,
                                                topology=str(topology),
                                                num_agents=num_agents,
                                                emergent_count=ec,
                                                num_patients=num_patients,
                                                emergent_target_role=role,
                                                prompt_version=str(prompt_version),
                                                seed=seed,
                                                out_dir=root,
                                            )
                                        )
                                        tasks.append(task)
                                        task_labels[task] = run_label

    # Execution Loop (Robust)
    with tqdm(total=len(tasks), desc="Experiments", unit="run") as pbar:
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                run_label = task_labels.get(finished, "unknown")
                pbar.set_postfix_str(run_label)
                try:
                    res = await finished
                    summaries.append(res)
                    completed += 1
                except Exception:
                    failed += 1
                    logger.exception("RUN FAILED %s", run_label)
                finally:
                    pbar.update(1)
                    _write_progress(root, {
                        "status": "running",
                        "total_runs": total_runs,
                        "completed_runs": completed,
                        "failed_runs": failed,
                        "last_run": run_label
                    })

    _write_json(root / "summary.json", summaries)
    
    # CSV generation
    csv_rows = []
    for row in summaries:
        flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        csv_rows.append(flat)
    if csv_rows:
        fieldnames = sorted(list(csv_rows[0].keys()))
        with open(root / "summary.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    # Optional: aggregate costs across all runs (best-effort).
    try:
        from experiments.emergent_misalignment.costs import compute_and_write_costs

        compute_and_write_costs(root, write_per_run=False)
    except Exception:
        logger.exception("Failed to write costs_summary.json")
            
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-concurrent-runs", default=None, type=int)
    args = parser.parse_args()
    
    asyncio.run(
        run_from_config(
            args.config, 
            out_dir=args.out_dir, 
            max_concurrent_runs=args.max_concurrent_runs
        )
    )

if __name__ == "__main__":
    main()
