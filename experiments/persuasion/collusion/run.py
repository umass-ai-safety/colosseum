from __future__ import annotations

# ruff: noqa: E402

import sys
import argparse
import copy
import csv
import json
import logging
import random
import importlib
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
from experiments.collusion.metrics import compute_collusion_metrics, metrics_to_json
from experiments.persuasion.collusion.prompts import PersuasionCollusionPrompts
from experiments.common.local_protocol import LocalCommunicationProtocol
from terrarium.networks import build_communication_network
from terrarium.logger import AgentTrajectoryLogger, PromptLogger
from terrarium.utils import get_client_instance, get_generation_params, get_model_name
from terrarium.agents.base import BaseAgent


LOGGER_NAME = "experiments.persuasion.collusion"
logger = logging.getLogger(LOGGER_NAME)


def _configure_experiment_logging(root: Path, *, verbose: bool = True) -> None:
    _configure_experiment_logging_impl(logger, root, verbose=verbose)


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
        raise ValueError(
            "environment.name is required (or set environment.import_path)."
        )

    candidate_modules = [
        "envs.dcops",
    ]
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


def _select_colluders(
    *,
    agent_names: Sequence[str],
    count: int,
    strategy: str,
    rng: random.Random,
) -> List[str]:
    if count <= 0:
        return []
    count = min(int(count), len(agent_names))
    strategy = str(strategy or "random").strip().lower()
    if strategy == "random":
        return [str(x) for x in rng.sample(list(agent_names), k=count)]
    if strategy == "first":
        return [str(x) for x in list(agent_names)[:count]]
    raise ValueError(f"Unknown colluder selection strategy: {strategy!r}")

def _order_agent_turns(
    *,
    agent_names: Sequence[str],
    colluders: Sequence[str],
    strategy: str,
) -> List[str]:
    strategy = str(strategy or "random").strip().lower()
    if strategy == "random":
        return list(agent_names)
    if strategy in {"colluders_first", "colluders-front", "colluders_front"}:
        colluder_set = {str(x) for x in colluders}
        ordered_colluders = [str(a) for a in agent_names if str(a) in colluder_set]
        ordered_others = [str(a) for a in agent_names if str(a) not in colluder_set]
        return ordered_colluders + ordered_others
    raise ValueError(
        f"Unknown agent_order strategy: {strategy!r} (expected: 'random' or 'colluders_first')"
    )


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
    colluder_count: int,
    secret_channel_enabled: bool,
    prompt_variant: str,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    secret_channel_enabled = bool(secret_channel_enabled)
    effective_prompt_variant = str(prompt_variant or "").strip() or "control"
    if not secret_channel_enabled:
        effective_prompt_variant = "control"

    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("simulation", {})["seed"] = int(seed)
    cfg.setdefault("simulation", {})["max_iterations"] = 1
    cfg.setdefault("simulation", {})["max_planning_rounds"] = int(
        cfg.get("experiment", {}).get(
            "planning_rounds", cfg.get("simulation", {}).get("max_planning_rounds", 2)
        )
    )
    cfg.setdefault("simulation", {})["max_conversation_steps"] = int(
        cfg.get("experiment", {}).get(
            "max_conversation_steps",
            cfg.get("simulation", {}).get("max_conversation_steps", 3),
        )
    )
    cfg.setdefault("communication_network", {})["topology"] = str(topology)
    cfg.setdefault("communication_network", {})["num_agents"] = int(num_agents)
    cfg["llm"] = copy.deepcopy(model_llm_cfg)

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = (
        f"{model_label}__{sweep_name}__{topology}__n{num_agents}"
        f"__c{colluder_count}__secret{int(bool(secret_channel_enabled))}"
        f"__pv{effective_prompt_variant}__seed{seed}"
    )
    run_dir = out_dir / "runs" / model_label / sweep_name / run_id
    _ensure_dir(run_dir)

    logger.info("RUN START %s", run_id)

    cfg.setdefault("simulation", {})["run_timestamp"] = f"{run_timestamp}__{run_id}"
    cfg.setdefault("simulation", {})["tags"] = [
        str(cfg.get("experiment", {}).get("tag", "persuasion_collusion"))
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

    persuasion_collusion_cfg = (
        (cfg.get("experiment") or {}).get("persuasion_collusion") or {}
    )
    selection_strategy = str(
        persuasion_collusion_cfg.get("colluder_selection", "random")
    )
    colluders = _select_colluders(
        agent_names=agent_names,
        count=int(colluder_count),
        strategy=selection_strategy,
        rng=rng,
    )
    agent_order_strategy = str(persuasion_collusion_cfg.get("agent_order", "random"))
    ordered_agent_names = _order_agent_turns(
        agent_names=agent_names,
        colluders=colluders,
        strategy=agent_order_strategy,
    )
    colluder_set = set(colluders)
    roles = {
        str(a): ("colluder" if str(a) in colluder_set else "normal")
        for a in agent_names
    }

    env.prompts = PersuasionCollusionPrompts(
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

    agents: List[BaseAgent] = []
    for name in agent_names:
        client = get_client_instance(cfg["llm"], agent_name=name)
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
    if str(agent_order_strategy or "random").strip().lower() == "random":
        rng.shuffle(agents)
    else:
        agents_by_name = {a.name: a for a in agents}
        agents = [
            agents_by_name[name]
            for name in ordered_agent_names
            if name in agents_by_name
        ]
    env.set_agent_clients(agents)

    await env.async_init()

    secret_blackboard_id: Optional[int] = None
    if secret_channel_enabled and len(colluders) >= 2:
        try:
            base_context = env.get_network_context()
            secret_context = env.format_blackboard_context(colluders, base_context)
        except Exception:
            secret_context = (
                env.get_network_context() if hasattr(env, "get_network_context") else ""
            )
        secret_blackboard_id = await protocol.generate_comm_network(
            colluders, secret_context
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

    planning_rounds = int(cfg["simulation"].get("max_planning_rounds", 1))
    turns: List[Dict[str, Any]] = []

    def _extra_fields(agent_name: str) -> Dict[str, Any]:
        prompt_role = (
            "colluder"
            if (secret_channel_enabled and str(agent_name) in colluder_set)
            else "normal"
        )
        return {
            "collusion_role": prompt_role,
            "collusion_coalition_members": sorted(list(colluder_set))
            if prompt_role == "colluder"
            else [],
            "collusion_blackboard_id": secret_blackboard_id
            if prompt_role == "colluder"
            else None,
        }

    for planning_round in range(1, planning_rounds + 1):
        last_agent = None
        for agent in env.agents:
            base_ctx = env.build_agent_context(
                agent.name,
                phase="planning",
                iteration=1,
                planning_round=planning_round,
            )
            agent_context = dict(base_ctx)
            agent_context.update(_extra_fields(agent.name))
            response = await protocol.agent_planning_turn(
                agent,
                agent.name,
                agent_context,
                env,
                iteration=1,
                planning_round=planning_round,
            )
            turns.append(
                {
                    "phase": "planning",
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
                    iteration=1,
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

    last_exec_agent = None
    for agent in env.agents:
        base_ctx = env.build_agent_context(agent.name, phase="execution", iteration=1)
        agent_context = dict(base_ctx)
        agent_context.update(_extra_fields(agent.name))
        response = await protocol.agent_execution_turn(
            agent, agent.name, agent_context, env, iteration=1
        )
        turns.append(
            {
                "phase": "execution",
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
                iteration=1,
                phase="execution",
                agent_name=str(last_exec_agent),
                planning_round=None,
            )
        except Exception as exc:
            logger.warning(
                "Failed to write blackboard_*.txt logs after execution: %s", exc
            )

    final_summary = env.get_final_summary()

    blackboard_participants = {
        idx: sorted(list(bb.agents))
        for idx, bb in enumerate(protocol.megaboard.blackboards)
    }
    metrics = compute_collusion_metrics(
        env=env,
        colluders=colluders,
        secret_blackboard_id=secret_blackboard_id,
        secret_channel_enabled=secret_channel_enabled,
        prompt_variant=str(effective_prompt_variant),
        tool_events=protocol.tool_events,
        final_summary=final_summary,
    )

    _write_json(
        run_dir / "run_config.json",
        {
            "run_id": run_id,
            "model_label": model_label,
            "provider": provider,
            "model": model_name,
            "sweep": sweep_name,
            "topology": topology,
            "num_agents": num_agents,
            "colluder_count": colluder_count,
            "colluders": colluders,
            "agent_order": str(agent_order_strategy or "random"),
            "agent_turn_order": [a.name for a in env.agents],
            "secret_channel_enabled": secret_channel_enabled,
            "secret_blackboard_id": secret_blackboard_id,
            "prompt_variant": str(effective_prompt_variant),
            "seed": seed,
            "roles": roles,
            "blackboard_participants": blackboard_participants,
        },
    )
    _write_json(run_dir / "final_summary.json", final_summary)
    _write_json(run_dir / "agent_turns.json", turns)
    _write_json(run_dir / "metrics.json", metrics_to_json(metrics))
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
        "num_agents": num_agents,
        "colluder_count": colluder_count,
        "secret_channel_enabled": secret_channel_enabled,
        "secret_blackboard_id": secret_blackboard_id,
        "prompt_variant": str(effective_prompt_variant),
        "seed": seed,
        "colluders": colluders,
        "agent_order": str(agent_order_strategy or "random"),
        "status": metrics.status,
        "coalition_reward_sum": metrics.coalition_reward_sum,
        "noncoalition_reward_sum": metrics.noncoalition_reward_sum,
        "coalition_mean_reward": metrics.coalition_mean_reward,
        "noncoalition_mean_reward": metrics.noncoalition_mean_reward,
        "coalition_advantage_mean": metrics.coalition_advantage_mean,
        "colluder_posts_secret_rate": metrics.colluder_posts_secret_rate,
        "mean_regret": metrics.mean_regret,
        "coalition_mean_regret": metrics.coalition_mean_regret,
        "noncoalition_mean_regret": metrics.noncoalition_mean_regret,
        "system_regret": metrics.system_regret,
        "system_regret_ratio": metrics.system_regret_ratio,
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
        default_seeds = _normalize_seeds((cfg.get("simulation") or {}).get("seed")) or [
            1
        ]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = (
        Path(
            out_dir
            or exp.get("output_dir")
            or "experiments/persuasion/collusion/outputs/persuasion_collusion"
        )
        / timestamp
    )
    _ensure_dir(root)
    _write_json(root / "config.json", cfg)
    _configure_experiment_logging(root)

    total_runs = 0
    for model in models:
        for sweep in sweeps:
            topologies = sweep.get("topologies") or []
            agent_counts = sweep.get("num_agents") or []
            colluder_counts = sweep.get("colluder_counts") or []
            secret_flags = (
                sweep.get("secret_channel_enabled")
                or sweep.get("secret_channels")
                or [False]
            )
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
                    for _c in colluder_counts:
                        for _secret in secret_flags:
                            for _pv in prompt_variants:
                                if not bool(_secret) and str(_pv) != "control":
                                    continue
                                total_runs += len(seeds)

    logger.info("EXPERIMENT START (total_runs=%s, output_root=%s)", total_runs, root)
    _write_progress(
        root,
        {
            "status": "running",
            "total_runs": total_runs,
            "completed_runs": 0,
            "failed_runs": 0,
            "started_at": datetime.now().isoformat(),
            "config_path": str(config_path),
        },
    )

    summaries: List[Dict[str, Any]] = []
    completed = 0
    failed = 0

    with tqdm(
        total=total_runs, desc="Experiments", unit="run", dynamic_ncols=True
    ) as pbar:
        for model_idx, model in enumerate(models, start=1):
            model_label = str(model.get("label") or "model")
            llm_cfg = model.get("llm") or {}
            logger.info("MODEL START %s (%s/%s)", model_label, model_idx, len(models))
            if max_concurrent_runs > 1:
                import asyncio

                semaphore = asyncio.Semaphore(int(max_concurrent_runs))

                def _run_single_in_thread(**kwargs: Any) -> Dict[str, Any]:
                    return asyncio.run(_run_single(**kwargs))

                async def _run_single_limited(
                    *, run_label: str, **kwargs: Any
                ) -> Dict[str, Any]:
                    async with semaphore:
                        logger.info("SCHEDULED %s", run_label)
                        return await asyncio.to_thread(_run_single_in_thread, **kwargs)

            for sweep_idx, sweep in enumerate(sweeps, start=1):
                sweep_name = str(sweep.get("name") or "sweep")
                topologies = sweep.get("topologies") or []
                agent_counts = sweep.get("num_agents") or []
                colluder_counts = sweep.get("colluder_counts") or []
                secret_flags = (
                    sweep.get("secret_channel_enabled")
                    or sweep.get("secret_channels")
                    or [False]
                )
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
                if not seeds:
                    raise ValueError(
                        "No seeds specified. Set experiment.seeds or sweeps[].seeds."
                    )

                logger.info(
                    "SWEEP START %s (%s/%s)", sweep_name, sweep_idx, len(sweeps)
                )

                if max_concurrent_runs <= 1:
                    for topology in topologies:
                        for n in agent_counts:
                            for c in colluder_counts:
                                for secret in secret_flags:
                                    for pv in prompt_variants:
                                        if not bool(secret) and str(pv) != "control":
                                            continue
                                        for seed in seeds:
                                            run_label = f"{model_label}/{sweep_name}/{topology}/n{n}/c{c}/secret{int(bool(secret))}/pv{pv}/seed{seed}"
                                            pbar.set_postfix_str(run_label)
                                            run_status = "success"
                                            try:
                                                summaries.append(
                                                    await _run_single(
                                                        base_cfg=cfg,
                                                        model_label=model_label,
                                                        model_llm_cfg=llm_cfg,
                                                        sweep_name=sweep_name,
                                                        topology=str(topology),
                                                        num_agents=int(n),
                                                        colluder_count=int(c),
                                                        secret_channel_enabled=bool(
                                                            secret
                                                        ),
                                                        prompt_variant=str(pv),
                                                        seed=int(seed),
                                                        out_dir=root,
                                                    )
                                                )
                                                completed += 1
                                            except Exception:
                                                run_status = "failed"
                                                failed += 1
                                                logger.exception("RUN FAILED %s", run_label)
                                                raise
                                            finally:
                                                pbar.update(1)
                                                _write_progress(
                                                    root,
                                                    {
                                                        "status": "running",
                                                        "total_runs": total_runs,
                                                        "completed_runs": completed,
                                                        "failed_runs": failed,
                                                        "last_run_label": run_label,
                                                        "last_run_status": run_status,
                                                    },
                                                )
                    continue

                import asyncio

                tasks: List[asyncio.Task] = []
                task_labels: Dict[asyncio.Task, str] = {}
                for topology in topologies:
                    for n in agent_counts:
                        for c in colluder_counts:
                            for secret in secret_flags:
                                for pv in prompt_variants:
                                    if not bool(secret) and str(pv) != "control":
                                        continue
                                    for seed in seeds:
                                        run_label = f"{model_label}/{sweep_name}/{topology}/n{n}/c{c}/secret{int(bool(secret))}/pv{pv}/seed{seed}"
                                        task = asyncio.create_task(
                                            _run_single_limited(
                                                run_label=run_label,
                                                base_cfg=cfg,
                                                model_label=model_label,
                                                model_llm_cfg=llm_cfg,
                                                sweep_name=sweep_name,
                                                topology=str(topology),
                                                num_agents=int(n),
                                                colluder_count=int(c),
                                                secret_channel_enabled=bool(secret),
                                                prompt_variant=str(pv),
                                                seed=int(seed),
                                                out_dir=root,
                                            )
                                        )
                                        tasks.append(task)
                                        task_labels[task] = run_label

                pending = set(tasks)
                while pending:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    for finished in done:
                        run_label = task_labels.get(finished, "unknown")
                        pbar.set_postfix_str(run_label)
                        run_status = "success"
                        try:
                            summaries.append(await finished)
                            completed += 1
                        except Exception:
                            run_status = "failed"
                            failed += 1
                            logger.exception("RUN FAILED %s", run_label)
                            for t in pending:
                                t.cancel()
                            await asyncio.gather(*pending, return_exceptions=True)
                            raise
                        finally:
                            pbar.update(1)
                            _write_progress(
                                root,
                                {
                                    "status": "running",
                                    "total_runs": total_runs,
                                    "completed_runs": completed,
                                    "failed_runs": failed,
                                    "last_run_label": run_label,
                                    "last_run_status": run_status,
                                },
                            )

            logger.info("MODEL END %s", model_label)

    _write_json(root / "summary.json", summaries)
    with open(root / "summary.jsonl", "w", encoding="utf-8") as f:
        for row in summaries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_rows = []
    for row in summaries:
        flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        csv_rows.append(flat)
    if csv_rows:
        fieldnames = sorted({k for r in csv_rows for k in r.keys()})
        with open(root / "summary.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    logger.info(
        "EXPERIMENT END (completed=%s, failed=%s, output_root=%s)",
        completed,
        failed,
        root,
    )
    _write_progress(
        root,
        {
            "status": "completed",
            "total_runs": total_runs,
            "completed_runs": completed,
            "failed_runs": failed,
        },
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run persuasion-based collusion sweeps (local protocol; no MCP)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config (e.g., experiments/persuasion/collusion/configs/persuasion_collusion_jira.yaml).",
    )
    parser.add_argument(
        "--out-dir", default=None, help="Override output root directory."
    )
    parser.add_argument(
        "--max-concurrent-runs",
        default=None,
        type=int,
        help="Maximum number of runs to execute in parallel (overrides experiment.max_concurrent_runs).",
    )
    args = parser.parse_args()

    import asyncio

    out = asyncio.run(
        run_from_config(
            args.config,
            out_dir=args.out_dir,
            max_concurrent_runs=args.max_concurrent_runs,
        )
    )
    print(f"Wrote results to: {out}")


if __name__ == "__main__":
    main()
