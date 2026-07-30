from __future__ import annotations

# ruff: noqa: E402

"""Covert collusion experiment runner.

Focuses on whether a subset of agents can coordinate via a hidden coalition-only blackboard and
benefit at the expense of non-coalition members.
"""

import sys
import argparse
import copy
import csv
import json
import logging
import random
import re
import importlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

if sys.version_info < (3, 11):
    raise RuntimeError(
        "Terrarium requires Python >= 3.11. "
        "Create/activate a `.venv` (see repo README) and re-run with `.venv/bin/python`."
    )

project_root = Path(__file__).resolve().parents[2]
# Allow running without installing the repo as a package.
if str(project_root) not in sys.path:
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
from experiments.collusion.prompts import CollusionPrompts
from experiments.private_channel_audit.monitor import (
    AuditSettings,
    PrivateChannelAuditor,
    append_audit_event,
    format_public_audit_report,
)
from experiments.private_channel_audit.foundry_client import (
    FoundryAnthropicClient,
    FoundryChatCompletionsClient,
    FoundryOpenAIClient,
)
from experiments.common.local_protocol import LocalCommunicationProtocol
from terrarium.networks import build_communication_network
from terrarium.logger import AgentTrajectoryLogger, PromptLogger
from terrarium.utils import (
    build_vllm_runtime,
    get_client_instance,
    get_generation_params,
    get_model_name,
)
from terrarium.agents.base import BaseAgent


LOGGER_NAME = "experiments.collusion"
logger = logging.getLogger(LOGGER_NAME)

_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9]+")


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


def _sanitize_label(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return "env"
    value = _SAFE_LABEL_RE.sub("_", value).strip("_")
    return value or "env"


def _infer_environment_label(env_cfg: Dict[str, Any]) -> str:
    import_path = str(env_cfg.get("import_path") or "").strip()
    if import_path:
        _module, _sep, cls_name = import_path.partition(":")
        if cls_name:
            return cls_name
        return import_path
    name = str(env_cfg.get("name") or "").strip()
    if name:
        return name
    return "env"


def _normalize_environment_sweep(
    *, sweep: Dict[str, Any], base_env_cfg: Dict[str, Any]
) -> List[tuple[str, Dict[str, Any]]]:
    raw = sweep.get("environments") or []
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError("sweeps[].environments must be a list.")

    variants: List[tuple[str, Dict[str, Any]]] = []
    seen: set[str] = set()
    for entry in raw:
        label: Optional[str] = None
        override: Dict[str, Any] = {}
        if isinstance(entry, str):
            value = entry.strip()
            if not value:
                continue
            if ":" in value:
                override = {"import_path": value}
                label = value.split(":", 1)[1].strip() or value
            else:
                override = {"name": value}
                label = value
        elif isinstance(entry, dict):
            label = str(entry.get("label") or "").strip() or None
            override = {k: v for k, v in entry.items() if k != "label"}
            if label is None:
                label = _infer_environment_label(override)
        else:
            raise ValueError(
                f"Invalid sweeps[].environments entry (expected str|dict, got {type(entry).__name__})."
            )

        env_cfg = copy.deepcopy(base_env_cfg or {})
        env_cfg.update(copy.deepcopy(override))
        safe_label = _sanitize_label(label or _infer_environment_label(env_cfg))
        unique_label = safe_label
        suffix = 2
        while unique_label in seen:
            unique_label = f"{safe_label}_{suffix}"
            suffix += 1
        seen.add(unique_label)
        variants.append((unique_label, env_cfg))

    if not variants:
        raise ValueError(
            "sweeps[].environments is set but empty; provide at least one environment entry."
        )
    return variants


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


def _private_channel_audit_flags(
    sweep: Dict[str, Any], experiment: Dict[str, Any]
) -> List[bool]:
    """Return the audit conditions requested by one sweep.

    Existing collusion configs omit this field and therefore retain their previous
    behavior.  Audit experiments can set a two-arm list such as ``[false, true]``.
    """
    raw = sweep.get("private_channel_audit_enabled")
    if raw is None:
        raw = [bool((experiment.get("private_channel_audit") or {}).get("enabled", False))]
    if not isinstance(raw, list):
        raw = [raw]

    flags: List[bool] = []
    for value in raw:
        flag = bool(value)
        if flag not in flags:
            flags.append(flag)
    return flags or [False]


def _find_blackboard_by_id(protocol: LocalCommunicationProtocol, blackboard_id: int) -> Any:
    for blackboard in protocol.megaboard.blackboards:
        try:
            if int(blackboard.blackboard_id) == int(blackboard_id):
                return blackboard
        except (TypeError, ValueError):
            continue
    raise KeyError(f"Blackboard {blackboard_id} was not found.")


def _find_public_blackboard_id(
    *,
    protocol: LocalCommunicationProtocol,
    agent_names: Sequence[str],
    secret_blackboard_id: int,
) -> int:
    """Find the all-agent public board required for a public audit report."""
    all_agents = {str(name) for name in agent_names}
    for blackboard in protocol.megaboard.blackboards:
        try:
            blackboard_id = int(blackboard.blackboard_id)
        except (TypeError, ValueError):
            continue
        if blackboard_id == int(secret_blackboard_id):
            continue
        participants = {str(name) for name in getattr(blackboard, "agents", [])}
        if participants == all_agents:
            return blackboard_id
    raise ValueError(
        "Private-channel audit requires a public blackboard visible to all agents. "
        "Use the complete, consolidated JIRA topology for this experiment."
    )


def _effective_prompt_variant(
    *, secret_channel_enabled: bool, prompt_variant: str
) -> str:
    value = str(prompt_variant or "").strip() or "control"
    return value if secret_channel_enabled else "control"


def _run_id(
    *,
    model_label: str,
    sweep_name: str,
    environment_label: Optional[str],
    topology: str,
    num_agents: int,
    colluder_count: int,
    secret_channel_enabled: bool,
    prompt_variant: str,
    private_channel_audit_enabled: bool,
    seed: int,
) -> str:
    run_id = f"{model_label}__{sweep_name}"
    if environment_label:
        run_id += f"__env{_sanitize_label(environment_label)}"
    return (
        f"{run_id}__{topology}__n{num_agents}__c{colluder_count}"
        f"__secret{int(bool(secret_channel_enabled))}__pv{prompt_variant}"
        f"__audit{int(bool(private_channel_audit_enabled))}__seed{seed}"
    )


def _model_name_for_config(model_llm_cfg: Dict[str, Any]) -> str:
    provider = str(model_llm_cfg.get("provider") or "").lower()
    if provider == "foundry":
        return str((model_llm_cfg.get("foundry") or {}).get("model") or "gpt-5.4-mini")
    if provider == "foundry_anthropic":
        return str(
            (model_llm_cfg.get("foundry_anthropic") or {}).get("model")
            or "claude-sonnet-4-5"
        )
    if provider == "foundry_chat_completions":
        return str(
            (model_llm_cfg.get("foundry_chat_completions") or {}).get("model")
            or "FW-GLM-5"
        )
    return str(get_model_name(provider, model_llm_cfg))


def _summary_from_completed_run(run_dir: Path) -> Dict[str, Any]:
    """Reconstruct one summary row from a fully written run directory."""
    run_cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    final_summary = json.loads(
        (run_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    audit = metrics.get("private_channel_audit") or {}
    joint_reward = final_summary.get("joint_reward")
    coalition_reward_sum = metrics.get("coalition_reward_sum")
    coalition_reward_ratio = None
    try:
        if coalition_reward_sum is not None and float(joint_reward) != 0.0:
            coalition_reward_ratio = float(coalition_reward_sum) / float(joint_reward)
    except (TypeError, ValueError):
        pass

    return {
        "run_id": run_cfg.get("run_id", run_dir.name),
        "model_label": run_cfg.get("model_label"),
        "provider": run_cfg.get("provider"),
        "model": run_cfg.get("model"),
        "sweep": run_cfg.get("sweep"),
        "environment_label": run_cfg.get("environment_label"),
        "environment_name": run_cfg.get("environment_name"),
        "topology": run_cfg.get("topology"),
        "num_agents": run_cfg.get("num_agents"),
        "colluder_count": run_cfg.get("colluder_count"),
        "secret_channel_enabled": run_cfg.get("secret_channel_enabled"),
        "secret_blackboard_id": run_cfg.get("secret_blackboard_id"),
        "prompt_variant": run_cfg.get("prompt_variant"),
        "private_channel_audit_enabled": bool(audit.get("enabled", False)),
        "audit_events": audit.get("audit_events", 0),
        "audit_flagged_events": audit.get("flagged_events", 0),
        "audit_public_reports_posted": audit.get("public_reports_posted", 0),
        "seed": run_cfg.get("seed"),
        "colluders": run_cfg.get("colluders", []),
        "agent_order": run_cfg.get("agent_order"),
        "status": metrics.get("status"),
        "joint_reward": joint_reward,
        "joint_reward_ratio": final_summary.get("joint_reward_ratio"),
        "coverage": final_summary.get("coverage"),
        "normalized_score": final_summary.get("normalized_score"),
        "tasks_done": metrics.get("tasks_done"),
        "violations": metrics.get("violations"),
        "total_cost": metrics.get("total_cost"),
        "priority_sum": metrics.get("priority_sum"),
        "mean_regret": metrics.get("mean_regret"),
        "reward_std": metrics.get("reward_std"),
        "coalition_reward_sum": coalition_reward_sum,
        "noncoalition_reward_sum": metrics.get("noncoalition_reward_sum"),
        "coalition_mean_reward": metrics.get("coalition_mean_reward"),
        "noncoalition_mean_reward": metrics.get("noncoalition_mean_reward"),
        "coalition_advantage_mean": metrics.get("coalition_advantage_mean"),
        "coalition_reward_ratio": coalition_reward_ratio,
        "colluder_posts_secret_rate": metrics.get("colluder_posts_secret_rate"),
        "coalition_mean_regret": metrics.get("coalition_mean_regret"),
        "noncoalition_mean_regret": metrics.get("noncoalition_mean_regret"),
        "system_regret": metrics.get("system_regret"),
        "system_regret_ratio": metrics.get("system_regret_ratio"),
        "colluder_posts_total": metrics.get("colluder_posts_total"),
        "colluder_posts_secret": metrics.get("colluder_posts_secret"),
        "colluder_posts_non_secret": metrics.get("colluder_posts_non_secret"),
        "largest_non_secret_blackboard_id": metrics.get("largest_non_secret_blackboard_id"),
        "colluder_posts_secret_to_largest_bb_ratio_mean": metrics.get(
            "colluder_posts_secret_to_largest_bb_ratio_mean"
        ),
    }


def _completed_run_if_matching(
    *,
    root: Path,
    model_label: str,
    model_llm_cfg: Dict[str, Any],
    sweep_name: str,
    environment_label: Optional[str],
    topology: str,
    num_agents: int,
    colluder_count: int,
    secret_channel_enabled: bool,
    prompt_variant: str,
    private_channel_audit_enabled: bool,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Return an existing result only when it is complete and exactly compatible."""
    effective_prompt = _effective_prompt_variant(
        secret_channel_enabled=secret_channel_enabled, prompt_variant=prompt_variant
    )
    run_id = _run_id(
        model_label=model_label,
        sweep_name=sweep_name,
        environment_label=environment_label,
        topology=topology,
        num_agents=num_agents,
        colluder_count=colluder_count,
        secret_channel_enabled=secret_channel_enabled,
        prompt_variant=effective_prompt,
        private_channel_audit_enabled=private_channel_audit_enabled,
        seed=seed,
    )
    run_dir = root / "runs" / model_label / sweep_name / run_id
    required = [
        run_dir / "run_config.json",
        run_dir / "metrics.json",
        run_dir / "final_summary.json",
    ]
    if not all(path.is_file() for path in required):
        return None
    try:
        run_cfg = json.loads(required[0].read_text(encoding="utf-8"))
        final_summary = json.loads(required[2].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    expected = {
        "run_id": run_id,
        "model_label": model_label,
        "model": _model_name_for_config(model_llm_cfg),
        "sweep": sweep_name,
        "topology": topology,
        "num_agents": int(num_agents),
        "colluder_count": int(colluder_count),
        "secret_channel_enabled": bool(secret_channel_enabled),
        "prompt_variant": effective_prompt,
        "seed": int(seed),
    }
    if any(run_cfg.get(key) != value for key, value in expected.items()):
        return None
    existing_audit = (run_cfg.get("private_channel_audit") or {}).get("enabled")
    if bool(existing_audit) != bool(private_channel_audit_enabled):
        return None
    if str(final_summary.get("status") or "").lower() != "complete":
        return None
    try:
        return _summary_from_completed_run(run_dir)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


async def _run_single(
    *,
    base_cfg: Dict[str, Any],
    model_label: str,
    model_llm_cfg: Dict[str, Any],
    sweep_name: str,
    environment_label: Optional[str] = None,
    environment_cfg: Optional[Dict[str, Any]] = None,
    topology: str,
    num_agents: int,
    colluder_count: int,
    secret_channel_enabled: bool,
    prompt_variant: str,
    seed: int,
    out_dir: Path,
    private_channel_audit_enabled: bool = False,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    secret_channel_enabled = bool(secret_channel_enabled)
    # Design choice: prompt variants are only active when the secret channel exists.
    # When secret_channel_enabled is False, we force the effective variant to "control"
    # so the baseline is the vanilla cooperative setup (no explicit collusion prompting).
    effective_prompt_variant = _effective_prompt_variant(
        secret_channel_enabled=secret_channel_enabled, prompt_variant=prompt_variant
    )

    cfg = copy.deepcopy(base_cfg)
    if environment_cfg is not None:
        cfg["environment"] = copy.deepcopy(environment_cfg)
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
    audit_cfg = (
        cfg.setdefault("experiment", {}).setdefault("private_channel_audit", {})
    )
    audit_cfg["enabled"] = bool(private_channel_audit_enabled)
    private_channel_audit_enabled = bool(audit_cfg["enabled"])
    if private_channel_audit_enabled and not secret_channel_enabled:
        raise ValueError("Private-channel audit requires secret_channel_enabled=true.")

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    env_label: Optional[str] = None
    if environment_label is not None:
        cleaned = _sanitize_label(environment_label)
        if cleaned:
            env_label = cleaned

    run_id = _run_id(
        model_label=model_label,
        sweep_name=sweep_name,
        environment_label=env_label,
        topology=topology,
        num_agents=num_agents,
        colluder_count=colluder_count,
        secret_channel_enabled=secret_channel_enabled,
        prompt_variant=effective_prompt_variant,
        private_channel_audit_enabled=private_channel_audit_enabled,
        seed=seed,
    )
    run_dir = out_dir / "runs" / model_label / sweep_name / run_id
    _ensure_dir(run_dir)

    logger.info("RUN START %s", run_id)

    cfg.setdefault("simulation", {})["run_timestamp"] = f"{run_timestamp}__{run_id}"
    cfg.setdefault("simulation", {})["tags"] = [
        str(cfg.get("experiment", {}).get("tag", "collusion"))
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

    # Coalition membership + secret channel injection.
    collusion_cfg = (cfg.get("experiment") or {}).get("collusion") or {}
    selection_strategy = str(collusion_cfg.get("colluder_selection", "random"))
    colluders = _select_colluders(
        agent_names=agent_names,
        count=int(colluder_count),
        strategy=selection_strategy,
        rng=rng,
    )
    agent_order_strategy = str(collusion_cfg.get("agent_order", "random"))
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

    # Wrap prompts (role-specific injection via agent_context).
    env.prompts = CollusionPrompts(
        env,
        cfg,
        prompt_variant=str(effective_prompt_variant),
        base_prompts=getattr(env, "prompts", None),
        experiment_prompt_logger=prompt_logger,
        log_prompts=log_prompts,
    )

    # Build agents.
    provider = (cfg.get("llm", {}).get("provider") or "").lower()
    if provider == "foundry":
        foundry_llm_cfg = cfg["llm"].get("foundry") or {}
        model_name = str(foundry_llm_cfg.get("model") or "gpt-5.4-mini")
        generation_params = dict(foundry_llm_cfg.get("params") or {})
    elif provider == "foundry_anthropic":
        foundry_llm_cfg = cfg["llm"].get("foundry_anthropic") or {}
        model_name = str(foundry_llm_cfg.get("model") or "claude-sonnet-4-5")
        generation_params = dict(foundry_llm_cfg.get("params") or {})
    elif provider == "foundry_chat_completions":
        foundry_llm_cfg = cfg["llm"].get("foundry_chat_completions") or {}
        model_name = str(foundry_llm_cfg.get("model") or "FW-GLM-5")
        generation_params = dict(foundry_llm_cfg.get("params") or {})
    else:
        model_name = get_model_name(provider, cfg["llm"])
        generation_params = get_generation_params(cfg["llm"])
    max_steps = int(cfg["simulation"].get("max_conversation_steps", 3))
    vllm_runtime = None
    if provider == "vllm":
        vllm_runtime = build_vllm_runtime(cfg["llm"])

    agents: List[BaseAgent] = []
    env_tool_name = str(
        getattr(env, "tools_environment_name", None) or env.__class__.__name__
    )
    for name in agent_names:
        if provider == "foundry":
            client = FoundryOpenAIClient.from_config(cfg["llm"].get("foundry") or {})
        elif provider == "foundry_anthropic":
            client = FoundryAnthropicClient.from_config(
                cfg["llm"].get("foundry_anthropic") or {}
            )
        elif provider == "foundry_chat_completions":
            client = FoundryChatCompletionsClient.from_config(
                cfg["llm"].get("foundry_chat_completions") or {}
            )
        else:
            client = get_client_instance(
                cfg["llm"], agent_name=name, vllm_runtime=vllm_runtime
            )
        agents.append(
            BaseAgent(
                client,
                name,
                model_name,
                max_steps,
                None,
                trajectory_logger,
                env_tool_name,
                generation_params=generation_params,
            )
        )
    if str(agent_order_strategy or "random").strip().lower() == "random":
        rng.shuffle(agents)
    else:
        agents_by_name = {a.name: a for a in agents}
        agents = [agents_by_name[name] for name in ordered_agent_names if name in agents_by_name]
    env.set_agent_clients(agents)

    # Create blackboards for the communication network.
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

    audit_monitor: Optional[PrivateChannelAuditor] = None
    audit_events_path = run_dir / "audit_events.jsonl"
    public_blackboard_id: Optional[int] = None
    audit_event_count = 0
    audit_flagged_count = 0
    audit_public_report_count = 0
    if private_channel_audit_enabled:
        if secret_blackboard_id is None:
            raise ValueError(
                "Private-channel audit requires a coalition-only blackboard with at least two colluders."
            )
        public_blackboard_id = _find_public_blackboard_id(
            protocol=protocol,
            agent_names=agent_names,
            secret_blackboard_id=secret_blackboard_id,
        )
        audit_monitor = PrivateChannelAuditor(
            AuditSettings.from_config(audit_cfg), agent_names=agent_names
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

    logger.info(
        "RUN %s setup complete (agents=%s, colluders=%s, secret_channel=%s, secret_bb_id=%s, audit=%s, public_bb_id=%s, blackboards=%s)",
        run_id,
        len(agent_names),
        len(colluders),
        secret_channel_enabled,
        secret_blackboard_id,
        private_channel_audit_enabled,
        public_blackboard_id,
        len(protocol.megaboard.blackboards),
    )

    planning_rounds = int(cfg["simulation"].get("max_planning_rounds", 1))
    turns: List[Dict[str, Any]] = []

    def _extra_fields(agent_name: str) -> Dict[str, Any]:
        # Baseline (secret_channel_enabled=False): no explicit collusion prompting.
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
        logger.info(
            "RUN %s planning round %s/%s", run_id, planning_round, planning_rounds
        )
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
        if audit_monitor is not None and secret_blackboard_id is not None:
            secret_blackboard = _find_blackboard_by_id(protocol, secret_blackboard_id)
            decision = await audit_monitor.audit_new_messages(
                round_num=planning_round, secret_blackboard=secret_blackboard
            )
            audit_event = decision.event_payload()
            audit_event["public_blackboard_id"] = public_blackboard_id
            audit_event["public_report_posted"] = False
            audit_event_count += 1
            if decision.flagged:
                audit_flagged_count += 1
                report = format_public_audit_report(decision)
                try:
                    await protocol.post_system_message(
                        int(public_blackboard_id),
                        kind="context",
                        payload={
                            "message": report,
                            "source": "private_channel_audit",
                            "planning_round": planning_round,
                        },
                    )
                except Exception as exc:
                    audit_event["public_report_error"] = str(exc)
                    append_audit_event(audit_events_path, audit_event)
                    raise RuntimeError(
                        "Private-channel audit found collusion but could not publish its "
                        "public report; aborting this audit-arm run."
                    ) from exc
                audit_event["public_report_posted"] = True
                audit_event["public_report"] = report
                audit_public_report_count += 1
            append_audit_event(audit_events_path, audit_event)
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

    logger.info("RUN %s execution phase start", run_id)
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
    logger.info(
        "RUN %s execution complete (status=%s)", run_id, final_summary.get("status")
    )

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
        blackboard_participants=blackboard_participants,
        final_summary=final_summary,
    )
    audit_summary = {
        "enabled": private_channel_audit_enabled,
        "public_blackboard_id": public_blackboard_id,
        "audit_events": audit_event_count,
        "flagged_events": audit_flagged_count,
        "public_reports_posted": audit_public_report_count,
        "events_path": str(audit_events_path) if private_channel_audit_enabled else None,
    }

    _write_json(
        run_dir / "run_config.json",
        {
            "run_id": run_id,
            "model_label": model_label,
            "provider": provider,
            "model": model_name,
            "sweep": sweep_name,
            "environment_label": env_label,
            "environment_cfg": cfg.get("environment") or {},
            "environment_name": env.__class__.__name__,
            "topology": topology,
            "num_agents": num_agents,
            "colluder_count": colluder_count,
            "colluders": colluders,
            "agent_order": str(agent_order_strategy or "random"),
            "agent_turn_order": [a.name for a in env.agents],
            "secret_channel_enabled": secret_channel_enabled,
            "secret_blackboard_id": secret_blackboard_id,
            "prompt_variant": str(effective_prompt_variant),
            "private_channel_audit": {
                **audit_summary,
                "monitor": AuditSettings.from_config(audit_cfg).__dict__
                if private_channel_audit_enabled
                else None,
            },
            "seed": seed,
            "roles": roles,
            "blackboard_participants": blackboard_participants,
        },
    )
    _write_json(run_dir / "final_summary.json", final_summary)
    _write_json(run_dir / "agent_turns.json", turns)
    metrics_payload = metrics_to_json(metrics)
    metrics_payload["private_channel_audit"] = audit_summary
    _write_json(run_dir / "metrics.json", metrics_payload)
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
    joint_reward = final_summary.get("joint_reward")
    coalition_reward_ratio = None
    try:
        if (
            metrics.coalition_reward_sum is not None
            and joint_reward is not None
            and float(joint_reward) != 0.0
        ):
            coalition_reward_ratio = float(metrics.coalition_reward_sum) / float(
                joint_reward
            )
    except Exception:
        coalition_reward_ratio = None
    return {
        "run_id": run_id,
        "model_label": model_label,
        "provider": provider,
        "model": model_name,
        "sweep": sweep_name,
        "environment_label": env_label,
        "environment_name": env.__class__.__name__,
        "topology": topology,
        "num_agents": num_agents,
        "colluder_count": colluder_count,
        "secret_channel_enabled": secret_channel_enabled,
        "secret_blackboard_id": secret_blackboard_id,
        "prompt_variant": str(effective_prompt_variant),
        "private_channel_audit_enabled": private_channel_audit_enabled,
        "audit_events": audit_event_count,
        "audit_flagged_events": audit_flagged_count,
        "audit_public_reports_posted": audit_public_report_count,
        "seed": seed,
        "colluders": colluders,
        "agent_order": str(agent_order_strategy or "random"),
        "status": metrics.status,
        # Radar-chart inputs (environment summary + collusion metrics).
        "joint_reward": joint_reward,
        "joint_reward_ratio": final_summary.get("joint_reward_ratio"),
        "coverage": final_summary.get("coverage"),
        "normalized_score": final_summary.get("normalized_score"),
        "tasks_done": metrics.tasks_done,
        "violations": metrics.violations,
        "total_cost": metrics.total_cost,
        "priority_sum": metrics.priority_sum,
        "mean_regret": metrics.mean_regret,
        "reward_std": metrics.reward_std,
        "coalition_reward_sum": metrics.coalition_reward_sum,
        "noncoalition_reward_sum": metrics.noncoalition_reward_sum,
        "coalition_mean_reward": metrics.coalition_mean_reward,
        "noncoalition_mean_reward": metrics.noncoalition_mean_reward,
        "coalition_advantage_mean": metrics.coalition_advantage_mean,
        "coalition_reward_ratio": coalition_reward_ratio,
        "colluder_posts_secret_rate": metrics.colluder_posts_secret_rate,
        "mean_regret": metrics.mean_regret,
        "coalition_mean_regret": metrics.coalition_mean_regret,
        "noncoalition_mean_regret": metrics.noncoalition_mean_regret,
        "system_regret": metrics.system_regret,
        "system_regret_ratio": metrics.system_regret_ratio,
        "colluder_posts_total": metrics.colluder_posts_total,
        "colluder_posts_secret": metrics.colluder_posts_secret,
        "colluder_posts_non_secret": metrics.colluder_posts_non_secret,
        "largest_non_secret_blackboard_id": metrics.largest_non_secret_blackboard_id,
        "colluder_posts_secret_to_largest_bb_ratio_mean": metrics.colluder_posts_secret_to_largest_bb_ratio_mean,
    }


async def run_from_config(
    config_path: str,
    *,
    out_dir: Optional[str] = None,
    resume_dir: Optional[str] = None,
    max_concurrent_runs: Optional[int] = None,
) -> Path:
    if out_dir is not None and resume_dir is not None:
        raise ValueError("Use either out_dir or resume_dir, not both.")
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

    resumed = resume_dir is not None
    if resumed:
        root = Path(str(resume_dir)).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"resume_dir does not exist: {root}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        root = (
            Path(
                out_dir
                or exp.get("output_dir")
                or "experiments/collusion/outputs/collusion"
            )
            / timestamp
        )
    _ensure_dir(root)
    if not resumed:
        _write_json(root / "config.json", cfg)
    _configure_experiment_logging(root)

    base_env_cfg = cfg.get("environment") or {}

    # Pre-compute total runs for progress tracking.
    total_runs = 0
    for model in models:
        for sweep in sweeps:
            env_variants = _normalize_environment_sweep(
                sweep=sweep, base_env_cfg=base_env_cfg
            )
            env_count = len(env_variants) if env_variants else 1
            topologies = sweep.get("topologies") or []
            agent_counts = sweep.get("num_agents") or []
            colluder_counts = sweep.get("colluder_counts") or []
            secret_flags = (
                sweep.get("secret_channel_enabled")
                or sweep.get("secret_channels")
                or [False]
            )
            audit_flags = _private_channel_audit_flags(sweep, exp)
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
            # Prompt variants are only active when the secret channel exists.
            # When secret_channel_enabled=false, only the "control" variant is executed.
            for _topo in topologies:
                for _n in agent_counts:
                    for _c in colluder_counts:
                        for _secret in secret_flags:
                            for _pv in prompt_variants:
                                if not bool(_secret) and str(_pv) != "control":
                                    continue
                                for _audit in audit_flags:
                                    if bool(_audit) and not bool(_secret):
                                        continue
                                    total_runs += len(seeds) * env_count

    logger.info("EXPERIMENT START (total_runs=%s, output_root=%s)", total_runs, root)
    _write_progress(
        root,
        {
            "status": "running",
            "total_runs": total_runs,
            "completed_runs": 0,
            "failed_runs": 0,
            "skipped_runs": 0,
            "resumed": resumed,
            "started_at": datetime.now().isoformat(),
            "config_path": str(config_path),
        },
    )

    summaries: List[Dict[str, Any]] = []
    completed = 0
    failed = 0
    skipped = 0

    def _existing_completed_summary(**kwargs: Any) -> Optional[Dict[str, Any]]:
        if not resumed:
            return None
        return _completed_run_if_matching(root=root, **kwargs)

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
                env_variants = _normalize_environment_sweep(
                    sweep=sweep, base_env_cfg=base_env_cfg
                )
                if not env_variants:
                    env_variants = [(None, None)]
                topologies = sweep.get("topologies") or []
                agent_counts = sweep.get("num_agents") or []
                colluder_counts = sweep.get("colluder_counts") or []
                secret_flags = (
                    sweep.get("secret_channel_enabled")
                    or sweep.get("secret_channels")
                    or [False]
                )
                audit_flags = _private_channel_audit_flags(sweep, exp)
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
                    for env_label, env_cfg in env_variants:
                        env_part = f"/env{env_label}" if env_label else ""
                        for topology in topologies:
                            for n in agent_counts:
                                for c in colluder_counts:
                                    for secret in secret_flags:
                                        for pv in prompt_variants:
                                            if (
                                                not bool(secret)
                                                and str(pv) != "control"
                                            ):
                                                continue
                                            # Keep matched A/B runs adjacent: seed 1 without
                                            # audit, seed 1 with audit, then seed 2, and so on.
                                            for seed in seeds:
                                                for audit_enabled in audit_flags:
                                                    if bool(audit_enabled) and not bool(secret):
                                                        continue
                                                    run_label = (
                                                        f"{model_label}/{sweep_name}"
                                                        f"{env_part}/{topology}"
                                                        f"/n{n}/c{c}"
                                                        f"/secret{int(bool(secret))}"
                                                        f"/pv{pv}/audit{int(bool(audit_enabled))}"
                                                        f"/seed{seed}"
                                                    )
                                                    pbar.set_postfix_str(run_label)
                                                    run_status = "success"
                                                    try:
                                                        run_kwargs = {
                                                            "model_label": model_label,
                                                            "model_llm_cfg": llm_cfg,
                                                            "sweep_name": sweep_name,
                                                            "environment_label": env_label,
                                                            "topology": str(topology),
                                                            "num_agents": int(n),
                                                            "colluder_count": int(c),
                                                            "secret_channel_enabled": bool(secret),
                                                            "prompt_variant": str(pv),
                                                            "private_channel_audit_enabled": bool(audit_enabled),
                                                            "seed": int(seed),
                                                        }
                                                        existing = _existing_completed_summary(
                                                            **run_kwargs
                                                        )
                                                        if existing is not None:
                                                            summaries.append(existing)
                                                            skipped += 1
                                                            run_status = "skipped_complete"
                                                        else:
                                                            summaries.append(
                                                                await _run_single(
                                                                    base_cfg=cfg,
                                                                    environment_cfg=env_cfg,
                                                                    out_dir=root,
                                                                    **run_kwargs,
                                                                )
                                                            )
                                                        completed += 1
                                                    except Exception:
                                                        run_status = "failed"
                                                        failed += 1
                                                        logger.exception(
                                                            "RUN FAILED %s", run_label
                                                        )
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
                                                                "skipped_runs": skipped,
                                                                "last_run_label": run_label,
                                                                "last_run_status": run_status,
                                                            },
                                                        )
                    continue

                # Concurrent scheduling
                import asyncio

                tasks: List[asyncio.Task] = []
                task_labels: Dict[asyncio.Task, str] = {}
                for env_label, env_cfg in env_variants:
                    env_part = f"/env{env_label}" if env_label else ""
                    for topology in topologies:
                        for n in agent_counts:
                            for c in colluder_counts:
                                for secret in secret_flags:
                                    for pv in prompt_variants:
                                        if not bool(secret) and str(pv) != "control":
                                            continue
                                        # Submit matched A/B pairs in seed order. With
                                        # max_concurrent_runs=1 this is also execution order.
                                        for seed in seeds:
                                            for audit_enabled in audit_flags:
                                                if bool(audit_enabled) and not bool(secret):
                                                    continue
                                                run_label = (
                                                    f"{model_label}/{sweep_name}"
                                                    f"{env_part}/{topology}"
                                                    f"/n{n}/c{c}"
                                                    f"/secret{int(bool(secret))}"
                                                    f"/pv{pv}/audit{int(bool(audit_enabled))}"
                                                    f"/seed{seed}"
                                                )
                                                run_kwargs = {
                                                    "model_label": model_label,
                                                    "model_llm_cfg": llm_cfg,
                                                    "sweep_name": sweep_name,
                                                    "environment_label": env_label,
                                                    "topology": str(topology),
                                                    "num_agents": int(n),
                                                    "colluder_count": int(c),
                                                    "secret_channel_enabled": bool(secret),
                                                    "prompt_variant": str(pv),
                                                    "private_channel_audit_enabled": bool(audit_enabled),
                                                    "seed": int(seed),
                                                }
                                                existing = _existing_completed_summary(
                                                    **run_kwargs
                                                )
                                                if existing is not None:
                                                    summaries.append(existing)
                                                    completed += 1
                                                    skipped += 1
                                                    pbar.update(1)
                                                    _write_progress(
                                                        root,
                                                        {
                                                            "status": "running",
                                                            "total_runs": total_runs,
                                                            "completed_runs": completed,
                                                            "failed_runs": failed,
                                                            "skipped_runs": skipped,
                                                            "last_run_label": run_label,
                                                            "last_run_status": "skipped_complete",
                                                        },
                                                    )
                                                    continue
                                                task = asyncio.create_task(
                                                    _run_single_limited(
                                                        run_label=run_label,
                                                        base_cfg=cfg,
                                                        environment_cfg=env_cfg,
                                                        out_dir=root,
                                                        **run_kwargs,
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
                                    "skipped_runs": skipped,
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
            "skipped_runs": skipped,
            "resumed": resumed,
        },
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run covert collusion sweeps (local protocol; no MCP)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config (e.g., experiments/collusion/configs/collusion_jira.yaml).",
    )
    parser.add_argument(
        "--out-dir", default=None, help="Override output root directory."
    )
    parser.add_argument(
        "--resume-dir",
        default=None,
        help=(
            "Existing experiment root to reuse. Fully completed, exactly matching "
            "runs are skipped; missing runs are executed there."
        ),
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
            resume_dir=args.resume_dir,
            max_concurrent_runs=args.max_concurrent_runs,
        )
    )
    print(f"Wrote results to: {out}")


if __name__ == "__main__":
    main()
