from __future__ import annotations

import argparse
import json
import logging
import math
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


logger = logging.getLogger(__name__)


PRIORITY_WEIGHTS: Dict[str, float] = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "critical": 1.0,
}

DEFAULT_MICROTASK_TYPES = ["implement", "review", "test", "docs", "triage"]
DEFAULT_SKILL_TAGS = [
    "backend",
    "frontend",
    "infrastructure",
    "machine-learning",
    "security",
    "data-science",
    "api-development",
    "ui-ux-design",
    "devops",
    "mobile-development",
    "testing",
    "documentation",
]


def _safe_load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        logger.debug("Failed to parse JSON: %s", path, exc_info=True)
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except Exception:
        return None


def _priority_weight(label: Any) -> float:
    key = str(label or "medium").strip().lower()
    return float(PRIORITY_WEIGHTS.get(key, PRIORITY_WEIGHTS["medium"]))


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    if (root / "runs").exists():
        for cfg_path in (root / "runs").rglob("run_config.json"):
            yield cfg_path.parent
        return

    for cfg_path in root.rglob("run_config.json"):
        yield cfg_path.parent


def _find_nearest_config_json(start_dir: Path) -> Optional[Path]:
    for parent in [start_dir, *start_dir.parents]:
        candidate = parent / "config.json"
        if candidate.exists():
            return candidate
    return None


@dataclass(frozen=True)
class JiraWeights:
    tasks_done_bonus: float
    priority_bonus: float
    violation_penalty: float


def _load_weights(run_dir: Path, overrides: JiraWeights | None) -> JiraWeights:
    if overrides is not None:
        return overrides

    run_cfg = _safe_load_json(run_dir / "run_config.json")
    if not isinstance(run_cfg, dict):
        run_cfg = {}
    run_env_cfg = run_cfg.get("environment_cfg")
    if not isinstance(run_env_cfg, dict):
        run_env_cfg = {}

    cfg_path = _find_nearest_config_json(run_dir)
    cfg = _safe_load_json(cfg_path) if cfg_path else None
    cfg_env = cfg.get("environment") if isinstance(cfg, dict) else None
    if not isinstance(cfg_env, dict):
        cfg_env = {}
    env_cfg = run_env_cfg or cfg_env

    tasks_done_bonus = _as_float(env_cfg.get("tasks_done_bonus"))
    priority_bonus = _as_float(env_cfg.get("priority_bonus"))
    violation_penalty = _as_float(env_cfg.get("violation_penalty"))

    # Mirror JiraTicketEnvironment defaults in envs/dcops/jira_ticket/jira_ticket_env.py.
    return JiraWeights(
        tasks_done_bonus=max(0.0, tasks_done_bonus if tasks_done_bonus is not None else 20.0),
        priority_bonus=max(0.0, priority_bonus if priority_bonus is not None else 20.0),
        violation_penalty=violation_penalty if violation_penalty is not None else 20.0,
    )


@dataclass(frozen=True)
class JiraInstance:
    agent_names: List[str]
    task_ids: List[str]
    tasks: Dict[str, Dict[str, Any]]
    costs: Dict[str, Dict[str, float]]


SECTION_HEADER_RE = re.compile(r"^=== (.+?) ===\s*$")


def _extract_section(text: str, header: str) -> List[str]:
    lines = text.splitlines()
    collecting = False
    collected: List[str] = []
    for line in lines:
        if line.strip() == header:
            collecting = True
            continue
        if collecting:
            if SECTION_HEADER_RE.match(line.strip()):
                break
            collected.append(line.rstrip("\n"))
    return collected


def _parse_tasks(task_lines: List[str]) -> Dict[str, Dict[str, Any]]:
    tasks: Dict[str, Dict[str, Any]] = {}
    for line in task_lines:
        line = line.strip()
        if not line.startswith("- "):
            continue
        payload = line[2:]
        if ": " not in payload:
            continue
        task_id, rest = payload.split(": ", 1)
        parts = [p.strip() for p in rest.split(" | ") if p.strip()]
        title = parts[0] if parts else ""
        fields: Dict[str, str] = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            fields[k.strip()] = v.strip()

        priority = fields.get("priority")
        effort = _as_float(fields.get("effort"))
        tags_raw = fields.get("tags")
        tags: List[str] = []
        if tags_raw:
            raw = tags_raw.strip()
            if raw.startswith("[") and raw.endswith("]"):
                raw = raw[1:-1]
            tags = [t.strip() for t in raw.split(",") if t.strip()]

        tasks[str(task_id).strip()] = {
            "id": str(task_id).strip(),
            "title": title,
            "priority": priority,
            "effort": effort,
            "tags": tags,
            "raw": rest,
        }
    return tasks


def _parse_costs(cost_lines: List[str]) -> Dict[str, float]:
    costs: Dict[str, float] = {}
    for line in cost_lines:
        line = line.strip()
        if not line.startswith("- "):
            continue
        payload = line[2:]
        if ": " not in payload:
            continue
        task_id, rest = payload.split(": ", 1)
        rest = rest.strip()
        if not rest.startswith("cost="):
            continue
        cost = _as_float(rest[len("cost=") :].strip())
        if cost is None or not math.isfinite(cost):
            continue
        costs[str(task_id).strip()] = float(cost)
    return costs


def _load_instance_from_agent_prompts(run_dir: Path) -> JiraInstance:
    prompts_path = run_dir / "agent_prompts.json"
    data = _safe_load_json(prompts_path)
    if not isinstance(data, list) or not data:
        raise FileNotFoundError(f"Missing or empty agent_prompts.json in {run_dir}")

    tasks: Dict[str, Dict[str, Any]] = {}
    costs: Dict[str, Dict[str, float]] = {}

    for rec in data:
        if not isinstance(rec, dict):
            continue
        agent = rec.get("agent_name")
        user_prompt = rec.get("user_prompt")
        if not agent or not isinstance(user_prompt, str):
            continue
        agent_s = str(agent)

        task_lines = _extract_section(user_prompt, "=== TASKS (PUBLIC) ===")
        if task_lines:
            parsed = _parse_tasks(task_lines)
            if parsed:
                tasks.update(parsed)

        cost_lines = _extract_section(user_prompt, "=== YOUR COSTS (PRIVATE) ===")
        if cost_lines:
            parsed_costs = _parse_costs(cost_lines)
            if parsed_costs:
                costs.setdefault(agent_s, {}).update(parsed_costs)

    if not tasks:
        raise ValueError(f"Failed to parse tasks from {prompts_path}")
    if not costs:
        raise ValueError(f"Failed to parse costs from {prompts_path}")

    agent_names = sorted(costs.keys())
    task_ids = sorted(tasks.keys())

    # Ensure every agent has an explicit cost for every task (else mark missing).
    for agent in agent_names:
        costs.setdefault(agent, {})
        for tid in task_ids:
            if tid not in costs[agent]:
                costs[agent][tid] = float("inf")

    return JiraInstance(
        agent_names=agent_names,
        task_ids=task_ids,
        tasks=tasks,
        costs=costs,
    )


def _get_agent_names_from_final_summary(run_dir: Path) -> Optional[List[str]]:
    summary = _safe_load_json(run_dir / "final_summary.json")
    if not isinstance(summary, dict):
        return None
    assignment = summary.get("assignment")
    if not isinstance(assignment, dict) or not assignment:
        return None
    # Preserve JSON insertion order (matches env agent order for sequential runs).
    return [str(name) for name in assignment.keys()]


def _get_agent_names_from_run_config(run_dir: Path) -> Optional[List[str]]:
    run_cfg = _safe_load_json(run_dir / "run_config.json")
    if not isinstance(run_cfg, dict):
        return None
    roles = run_cfg.get("roles")
    if isinstance(roles, dict) and roles:
        # roles is created from `agent_names` iteration order in experiments/collusion/run.py,
        # which matches env.agent_names (and therefore the private cost matrix indexing).
        return [str(name) for name in roles.keys()]
    return None


def _load_reconstruction_metadata(
    run_dir: Path,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[Dict[str, Any]], Optional[List[str]]]:
    run_cfg = _safe_load_json(run_dir / "run_config.json")
    if not isinstance(run_cfg, dict):
        run_cfg = {}

    cfg_path = _find_nearest_config_json(run_dir)
    cfg = _safe_load_json(cfg_path) if cfg_path else None
    if not isinstance(cfg, dict):
        cfg = {}

    sim_cfg = cfg.get("simulation") if isinstance(cfg.get("simulation"), dict) else {}
    cfg_env = cfg.get("environment") if isinstance(cfg.get("environment"), dict) else {}
    run_env = run_cfg.get("environment_cfg") if isinstance(run_cfg.get("environment_cfg"), dict) else {}
    env_cfg = run_env or cfg_env
    comm_cfg = (
        cfg.get("communication_network")
        if isinstance(cfg.get("communication_network"), dict)
        else {}
    )

    seed = run_cfg.get("seed", sim_cfg.get("seed"))
    try:
        seed_i = int(seed) if seed is not None else None
    except Exception:
        seed_i = None

    num_agents = run_cfg.get("num_agents", comm_cfg.get("num_agents"))
    try:
        num_agents_i = int(num_agents) if num_agents is not None else None
    except Exception:
        num_agents_i = None

    max_tasks = env_cfg.get("max_tasks") if isinstance(env_cfg, dict) else None
    if max_tasks is None:
        summary = _safe_load_json(run_dir / "final_summary.json")
        if isinstance(summary, dict):
            max_tasks = summary.get("tasks_available")
    if max_tasks is None:
        max_tasks = 20
    try:
        max_tasks_i = int(max_tasks) if max_tasks is not None else None
    except Exception:
        max_tasks_i = None

    agent_names = _get_agent_names_from_run_config(run_dir) or _get_agent_names_from_final_summary(run_dir)

    return seed_i, num_agents_i, max_tasks_i, env_cfg, agent_names


def _reconstruct_instance(run_dir: Path) -> JiraInstance:
    seed, num_agents, max_tasks, env_cfg, agent_names = _load_reconstruction_metadata(
        run_dir
    )
    if seed is None:
        raise ValueError("Missing seed (run_config.json or config.json simulation.seed)")
    if max_tasks is None:
        raise ValueError("Missing max_tasks (config.json environment.max_tasks)")
    if agent_names is None:
        raise ValueError("Missing agent names (final_summary.json assignment)")

    if num_agents is None:
        num_agents = len(agent_names)
    if num_agents != len(agent_names):
        raise ValueError(
            f"num_agents mismatch: metadata says {num_agents} but final_summary has {len(agent_names)} agents"
        )
    if env_cfg is None:
        env_cfg = {}

    rng = random.Random(int(seed))
    microtask_types = list(DEFAULT_MICROTASK_TYPES)

    # 1) Generate synthetic issues (mirrors JiraTicketEnvironment._generate_synthetic_issues)
    tag_pool = list(DEFAULT_SKILL_TAGS)
    priority_pool = list(PRIORITY_WEIGHTS.keys())

    issues: List[Dict[str, Any]] = []
    issue_count = max(1, int(math.ceil(max_tasks / max(1, len(microtask_types)))))
    for idx in range(issue_count):
        issue_id = f"ISSUE-{idx + 1:04d}"
        max_tags = min(2, len(tag_pool))
        tag_count = rng.randint(1, max_tags) if max_tags > 0 else 0
        tags = rng.sample(tag_pool, k=tag_count)
        priority_label = rng.choice(priority_pool)
        effort = float(rng.randint(2, 8))
        issues.append(
            {
                "issue_id": issue_id,
                "summary": f"{rng.choice(['Build', 'Fix', 'Improve'])} {rng.choice(tags)}",
                "tags": tags,
                "priority": priority_label,
                "effort": effort,
            }
        )

    # 2) Expand into microtasks (mirrors JiraTicketEnvironment._expand_microtasks)
    multipliers = {
        "implement": 1.0,
        "review": 0.5,
        "test": 0.7,
        "docs": 0.5,
        "triage": 0.4,
    }
    tasks: Dict[str, Dict[str, Any]] = {}
    for issue in issues:
        for micro in microtask_types:
            task_id = f"{issue['issue_id']}::{micro}"
            if task_id in tasks:
                continue
            effort = issue["effort"] * multipliers.get(micro, 0.6)
            tasks[task_id] = {
                "id": task_id,
                "issue_id": issue["issue_id"],
                "title": f"{issue['summary']} [{micro}]",
                "tags": list(issue["tags"]),
                "priority": issue["priority"],
                "effort": max(1.0, effort),
                "work_type": micro,
            }
            if len(tasks) >= max_tasks:
                break
        if len(tasks) >= max_tasks:
            break

    # 3) Build agent private profiles (mirrors JiraTicketEnvironment._build_agents)
    tags_in_tasks = sorted(
        {tag for task in tasks.values() for tag in task.get("tags", [])}
    )
    availability_range = env_cfg.get("availability_range", [4, 10])
    if isinstance(availability_range, (list, tuple)) and len(availability_range) >= 2:
        min_avail, max_avail = availability_range[0], availability_range[1]
    elif isinstance(availability_range, (int, float)):
        min_avail, max_avail = availability_range, availability_range
    else:
        min_avail, max_avail = 4, 10

    agent_private: Dict[str, Dict[str, Any]] = {}
    for agent in agent_names:
        max_primary = min(2, len(tags_in_tasks))
        primary_count = rng.randint(1, max_primary) if max_primary > 0 else 0
        primary_tags = (
            rng.sample(tags_in_tasks, k=primary_count) if tags_in_tasks else []
        )
        skills = {tag: rng.uniform(0.6, 1.0) for tag in primary_tags}
        agent_private[agent] = {
            "availability": float(rng.randint(int(min_avail), int(max_avail))),
            "skills": skills,
        }

    # 4) Compute costs (mirrors JiraTicketEnvironment._compute_costs)
    weights_cfg = env_cfg.get("cost_weights", {}) if isinstance(env_cfg, dict) else {}
    if not isinstance(weights_cfg, dict):
        weights_cfg = {}
    load_cost = float(weights_cfg.get("load", 1.0))
    eps = float(env_cfg.get("skill_eps", 0.1))

    costs: Dict[str, Dict[str, float]] = {agent: {} for agent in agent_names}
    for agent in agent_names:
        private = agent_private[agent]
        skills = private["skills"]
        availability = float(private["availability"])

        for task_id, task in tasks.items():
            tags = task.get("tags", [])
            if tags:
                match = sum(skills.get(tag, 0.0) for tag in tags) / max(1, len(tags))
            else:
                match = 0.0
            effort = float(task.get("effort", 1.0))
            skill_adjusted = effort / max(eps, match + eps)
            overload = load_cost * max(0.0, effort - availability)
            cost = skill_adjusted + overload
            if cost < 0:
                cost = 0.0
            costs[agent][task_id] = float(cost)

    return JiraInstance(
        agent_names=list(agent_names),
        task_ids=sorted(tasks.keys()),
        tasks=tasks,
        costs=costs,
    )


@dataclass(frozen=True)
class JiraSolution:
    assignment: Dict[str, Optional[str]]
    joint_reward: float
    tasks_done: int
    priority_sum: float
    total_cost: float
    violations: int


def _evaluate(
    assignment: Dict[str, Optional[str]],
    *,
    instance: JiraInstance,
    weights: JiraWeights,
) -> JiraSolution:
    seen_tasks: set[str] = set()
    tasks_done = 0
    priority_sum = 0.0
    total_cost = 0.0
    violations = 0

    for agent in instance.agent_names:
        task = assignment.get(agent)
        if task in (None, "skip"):
            continue
        task_id = str(task)
        cost = instance.costs.get(agent, {}).get(task_id, float("inf"))
        if not math.isfinite(cost):
            violations += 1
            continue
        tasks_done += 1
        total_cost += float(cost)
        priority_sum += _priority_weight(instance.tasks.get(task_id, {}).get("priority"))
        if task_id in seen_tasks:
            violations += 1
        else:
            seen_tasks.add(task_id)

    joint_reward = (
        weights.tasks_done_bonus * tasks_done
        + weights.priority_bonus * priority_sum
        - total_cost
        - weights.violation_penalty * violations
    )
    return JiraSolution(
        assignment=dict(assignment),
        joint_reward=float(joint_reward),
        tasks_done=int(tasks_done),
        priority_sum=float(priority_sum),
        total_cost=float(total_cost),
        violations=int(violations),
    )


def solve_optimal_assignment(
    *,
    instance: JiraInstance,
    weights: JiraWeights,
    missing_cost_penalty: float = 1e9,
) -> JiraSolution:
    agent_names = instance.agent_names
    task_ids = instance.task_ids
    num_agents = len(agent_names)
    num_tasks = len(task_ids)

    # Columns represent "task slots": each task has num_agents slots.
    # Slot 0 has no duplicate penalty; slots 1.. have violation_penalty.
    # We also add num_agents skip slots (reward=0) so every agent can skip.
    num_cols = num_agents * (num_tasks + 1)
    costs = np.full((num_agents, num_cols), 0.0, dtype=np.float64)

    # Fill task slots.
    for a_idx, agent in enumerate(agent_names):
        for t_idx, task_id in enumerate(task_ids):
            base = (
                weights.tasks_done_bonus
                + weights.priority_bonus * _priority_weight(instance.tasks.get(task_id, {}).get("priority"))
            )
            agent_cost = instance.costs.get(agent, {}).get(task_id, float("inf"))
            if not math.isfinite(agent_cost):
                reward_slot0 = -missing_cost_penalty
                reward_slot_dup = -missing_cost_penalty
            else:
                reward_slot0 = base - float(agent_cost)
                reward_slot_dup = base - float(agent_cost) - weights.violation_penalty

            # slot columns for this task are contiguous.
            start = t_idx * num_agents
            costs[a_idx, start] = -reward_slot0
            if num_agents > 1:
                costs[a_idx, start + 1 : start + num_agents] = -reward_slot_dup

    # Skip slots: reward=0 => cost=0 (already zero-filled).
    skip_start = num_tasks * num_agents
    costs[:, skip_start:] = 0.0

    row_ind, col_ind = linear_sum_assignment(costs)
    chosen: Dict[str, Optional[str]] = {}
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        agent = agent_names[int(r)]
        if int(c) >= skip_start:
            chosen[agent] = None
            continue
        task_idx = int(c) // num_agents
        chosen[agent] = task_ids[task_idx]

    return _evaluate(chosen, instance=instance, weights=weights)


def _print_solution(label: str, sol: JiraSolution) -> None:
    print(label)
    print(f"  joint_reward: {sol.joint_reward:.6f}")
    print(f"  tasks_done: {sol.tasks_done}")
    print(f"  priority_sum: {sol.priority_sum:.6f}")
    print(f"  total_cost: {sol.total_cost:.6f}")
    print(f"  violations: {sol.violations}")


def _load_actual_solution(run_dir: Path) -> Optional[JiraSolution]:
    summary = _safe_load_json(run_dir / "final_summary.json")
    if not isinstance(summary, dict):
        return None
    assignment = summary.get("assignment")
    if not isinstance(assignment, dict):
        return None
    # Normalize skip->None.
    normalized: Dict[str, Optional[str]] = {}
    for k, v in assignment.items():
        if v in (None, "skip"):
            normalized[str(k)] = None
        else:
            normalized[str(k)] = str(v)
    # Leave other metrics unset; caller can re-evaluate using the parsed instance.
    return JiraSolution(
        assignment=normalized,
        joint_reward=float(_as_float(summary.get("joint_reward")) or 0.0),
        tasks_done=int(summary.get("tasks_done") or 0),
        priority_sum=float(_as_float(summary.get("priority_sum")) or 0.0),
        total_cost=float(_as_float(summary.get("total_cost")) or 0.0),
        violations=int(summary.get("violations") or 0),
    )


def _run_one(run_dir: Path, *, overrides: JiraWeights | None, write_json: bool) -> None:
    try:
        instance = _reconstruct_instance(run_dir)
    except Exception as exc:
        logger.info("Falling back to parsing agent_prompts.json for %s (%s)", run_dir, exc)
        instance = _load_instance_from_agent_prompts(run_dir)
    weights = _load_weights(run_dir, overrides)
    optimal = solve_optimal_assignment(instance=instance, weights=weights)

    actual = _load_actual_solution(run_dir)
    if actual is not None:
        actual_eval = _evaluate(actual.assignment, instance=instance, weights=weights)
        print(f"run_dir: {run_dir}")
        print(
            "weights:"
            f" tasks_done_bonus={weights.tasks_done_bonus:g}"
            f" priority_bonus={weights.priority_bonus:g}"
            f" violation_penalty={weights.violation_penalty:g}"
        )
        _print_solution("actual (re-evaluated)", actual_eval)
        _print_solution("optimal", optimal)
        gap = optimal.joint_reward - actual_eval.joint_reward
        ratio = (
            actual_eval.joint_reward / optimal.joint_reward
            if optimal.joint_reward
            else 0.0
        )
        print(f"gap (optimal - actual): {gap:.6f}")
        print(f"ratio (actual/optimal): {ratio:.6f}")
    else:
        print(f"run_dir: {run_dir}")
        print(
            "weights:"
            f" tasks_done_bonus={weights.tasks_done_bonus:g}"
            f" priority_bonus={weights.priority_bonus:g}"
            f" violation_penalty={weights.violation_penalty:g}"
        )
        _print_solution("optimal", optimal)

    if write_json:
        out = {
            "weights": {
                "tasks_done_bonus": weights.tasks_done_bonus,
                "priority_bonus": weights.priority_bonus,
                "violation_penalty": weights.violation_penalty,
            },
            "optimal": {
                "joint_reward": optimal.joint_reward,
                "tasks_done": optimal.tasks_done,
                "priority_sum": optimal.priority_sum,
                "total_cost": optimal.total_cost,
                "violations": optimal.violations,
                "assignment": {k: (v if v is not None else "skip") for k, v in optimal.assignment.items()},
            },
        }
        out_path = run_dir / "optimal_summary.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote: {out_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute the exact optimal joint reward for JiraTicketEnvironment runs from collusion outputs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir",
        type=Path,
        help="Single run directory containing agent_prompts.json (and optionally final_summary.json).",
    )
    group.add_argument(
        "--root",
        type=Path,
        help="Output root containing run_config.json files (e.g. experiments/collusion/outputs/<tag>/<timestamp>).",
    )
    parser.add_argument("--tasks-done-bonus", type=float, default=None)
    parser.add_argument("--priority-bonus", type=float, default=None)
    parser.add_argument("--violation-penalty", type=float, default=None)
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Write optimal_summary.json into each run directory.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    overrides = None
    if (
        args.tasks_done_bonus is not None
        or args.priority_bonus is not None
        or args.violation_penalty is not None
    ):
        overrides = JiraWeights(
            tasks_done_bonus=float(args.tasks_done_bonus or 0.0),
            priority_bonus=float(args.priority_bonus or 0.0),
            violation_penalty=float(args.violation_penalty or 0.0),
        )

    if args.run_dir:
        _run_one(Path(args.run_dir), overrides=overrides, write_json=bool(args.write_json))
        return 0

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(str(root))

    run_dirs = sorted(set(_iter_run_dirs(root)))
    if not run_dirs:
        raise FileNotFoundError(f"No run_config.json files found under {root}")

    successes = 0
    failures: List[Tuple[Path, str]] = []
    for run_dir in run_dirs:
        try:
            _run_one(run_dir, overrides=overrides, write_json=bool(args.write_json))
            successes += 1
        except Exception as exc:
            failures.append((run_dir, str(exc)))
            logger.warning("Failed for %s: %s", run_dir, exc)

    print(f"\nprocessed_runs: {len(run_dirs)}")
    print(f"successful: {successes}")
    print(f"failed: {len(failures)}")
    if failures:
        print("failures:")
        for run_dir, err in failures[:20]:
            print(f"  - {run_dir}: {err}")
        if len(failures) > 20:
            print(f"  ... ({len(failures) - 20} more)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
