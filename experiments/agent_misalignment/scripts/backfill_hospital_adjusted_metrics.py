from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from experiments.agent_misalignment.metrics import compute_misalignment_metrics


NEW_KEYS = {
    "agent_rewards_hospital_adjusted",
    "hospital_holding_cost",
    "hospital_resource_failure_penalty",
    "hospital_missed_step_penalty",
    "hospital_estimated_missed_steps",
    "agent_resource_failure_penalty",
    "coalition_minus_noncoalition_avg_hospital_adjusted",
    "coalition_reward_sum_hospital_adjusted",
    "coalition_reward_avg_hospital_adjusted",
    "coalition_reward_regret_hospital_adjusted",
    "agent_rewards_hospital_adjusted_normalized",
    "agent_rewards_hospital_adjusted_normalized_offset",
    "coalition_minus_noncoalition_avg_hospital_adjusted_normalized",
    "coalition_minus_noncoalition_hospital_avg",
    "coalition_minus_noncoalition_hospital_avg_hospital_adjusted",
    "coalition_minus_noncoalition_hospital_avg_hospital_adjusted_normalized",
    "coalition_minus_noncoalition_hospital_avg_actual",
}


class _ToolEvent:
    def __init__(self, d: Dict[str, Any]):
        self.tool_name = d.get("tool_name")
        self.agent_name = d.get("agent_name")
        self.arguments = d.get("arguments")
        self.result = d.get("result")
        self.phase = d.get("phase")
        self.iteration = d.get("iteration")
        self.planning_round = d.get("planning_round")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_run_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("metrics.json"):
        run_dir = p.parent
        if (run_dir / "final_summary.json").exists() and (run_dir / "run_config.json").exists():
            out.append(run_dir)
    return sorted(out)


def _extract_new_fields(metrics_blob: Dict[str, Any]) -> Dict[str, Any]:
    return {k: metrics_blob.get(k) for k in NEW_KEYS if k in metrics_blob}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing hospital-adjusted fields if already present.",
    )
    args = ap.parse_args()

    run_dirs = _iter_run_dirs(args.sweep_dir)
    updated = 0
    skipped = 0

    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        run_config_path = run_dir / "run_config.json"
        final_summary_path = run_dir / "final_summary.json"
        tool_events_path = run_dir / "tool_events.json"
        agent_rewards_path = run_dir / "agent_rewards.json"

        metrics = _read_json(metrics_path)
        if not isinstance(metrics, dict):
            skipped += 1
            continue

        if metrics.get("agent_rewards") is None:
            skipped += 1
            continue
        if not args.overwrite and any(k in metrics for k in NEW_KEYS):
            skipped += 1
            continue

        rc = _read_json(run_config_path)
        fs = _read_json(final_summary_path)
        te_raw = _read_json(tool_events_path) if tool_events_path.exists() else []
        ar = _read_json(agent_rewards_path) if agent_rewards_path.exists() else metrics.get("agent_rewards")

        if not isinstance(rc, dict) or not isinstance(fs, dict) or not isinstance(te_raw, list) or not isinstance(ar, dict):
            skipped += 1
            continue

        roles = rc.get("roles") or {}
        adversaries = rc.get("adversaries") or []
        if not isinstance(roles, dict) or not isinstance(adversaries, list):
            skipped += 1
            continue

        judgements = {
            str(a): {"suspects_foul_play": False, "suspected_agents": [], "frustration_level": 0}
            for a in roles.keys()
        }

        g = nx.Graph()
        for a in roles.keys():
            g.add_node(str(a))

        tool_events = [_ToolEvent(d) for d in te_raw if isinstance(d, dict)]

        computed = compute_misalignment_metrics(
            graph=g,
            adversaries=[str(a) for a in adversaries],
            tool_events=tool_events,
            judgements=judgements,
            blackboard_participants={},
            roles={str(k): str(v) for k, v in roles.items()},
            final_summary=fs,
            agent_rewards={str(k): float(v) for k, v in ar.items()},
            attack_type=str(metrics.get("attack_type") or ""),
        )
        computed_blob = asdict(computed)
        new_fields = _extract_new_fields(computed_blob)

        if not new_fields:
            skipped += 1
            continue

        metrics.update(new_fields)
        if isinstance(rc.get("metrics"), dict):
            rc["metrics"].update(new_fields)

        if not args.dry_run:
            _write_json(metrics_path, metrics)
            _write_json(run_config_path, rc)

        updated += 1

    print(f"runs_total={len(run_dirs)} updated={updated} skipped={skipped} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
