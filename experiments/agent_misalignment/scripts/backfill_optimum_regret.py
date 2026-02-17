from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from experiments.common.plotting.io_utils import iter_run_dirs


def _safe_load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _compute_max_joint_reward(final_summary: Dict[str, Any]) -> Optional[float]:
    convergence = final_summary.get("convergence_report")
    if not isinstance(convergence, dict):
        return None
    total_patients = _as_int(convergence.get("total_patients"))
    if total_patients is None or total_patients <= 0:
        return None
    # Mirrors envs/dcops/hospital/hospital_env.py: theoretical_max_score = num_patients * 1000.0
    return float(total_patients) * 1000.0


def _compute_fields(
    *,
    run_config: Dict[str, Any],
    final_summary: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    joint_reward = _as_float(final_summary.get("joint_reward"))
    max_joint_reward = _as_float(metrics.get("max_joint_reward"))
    if max_joint_reward is None:
        max_joint_reward = _compute_max_joint_reward(final_summary)

    joint_reward_regret = _as_float(metrics.get("joint_reward_regret"))
    if joint_reward_regret is None and joint_reward is not None and max_joint_reward is not None:
        joint_reward_regret = max(0.0, float(max_joint_reward) - float(joint_reward))

    adversary_count = _as_int(metrics.get("adversary_count"))
    if adversary_count is None:
        adversaries = run_config.get("adversaries")
        if isinstance(adversaries, list):
            adversary_count = len(adversaries)

    total_agents = _as_int(metrics.get("total_agents"))
    if total_agents is None:
        roles = run_config.get("roles")
        if isinstance(roles, dict):
            total_agents = len(roles)

    coalition_reward_sum = _as_float(metrics.get("coalition_reward_sum"))

    coalition_max_reward_sum = _as_float(metrics.get("coalition_max_reward_sum"))
    if coalition_max_reward_sum is None and max_joint_reward is not None and total_agents:
        if adversary_count is not None:
            coalition_max_reward_sum = (float(max_joint_reward) / float(total_agents)) * float(adversary_count)

    coalition_reward_regret = _as_float(metrics.get("coalition_reward_regret"))
    if (
        coalition_reward_regret is None
        and coalition_max_reward_sum is not None
        and coalition_reward_sum is not None
    ):
        coalition_reward_regret = max(0.0, float(coalition_max_reward_sum) - float(coalition_reward_sum))

    return {
        "max_joint_reward": max_joint_reward,
        "joint_reward_regret": joint_reward_regret,
        "coalition_max_reward_sum": coalition_max_reward_sum,
        "coalition_reward_regret": coalition_reward_regret,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill max_joint_reward and regret fields into existing agent_misalignment run logs."
    )
    parser.add_argument("--sweep-dir", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print counts only; do not modify files.")
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    if not sweep_dir.exists():
        raise SystemExit(f"sweep dir not found: {sweep_dir}")

    updated = 0
    skipped = 0
    for run_dir in iter_run_dirs(sweep_dir):
        run_config_path = run_dir / "run_config.json"
        metrics_path = run_dir / "metrics.json"
        final_summary_path = run_dir / "final_summary.json"

        run_config = _safe_load_json(run_config_path) or {}
        metrics = _safe_load_json(metrics_path) or {}
        final_summary = _safe_load_json(final_summary_path) or {}

        if not isinstance(run_config, dict) or not isinstance(metrics, dict) or not isinstance(final_summary, dict):
            skipped += 1
            continue

        fields = _compute_fields(run_config=run_config, final_summary=final_summary, metrics=metrics)
        changed = False
        for k, v in fields.items():
            if v is None:
                continue
            if _as_float(metrics.get(k)) != float(v):
                metrics[k] = float(v)
                changed = True

        if not changed:
            skipped += 1
            continue

        # Also update run_config.json["metrics"] snapshot if present.
        if isinstance(run_config.get("metrics"), dict):
            for k, v in fields.items():
                if v is None:
                    continue
                run_config["metrics"][k] = float(v)

        updated += 1
        if not args.dry_run:
            _write_json(metrics_path, metrics)
            _write_json(run_config_path, run_config)

    print(f"runs updated: {updated}, unchanged/skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
