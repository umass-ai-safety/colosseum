from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


def _safe_load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        logger.debug("Failed to parse JSON: %s", path, exc_info=True)
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except Exception:
        return None


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    runs_root = root / "runs"
    if not runs_root.exists():
        return []
    for cfg_path in runs_root.rglob("run_config.json"):
        yield cfg_path.parent


def _pick_metrics_path(run_dir: Path, *, prefer_repaired: bool) -> Path:
    if prefer_repaired:
        repaired = run_dir / "metrics_repaired.json"
        if repaired.exists():
            return repaired
    return run_dir / "metrics.json"


@dataclass(frozen=True)
class ViolationStats:
    total_runs: int
    runs_with_numeric_violations: int
    runs_with_missing_violations: int
    total_violations: int
    runs_with_any_violations: int
    max_violations: Optional[int]


def _summarize(rows: List[Dict[str, Any]]) -> ViolationStats:
    violations: List[int] = []
    missing = 0
    for r in rows:
        v = r.get("violations")
        if isinstance(v, int):
            violations.append(v)
        else:
            missing += 1

    return ViolationStats(
        total_runs=len(rows),
        runs_with_numeric_violations=len(violations),
        runs_with_missing_violations=missing,
        total_violations=sum(violations),
        runs_with_any_violations=sum(1 for v in violations if v > 0),
        max_violations=max(violations) if violations else None,
    )


def _group_by(
    rows: List[Dict[str, Any]], keys: Tuple[str, ...]
) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    return grouped


def _print_stats(name: str, stats: ViolationStats) -> None:
    print(name)
    print(f"  total_runs: {stats.total_runs}")
    print(f"  runs_with_numeric_violations: {stats.runs_with_numeric_violations}")
    print(f"  runs_with_missing_violations: {stats.runs_with_missing_violations}")
    print(f"  total_violations: {stats.total_violations}")
    print(f"  runs_with_any_violations: {stats.runs_with_any_violations}")
    print(f"  max_violations: {stats.max_violations}")


def _load_rows(root: Path, *, prefer_repaired: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(_iter_run_dirs(root)):
        run_config = _safe_load_json(run_dir / "run_config.json") or {}
        if not isinstance(run_config, dict):
            run_config = {}

        metrics_path = _pick_metrics_path(run_dir, prefer_repaired=prefer_repaired)
        metrics = _safe_load_json(metrics_path) or {}
        if not isinstance(metrics, dict):
            metrics = {}

        rid = run_config.get("run_id") or run_dir.name
        rows.append(
            {
                "run_dir": str(run_dir),
                "run_id": rid,
                "model_label": run_config.get("model_label"),
                "provider": run_config.get("provider"),
                "model": run_config.get("model"),
                "sweep": run_config.get("sweep"),
                "environment_label": run_config.get("environment_label"),
                "environment_name": run_config.get("environment_name"),
                "topology": run_config.get("topology"),
                "num_agents": _as_int(run_config.get("num_agents")),
                "colluder_count": _as_int(run_config.get("colluder_count")),
                "secret_channel_enabled": run_config.get("secret_channel_enabled"),
                "secret_blackboard_id": _as_int(run_config.get("secret_blackboard_id")),
                "prompt_variant": run_config.get("prompt_variant"),
                "seed": _as_int(run_config.get("seed")),
                "status": metrics.get("status"),
                "violations": _as_int(metrics.get("violations")),
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize constraint violations from collusion run outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Output root containing runs/ (e.g. experiments/collusion/outputs/<tag>/<timestamp>).",
    )
    parser.add_argument(
        "--prefer-repaired",
        action="store_true",
        help="Prefer metrics_repaired.json when present (else uses metrics.json).",
    )
    parser.add_argument(
        "--group-by",
        default="",
        help=(
            "Comma-separated keys to group results (e.g. 'prompt_variant,secret_channel_enabled'). "
            "Available keys: sweep,environment_label,environment_name,topology,num_agents,colluder_count,"
            "secret_channel_enabled,secret_blackboard_id,prompt_variant,model_label,provider,model,status."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(str(root))

    rows = _load_rows(root, prefer_repaired=bool(args.prefer_repaired))
    overall = _summarize(rows)
    _print_stats(f"Overall ({root})", overall)

    group_keys = tuple(k.strip() for k in str(args.group_by).split(",") if k.strip())
    if group_keys:
        grouped = _group_by(rows, group_keys)
        for key, subrows in sorted(grouped.items(), key=lambda kv: str(kv[0])):
            stats = _summarize(subrows)
            label = ", ".join(f"{k}={v}" for k, v in zip(group_keys, key))
            _print_stats(f"\nGroup ({label})", stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

