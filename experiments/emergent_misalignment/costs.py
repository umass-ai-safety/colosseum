from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TokenUsage":
        if not isinstance(d, dict):
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=int(d.get("prompt_tokens") or 0),
            completion_tokens=int(d.get("completion_tokens") or 0),
            total_tokens=int(d.get("total_tokens") or 0),
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_run_dirs(output_root: Path) -> Iterable[Path]:
    runs = output_root / "runs"
    if not runs.exists():
        return []
    # run.py layout: runs/<model_label>/<sweep>/<run_id>/
    return (p for p in runs.rglob("*") if p.is_dir() and (p / "run_config.json").exists())


def _sum_agent_turn_usage(agent_turns: list[dict[str, Any]]) -> TokenUsage:
    total = TokenUsage()
    for t in agent_turns:
        total += TokenUsage.from_dict(t.get("usage") or {})
    return total


def _sum_judge_usage(run_dir: Path) -> Tuple[TokenUsage, Optional[dict[str, Any]]]:
    judge_path = run_dir / "judge_usage.json"
    if not judge_path.exists():
        return TokenUsage(), None
    data = _load_json(judge_path)
    usage = TokenUsage.from_dict((data or {}).get("total_usage") or {})
    return usage, data


def _resolve_model_pricing(
    *,
    config_root: Optional[Path],
    provider: str,
    model_name: str,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Returns (input_per_1m_usd, output_per_1m_usd, source_note).

    Pricing resolution order:
      1) <output_root>/config.json: top-level `pricing.<provider>.<model_name>` (override)
      2) Terrarium/pricing.json (central registry; default)
      3) None (unknown)
    """
    provider = str(provider or "").strip().lower()
    model_name = str(model_name or "").strip()
    if not provider or not model_name:
        return None, None, "missing_provider_or_model"

    if config_root:
        cfg_path = config_root / "config.json"
        if cfg_path.exists():
            cfg = _load_json(cfg_path) or {}
            pricing = (cfg.get("pricing") or {}).get(provider) or {}
            model_block = pricing.get(model_name) or {}
            if isinstance(model_block, dict):
                inp = model_block.get("input_per_1m_usd")
                out = model_block.get("output_per_1m_usd")
                if inp is not None and out is not None:
                    return float(inp), float(out), "config.json:pricing"

    pricing_path = os.getenv("TERRARIUM_PRICING_PATH")
    if pricing_path:
        registry_path = Path(pricing_path).expanduser()
    else:
        registry_path = Path(__file__).resolve().parents[3] / "pricing.json"

    if registry_path.exists():
        registry = _load_json(registry_path) or {}
        pricing = (registry.get(provider) or {}) if isinstance(registry, dict) else {}
        model_block = pricing.get(model_name) if isinstance(pricing, dict) else None
        if isinstance(model_block, dict):
            inp = model_block.get("input_per_1m_usd")
            out = model_block.get("output_per_1m_usd")
            if inp is not None and out is not None:
                return float(inp), float(out), f"{registry_path}:pricing"

    return None, None, "pricing_missing_for_model"


def _cost_usd(usage: TokenUsage, *, input_per_1m_usd: float, output_per_1m_usd: float) -> float:
    return (usage.prompt_tokens / 1_000_000.0) * input_per_1m_usd + (
        usage.completion_tokens / 1_000_000.0
    ) * output_per_1m_usd


def compute_and_write_costs(output_root: Path, *, write_per_run: bool = True) -> dict[str, Any]:
    output_root = output_root.resolve()
    config_root = output_root if (output_root / "config.json").exists() else None

    results: list[dict[str, Any]] = []
    for run_dir in _iter_run_dirs(output_root):
        run_cfg = _load_json(run_dir / "run_config.json") or {}
        run_id = str(run_cfg.get("run_id") or run_dir.name)

        turns_path = run_dir / "agent_turns.json"
        agent_turns = _load_json(turns_path) if turns_path.exists() else []
        agent_usage = _sum_agent_turn_usage(agent_turns or [])

        judge_usage, judge_blob = _sum_judge_usage(run_dir)

        # Resolve model names from run_id prefix; detailed pricing uses config.json mapping.
        # run_id format: <model_label>__...
        model_label = run_id.split("__", 1)[0]

        # Map model_label -> provider/model_name using config.json if present.
        provider = None
        model_name = None
        if config_root:
            cfg = _load_json(config_root / "config.json") or {}
            for m in cfg.get("llm_models") or []:
                if m.get("label") == model_label:
                    llm = m.get("llm") or {}
                    provider = (llm.get("provider") or "").lower() or None
                    if provider:
                        model_name = (llm.get(provider) or {}).get("model")
                    break

        provider = provider or "unknown"
        model_name = model_name or model_label

        inp_rate, out_rate, pricing_source = _resolve_model_pricing(
            config_root=config_root,
            provider=provider,
            model_name=str(model_name),
        )

        agent_cost = None
        judge_cost = None
        total_cost = None
        if inp_rate is not None and out_rate is not None:
            agent_cost = _cost_usd(agent_usage, input_per_1m_usd=inp_rate, output_per_1m_usd=out_rate)
            judge_cost = _cost_usd(judge_usage, input_per_1m_usd=inp_rate, output_per_1m_usd=out_rate)
            total_cost = agent_cost + judge_cost

        payload = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "provider": provider,
            "model_name": model_name,
            "pricing_source": pricing_source,
            "pricing": (
                None
                if inp_rate is None or out_rate is None
                else {"input_per_1m_usd": inp_rate, "output_per_1m_usd": out_rate}
            ),
            "agent_usage": {
                "prompt_tokens": agent_usage.prompt_tokens,
                "completion_tokens": agent_usage.completion_tokens,
                "total_tokens": agent_usage.total_tokens,
            },
            "judge_usage": {
                "prompt_tokens": judge_usage.prompt_tokens,
                "completion_tokens": judge_usage.completion_tokens,
                "total_tokens": judge_usage.total_tokens,
            },
            "agent_cost_usd": agent_cost,
            "judge_cost_usd": judge_cost,
            "total_cost_usd": total_cost,
        }

        if judge_blob is not None:
            payload["judge_details"] = judge_blob

        if write_per_run:
            with open(run_dir / "costs.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)

        results.append(payload)

    summary = {"output_root": str(output_root), "runs": results}
    with open(output_root / "costs_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        required=True,
        help="Experiment output root (the timestamp dir containing config.json, runs/, summary.json).",
    )
    parser.add_argument("--no-per-run", action="store_true")
    args = parser.parse_args()

    compute_and_write_costs(Path(args.output_root), write_per_run=not args.no_per_run)


if __name__ == "__main__":
    main()
