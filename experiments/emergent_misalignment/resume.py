from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from experiments.agent_misalignment import run as run_mod  # noqa: E402


REQUIRED_RUN_FILES: Tuple[str, ...] = (
    "run_config.json",
    "metrics.json",
    "final_summary.json",
    "agent_turns.json",
    "survey_responses.json",
    "judge_results.json",
    "tool_events.json",
    "blackboards.json",
)


@dataclass(frozen=True)
class RunSpec:
    model_label: str
    model_llm_cfg: Dict[str, Any]
    sweep_name: str
    topology: str
    num_agents: int
    adversary_count: int
    seed: int

    @property
    def run_id(self) -> str:
        return (
            f"{self.model_label}__{self.sweep_name}__{self.topology}"
            f"__n{self.num_agents}__a{self.adversary_count}__seed{self.seed}"
        )

    @property
    def run_label(self) -> str:
        return f"{self.model_label}/{self.sweep_name}/{self.topology}/n{self.num_agents}/a{self.adversary_count}/seed{self.seed}"


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8")) or {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _is_run_complete(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    return all((run_dir / fname).exists() for fname in REQUIRED_RUN_FILES)


def _iter_expected_run_specs(cfg: Dict[str, Any]) -> Iterable[RunSpec]:
    exp = cfg.get("experiment") or {}
    models = cfg.get("llm_models") or []
    sweeps = exp.get("sweeps") or []

    runs_per_setting = exp.get("runs_per_setting")
    if runs_per_setting is not None:
        runs_per_setting = int(runs_per_setting)
        if runs_per_setting <= 0:
            raise ValueError("experiment.runs_per_setting must be a positive integer")

    default_seeds = run_mod._normalize_seeds(exp.get("seeds"))
    if not default_seeds:
        default_seeds = run_mod._normalize_seeds(
            (cfg.get("simulation") or {}).get("seed")
        ) or [1]

    for model in models:
        model_label = str(model.get("label") or "model")
        llm_cfg = model.get("llm") or {}
        for sweep in sweeps:
            sweep_name = str(sweep.get("name") or "sweep")
            topologies = sweep.get("topologies") or []
            agent_counts = sweep.get("num_agents") or []
            adversary_counts = sweep.get("adversary_counts") or []
            seeds = run_mod._normalize_seeds(sweep.get("seeds")) or list(default_seeds)
            if runs_per_setting is not None:
                seeds = seeds[:runs_per_setting]
            if not seeds:
                raise ValueError(
                    "No seeds specified. Set experiment.seeds or sweeps[].seeds."
                )

            for topology in topologies:
                for n in agent_counts:
                    for a in adversary_counts:
                        for seed in seeds:
                            yield RunSpec(
                                model_label=model_label,
                                model_llm_cfg=llm_cfg,
                                sweep_name=sweep_name,
                                topology=str(topology),
                                num_agents=int(n),
                                adversary_count=int(a),
                                seed=int(seed),
                            )


def _select_incomplete_runs(
    *, root: Path, cfg: Dict[str, Any]
) -> Tuple[List[RunSpec], int, int]:
    expected = list(_iter_expected_run_specs(cfg))
    total_runs = len(expected)
    completed = 0
    incomplete: List[RunSpec] = []
    for spec in expected:
        run_dir = root / "runs" / spec.model_label / spec.sweep_name / spec.run_id
        if _is_run_complete(run_dir):
            completed += 1
        else:
            incomplete.append(spec)
    return incomplete, completed, total_runs


async def resume_runs(
    *,
    root: Path,
    cfg: Dict[str, Any],
    max_concurrent_runs: int,
    stop_on_error: bool,
    dry_run: bool,
) -> None:
    run_mod._ensure_dir(root)
    run_mod._configure_experiment_logging(root)

    to_run, completed, total_runs = _select_incomplete_runs(root=root, cfg=cfg)
    failed = 0

    run_mod.logger.info(
        "RESUME START (total_runs=%s, completed=%s, remaining=%s, output_root=%s)",
        total_runs,
        completed,
        len(to_run),
        root,
    )

    if dry_run:
        print(f"Output root: {root}")
        print(f"Total runs: {total_runs}")
        print(f"Already complete: {completed}")
        print(f"Remaining (incomplete): {len(to_run)}")
        if to_run:
            print("Next 10 runs to execute:")
            for spec in to_run[:10]:
                print(f"  - {spec.run_label}")
        return

    run_mod._write_progress(
        root,
        {
            "status": "running",
            "total_runs": total_runs,
            "completed_runs": completed,
            "failed_runs": failed,
            "resumed_at": datetime.now().isoformat(),
        },
    )

    if not to_run:
        run_mod.logger.info("RESUME DONE (nothing to do; all runs appear complete)")
        run_mod._write_progress(
            root,
            {
                "status": "completed",
                "total_runs": total_runs,
                "completed_runs": completed,
                "failed_runs": failed,
            },
        )
        return

    max_concurrent_runs = int(max_concurrent_runs)
    if max_concurrent_runs <= 0:
        raise ValueError("max_concurrent_runs must be a positive integer")

    with tqdm(
        total=total_runs,
        initial=completed,
        desc="Experiments (resume)",
        unit="run",
        dynamic_ncols=True,
    ) as pbar:
        if max_concurrent_runs <= 1:
            for spec in to_run:
                pbar.set_postfix_str(spec.run_label)
                run_status = "success"
                try:
                    await run_mod._run_single(
                        base_cfg=cfg,
                        model_label=spec.model_label,
                        model_llm_cfg=spec.model_llm_cfg,
                        sweep_name=spec.sweep_name,
                        topology=spec.topology,
                        num_agents=spec.num_agents,
                        adversary_count=spec.adversary_count,
                        seed=spec.seed,
                        out_dir=root,
                    )
                    completed += 1
                except Exception:
                    run_status = "failed"
                    failed += 1
                    run_mod.logger.exception("RUN FAILED %s", spec.run_label)
                    if stop_on_error:
                        raise
                finally:
                    pbar.update(1)
                    run_mod._write_progress(
                        root,
                        {
                            "status": "running",
                            "total_runs": total_runs,
                            "completed_runs": completed,
                            "failed_runs": failed,
                            "last_run_label": spec.run_label,
                            "last_run_status": run_status,
                        },
                    )
        else:
            semaphore = asyncio.Semaphore(max_concurrent_runs)

            def _run_single_in_thread(**kwargs: Any) -> Dict[str, Any]:
                return asyncio.run(run_mod._run_single(**kwargs))

            async def _run_single_limited(*, spec: RunSpec) -> Dict[str, Any]:
                async with semaphore:
                    run_mod.logger.info("SCHEDULED %s", spec.run_label)
                    return await asyncio.to_thread(
                        _run_single_in_thread,
                        base_cfg=cfg,
                        model_label=spec.model_label,
                        model_llm_cfg=spec.model_llm_cfg,
                        sweep_name=spec.sweep_name,
                        topology=spec.topology,
                        num_agents=spec.num_agents,
                        adversary_count=spec.adversary_count,
                        seed=spec.seed,
                        out_dir=root,
                    )

            tasks = []
            task_specs: Dict[asyncio.Task[Any], RunSpec] = {}
            for spec in to_run:
                task = asyncio.create_task(_run_single_limited(spec=spec))
                tasks.append(task)
                task_specs[task] = spec

            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for finished in done:
                    spec = task_specs.get(finished)
                    if spec is None:
                        continue
                    pbar.set_postfix_str(spec.run_label)
                    run_status = "success"
                    try:
                        await finished
                        completed += 1
                    except Exception:
                        run_status = "failed"
                        failed += 1
                        run_mod.logger.exception("RUN FAILED %s", spec.run_label)
                        if stop_on_error:
                            # Cancel remaining tasks so we don't keep spending tokens/compute.
                            for t in pending:
                                t.cancel()
                            await asyncio.gather(*pending, return_exceptions=True)
                            raise
                    finally:
                        pbar.update(1)
                        run_mod._write_progress(
                            root,
                            {
                                "status": "running",
                                "total_runs": total_runs,
                                "completed_runs": completed,
                                "failed_runs": failed,
                                "last_run_label": spec.run_label,
                                "last_run_status": run_status,
                            },
                        )

    status = "completed" if completed == total_runs else "completed_with_failures"
    run_mod.logger.info(
        "RESUME END (completed=%s, failed=%s, total_runs=%s, output_root=%s)",
        completed,
        failed,
        total_runs,
        root,
    )
    run_mod._write_progress(
        root,
        {
            "status": status,
            "total_runs": total_runs,
            "completed_runs": completed,
            "failed_runs": failed,
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Resume a previously started network influence experiment in-place."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Existing timestamp output root (e.g., experiments/agent_misalignment/outputs/<tag>/<timestamp>).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config override. Defaults to <root>/config.json if present.",
    )
    parser.add_argument(
        "--max-concurrent-runs",
        default=None,
        type=int,
        help="Override experiment.max_concurrent_runs when resuming.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort immediately on the first failed run (default: continue and record failures).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would run and exit."
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    config_path = Path(args.config) if args.config else (root / "config.json")
    cfg = _load_config(config_path)

    exp = cfg.get("experiment") or {}
    max_concurrent_runs = (
        int(args.max_concurrent_runs)
        if args.max_concurrent_runs is not None
        else int(exp.get("max_concurrent_runs", 1))
    )

    asyncio.run(
        resume_runs(
            root=root,
            cfg=cfg,
            max_concurrent_runs=max_concurrent_runs,
            stop_on_error=bool(args.stop_on_error),
            dry_run=bool(args.dry_run),
        )
    )


if __name__ == "__main__":
    main()
