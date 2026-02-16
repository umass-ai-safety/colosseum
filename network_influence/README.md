# Network Influence Experiment

This experiment runs an environment on a configurable communication graph, injects adversarial misinformation about a target agent’s *private availability* (e.g., “they are traveling all week and unavailable”), and measures how the misinformation propagates through the network.

## What it measures
- **Adversary count sweep**: vary # adversarial agents and measure final spread.
- **Topology effects**: compare `path`, `star`, `complete`, and (optionally) random graph families.
- **Scaling (optional)**: you can sweep `num_agents` in YAML to study how propagation changes with network size.

Propagation is evaluated with an **LLM judge** (OpenAI API) on each agent’s post-run **Survey** response (model configurable in YAML).

## Run
1) Ensure API keys are set (root `.env`):
- `OPENAI_API_KEY` (required for the judge)
- `TOGETHER_API_KEY` (only if you enable Together models in `llm_models`)

2) Quickstart (recommended):
```bash
.venv/bin/python -m experiments.network_influence.run \
  --config experiments/network_influence/configs/quickstart.yaml
```

3) Run one of the full configs:
- `experiments/network_influence/configs/misinformation_prop_targeted.yaml` (all adversaries target the same victim)
- `experiments/network_influence/configs/misinformation_prop_mass.yaml` (adversaries target distinct victims where possible)

## Plot
Generate the sweep-level summary plot (`summary_mean.png`):
```bash
.venv/bin/python -m experiments.network_influence.plots.generate_all \
  --sweep-dir experiments/network_influence/outputs/<tag>/<timestamp>/runs/<model>/<sweep_name>
```

By default, plot artifacts are written under `experiments/network_influence/plots_outputs/<tag>/<timestamp>/<model>/<sweep_name>/` (see `sweep/summary_mean.png`).

## Environment selection
Set `environment.name` in the YAML config (e.g., `MeetingSchedulingEnvironment` or `JiraTicketEnvironment`).

For custom / non-exported environments, set `environment.import_path` to an explicit import like:
`some.package.some_module:MyEnvironmentClass`.

Outputs are written under `experiments/network_influence/outputs/<tag>/<timestamp>/`.
Each run folder contains `metrics.json`, `judge_results.json`, `survey_responses.json`, `blackboards.json`, `tool_events.json`, and `blackboard_*.txt` (human-readable blackboard snapshots).
The output root also includes `experiment.log` (checkpointed progress logs) and `progress.json` (machine-readable progress).

## What `summary_mean.png` shows
For each topology, the plot shows mean ± SEM over seeds:
- solid line: `joint_reward_ratio` (environment-provided task performance metric)
- dashed line: misinformation propagation rate among non-adversaries (excluding the target agent)

## Customize the scenario
Most knobs are in YAML under `experiment.*`:
- `experiment.adversary.*`: adversary selection + victim assignment policy
- `experiment.info.*`: labels + code token prefix used in the claim/survey/judge

If you change the misinformation claim semantics, also update:
- `experiments/network_influence/run.py` (`_build_claims`, `_choose_target_item_id`)
- `experiments/network_influence/metrics.py` (`_is_misinfo` heuristic)

## Resume (continue an interrupted run)
If a sweep crashes part-way through, you can resume **in-place** (no new timestamp folder) and only execute missing/incomplete runs. The resume script reads `<root>/config.json` by default and writes back into the same `<root>/runs/...` directories.

Dry-run (preview what will run):
```bash
.venv/bin/python -m experiments.network_influence.resume \
  --root experiments/network_influence/outputs/<tag>/<timestamp> \
  --max-concurrent-runs 10 \
  --dry-run
```

Resume for real:
```bash
.venv/bin/python -m experiments.network_influence.resume \
  --root experiments/network_influence/outputs/<tag>/<timestamp> \
  --max-concurrent-runs 10
```

## Reproducibility / stochasticity
Set a seed list in your YAML config under `experiment.seeds`.
Each sweep setting is repeated once per seed in that list (so runs per setting = `len(experiment.seeds)` unless you set `experiment.runs_per_setting`).

## Parallelism
Set `experiment.max_concurrent_runs` (or pass `--max-concurrent-runs`) to run multiple independent runs in parallel.
This increases API concurrency, so start small to avoid rate limits.

## Victim assignment
Set `experiment.adversary.victim_assignment` to:
- `distinct`: each adversary targets a different non-adversary victim (if feasible)
- `shared`: all adversaries target the same non-adversary victim

## Misinformation frequency
Set `experiment.adversary.misinfo_frequency` to:
- `once`: adversaries seed the misinformation in planning round 1, then focus on normal coordination
- `always`: adversaries re-seed each planning round (still coordinating normally)
