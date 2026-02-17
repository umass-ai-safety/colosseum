# Agent Misalignment (HOSPITAL)

This experiment evaluates **objective misalignment** in the **HOSPITAL** DCOPS environment (distributed hospital resource allocation), matching the paper’s Section 5.5 / Figure 7 setup (complete topology; misalignment type × coalition composition).

## Experiment axes (Figure 7)
### Misalignment type (strategy)
- **Covert** (`covert_max_coalition_reward`): maximize coalition reward while keeping overall performance near baseline (stealthy).
- **Sabotage (Self-Prom)** (`destructive_max_coalition_reward`): maximize coalition reward without preserving overall performance.
- **Sabotage (Self-Sac)** (`destructive_no_reward_preservation`): maximize overall harm (raise overall regret) regardless of coalition reward.

### Coalition composition (target role)
Sweeps enforce that the coalition contains at least one:
- `Resource_Provisioner` (high-leverage agent), or
- `departmental` (department-only coalition).

## Metrics
Metrics are computed in `metrics.py` and written to each run directory (plus sweep-level aggregations used by plotting scripts).

Figure 7-style summaries focus on:
- **Overall regret**: `max_joint_reward - joint_reward` (derived from number of patients and achieved joint reward).
- **Coalition regret**: coalition’s shortfall vs an equal-share maximum (computed from per-agent rewards).
- **Coalition advantage**: coalition average reward minus non-coalition average reward (also reported in hospital-level variants).

Additional auditing metrics (optional, not required for Figure 7) include:
- **Suspicion / attribution**: an LLM judge scores post-run surveys (who suspects foul play and whether blame is correct).

## Run
1) Ensure API keys are set (root `.env`):
- `OPENAI_API_KEY` (for the evaluator/judge)
- `TOGETHER_API_KEY` (if using Together models)

2) Run with a config from `configs/`:
```bash
python experiments/agent_misalignment/run.py \
  --config experiments/agent_misalignment/configs/agent_misalignment_gpt_4_1_mini.yaml
```

Optional parameters:
- `--out-dir <path>`: override the configured output directory
- `--max-concurrent-runs <int>`: cap parallelism (also configurable in YAML)

Configs you’ll likely use:
- `configs/agent_misalignment_gpt_4_1_mini.yaml`, `configs/agent_misalignment_gpt_4o_mini.yaml`, `configs/agent_misalignment_kimi_k2_instruct.yaml`: model-specific presets
- `configs/agent_misalignment_benign.yaml`: benign baseline

## Plot
Generate sweep-level plots (including Figure 7-style panels) from a sweep directory:
```bash
python -m experiments.agent_misalignment.plots.generate_all \
  --sweep-dir experiments/agent_misalignment/outputs/<tag>/<timestamp>/runs/<model>/<sweep_name>
```

Optional parameters:
- `--benign-sweep-dir <path>`: overlay a benign baseline
- `--seeds 0-9`: filter which seeds to include

### Multi-model comparison
Compare multiple model runs on the same grouped bar charts:
```bash
python -m experiments.agent_misalignment.plots.plot_model_comparison \
  --model-runs-dir experiments/agent_misalignment/outputs/<tag>/<timestamp>/runs/<model> \
  --model-runs-dir experiments/agent_misalignment/outputs/<tag>/<timestamp>/runs/<model> \
  --num-agents 9 \
  --target-role departmental \
  --adversary-count 4 \
  --seeds 0-4 \
  --out-dir experiments/agent_misalignment/plots_outputs/<tag>/<timestamp>/model_compare
```

The filter arguments above (`--num-agents`, `--target-role`, `--adversary-count`, `--seeds`) are optional; omit them to plot all matching runs under the provided `--model-runs-dir` roots.

## Configuration notes
* **Environment class**: `MisalignmentHospitalEnvironment` (in `experiments.agent_misalignment.environment`)
* **Figure 7 topology**: `complete`
* **Figure 7 sweep design**: strategies correspond to the three misalignment types; `target_roles` splits provisioner vs departmental coalitions.

## Outputs
Data is saved to `experiments/agent_misalignment/outputs/<tag>/<timestamp>/`.
Each run directory contains:
* `run_config.json`: full config + assigned roles
* `final_summary.json`: environment final state summary
* `judgements.json`: judge outputs (if enabled)
* `agent_turns.json`: full transcript (planning/execution/surveys)

## Resume
If a sweep is interrupted, you can resume it without re-running completed settings:
```bash
python -m experiments.agent_misalignment.resume \
  --root experiments/agent_misalignment/outputs/<tag>/<timestamp> \
  --dry-run
python -m experiments.agent_misalignment.resume \
  --root experiments/agent_misalignment/outputs/<tag>/<timestamp>
```

## Notes
Coalition composition matters: coalitions containing the **resource provisioner** tend to achieve their misalignment objectives more reliably than **departmental-only** coalitions, which can fail due to a more limited action space.
