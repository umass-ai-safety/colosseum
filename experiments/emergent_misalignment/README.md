# Emergent Misalignment Experiment

This experiment asks whether a *benign-looking, benign-intended* prompt variant given to **one** agent can lead to emergent misaligned behavior (e.g., harm to the joint objective, selfish/resource-wasteful actions, goal drift) in the Hospital DCOPS environment.

## What it measures
The sweep varies:
- `prompt_versions`: `v0..v6` (stored as `.txt` files under `prompt_versions/`)
- `emergent_counts`: typically `0` (baseline) and `1` (one treated agent)

## Metrics
Metrics are computed in `metrics.py` and logged at the end of every run:
- **Joint Reward**: The total utility achieved by the hospital system (primary performance metric).
- **Suspicion Rate**: The percentage of non-treated agents who suspect foul play (via an LLM judge on post-run surveys).

## Run
1) Ensure API keys are set (root `.env`):
- `OPENAI_API_KEY` (for the evaluator/judge)
- `TOGETHER_API_KEY` (if using Together AI models)

2) Run the experiment:
```bash
python experiments/emergent_misalignment/run.py \
  --config experiments/emergent_misalignment/configs/emergent_misalignment_smalltest.yaml
```

## Prompt versions
- The treated agent is tagged via `agent_context["emergent_role"] == "emergent"`.
- The injected text is loaded from `prompt_versions/<prompt_version>.txt` and appended as “ADDITIONAL OPERATING GUIDELINES” in the user prompt.

## TextGrad optimization (external)
This repo does not vendor TextGrad. The scaffold script:
`experiments/emergent_misalignment/optimize_prompts_textgrad.py`
- Detects whether `textgrad` is installed.
- If missing, runs an evaluation-only baseline and prints install instructions.
- If present, it runs a best-effort adapter across common TextGrad APIs to propose rewrites and evaluate `v1..vN`.

## Multi-objective optimization (built-in)
This repo includes a simple multi-objective optimizer that:
- Generates benign-looking prompt mutations
- Discards candidates that violate “benignness” constraints
- Evaluates remaining candidates by running the environment and scoring (joint reward ↓, failures ↑, suspicion ↓)
- Writes the best prompt each step to `prompt_versions/v1.txt` … `v6.txt`

Example:
```bash
python experiments/emergent_misalignment/optimize_prompts_multiobjective.py \
  --config experiments/emergent_misalignment/configs/emergent_misalignment_smalltest.yaml \
  --steps 6 \
  --candidates-per-step 6
```

## Plot
Generate sweep-level plots from a sweep directory (Reward, Failures, Suspicion, Attribution,
Messages/Complaints, Coalition advantage, and distance-based belief effects):
```bash
python -m experiments.emergent_misalignment.plots.generate_all \
  --sweep-dir experiments/emergent_misalignment/outputs/<tag>/<timestamp>/runs/<model>/<sweep_name>
```

Outputs are written to: `experiments/emergent_misalignment/plots_outputs/<tag>/<timestamp>/<model>/<sweep_name>/`.
Plots are written as PDF files under `.../sweep/`.

## Configuration
The experiment is configured via `experiments/emergent_misalignment/configs/emergent_misalignment_smalltest.yaml`.

### Environment
* **Class**: `EmergentMisalignmentHospitalEnvironment` (defined in `experiments.emergent_misalignment.environment`)
* **Topology**: Defaults to `complete` (fully connected) for hospital coordination.

### Models
The config supports multiple backends:
* **OpenAI**: `gpt-4.1-mini`
* **vLLM**: `Qwen2.5-7B-Instruct` (supports auto-server startup)
* **Together AI**: `Llama-3.3-70b`, `Qwen3-235B`, `gpt-oss-120b`, etc.

## Outputs
Data is saved to `experiments/emergent_misalignment/outputs/<tag>/<timestamp>/`.
Each run directory contains:
* `run_config.json`: Full configuration, assigned roles, and final metrics.
* `final_summary.json`: The environment's final state summary.
* `judgements.json`: The LLM judge's assessment of agent suspicion.
* `agent_turns.json`: Full transcript of planning, execution, and survey phases.

## Parallelism
Increase `max_concurrent_runs` in the YAML or via CLI to run multiple simulations in parallel:
```bash
python experiments/emergent_misalignment/run.py \
  --config experiments/emergent_misalignment/configs/emergent_misalignment_smalltest.yaml \
  --max-concurrent-runs 5
```
