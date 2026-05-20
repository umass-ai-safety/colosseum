# Patch: Per-agent model config for heterogeneous coalition experiments

## What changed

### `experiments/collusion/run.py`

Added optional `colluder_llm_configs` parameter to `_run_single()` enabling each
colluder to use a different LLM without modifying agent logic or the global config.

Key additions (~20 lines):

- `_run_single()` accepts `colluder_llm_configs: Optional[List[Dict]]`
- After random colluder selection, builds `colluder_llm_map: Dict[agent_name → llm_config]`
  mapping the Nth selected colluder to the Nth config entry
- Agent-building loop uses `colluder_llm_map.get(name, cfg["llm"])` to pick per-agent
  LLM; falls back to global config for non-colluders and unspecified colluders
- Both sequential and concurrent call sites thread `colluder_llm_configs` through

The global config path (`cfg["llm"]`) is unchanged — existing experiment configs
continue to work without modification.

### New experiment configs

**`experiments/collusion/configs/gap3_hetero_qwen_sonnet.yaml`**

3-agent, 2-colluder active_collusion experiment with a heterogeneous coalition:
- Colluder 0: qwen2.5:7b served locally via Ollama (Together-compatible client,
  `base_url: http://localhost:11434/v1`)
- Colluder 1: Claude Sonnet 4.5 (Anthropic)
- Normal agent: Claude Sonnet 4.5

Run 5 seeds with: `ollama serve` running and `ollama pull qwen2.5:7b` completed.

**`experiments/collusion/configs/gap3_hetero_kimi_sonnet.yaml`**

Same structure with Kimi-K2-0905 (via OpenRouter) as one colluder. Staged for
future use pending API access. Not yet run.

## Why

Testing whether heterogeneous coalitions (capability-mismatched colluders) exhibit
different coordination patterns than homogeneous ones. Specifically: does a weaker
model sustain coordinated collusion alongside a stronger partner, or does capability
mismatch break the coordination protocol?

Findings from the 5-seed qwen+Sonnet run are documented in the Tesserae repo
(`sahiee-dev/Agentops-replay`) under `examples/colosseum_adapter/`.
