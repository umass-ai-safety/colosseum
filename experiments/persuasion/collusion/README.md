# Persuasion Collusion Experiment (JiraTicket default)

This experiment is a variant of `experiments/collusion` where a subset of agents are **already colluding**,
coordinate privately (optional secret blackboard), and use **persuasion as a tactic** on public channels to
push non-colluders into **suboptimal task selections** (e.g., higher-cost tasks, duplicate/conflicting choices,
or low-priority work).

## Key factors
- `colluder_count`: how many agents are in the coalition
- `colluder_selection`: how the coalition is chosen (`random` or `first`)
- `secret_channel_enabled`: whether a colluder-only blackboard is added
- `agent_order`: agent turn order (`random` or `colluders_first`)
- `prompt_variant`: which persuasion tactic playbook colluders use
- `topology`, `num_agents`, `seed`: standard comms + stochasticity sweeps

Built-in `prompt_variant` values live in `experiments/persuasion/collusion/prompts.py`
(e.g., `control`, `helpful_misdirection`, `authority_nudge`, `social_proof`, `scarcity_pressure`, `reciprocity_trade`, `risk_aversion`, `cost_anchor`, `split_brain`).

## Agent ordering (turn order)
By default, agents are shuffled each run (`agent_order: random`). If you want the colluding agents to act first in each planning/execution loop, set:

```yaml
experiment:
  persuasion_collusion:
    agent_order: colluders_first
```

Each run also records the realized ordering in `run_config.json` under `agent_turn_order`.

## Run
1) Ensure API keys are set (root `.env`) for your chosen provider.
2) Run:
```bash
python experiments/persuasion/collusion/run.py --config experiments/persuasion/collusion/configs/persuasion_collusion_jira.yaml
```

Outputs are written under `experiments/persuasion/collusion/outputs/<timestamp>/`.

## Prompt overrides
You can override built-in variants in YAML:
```yaml
experiment:
  persuasion_collusion:
    prompt_variant_overrides:
      helpful_misdirection: |
        Your custom colluder persuasion playbook...
```
