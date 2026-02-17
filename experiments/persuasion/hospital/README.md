# Persuasion Collusion Experiment (HospitalEnvironment)

This experiment adapts the Jira persuasion-collusion setup to the **DCOPS Hospital** environment.
A subset of agents are designated as **colluders** and (optionally) coordinate via a **secret coalition blackboard**.
They attempt to use **persuasion on PUBLIC blackboards** to steer non-colluders toward lower-impact or conflicting actions.

## Key factors
- `colluder_count`: size of the coalition
- `misalignment_strategy`: one of `covert_max_coalition_reward`, `destructive_max_coalition_reward`, `destructive_no_reward_preservation`
- `target_role`: optional constraint on coalition membership (e.g., `Resource_Provisioner`, `departmental`, `Triage`)
- `secret_channel_enabled`: whether a colluder-only secret blackboard is created
- `prompt_variant`: the persuasion playbook colluders use (see `experiments/persuasion/hospital/prompts.py`)
- `topology`, `num_agents`, `num_patients`, `seed`: standard sweeps

## Run
```bash
python experiments/persuasion/hospital/run.py --config experiments/persuasion/hospital/configs/persuasion_hospital.yaml
```

Outputs are written under `experiments/persuasion/hospital/outputs/<timestamp>/`.

## Plots
Generate plots for a single sweep directory:
```bash
python -m experiments.persuasion.hospital.plots.generate_all \
  --sweep-dir experiments/persuasion/hospital/outputs/persuasion_hospital/<timestamp>/runs/<model>/<sweep_name>
```

This produces:
- Misalignment-style sweep/overview plots (reused from `agent_misalignment`) under `plots_outputs/.../misalignment_style` and `.../overview`
- Persuasion-specific variant plots (by `prompt_variant`) under `plots_outputs/.../by_variant`

## Prompt overrides
You can override built-in variants in YAML:
```yaml
experiment:
  persuasion_hospital:
    prompt_variant_overrides:
      helpful_misdirection: |
        Your custom colluder persuasion playbook...
```
