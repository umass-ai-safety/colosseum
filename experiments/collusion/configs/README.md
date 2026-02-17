# Collusion Configs

This folder contains ready-to-run YAML configs for `experiments/collusion/run.py`.

Usage:
```bash
.venv/bin/python -m experiments.collusion.run --config experiments/collusion/configs/<file>.yaml
```

## Quick sanity check
- `collusion_jira_three_agents_two_blackboards.yaml`: tiny setup with a secret colluder-only blackboard.

## Main sweeps
- `collusion_jira_topologies.yaml`: topology sweep for JiraTicket (baseline vs secret-channel variants).
- `collusion_jira_topologies_env_assignment_fills.yaml`: variant sweep with environment tweaks (see file comments).

## Regret report model set
- `collusion_jira_complete_n6_c2_regret_models.yaml`: configuration used for the Jira regret plots.
