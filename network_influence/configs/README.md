# Network Influence Configs

This folder contains ready-to-run YAML configs for `experiments/network_influence/run.py`.

## Quickstart
- `quickstart.yaml`: minimal sweep intended to validate that keys, models, and plotting work end-to-end.

## Experiment configs
- `misinformation_prop_targeted.yaml`: targeted attack (all adversaries target the same victim via `victim_assignment: shared`).
- `misinformation_prop_mass.yaml`: broader attack (adversaries target distinct victims where possible via `victim_assignment: distinct`).
