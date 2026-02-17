# Colosseum
The complete code will be available soon!!

This repo contains experiment drivers under `experiments/` that depend on Terrarium (`terrarium-agents`).

## Setup

1) Install CoLLAB (required for certain environments):
- Recommended (submodule):
  - `git submodule add https://github.com/Saad-Mahmud/CoLLAB_SEA.git external/CoLLAB`
  - `git submodule update --init --recursive`
- Alternative: clone anywhere and set `TERRARIUM_COLLAB_PATH` in `.env`.

2) Create env + install deps:
- `uv venv --python 3.11 .venv`
- `source .venv/bin/activate`
- `uv sync --no-install-project`

3) Configure environment and API keys (if needed)(never put real keys in `.env.example`):
- `cp .env.example .env`
- Fill API keys

## Run

Run from the repo root, e.g.
- `uv run --env-file .env python experiments/persuasion/collusion/run.py --config experiments/persuasion/collusion/configs/persuasion_collusion_jira.yaml`
