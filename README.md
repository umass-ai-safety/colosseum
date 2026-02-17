# <sub><img src="dev/colosseum_icon.png" alt="Colosseum icon" width="45" /></sub> Colosseum

This repo contains experiments for Colosseum under `experiments/` that depend on the Terrarium (`terrarium-agents`) framework.

If you find this repo useful or interesting please :star: it!

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


## Paper Citation
Coming soon!
<!-- ```bibtex
@article{nakamura2025terrarium,
  title={Terrarium: Revisiting the Blackboard for Multi-Agent Safety, Privacy, and Security Studies},
  author={Nakamura, Mason and Kumar, Abhinav and Mahmud, Saaduddin and Abdelnabi, Sahar and Zilberstein, Shlomo and Bagdasarian, Eugene},
  journal={arXiv preprint arXiv:2510.14312},
  year={2025}
}
``` -->
