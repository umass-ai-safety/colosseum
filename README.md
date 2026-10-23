# Colosseum <sub><sub><img src="dev/colosseum_icon.png" alt="Colosseum icon" width="45" /></sub></sub>

> [!NOTE]
> :tada: **Colosseum has been accepted to the NeurIPS 2026 Evaluations and Datasets Track!** Read the paper on [arXiv](https://arxiv.org/abs/2602.15198).

If you find the repo or paper useful or interesting please :star: it!
## Description
<!-- > [!NOTE] -->
> Colosseum is a framework for objectively auditing cooperative multi-agent systems composed of tool-calling, LLM-based agents. We study collusion across varying models, network topologies, and environments. Below is a taxonomy of various collusion types Colosseum can identify:

<!-- > [!FINDING] -->
<!-- > [!TIP] -->

<p align="center">
  <a href="dev/colosseum-motivation-v3.pdf">
    <img src="dev/colosseum-motivation-v3-preview.png" alt="Colosseum motivation slides" width="70%" />
  </a>
</p>

This repo contains experiments for Colosseum under `experiments/` that utilizes the [Terrarium](https://github.com/umass-aisec/Terrarium) (`terrarium-agents==0.2.1`) framework.


## Setup

1) Clone with submodules (CoLLAB is used by the DCOP environments):
- `git clone --recurse-submodules https://github.com/umass-ai-safety/colosseum.git && cd colosseum`
- Already cloned without them? Run `git submodule update --init --recursive`.
- Alternative: clone [CoLLAB](https://github.com/Saad-Mahmud/CoLLAB_SEA) anywhere and `export TERRARIUM_COLLAB_PATH=/path/to/CoLLAB` in your shell. (Setting it in `.env` does not work: Terrarium reads it when the environments are imported, before `.env` is loaded.)

2) Create env + install deps:
- `uv venv --python 3.11 .venv`
- `source .venv/bin/activate`
- `uv sync --no-install-project`
- For local models via vLLM (Linux + CUDA GPU), use `uv sync --no-install-project --extra vllm` instead.

3) Configure API keys (never put real keys in `.env.example`):
- `cp .env.example .env`
- Uncomment and fill the keys you use. Values in `.env` override exported shell variables, so leave a key commented out if you set it in your shell.

## Run

> [!TIP]
> If you are using APIs, actively track API usage since API costs accumulate linearly with the number of agents in the run.

> [!NOTE]
> The paper's results were produced with `terrarium-agents==0.1.1` (tag [`paper-v0.1.1`](https://github.com/umass-ai-safety/colosseum/tree/paper-v0.1.1), commit `b077fe7`). Terrarium 0.2.x changes some behavior, so check out that tag to reproduce the paper's numbers exactly. Changes we know of:
> - An agent's turn ends after a step with no tool calls; 0.1.1 prompted the model again until the step limit.
> - A secret-channel blackboard is always a new board; 0.1.1 reused an existing board with the same members (e.g., a pairwise board between two colluders in sparse topologies).

Run from the repo root with your virtualenv active (e.g., `source .venv/bin/activate`). Most configs below are full sweeps; start with the quickstart to check your setup cheaply. Here is a non-exhaustive list of run commands:

```bash
# Quickstart: one short run to validate keys, models, and outputs (OpenAI)
python experiments/network_influence/run.py \
  --config experiments/network_influence/configs/quickstart.yaml

# Persuasion affects on collusion (JiraTicket)
python experiments/persuasion/collusion/run.py \
  --config experiments/persuasion/collusion/configs/persuasion_collusion_jira.yaml

# Persuasion affects on collusion (Hospital)
python experiments/persuasion/hospital/run.py \
  --config experiments/persuasion/hospital/configs/persuasion_hospital.yaml

# Collusion (JiraTicket topology sweeps; local Qwen2.5-7B via vLLM, needs `--extra vllm` and a GPU)
python experiments/collusion/run.py \
  --config experiments/collusion/configs/collusion_jira_topologies.yaml

# Network influence / misinformation propagation (Mass policy)
python experiments/network_influence/run.py \
  --config experiments/network_influence/configs/misinformation_prop_mass.yaml

# Agent objective misalignment (Hospital)
python experiments/agent_misalignment/run.py \
  --config experiments/agent_misalignment/configs/agent_misalignment_gpt_4_1_mini.yaml
```

## Customizing environments and the MAS harness

To change an environment's prompts, rewards, or other logic, subclass it inside this repo and point a config at your class with `environment.import_path` (supported by every runner). Tools are looked up by name from Terrarium's built-in toolsets, so a subclass reuses one via `tools_environment_name` rather than defining its own. [`experiments/collusion/choice/`](experiments/collusion/choice/) is a working example:

```python
# experiments/my_envs/jira_variant.py
from terrarium.environments.dcops.jira_ticket.jira_ticket_env import JiraTicketEnvironment

class MyJira(JiraTicketEnvironment):
    tools_environment_name = "JiraTicketEnvironment"  # reuse the existing toolset
```
```yaml
environment:
  import_path: "experiments.my_envs.jira_variant:MyJira"
```

To edit Terrarium itself, install an editable checkout of the matching release:

```bash
git clone --recurse-submodules --branch v0.2.1 https://github.com/umass-aisec/Terrarium.git ../Terrarium
```
```toml
# pyproject.toml
[tool.uv.sources]
terrarium-agents = { path = "../Terrarium", editable = true }
```

Then recreate the venv so no files from the PyPI install shadow the checkout (Terrarium writes generated CoLLAB instances inside its installed package, and those survive an uninstall). Add `--extra vllm` to the sync if you use vLLM:

```bash
rm -rf .venv && uv venv --python 3.11 .venv && uv sync --no-install-project
```

Edits under `../Terrarium/terrarium/` then take effect without reinstalling. (A plain `uv pip install -e` is undone by the next `uv sync`.) This also rewrites `uv.lock` to point at your local checkout, so keep the `pyproject.toml` and `uv.lock` changes out of commits (`git restore pyproject.toml uv.lock` before committing).

## Citation
Please cite us using:
```bibtex
@article{nakamura2026colosseumauditingcollusioncooperative,
      title={Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems}, 
      author={Mason Nakamura and Abhinav Kumar and Saswat Das and Sahar Abdelnabi and Saaduddin Mahmud and Ferdinando Fioretto and Shlomo Zilberstein and Eugene Bagdasarian},
      year={2026},
      eprint={2602.15198},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2602.15198}, 
}
```
