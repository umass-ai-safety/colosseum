"""Self-Contained Experiments."""

import os
from pathlib import Path

from dotenv import load_dotenv

# The repo-root .env is the only .env: load it before Terrarium creates any LLM client,
# then turn off later load_dotenv() calls. Terrarium reloads .env with override=True from
# wherever it is installed, so an editable checkout's own .env (see README) would
# otherwise blank or replace these keys.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
