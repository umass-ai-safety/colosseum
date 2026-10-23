"""Self-Contained Experiments."""

from pathlib import Path

from dotenv import load_dotenv

# Load the repo-root .env before Terrarium creates any LLM client. Terrarium's own
# load_dotenv() searches upward from where Terrarium is installed, so it misses this
# file when Terrarium is an editable checkout outside the repo (see README).
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
