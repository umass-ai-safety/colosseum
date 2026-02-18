from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from terrarium.logger import BlackboardLogger


class ExperimentBlackboardLogger(BlackboardLogger):
    """BlackboardLogger that writes blackboard_*.txt into an experiment run directory.

    This avoids using the default `logs/<env>/<tag_model>/<run_timestamp>/seed_<seed>/` layout so
    experiment artifacts stay self-contained under `experiments/<project>/outputs/.../<run>/`.
    """

    def __init__(self, config: Dict[str, Any], *, log_root: Path) -> None:
        # Intentionally do not call super().__init__ (which builds `logs/...`).
        self.session_start = time.time()
        self.config = config
        self.environment_name = str(
            (config.get("environment") or {}).get("name") or "Unknown"
        )
        self.seed = int((config.get("simulation") or {}).get("seed") or 0)
        self.tag_model = ""
        self.run_timestamp = (config.get("simulation") or {}).get("run_timestamp")
        self.log_root = Path(log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.blackboard_log_files = {}
