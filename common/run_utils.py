from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from tqdm import tqdm


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, data)
    tmp.replace(path)


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass


def configure_experiment_logging(
    logger: logging.Logger, root: Path, *, verbose: bool = True
) -> None:
    """Configure a dedicated experiment logger (file + tqdm-safe console)."""
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Reset handlers so repeated calls (or notebooks) re-point to the new output root.
    for handler in list(logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)

    log_path = root / "experiment.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(file_handler)

    if verbose:
        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(console_handler)


def write_progress(root: Path, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = datetime.now().isoformat()
    atomic_write_json(root / "progress.json", payload)


def normalize_seeds(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, str):
        # Allow comma-separated values as a convenience.
        parts = [p.strip() for p in raw.split(",")]
        seeds: List[int] = []
        for part in parts:
            if not part:
                continue
            try:
                seeds.append(int(part))
            except Exception:
                continue
        return seeds
    if isinstance(raw, list):
        seeds = []
        for item in raw:
            if item is None:
                continue
            try:
                seeds.append(int(item))
            except Exception:
                continue
        return seeds
    return []
