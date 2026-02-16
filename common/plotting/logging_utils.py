from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def configure_basic_logging(*, level: int = logging.INFO) -> None:
    """Configure default console logging if the app hasn't set up handlers yet."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _format_path(path: PathLike) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except Exception:
        return str(path)


def log_saved_plot(path: PathLike, *, logger: Optional[logging.Logger] = None) -> None:
    (logger or logging.getLogger(__name__)).info("Saved plot: %s", _format_path(path))
