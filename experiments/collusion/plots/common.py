from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from experiments.common.plotting.io_utils import infer_labels_from_sweep_dir


def canonical_variant(name: Any) -> str:
    return str(name or "").strip()


def default_out_dir(*, sweep_dir: Path, requested_out_dir: Optional[str]) -> Path:
    labels = infer_labels_from_sweep_dir(sweep_dir)
    return (
        Path(requested_out_dir).expanduser().resolve()
        if requested_out_dir
        else Path("experiments/collusion/plots_outputs")
        / labels.experiment_tag
        / labels.timestamp
        / labels.model_label
        / labels.sweep_name
    )
