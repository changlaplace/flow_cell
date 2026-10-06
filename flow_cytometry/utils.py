"""Small, domain-independent file and reproducibility helpers."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from .config import Config


def save_figure(fig: Figure, path: Path) -> None:
    """Save and close a Matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_json(data: Any, path: Path) -> None:
    """Serialize JSON consistently across all pipeline outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_config(config: Config, path: Path) -> None:
    """Save a run configuration as JSON."""
    save_json(asdict(config), path)


def set_random_seed(seed: int) -> None:
    """Seed NumPy and PyTorch for reproducible initialization and data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
