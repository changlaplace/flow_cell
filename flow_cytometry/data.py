"""Synthetic thin-phase cell generation and datasets."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config


SYNTHETIC_CLASS_NAMES = (
    "small round",
    "large ellipse",
    "bilobed",
    "ring / granular",
)


def xy_grid(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a normalized square coordinate grid."""
    axis = torch.linspace(-1, 1, size)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return x, y


def toy_phase(
    label: int,
    size: int,
    rng: np.random.Generator,
    depth: float,
) -> torch.Tensor:
    """Generate a visibly distinct class-conditioned synthetic phase map."""
    x, y = xy_grid(size)
    center_x, center_y = rng.uniform(-0.07, 0.07, size=2)
    angle = rng.uniform(-0.25, 0.25)
    scale_jitter = rng.uniform(0.90, 1.10)

    shifted_x = x - center_x
    shifted_y = y - center_y
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotated_x = cos_angle * shifted_x + sin_angle * shifted_y
    rotated_y = -sin_angle * shifted_x + cos_angle * shifted_y

    def gaussian(
        local_x: torch.Tensor,
        local_y: torch.Tensor,
        width_x: float,
        width_y: float,
    ) -> torch.Tensor:
        return torch.exp(
            -0.5
            * ((local_x / width_x) ** 2 + (local_y / width_y) ** 2)
        )

    if label == 0:
        # Compact, nearly circular cell.
        phase = gaussian(
            rotated_x,
            rotated_y,
            0.105 * scale_jitter,
            0.105 * scale_jitter,
        )
    elif label == 1:
        # A much larger elongated cell with an off-center dense core.
        body = gaussian(
            rotated_x,
            rotated_y,
            0.34 * scale_jitter,
            0.15 * scale_jitter,
        )
        core = gaussian(
            rotated_x - 0.10 * scale_jitter,
            rotated_y + 0.02 * scale_jitter,
            0.075 * scale_jitter,
            0.070 * scale_jitter,
        )
        phase = 0.62 * body + 0.72 * core
    elif label == 2:
        # Two separated lobes with a faint bridge.
        separation = 0.17 * scale_jitter
        left_lobe = gaussian(
            rotated_x + separation,
            rotated_y,
            0.105 * scale_jitter,
            0.145 * scale_jitter,
        )
        right_lobe = gaussian(
            rotated_x - separation,
            rotated_y,
            0.105 * scale_jitter,
            0.145 * scale_jitter,
        )
        bridge = gaussian(
            rotated_x,
            rotated_y,
            0.25 * scale_jitter,
            0.075 * scale_jitter,
        )
        phase = left_lobe + right_lobe + 0.18 * bridge
    else:
        # Large annular cell with several reproducibly random granules.
        radius = torch.sqrt(
            (rotated_x / (0.30 * scale_jitter)) ** 2
            + (rotated_y / (0.23 * scale_jitter)) ** 2
        )
        phase = torch.exp(-0.5 * ((radius - 0.72) / 0.16) ** 2)
        for _ in range(4):
            granule_angle = rng.uniform(0, 2 * np.pi)
            granule_x = 0.20 * scale_jitter * np.cos(granule_angle)
            granule_y = 0.15 * scale_jitter * np.sin(granule_angle)
            phase = phase + 0.22 * gaussian(
                rotated_x - granule_x,
                rotated_y - granule_y,
                0.040 * scale_jitter,
                0.040 * scale_jitter,
            )

    return (phase / (phase.max() + 1e-8) * depth).float()


class ToyCells(Dataset):
    """In-memory synthetic phase-cell dataset."""

    def __init__(self, samples_per_class: int, config: Config, seed: int):
        rng = np.random.default_rng(seed)
        self.data = [
            (
                toy_phase(
                    label,
                    config.grid_N,
                    rng,
                    config.phase_depth_rad,
                ),
                label,
            )
            for label in range(config.n_classes)
            for _ in range(samples_per_class)
        ]
        rng.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.data[index]
