"""Combined optical frontend and lightweight digital classifier."""

import torch
import torch.nn as nn

from .config import Config
from .optics import OpticalStation


class OpticalClassifier(nn.Module):
    """Concatenate station readouts and classify the compressed features."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.stations = nn.ModuleList(
            [OpticalStation(config) for _ in range(config.n_stations_sim)]
        )

        feature_count = config.n_stations_sim * config.sensor_N**2
        if config.backend == "linear":
            self.backend = nn.Linear(feature_count, config.n_classes)
        elif config.backend == "mlp":
            self.backend = nn.Sequential(
                nn.Linear(feature_count, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, config.n_classes),
            )
        else:
            raise ValueError("backend must be 'linear' or 'mlp'")

    def forward(
        self, cell_phase: torch.Tensor, details: bool = False
    ) -> torch.Tensor | tuple[
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        detector_outputs = []
        intensities = []
        for station in self.stations:
            detector, intensity = station(cell_phase)
            detector_outputs.append(detector.flatten(1))
            intensities.append(intensity)

        features = torch.cat(detector_outputs, dim=1)
        features = (features - features.mean(0, keepdim=True)) / (
            features.std(0, keepdim=True) + 1e-5
        )
        logits = self.backend(features)

        if details:
            return logits, features, detector_outputs, intensities
        return logits
