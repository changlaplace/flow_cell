"""Configuration for the optical simulator and training pipeline."""

from dataclasses import dataclass


@dataclass
class Config:
    """All optical, data, and optimization parameters for one run."""

    wavelength_m: float = 632.8e-9
    dx_m: float = 0.35e-6
    paper_grid_N: int = 1200
    grid_N: int = 192

    z_m1_cell_m: float = 8e-3
    z_cell_m2_m: float = 8e-3
    z_m2_sensor_m: float = 4e-3

    sensor_N: int = 4
    n_stations_sim: int = 3
    target_stations: int = 10
    phase_ctrl_N: int = 32
    cell_speed_m_s: float = 1.0

    n_classes: int = 4
    phase_depth_rad: float = 1.7
    train_samples_per_class: int = 48
    test_samples_per_class: int = 16

    batch_size: int = 16
    epochs: int = 8
    lr: float = 3e-3
    backend: str = "linear"
    seed: int = 7

    def validate(self) -> None:
        """Fail early for invalid command-line configurations."""
        positive_ints = {
            "grid_N": self.grid_N,
            "sensor_N": self.sensor_N,
            "n_stations_sim": self.n_stations_sim,
            "phase_ctrl_N": self.phase_ctrl_N,
            "n_classes": self.n_classes,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }
        invalid = [name for name, value in positive_ints.items() if value <= 0]
        if invalid:
            raise ValueError(f"Expected positive values for: {', '.join(invalid)}")
        if self.backend not in {"linear", "mlp"}:
            raise ValueError("backend must be 'linear' or 'mlp'")
