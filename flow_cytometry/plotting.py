"""Plot generation for optical setup, diagnostics, and training outputs."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch, Rectangle

from .config import Config
from .data import SYNTHETIC_CLASS_NAMES, ToyCells
from .engine import TrainingHistory
from .model import OpticalClassifier
from .utils import save_figure


def draw_setup(output_path: Path, station_count: int = 5) -> None:
    """Draw the proposal-style array of paired optical stations."""
    fig, axis = plt.subplots(figsize=(14, 6))
    axis.set_xlim(-1, station_count * 2.2 + 4)
    axis.set_ylim(-4.3, 4.6)
    axis.axis("off")
    axis.add_patch(
        Rectangle((-0.4, -0.45), station_count * 2.2 + 0.8, 0.9, alpha=0.18)
    )
    axis.add_patch(
        FancyArrowPatch(
            (-0.2, 0),
            (station_count * 2.2 + 0.2, 0),
            arrowstyle="->",
            mutation_scale=18,
            lw=2,
        )
    )
    axis.text(-0.25, 0.8, "microfluidic channel", fontsize=12, weight="bold")
    axis.text(station_count * 2.2 + 0.35, 0, "flow ~1 m/s", va="center")

    for index in range(station_count):
        x_position = 0.6 + index * 2.2
        axis.add_patch(
            FancyArrowPatch(
                (x_position, 4.0),
                (x_position, 3.05),
                arrowstyle="->",
                mutation_scale=14,
                lw=1.5,
            )
        )
        axis.add_patch(
            Rectangle((x_position - 0.55, 2.35), 1.1, 0.45, alpha=0.45)
        )
        axis.text(
            x_position,
            2.58,
            f"M1[{index}]",
            ha="center",
            va="center",
            weight="bold",
        )
        axis.add_patch(
            FancyArrowPatch(
                (x_position, 2.3),
                (x_position, 0.55),
                arrowstyle="->",
                mutation_scale=14,
                lw=1.2,
            )
        )
        axis.scatter([x_position], [0], s=90, zorder=3)
        axis.add_patch(
            FancyArrowPatch(
                (x_position, -0.55),
                (x_position, -2.15),
                arrowstyle="->",
                mutation_scale=14,
                lw=1.2,
            )
        )
        axis.add_patch(
            Rectangle((x_position - 0.55, -2.75), 1.1, 0.45, alpha=0.45)
        )
        axis.text(
            x_position,
            -2.53,
            f"M2[{index}]",
            ha="center",
            va="center",
            weight="bold",
        )
        axis.add_patch(
            FancyArrowPatch(
                (x_position, -2.85),
                (x_position, -3.55),
                arrowstyle="->",
                mutation_scale=14,
                lw=1.2,
            )
        )
        axis.add_patch(
            Rectangle((x_position - 0.55, -4.0), 1.1, 0.28, alpha=0.35)
        )

    axis.text(
        0.55,
        4.2,
        "coherent visible laser -> plane-wave input to each station",
        fontsize=12,
        weight="bold",
    )
    axis.text(
        station_count * 2.2 + 1.0,
        2.55,
        "M1 array\nstructured illumination",
        va="center",
    )
    axis.text(
        station_count * 2.2 + 1.0,
        -2.5,
        "M2 array\ncoherent mixing / compression",
        va="center",
    )
    axis.text(
        station_count * 2.2 + 1.0,
        -3.85,
        "low-res intensity sensor",
        va="center",
    )
    axis.text(
        0.6,
        -4.45,
        "Each vertical column is one paired optical transform; "
        "the proposal scales toward ~10-100 stations.",
    )
    save_figure(fig, output_path)


def save_toy_cells(config: Config, output_dir: Path) -> ToyCells:
    """Save phase maps and peak-crossing profiles for every synthetic class."""
    demo = ToyCells(2, config, config.seed)
    fig, axes = plt.subplots(
        2,
        config.n_classes,
        figsize=(3.2 * config.n_classes, 5.7),
        squeeze=False,
        constrained_layout=True,
    )
    images = []
    profile_x = torch.linspace(-1, 1, config.grid_N)

    for label in range(config.n_classes):
        phase = next(item for item, item_label in demo if item_label == label)
        image = axes[0, label].imshow(
            phase,
            cmap="viridis",
            vmin=0,
            vmax=config.phase_depth_rad,
            extent=(-1, 1, -1, 1),
        )
        images.append(image)
        class_name = (
            SYNTHETIC_CLASS_NAMES[label]
            if label < len(SYNTHETIC_CLASS_NAMES)
            else f"class {label}"
        )
        axes[0, label].set_title(f"class {label}: {class_name}")
        axes[0, label].set_xlabel("normalized x")
        if label == 0:
            axes[0, label].set_ylabel("normalized y")

        peak_row = int(torch.argmax(phase).item() // config.grid_N)
        axes[1, label].plot(profile_x, phase[peak_row], color="tab:blue")
        axes[1, label].set_ylim(0, config.phase_depth_rad * 1.05)
        axes[1, label].set_xlabel("normalized x")
        axes[1, label].grid(alpha=0.25)
        if label == 0:
            axes[1, label].set_ylabel("phase delay [rad]")

    fig.colorbar(
        images[-1],
        ax=axes[0].tolist(),
        label="phase delay [rad]",
        fraction=0.025,
        pad=0.02,
    )
    fig.suptitle("Synthetic thin phase-object cells")
    save_figure(fig, output_dir / "toy_phase_objects.png")
    return demo


def save_forward_sanity(
    model: OpticalClassifier,
    demo: ToyCells,
    device: torch.device,
    config: Config,
    output_dir: Path,
) -> torch.Tensor:
    """Save representative phase, intensity, and detector readout images."""
    phase = torch.stack(
        [demo[index][0] for index in range(min(8, len(demo)))]
    ).to(device)
    model.eval()
    with torch.no_grad():
        _, features, detector_outputs, intensities = model(phase, details=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    axes[0].imshow(phase[0].cpu(), cmap="viridis")
    axes[0].set_title("cell phase")
    axes[1].imshow(intensities[0][0].cpu(), cmap="magma")
    axes[1].set_title("coherent intensity")
    axes[2].imshow(
        detector_outputs[0][0]
        .reshape(config.sensor_N, config.sensor_N)
        .cpu(),
        cmap="magma",
    )
    axes[2].set_title(f"{config.sensor_N}x{config.sensor_N} detector")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    save_figure(fig, output_dir / "forward_sanity.png")
    return features.detach().cpu()


def save_phase_masks(
    model: OpticalClassifier, config: Config, output_dir: Path
) -> None:
    """Save both learned phase masks for up to four stations."""
    station_count = min(config.n_stations_sim, 4)
    fig, axes = plt.subplots(
        station_count, 2, figsize=(8, 3 * station_count), squeeze=False
    )
    for index in range(station_count):
        station = model.stations[index]
        phase_1 = station.m1.phase(config.grid_N).detach().cpu()
        phase_2 = station.m2.phase(config.grid_N).detach().cpu()
        axes[index, 0].imshow(
            phase_1, cmap="twilight", vmin=-math.pi, vmax=math.pi
        )
        axes[index, 0].set_title(f"M1[{index}] phase")
        axes[index, 1].imshow(
            phase_2, cmap="twilight", vmin=-math.pi, vmax=math.pi
        )
        axes[index, 1].set_title(f"M2[{index}] phase")
        axes[index, 0].axis("off")
        axes[index, 1].axis("off")
    fig.tight_layout()
    save_figure(fig, output_dir / "phase_masks.png")


def save_feature_correlation(features: torch.Tensor, output_dir: Path) -> None:
    """Save the detector-feature correlation matrix."""
    correlation = torch.corrcoef(features.T)
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(image, ax=axis, label="feature correlation")
    axis.set_title("Joint detector readout correlations")
    axis.set_xlabel("detector feature")
    axis.set_ylabel("detector feature")
    fig.tight_layout()
    save_figure(fig, output_dir / "feature_correlation.png")


def save_training_curves(
    history: TrainingHistory, output_dir: Path
) -> None:
    """Save loss, accuracy, and separability curves."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    axes[0].plot(history["loss"])
    axes[0].set_title("cross-entropy")
    axes[0].set_xlabel("epoch")
    axes[1].plot(history["accuracy"])
    axes[1].set_title("test accuracy")
    axes[1].set_xlabel("epoch")
    axes[2].plot(history["joint_bhattacharyya"])
    axes[2].set_title("joint Bhattacharyya distance")
    axes[2].set_xlabel("epoch")
    fig.tight_layout()
    save_figure(fig, output_dir / "training_curves.png")
