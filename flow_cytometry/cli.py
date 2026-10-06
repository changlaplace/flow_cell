"""Command-line orchestration for simulation, training, and saved outputs."""

import argparse
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import ToyCells
from .engine import evaluate, train
from .model import OpticalClassifier
from .plotting import (
    draw_setup,
    save_feature_correlation,
    save_forward_sanity,
    save_phase_masks,
    save_toy_cells,
    save_training_curves,
)
from .utils import save_config, save_json, set_random_seed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse public CLI options while retaining the original interface."""
    parser = argparse.ArgumentParser(
        description="Coherent two-metasurface flow-cytometry prototype"
    )
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("flow_cytometry_outputs")
    )
    parser.add_argument("--grid-N", type=int, default=192)
    parser.add_argument("--stations", type=int, default=3)
    parser.add_argument("--sensor-N", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--backend", choices=["linear", "mlp"], default="linear")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    """Create and validate a run configuration from CLI arguments."""
    config = Config(
        grid_N=args.grid_N,
        n_stations_sim=args.stations,
        sensor_N=args.sensor_N,
        epochs=args.epochs,
        backend=args.backend,
    )
    config.validate()
    return config


def build_data_loaders(
    config: Config,
) -> tuple[DataLoader, DataLoader]:
    """Build deterministic synthetic training and test loaders."""
    train_dataset = ToyCells(
        config.train_samples_per_class, config, config.seed
    )
    test_dataset = ToyCells(
        config.test_samples_per_class, config, config.seed + 1000
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )
    return train_loader, test_loader


def report_environment(config: Config, device: torch.device) -> None:
    """Print the hardware and simulation dimensions for a run."""
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Grid:", config.grid_N)
    print("Stations:", config.n_stations_sim)
    print("Sensor:", f"{config.sensor_N}x{config.sensor_N}")
    print("Fast-mode FOV [um]:", config.grid_N * config.dx_m * 1e6)


def run(config: Config, output_dir: Path, run_training: bool) -> None:
    """Execute one complete simulation or training run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(config.seed)
    save_config(config, output_dir / "config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_environment(config, device)
    draw_setup(
        output_dir / "optical_setup.png",
        station_count=min(max(config.n_stations_sim, 5), 10),
    )
    demo = save_toy_cells(config, output_dir)
    model = OpticalClassifier(config).to(device)
    save_forward_sanity(model, demo, device, config, output_dir)
    save_phase_masks(model, config, output_dir)
    train_loader, test_loader = build_data_loaders(config)

    if run_training:
        history = train(model, train_loader, test_loader, device, config)
        save_training_curves(history, output_dir)
        save_phase_masks(model, config, output_dir)
        metrics = evaluate(model, test_loader, device, config)
        save_feature_correlation(metrics["features"], output_dir)
        torch.save(model.state_dict(), output_dir / "model.pt")
        save_json(
            {
                "accuracy": metrics["accuracy"],
                "joint_bhattacharyya": metrics["joint_bhattacharyya"],
                "history": history,
            },
            output_dir / "metrics.json",
        )
        print("\nFinal accuracy:", metrics["accuracy"])
        print("Final joint Bhattacharyya:", metrics["joint_bhattacharyya"])
    else:
        metrics = evaluate(model, test_loader, device, config)
        save_feature_correlation(metrics["features"], output_dir)
        save_json(
            {
                "accuracy": metrics["accuracy"],
                "joint_bhattacharyya": metrics["joint_bhattacharyya"],
                "note": (
                    "Untrained optical/frontend baseline. Run with "
                    "--run-training for end-to-end optimization."
                ),
            },
            output_dir / "metrics_untrained.json",
        )
        print("\nTraining was not run.")
        print("Use --run-training to optimize M1, M2, and the backend.")

    print("\nSaved outputs to:", output_dir.resolve())
    for path in sorted(output_dir.iterdir()):
        print(" -", path.name)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    run(config_from_args(args), args.output_dir, args.run_training)
