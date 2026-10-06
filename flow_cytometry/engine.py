"""Evaluation and optimization loops."""

from typing import TypedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .metrics import mean_joint_bhattacharyya_distance
from .model import OpticalClassifier


class EvaluationMetrics(TypedDict):
    accuracy: float
    joint_bhattacharyya: float
    features: torch.Tensor
    labels: torch.Tensor


class TrainingHistory(TypedDict):
    loss: list[float]
    accuracy: list[float]
    joint_bhattacharyya: list[float]


def evaluate(
    model: OpticalClassifier,
    loader: DataLoader,
    device: torch.device,
    config: Config,
) -> EvaluationMetrics:
    """Evaluate classification and feature-space separability."""
    model.eval()
    correct = 0
    total = 0
    feature_batches = []
    label_batches = []

    with torch.no_grad():
        for cell_phase, labels in loader:
            cell_phase = cell_phase.to(device)
            labels = labels.to(device)
            logits, features, _, _ = model(cell_phase, details=True)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
            feature_batches.append(features.cpu())
            label_batches.append(labels.cpu())

    features = torch.cat(feature_batches)
    labels = torch.cat(label_batches)
    return {
        "accuracy": correct / total,
        "joint_bhattacharyya": float(
            mean_joint_bhattacharyya_distance(
                features, labels, config.n_classes
            )
        ),
        "features": features,
        "labels": labels,
    }


def train(
    model: OpticalClassifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config: Config,
) -> TrainingHistory:
    """Jointly optimize phase masks and the digital backend."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    history: TrainingHistory = {
        "loss": [],
        "accuracy": [],
        "joint_bhattacharyya": [],
    }

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for cell_phase, labels in train_loader:
            cell_phase = cell_phase.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(cell_phase), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        metrics = evaluate(model, test_loader, device, config)
        history["loss"].append(running_loss / len(train_loader))
        history["accuracy"].append(metrics["accuracy"])
        history["joint_bhattacharyya"].append(
            metrics["joint_bhattacharyya"]
        )
        print(
            f"Epoch {epoch + 1:02d}/{config.epochs}: "
            f"loss={history['loss'][-1]:.4f}, "
            f"acc={metrics['accuracy']:.4f}, "
            f"joint_BD={metrics['joint_bhattacharyya']:.4f}"
        )

    return history
