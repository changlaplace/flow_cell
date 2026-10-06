"""Feature-space separability metrics."""

import torch


def bhattacharyya_distance(
    mean_1: torch.Tensor,
    covariance_1: torch.Tensor,
    mean_2: torch.Tensor,
    covariance_2: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Compute multivariate Gaussian Bhattacharyya distance."""
    dimensions = mean_1.numel()
    identity = torch.eye(
        dimensions, dtype=mean_1.dtype, device=mean_1.device
    )
    covariance_1 = covariance_1 + epsilon * identity
    covariance_2 = covariance_2 + epsilon * identity
    covariance = 0.5 * (covariance_1 + covariance_2)
    delta = (mean_1 - mean_2)[:, None]

    mean_term = 0.125 * (
        delta.T @ torch.linalg.solve(covariance, delta)
    ).squeeze()
    _, logdet = torch.linalg.slogdet(covariance)
    _, logdet_1 = torch.linalg.slogdet(covariance_1)
    _, logdet_2 = torch.linalg.slogdet(covariance_2)
    return mean_term + 0.5 * (logdet - 0.5 * (logdet_1 + logdet_2))


def mean_joint_bhattacharyya_distance(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_count: int,
) -> torch.Tensor:
    """Average pairwise class distance in the joint detector feature space."""
    statistics = []
    for label in range(class_count):
        class_features = features[labels == label]
        if len(class_features) < 3:
            continue
        mean = class_features.mean(0)
        centered = class_features - mean
        covariance = centered.T @ centered / max(len(class_features) - 1, 1)
        statistics.append((mean, covariance))

    distances = [
        bhattacharyya_distance(*statistics[i], *statistics[j])
        for i in range(len(statistics))
        for j in range(i + 1, len(statistics))
    ]
    if not distances:
        raise ValueError("At least two classes with three samples each are required")
    return torch.stack(distances).mean()


bhatta = bhattacharyya_distance
mean_joint_bd = mean_joint_bhattacharyya_distance
