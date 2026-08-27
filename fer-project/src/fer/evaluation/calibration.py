"""Probability calibration using temperature scaling and Expected Calibration Error (ECE)."""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """Learns a single temperature parameter to calibrate model probabilities."""

    def __init__(self):
        """Initialize TemperatureScaling with initial T=1.5."""
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.logger = logging.getLogger(__name__)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature.

        Args:
            logits: Raw logits tensor of shape [B, num_classes].

        Returns:
            Temperature-scaled logits tensor of shape [B, num_classes].
        """
        return logits / self.temperature

    def fit(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Learn optimal temperature parameter on validation set using LBFGS.

        Args:
            logits: Validation logits array of shape [N, num_classes].
            targets: Validation ground truth targets array of shape [N].
            lr: Learning rate for LBFGS optimizer.
            max_iter: Maximum iterations.

        Returns:
            Learned temperature float value.
        """
        logits_tensor = torch.from_numpy(logits).float()
        targets_tensor = torch.from_numpy(targets).long()

        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        loss_fn = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(logits_tensor)
            loss = loss_fn(scaled, targets_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)

        learned_temp = float(self.temperature.item())
        self.logger.info(f"Learned temperature scaling parameter: T = {learned_temp:.4f}")
        return learned_temp

    def get_calibrated_probs(self, logits: np.ndarray) -> np.ndarray:
        """Apply learned temperature and return calibrated probability distribution.

        Args:
            logits: Raw model logits array of shape [N, num_classes].

        Returns:
            Calibrated probability distribution array of shape [N, num_classes].
        """
        with torch.no_grad():
            logits_tensor = torch.from_numpy(logits).float()
            scaled_logits = self.forward(logits_tensor)
            probs_tensor = torch.softmax(scaled_logits, dim=1)
        return probs_tensor.numpy()


def compute_ece(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 15
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error (ECE) and bin statistics.

    Args:
        probs: Predicted probability array of shape [N, num_classes].
        targets: Target integer labels array of shape [N].
        n_bins: Number of confidence bins.

    Returns:
        Tuple of (ece_float, bin_confidences_array, bin_accuracies_array).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            mask = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            mask = (confidences >= bin_lower) & (confidences < bin_upper)

        count = int(mask.sum())
        if count > 0:
            bin_conf = float(confidences[mask].mean())
            bin_acc = float(accuracies[mask].mean())
        else:
            bin_conf = float((bin_lower + bin_upper) / 2.0)
            bin_acc = 0.0

        bin_confidences.append(bin_conf)
        bin_accuracies.append(bin_acc)
        bin_counts.append(count)

    total_samples = float(sum(bin_counts))
    if total_samples == 0:
        return 0.0, np.array(bin_confidences), np.array(bin_accuracies)

    ece = sum(
        (count / total_samples) * abs(conf - acc)
        for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts)
    )

    return float(ece), np.array(bin_confidences), np.array(bin_accuracies)
