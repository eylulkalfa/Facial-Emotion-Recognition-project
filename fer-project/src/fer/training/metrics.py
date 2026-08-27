"""MetricTracker for epoch-level evaluation using scikit-learn."""

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

from fer.data.label_mapping import EMOTION_NAMES, NUM_CLASSES


class MetricTracker:
    """Accumulates batch predictions and computes epoch-level classification metrics."""

    def __init__(self):
        """Initialize MetricTracker."""
        self.all_logits: List[np.ndarray] = []
        self.all_targets: List[np.ndarray] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate a batch of predictions and targets.

        Args:
            logits: Raw logits tensor of shape [B, num_classes].
            targets: Ground truth target labels tensor of shape [B].
        """
        logits_np = logits.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        self.all_logits.append(logits_np)
        self.all_targets.append(targets_np)

    def compute(self) -> Dict[str, Any]:
        """Compute all classification metrics from accumulated predictions.

        Returns:
            Dict containing accuracy, macro_f1, weighted_f1, per-class metrics,
            confusion_matrix, and roc_auc.
        """
        if not self.all_logits:
            raise ValueError("No predictions accumulated. Call update() before compute().")

        concat_logits = np.concatenate(self.all_logits, axis=0)
        concat_targets = np.concatenate(self.all_targets, axis=0)

        predictions = np.argmax(concat_logits, axis=1)

        # Stable softmax for probabilities
        shifted_logits = concat_logits - np.max(concat_logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        accuracy = float(accuracy_score(concat_targets, predictions))
        macro_f1 = float(
            f1_score(concat_targets, predictions, average="macro", zero_division=0)
        )
        weighted_f1 = float(
            f1_score(concat_targets, predictions, average="weighted", zero_division=0)
        )

        precision_per_class = precision_score(
            concat_targets,
            predictions,
            average=None,
            labels=range(NUM_CLASSES),
            zero_division=0,
        )
        recall_per_class = recall_score(
            concat_targets,
            predictions,
            average=None,
            labels=range(NUM_CLASSES),
            zero_division=0,
        )
        f1_per_class = f1_score(
            concat_targets,
            predictions,
            average=None,
            labels=range(NUM_CLASSES),
            zero_division=0,
        )

        per_class_precision = {
            EMOTION_NAMES[i]: float(precision_per_class[i]) for i in range(NUM_CLASSES)
        }
        per_class_recall = {
            EMOTION_NAMES[i]: float(recall_per_class[i]) for i in range(NUM_CLASSES)
        }
        per_class_f1 = {
            EMOTION_NAMES[i]: float(f1_per_class[i]) for i in range(NUM_CLASSES)
        }

        cm = confusion_matrix(concat_targets, predictions, labels=range(NUM_CLASSES))

        roc_auc: Any = None
        try:
            roc_auc = float(
                roc_auc_score(
                    concat_targets,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                    labels=range(NUM_CLASSES),
                )
            )
        except Exception:
            roc_auc = None

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm,
            "roc_auc": roc_auc,
        }

    def reset(self) -> None:
        """Clear accumulated batch predictions."""
        self.all_logits = []
        self.all_targets = []

    def get_predictions_and_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return raw concatenated logits and targets for calibration or post-processing."""
        if not self.all_logits:
            return np.empty((0, NUM_CLASSES)), np.empty((0,))
        return np.concatenate(self.all_logits, axis=0), np.concatenate(
            self.all_targets, axis=0
        )
