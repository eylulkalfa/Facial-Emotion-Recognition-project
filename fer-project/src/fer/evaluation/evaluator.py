"""Evaluator module for evaluating trained FER models on held-out test datasets."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fer.models.fer_model import FERModel
from fer.training.metrics import MetricTracker
from fer.utils.device import get_device
from fer.utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates a trained FERModel instance on test/validation DataLoaders."""

    def __init__(self, model: FERModel, device: Optional[torch.device] = None):
        """Initialize Evaluator.

        Args:
            model: Instantiated FERModel.
            device: PyTorch compute device. Auto-detected if None.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()
        self.logger = logging.getLogger(__name__)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Run evaluation loop on a dataloader.

        Args:
            dataloader: Test or validation DataLoader.

        Returns:
            Dict containing computed metrics (accuracy, macro_f1, confusion_matrix, etc.).
        """
        tracker = MetricTracker()
        self.model.eval()

        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(images)
                tracker.update(logits, targets)

        return tracker.compute()

    def evaluate_and_save(
        self, dataloader: DataLoader, output_dir: str
    ) -> Dict[str, Any]:
        """Run evaluation, log summary, and save metrics JSON to disk.

        Args:
            dataloader: DataLoader to evaluate.
            output_dir: Output directory path.

        Returns:
            Dict of metrics.
        """
        metrics = self.evaluate(dataloader)
        out_path = Path(output_dir)
        ensure_dir(out_path)

        serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items()
        }
        save_json(serializable, out_path / "metrics.json")

        self.logger.info(f"Saved evaluation metrics to {out_path / 'metrics.json'}")
        self._print_summary(metrics)
        return metrics

    def get_all_predictions(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get raw logits and target arrays for calibration or post-processing analysis.

        Args:
            dataloader: DataLoader.

        Returns:
            Tuple of (logits [N, 7], targets [N]).
        """
        tracker = MetricTracker()
        self.model.eval()

        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Collecting predictions", leave=False):
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(images)
                tracker.update(logits, targets)

        return tracker.get_predictions_and_targets()

    @staticmethod
    def load_from_checkpoint(checkpoint_path: str, model_config) -> "Evaluator":
        """Load trained FERModel from checkpoint file and wrap in Evaluator.

        Args:
            checkpoint_path: Path to PyTorch .pt checkpoint file.
            model_config: ModelConfig instance for model construction.

        Returns:
            Evaluator instance.
        """
        model = FERModel(config=model_config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        return Evaluator(model)

    def _print_summary(self, metrics: Dict[str, Any]) -> None:
        """Log formatted evaluation summary to console."""
        self.logger.info("=" * 60)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"  Accuracy:    {metrics.get('accuracy', 0.0):.4f}")
        self.logger.info(f"  Macro-F1:    {metrics.get('macro_f1', 0.0):.4f}")
        self.logger.info(f"  Weighted-F1: {metrics.get('weighted_f1', 0.0):.4f}")
        if metrics.get("roc_auc") is not None:
            self.logger.info(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        self.logger.info("-" * 60)
        self.logger.info("Per-Class F1 Scores:")
        per_class_f1 = metrics.get("per_class_f1", {})
        for emotion, score in per_class_f1.items():
            self.logger.info(f"  {emotion:12s}: {score:.4f}")
        self.logger.info("=" * 60)
