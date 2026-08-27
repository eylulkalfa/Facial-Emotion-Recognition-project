"""Training callbacks: EarlyStopping and ModelCheckpointer."""

import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to halt training when monitored metric plateaus."""

    def __init__(
        self, patience: int = 10, min_delta: float = 0.001, mode: str = "max"
    ):
        """Initialize EarlyStopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change in monitored metric to qualify as an improvement.
            mode: "max" for metrics where higher is better (e.g. F1), "min" for loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False
        self.logger = logging.getLogger(__name__)

    def step(self, current_value: float) -> bool:
        """Update early stopping state with current epoch metric.

        Args:
            current_value: Metric value from current epoch.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == "max":
            improved = current_value > (self.best_value + self.min_delta)
        elif self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Expected 'max' or 'min'.")

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(
                f"EarlyStopping counter: {self.counter}/{self.patience} (best: {self.best_value:.4f})"
            )

        self.should_stop = self.counter >= self.patience
        return self.should_stop

    def reset(self) -> None:
        """Reset counter and best value (used during phase transitions)."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class ModelCheckpointer:
    """Model Checkpointer callback to save best and top-K model checkpoints."""

    def __init__(
        self,
        save_dir: str,
        monitor: str = "macro_f1",
        mode: str = "max",
        top_k: int = 3,
    ):
        """Initialize ModelCheckpointer.

        Args:
            save_dir: Directory where checkpoints should be saved.
            monitor: Name of the metric key to monitor.
            mode: "max" or "min".
            top_k: Number of best checkpoints to retain on disk.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode.lower()
        self.top_k = top_k
        self.best_value: Optional[float] = None
        self.checkpoints: List[Tuple[float, str]] = []
        self.logger = logging.getLogger(__name__)

    def step(
        self, metrics: Dict[str, Any], state: Dict[str, Any], epoch: int
    ) -> Optional[str]:
        """Save checkpoint and manage top-K files on disk.

        Args:
            metrics: Current epoch metrics dictionary containing self.monitor.
            state: Dictionary of states to save (model_state_dict, optimizer, epoch, etc.).
            epoch: Current epoch integer.

        Returns:
            Path string to saved epoch checkpoint.
        """
        if self.monitor not in metrics:
            self.logger.warning(
                f"Monitored metric '{self.monitor}' not found in metrics dict."
            )
            return None

        current_value = float(metrics[self.monitor])

        if self.mode == "max":
            is_best = (self.best_value is None) or (current_value > self.best_value)
        else:
            is_best = (self.best_value is None) or (current_value < self.best_value)

        epoch_path = self.save_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(state, epoch_path)
        self.checkpoints.append((current_value, str(epoch_path)))
        self.logger.info(f"Saved checkpoint: {epoch_path}")

        if is_best:
            self.best_value = current_value
            best_path = self.save_dir / "best_model.pt"
            shutil.copy2(str(epoch_path), str(best_path))
            self.logger.info(
                f"New best model saved to {best_path} ({self.monitor}={current_value:.4f})"
            )

        last_path = self.save_dir / "last_model.pt"
        shutil.copy2(str(epoch_path), str(last_path))

        # Top-K cleanup
        if len(self.checkpoints) > self.top_k:
            reverse_sort = self.mode == "max"
            self.checkpoints.sort(key=lambda x: x[0], reverse=reverse_sort)

            while len(self.checkpoints) > self.top_k:
                worst_val, worst_str_path = self.checkpoints.pop()
                p = Path(worst_str_path)
                if p.exists() and p.name not in ("best_model.pt", "last_model.pt"):
                    p.unlink()
                    self.logger.info(f"Removed old top-K checkpoint: {worst_str_path}")

        return str(epoch_path)
