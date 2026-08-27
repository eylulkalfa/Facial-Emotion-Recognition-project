"""Multi-phase training orchestrator with AMP, TensorBoard logging, and callbacks."""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fer.config import ProjectConfig, save_config
from fer.models.fer_model import FERModel
from fer.training.callbacks import EarlyStopping, ModelCheckpointer
from fer.training.losses import create_loss
from fer.training.metrics import MetricTracker
from fer.training.optimizer import create_optimizer, create_scheduler
from fer.utils.device import get_device
from fer.utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)


class Trainer:
    """Multi-phase training orchestrator."""

    def __init__(
        self,
        model: FERModel,
        config: ProjectConfig,
        train_loader,
        val_loader,
        class_weights: Optional[torch.Tensor] = None,
        experiment_dir: Optional[str] = None,
    ):
        """Initialize Trainer.

        Args:
            model: FERModel instance.
            config: ProjectConfig configuration.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            class_weights: Inverse-frequency class weights tensor.
            experiment_dir: Output experiment directory path. Auto-generated if None.
        """
        self.model = model
        self.config = config
        self.device = get_device()
        self.model.to(self.device)
        self.logger = logging.getLogger(__name__)

        if experiment_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{config.model.backbone}_{config.data.dataset}_{timestamp}"
            experiment_dir = str(Path(config.experiment.output_dir) / exp_name)

        self.experiment_dir = Path(experiment_dir)
        ensure_dir(self.experiment_dir / "checkpoints")
        ensure_dir(self.experiment_dir / "logs")
        ensure_dir(self.experiment_dir / "results")
        ensure_dir(self.experiment_dir / "exports")

        save_config(config, str(self.experiment_dir / "config.yaml"))

        self.train_loader = train_loader
        self.val_loader = val_loader

        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        label_smoothing = getattr(config.training, "label_smoothing", 0.0)
        self.loss_fn = create_loss(
            config.training.loss,
            class_weights=class_weights,
            focal_gamma=config.training.focal_gamma,
            label_smoothing=label_smoothing,
        )

        self.writer = SummaryWriter(str(self.experiment_dir / "logs"))

        self.checkpointer = ModelCheckpointer(
            save_dir=str(self.experiment_dir / "checkpoints"),
            monitor="macro_f1",
            mode="max",
            top_k=3,
        )

        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
            mode="max",
        )

        self.metric_tracker = MetricTracker()
        self.global_step = 0

    def train(self) -> Dict[str, Any]:
        """Execute multi-phase training loop.

        Returns:
            Dict containing best validation metrics achieved.
        """
        best_metrics: Dict[str, Any] = {}
        total_epoch = 0

        for phase_idx, phase in enumerate(self.config.training.phases):
            self.logger.info(
                f"=== Phase {phase_idx + 1}/{len(self.config.training.phases)}: "
                f"{phase.name} (lr={phase.lr}, freeze_backbone={phase.freeze_backbone}) ==="
            )

            if phase.freeze_backbone:
                self.model.freeze_backbone()
            else:
                self.model.unfreeze_backbone()

            optimizer = create_optimizer(
                self.model,
                self.config.training.optimizer,
                lr=phase.lr,
                weight_decay=self.config.training.weight_decay,
            )
            scheduler = create_scheduler(
                optimizer,
                self.config.training.scheduler,
                factor=self.config.training.scheduler_factor,
                patience=self.config.training.scheduler_patience,
            )

            self.early_stopping.reset()

            use_amp = (
                self.config.training.mixed_precision and self.device.type == "cuda"
            )
            scaler = torch.amp.GradScaler("cuda") if use_amp else None

            for epoch in range(phase.epochs):
                total_epoch += 1

                train_loss = self._train_epoch(optimizer, scaler, use_amp)
                val_metrics = self._val_epoch()
                val_metrics["train_loss"] = train_loss

                for key, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f"val/{key}", value, total_epoch)
                self.writer.add_scalar("train/loss", train_loss, total_epoch)
                self.writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], total_epoch
                )

                self.logger.info(
                    f"Epoch {total_epoch:02d} [{phase.name}] - "
                    f"train_loss={train_loss:.4f}, "
                    f"val_acc={val_metrics['accuracy']:.4f}, "
                    f"val_macro_f1={val_metrics['macro_f1']:.4f}"
                )

                state = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": total_epoch,
                    "phase": phase.name,
                    "metrics": val_metrics,
                    "config": self.config,
                }
                self.checkpointer.step(val_metrics, state, total_epoch)

                if not best_metrics or val_metrics["macro_f1"] > best_metrics.get(
                    "macro_f1", 0
                ):
                    best_metrics = val_metrics.copy()

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["macro_f1"])
                else:
                    scheduler.step()

                if self.early_stopping.step(val_metrics["macro_f1"]):
                    self.logger.info(
                        f"Early stopping triggered in phase '{phase.name}' at epoch {total_epoch}"
                    )
                    break

        self.writer.close()

        serializable_metrics = {
            k: v.tolist() if isinstance(v, torch.Tensor) or hasattr(v, "tolist") else v
            for k, v in best_metrics.items()
        }
        save_json(
            serializable_metrics,
            str(self.experiment_dir / "results" / "best_metrics.json"),
        )
        return best_metrics

    def _train_epoch(self, optimizer, scaler, use_amp: bool) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()

            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            self.global_step += 1

        return total_loss / max(num_batches, 1)

    def _val_epoch(self) -> Dict[str, Any]:
        self.model.eval()
        self.metric_tracker.reset()

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation", leave=False):
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(images)
                self.metric_tracker.update(logits, targets)

        return self.metric_tracker.compute()
