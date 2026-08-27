"""CLI entry point for training Facial Emotion Recognition models."""

import argparse
import logging
import sys

from fer.config import load_config
from fer.data import create_dataloaders
from fer.models.fer_model import FERModel
from fer.training.trainer import Trainer
from fer.utils.logging import setup_logger
from fer.utils.seeding import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train Facial Emotion Recognition Model")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Path to one or more YAML config files (base config first, then overrides).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed specified in configuration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(*args.config)

    # Seed override if specified
    if args.seed is not None:
        config.experiment.seed = args.seed

    # Set random seed
    seed_everything(config.experiment.seed)

    # Setup main logger
    logger = setup_logger("fer")
    logger.info(f"Starting training run: {config.experiment.name}")
    logger.info(f"Backbone: {config.model.backbone}, Dataset: {config.data.dataset}")

    # DataLoaders
    train_loader, val_loader, _ = create_dataloaders(config)
    class_weights = train_loader.dataset.get_class_weights()

    # Model
    model = FERModel(config=config.model)

    # Trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
    )

    # Execute training
    best_metrics = trainer.train()

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best Macro-F1: {best_metrics.get('macro_f1', 0.0):.4f}")
    logger.info(f"Best Accuracy: {best_metrics.get('accuracy', 0.0):.4f}")
    logger.info(f"Results saved to: {trainer.experiment_dir}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
