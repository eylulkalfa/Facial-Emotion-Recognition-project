"""CLI script for evaluating trained FER models, running calibration, and saving visualizations."""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np

from fer.config import load_config
from fer.data import create_dataloaders
from fer.evaluation.calibration import TemperatureScaling, compute_ece
from fer.evaluation.evaluator import Evaluator
from fer.evaluation.visualization import (
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_reliability_diagram,
    plot_roc_curves,
)
from fer.utils.io import ensure_dir, save_json
from fer.utils.logging import setup_logger
from fer.utils.seeding import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained FER Model")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Path to YAML configuration files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained PyTorch .pt model checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation results and plots.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Data split to evaluate on.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run temperature scaling calibration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(*args.config)
    seed_everything(config.experiment.seed)

    logger = setup_logger("fer_eval")
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")

    # Determine output directory
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        ckpt_path = Path(args.checkpoint)
        out_dir = ckpt_path.parent.parent / "results"

    ensure_dir(out_dir)

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(config)

    if args.split == "test" and test_loader is not None:
        eval_loader = test_loader
    else:
        eval_loader = val_loader

    # Load evaluator from checkpoint
    evaluator = Evaluator.load_from_checkpoint(args.checkpoint, config.model)

    # Execute evaluation
    metrics = evaluator.evaluate(eval_loader)
    logits, targets = evaluator.get_all_predictions(eval_loader)

    # Compute uncalibrated probabilities and ECE
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs_uncalibrated = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    ece_before, bin_confs_before, bin_accs_before = compute_ece(
        probs_uncalibrated, targets
    )
    metrics["ece_uncalibrated"] = float(ece_before)

    # Calibration if requested
    if args.calibrate:
        logger.info("Running temperature scaling calibration on validation set...")
        val_logits, val_targets = evaluator.get_all_predictions(val_loader)
        ts = TemperatureScaling()
        learned_temp = ts.fit(val_logits, val_targets)
        metrics["learned_temperature"] = float(learned_temp)

        probs_calibrated = ts.get_calibrated_probs(logits)
        ece_after, bin_confs_after, bin_accs_after = compute_ece(
            probs_calibrated, targets
        )
        metrics["ece_calibrated"] = float(ece_after)
        logger.info(
            f"ECE Before: {ece_before:.4f} -> ECE After: {ece_after:.4f} (T={learned_temp:.4f})"
        )

        plot_reliability_diagram(
            bin_confs_after,
            bin_accs_after,
            ece_after,
            str(out_dir / "reliability_diagram.png"),
        )
    else:
        plot_reliability_diagram(
            bin_confs_before,
            bin_accs_before,
            ece_before,
            str(out_dir / "reliability_diagram.png"),
        )

    # Generate plots
    if "confusion_matrix" in metrics:
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(cm, str(out_dir / "confusion_matrix.png"))

    if "per_class_f1" in metrics:
        plot_per_class_f1(metrics["per_class_f1"], str(out_dir / "per_class_f1.png"))

    plot_roc_curves(probs_uncalibrated, targets, str(out_dir / "roc_curves.png"))

    # Save metrics JSON
    serializable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in metrics.items()
    }
    save_json(serializable, out_dir / "metrics.json")
    logger.info(f"Evaluation complete. Results saved to {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
