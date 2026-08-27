"""Visualization tools for confusion matrices, reliability diagrams, F1 scores, and ROC curves."""

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from fer.data.label_mapping import EMOTION_NAMES, NUM_CLASSES


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix as a heatmap.

    Args:
        cm: Raw confusion matrix array [num_classes, num_classes].
        save_path: Output image path.
        normalize: If True, normalize matrix rows to fractions [0, 1].
        title: Plot title string.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1)  # Avoid division by zero
        cm_display = cm.astype(float) / row_sums
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=EMOTION_NAMES,
        yticklabels=EMOTION_NAMES,
        ax=ax,
        vmin=0.0,
        vmax=1.0 if normalize else None,
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_diagram(
    bin_confidences: np.ndarray,
    bin_accuracies: np.ndarray,
    ece: float,
    save_path: str,
) -> None:
    """Plot reliability diagram (calibration curve).

    Args:
        bin_confidences: Mean confidence per bin.
        bin_accuracies: Accuracy per bin.
        ece: Expected Calibration Error value.
        save_path: Output image path.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    n_bins = len(bin_confidences)
    bin_width = 1.0 / max(n_bins, 1)

    ax.bar(
        bin_confidences,
        bin_accuracies,
        width=bin_width,
        alpha=0.7,
        edgecolor="black",
        align="center",
        label="Model Calibration",
    )
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect Calibration")

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_title(f"Reliability Diagram (ECE = {ece:.4f})")
    ax.legend(loc="upper left")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(
    per_class_f1: Dict[str, float],
    save_path: str,
    title: str = "Per-Class F1 Score",
) -> None:
    """Plot horizontal bar chart of per-class F1 scores.

    Args:
        per_class_f1: Dict mapping emotion name to F1 score.
        save_path: Output image path.
        title: Plot title string.
    """
    emotions = list(per_class_f1.keys())
    scores = list(per_class_f1.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(emotions)))
    bars = ax.barh(emotions, scores, color=colors, edgecolor="black")

    ax.set_xlabel("F1 Score")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    probs: np.ndarray, targets: np.ndarray, save_path: str
) -> None:
    """Plot one-vs-rest ROC curves for all classes.

    Args:
        probs: Predicted probabilities [N, num_classes].
        targets: Target integer labels [N].
        save_path: Output image path.
    """
    targets_bin = label_binarize(targets, classes=range(NUM_CLASSES))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(NUM_CLASSES):
        if targets_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(targets_bin[:, i], probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{EMOTION_NAMES[i]} (AUC = {roc_auc_val:.2f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    plt.tight_layout()

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close(fig)
