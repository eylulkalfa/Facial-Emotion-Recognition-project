"""Optimizer and LR scheduler factory functions."""

import logging

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 0.001,
    weight_decay: float = 0.0001,
) -> optim.Optimizer:
    """Create optimizer filtering only trainable parameters.

    Args:
        model: PyTorch model instance.
        optimizer_name: "adamw", "adam", or "sgd".
        lr: Learning rate.
        weight_decay: Weight decay penalty factor.

    Returns:
        optim.Optimizer instance.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found in model.")

    name = optimizer_name.lower()
    if name == "adamw":
        return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return optim.SGD(
            trainable_params, lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        raise ValueError(
            f"Unknown optimizer: '{optimizer_name}'. Supported: 'adamw', 'adam', 'sgd'"
        )


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = "reduce_on_plateau",
    factor: float = 0.5,
    patience: int = 4,
):
    """Create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer instance.
        scheduler_name: "reduce_on_plateau" or "cosine".
        factor: Reduction factor for ReduceLROnPlateau.
        patience: Epoch patience for ReduceLROnPlateau.

    Returns:
        LR scheduler instance.
    """
    name = scheduler_name.lower()
    if name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience
        )
    elif name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    else:
        raise ValueError(
            f"Unknown scheduler: '{scheduler_name}'. Supported: 'reduce_on_plateau', 'cosine'"
        )
