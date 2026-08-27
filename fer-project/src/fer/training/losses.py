"""Loss functions for class-imbalanced Facial Emotion Recognition."""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance by down-weighting easy examples."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        p_t = torch.exp(-ce_loss)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss (ArcFace) for deep face/emotion discrimination.

    Formula: cos(theta + m) penalty on ground truth class logits.
    Default parameters matching SOTA: s=64, m=0.5.
    """

    def __init__(
        self,
        s: float = 64.0,
        m: float = 0.5,
        label_smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.label_smoothing = label_smoothing
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert logits to cosine similarity normalized [-1, 1]
        cosine = F.normalize(logits, p=2, dim=1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=logits.device)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return F.cross_entropy(
            output, targets, weight=self.weight, label_smoothing=self.label_smoothing
        )


def create_loss(
    config_loss: str,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Factory function to instantiate loss functions."""
    loss_key = config_loss.lower()
    if loss_key == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    elif loss_key == "focal":
        alpha = focal_alpha if focal_alpha is not None else class_weights
        return FocalLoss(alpha=alpha, gamma=focal_gamma, label_smoothing=label_smoothing)
    elif loss_key in ("arcface", "ce_arcface"):
        return ArcFaceLoss(s=64.0, m=0.5, label_smoothing=label_smoothing, weight=class_weights)
    else:
        raise ValueError(
            f"Unknown loss type: '{config_loss}'. Supported: 'weighted_ce', 'focal', 'arcface'"
        )
