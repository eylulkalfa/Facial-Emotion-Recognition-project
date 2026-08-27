"""End-to-end Facial Emotion Recognition model combining backbone and classification head."""

import logging
from typing import Optional

import torch
import torch.nn as nn

from fer.config import ModelConfig
from fer.models.backbone_factory import create_backbone
from fer.models.heads import FERHead

logger = logging.getLogger(__name__)


class FERModel(nn.Module):
    """Complete Facial Emotion Recognition model.

    Combines a timm feature extractor backbone with an architecture-independent
    FER classification head. Supports backbone freezing/unfreezing for multi-phase
    transfer learning.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        backbone_name: str = "mobilenetv3_large_100",
        pretrained: bool = True,
        num_classes: int = 7,
        dropout: float = 0.2,
    ):
        """Initialize FERModel.

        Can be initialized either from a ModelConfig object or from explicit parameters.
        If config is provided, its parameters override explicit arguments.
        """
        super().__init__()

        if config is not None:
            backbone_name = config.backbone
            pretrained = config.pretrained
            num_classes = config.num_classes
            dropout = config.dropout

        self.backbone_name = backbone_name
        self.num_classes = num_classes

        self.backbone, self.feature_dim = create_backbone(
            backbone_name, pretrained=pretrained
        )
        self.head = FERHead(
            feature_dim=self.feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.logger = logging.getLogger(__name__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor of shape [B, 3, 224, 224].

        Returns:
            Raw logits tensor of shape [B, num_classes].
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def forward_with_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting calibrated probabilities.

        Args:
            x: Input image tensor of shape [B, 3, 224, 224].

        Returns:
            Probability tensor of shape [B, num_classes] (softmax applied).
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters for head-only warmup training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.logger.info(f"Backbone '{self.backbone_name}' parameters frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full backbone training/fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.logger.info(f"Backbone '{self.backbone_name}' parameters unfrozen")

    def get_trainable_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_model_size_mb(self) -> float:
        """Estimate model size in megabytes (assuming FP32 precision)."""
        total_params = self.get_total_params()
        return (total_params * 4) / (1024 * 1024)


def create_fer_model(config: ModelConfig) -> FERModel:
    """Convenience factory function creating FERModel from ModelConfig."""
    return FERModel(config=config)
