"""Timm backbone factory and output shape normalization wrapper."""

import logging
from typing import Set, Tuple

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

SUPPORTED_BACKBONES: Set[str] = {
    "mobilenetv3_large_100",
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnetv2_rw_m",
    "convnext_tiny",
    "convnext_base",
    "vit_tiny_patch16_224",
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "swin_base_patch4_window7_224",
    "resnet50d",
    "mobilevit_xs",
}


class BackboneWrapper(nn.Module):
    """Wrapper that normalizes timm backbone outputs to a 2D tensor [B, feature_dim]."""

    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        if features.ndim == 4:
            features = self.pool(features)  # [B, C, 1, 1]
            features = features.flatten(1)  # [B, C]
        elif features.ndim == 3:
            features = features.mean(dim=1)  # [B, C]

        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected output shape [B, {self.feature_dim}], got {features.shape}"
            )

        return features


def create_backbone(
    name: str, pretrained: bool = True
) -> Tuple[BackboneWrapper, int]:
    """Create a timm feature extractor backbone wrapped in BackboneWrapper."""
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone: '{name}'. "
            f"Supported backbones: {sorted(list(SUPPORTED_BACKBONES))}"
        )

    backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)

    # Discover feature dimension via a dummy forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_out = backbone(dummy_input)

    if dummy_out.ndim == 2:
        feature_dim = dummy_out.shape[1]
    elif dummy_out.ndim in (3, 4):
        feature_dim = dummy_out.shape[1]
    else:
        raise ValueError(
            f"Unexpected backbone output dimension: {dummy_out.ndim}D"
        )

    wrapper = BackboneWrapper(backbone, feature_dim)
    logger.info(f"Created backbone '{name}' with feature_dim={feature_dim}")

    return wrapper, feature_dim
