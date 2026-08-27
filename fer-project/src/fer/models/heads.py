"""Classification head for Facial Emotion Recognition."""

import torch
import torch.nn as nn


class FERHead(nn.Module):
    """Architecture-independent FER classification head.

    Architecture: Dropout -> Linear(feature_dim, num_classes)
    Outputs raw logits [B, num_classes] without softmax.
    """

    def __init__(
        self, feature_dim: int, num_classes: int = 7, dropout: float = 0.2
    ):
        """Initialize FERHead.

        Args:
            feature_dim: Input feature dimension size from backbone.
            num_classes: Number of output emotion classes (default 7).
            dropout: Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature tensor of shape [B, feature_dim].

        Returns:
            Raw logits tensor of shape [B, num_classes].
        """
        x = self.dropout(x)
        x = self.fc(x)
        return x
