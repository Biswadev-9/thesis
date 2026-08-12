"""Convolutional networks trained from scratch.

Two of the specification's models live here. The pretrained transfer-learning backbones
(ResNet50, EfficientNet-B0, ViT, Swin) arrive in Phase 3 alongside the rest of Step 9.

- :class:`SimpleCNN` is Step 9's Baseline 1.
- :class:`SmallCNN` is the lightweight proxy used by the Step 6 preprocessing sweep and
  the Step 8 imbalance ablation, both of which evaluate many configurations and so run
  at reduced resolution on a balanced subset.
"""

from typing import Dict

import torch
from torch import nn

from src.models.components.base import FeatureNet


class SimpleCNN(FeatureNet):
    """Baseline 1: a plain four-block convolutional network trained from scratch.

    Provides the "no pretrained prior" reference point. Without it, every baseline would
    carry ImageNet weights and the study could not separate architectural benefit from
    transfer-learning benefit.

    Global average pooling before the classifier keeps the network valid at any input
    size, so the same class serves both the 224 px main pipeline and any reduced-size
    sweep.

    :param num_classes: Number of output classes.
    :param channels: Output channels of each convolutional block.
    :param dropout: Dropout applied before the classifier.
    :param in_channels: Input channel count; 3 for the RGB-replicated MRI pipeline.
    """

    def __init__(
        self,
        num_classes: int = 4,
        channels: tuple = (32, 64, 128, 256),
        dropout: float = 0.3,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        blocks = []
        previous = in_channels
        for width in channels:
            blocks.extend(
                [
                    nn.Conv2d(previous, width, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            previous = width

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(previous, num_classes)
        self.feature_dim = previous

    def extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """:param x: Image batch, ``(B, C, H, W)``.
        :return: ``{"logits", "features"}``.
        """
        maps = self.features(x)
        embedding = torch.flatten(self.pool(maps), 1)
        return {"logits": self.classifier(self.dropout(embedding)), "features": embedding}


class SmallCNN(FeatureNet):
    """Compact proxy network for the Step 6 and Step 8 sweeps.

    The notebook's version ended in ``Linear(64 * 16 * 16, 128)``, which silently
    hard-coded a 128 px input - the module raised a shape error at any other resolution
    and could not be reused at the 224 px pipeline size. Global average pooling replaces
    the flattened head, so the proxy is resolution-independent while keeping the same
    three-block shape and parameter scale.

    :param num_classes: Number of output classes.
    :param channels: Output channels of each convolutional block.
    :param hidden_dim: Width of the penultimate dense layer.
    :param dropout: Dropout before the classifier.
    :param in_channels: Input channel count.
    """

    def __init__(
        self,
        num_classes: int = 4,
        channels: tuple = (16, 32, 64),
        hidden_dim: int = 128,
        dropout: float = 0.3,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        blocks = []
        previous = in_channels
        for width in channels:
            blocks.extend(
                [
                    nn.Conv2d(previous, width, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            previous = width

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(nn.Linear(previous, hidden_dim), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.feature_dim = hidden_dim

    def extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """:param x: Image batch, ``(B, C, H, W)``.
        :return: ``{"logits", "features"}``.
        """
        maps = self.features(x)
        embedding = self.projection(torch.flatten(self.pool(maps), 1))
        return {"logits": self.classifier(self.dropout(embedding)), "features": embedding}
