"""Pretrained transfer-learning backbones (Step 9, Baselines 2-4b).

The specification asks for a CNN baseline (ResNet50 or DenseNet121), an EfficientNet
baseline, and a Transformer baseline (ViT or Swin). Without strong baselines the study
"cannot support claims about improvement or quantum advantage", so these are the
reference points every later result is measured against.

All four share one wrapper. Each torchvision model's classification head is replaced with
``Identity`` so the backbone emits a feature vector, our own dropout-plus-linear head sits
on top, and the backbone is frozen except for the final blocks. That gives a uniform
``extract`` contract across four architectures whose native forward signatures differ
considerably - ViT and Swin in particular do their own token pooling internally.

Partial fine-tuning rather than full is deliberate and matches the reference: this dataset
has a few thousand images, and unfreezing 25M+ parameters would overfit long before it
helped. The number of unfrozen blocks per architecture is the reference's choice.
"""

from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn
from torchvision import models

from src.models.components.base import FeatureNet
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _last_linear_in_features(head: nn.Module) -> int:
    """Find the input width of a classification head.

    Heads vary: ``Linear`` for ResNet and Swin, ``Sequential`` for EfficientNet and ViT.
    Reading the width from the head itself avoids hard-coding a dimension per
    architecture, which would silently break if torchvision changed a definition.

    :param head: The model's classification head.
    :return: Input feature count.
    :raises TypeError: If no ``Linear`` layer can be found.
    """
    if isinstance(head, nn.Linear):
        return head.in_features

    linears = [module for module in head.modules() if isinstance(module, nn.Linear)]
    if not linears:
        raise TypeError(f"Could not locate a Linear layer inside head {type(head).__name__}")
    return linears[0].in_features


def _resnet_blocks(model: nn.Module) -> List[nn.Module]:
    """:param model: A torchvision ResNet.
    :return: The final residual stage, matching the reference's fine-tuning depth.
    """
    return [model.layer4]


def _efficientnet_blocks(model: nn.Module) -> List[nn.Module]:
    """:param model: A torchvision EfficientNet.
    :return: The last three feature blocks.
    """
    return list(model.features[-3:])


def _vit_blocks(model: nn.Module) -> List[nn.Module]:
    """:param model: A torchvision ViT.
    :return: The last two transformer encoder layers.
    """
    return list(model.encoder.layers[-2:])


def _swin_blocks(model: nn.Module) -> List[nn.Module]:
    """:param model: A torchvision Swin Transformer.
    :return: The final stage.
    """
    return list(model.features[-2:])


#: Per-architecture head attribute and the blocks left trainable.
ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "resnet50": {"head": "fc", "unfreeze": _resnet_blocks, "family": "cnn"},
    "densenet121": {"head": "classifier", "unfreeze": lambda m: [m.features.denseblock4], "family": "cnn"},
    "efficientnet_b0": {"head": "classifier", "unfreeze": _efficientnet_blocks, "family": "cnn"},
    "efficientnet_v2_s": {"head": "classifier", "unfreeze": _efficientnet_blocks, "family": "cnn"},
    "convnext_tiny": {"head": "classifier", "unfreeze": lambda m: [m.features[-1]], "family": "cnn"},
    "vit_b_16": {"head": "heads", "unfreeze": _vit_blocks, "family": "transformer"},
    "swin_t": {"head": "head", "unfreeze": _swin_blocks, "family": "transformer"},
}


class TransferBackbone(FeatureNet):
    """A pretrained torchvision backbone with a fresh classification head.

    :param arch: Key into :data:`ARCHITECTURES`.
    :param num_classes: Number of output classes.
    :param weights: torchvision weights enum name, ``"DEFAULT"`` for the recommended
        ImageNet weights, or ``None`` for random initialisation. ``None`` is used in
        tests and smoke runs so they need no weight download.
    :param freeze_backbone: Freeze the backbone and fine-tune only its final blocks. The
        specification's Step 10 guidance - "fine-tune it carefully" - applies here too.
    :param dropout: Dropout before the classification head.
    :raises ValueError: If ``arch`` is unknown.
    """

    def __init__(
        self,
        arch: str = "efficientnet_b0",
        num_classes: int = 4,
        weights: Optional[str] = "DEFAULT",
        freeze_backbone: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if arch not in ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture {arch!r}. Available: {sorted(ARCHITECTURES)}"
            )

        spec = ARCHITECTURES[arch]
        self.arch = arch
        self.family = spec["family"]

        backbone = models.get_model(arch, weights=weights)

        head_attr = spec["head"]
        self.feature_dim = _last_linear_in_features(getattr(backbone, head_attr))
        # Strip the ImageNet head; the backbone now emits a feature vector directly, so
        # every architecture exposes the same interface regardless of how it pools.
        setattr(backbone, head_attr, nn.Identity())

        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        if freeze_backbone:
            self._freeze_except(spec["unfreeze"])

    def _freeze_except(self, select_blocks: Callable[[nn.Module], List[nn.Module]]) -> None:
        """Freeze the backbone, then re-enable gradients on the selected blocks.

        :param select_blocks: Returns the blocks to leave trainable.
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for block in select_blocks(self.backbone):
            for parameter in block.parameters():
                parameter.requires_grad = True

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        log.info(
            f"{self.arch}: fine-tuning {trainable:,}/{total:,} backbone parameters "
            f"({100 * trainable / max(total, 1):.1f}%)"
        )

    def extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """:param x: Image batch, ``(B, 3, H, W)``.
        :return: ``{"logits", "features"}`` where features is the pre-head embedding.
        """
        features = self.backbone(x)
        # Some backbones return spatial maps if their head was not the only pooling
        # stage; flatten defensively so the head always receives (B, feature_dim).
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return {"logits": self.classifier(self.dropout(features)), "features": features}


class FixedMultiscaleCNN(FeatureNet):
    """Baseline 6: parallel fixed kernels, concatenated without any gating.

    This is the control the Step 11 adaptive multiscale branch must beat. It uses the
    same idea - several receptive fields in parallel - but merges them by plain
    concatenation, with no attention or gating deciding which scale matters where. Any
    gain the proposed branch shows over this baseline is attributable to the *gating*,
    not merely to being multiscale.

    :param num_classes: Number of output classes.
    :param channels: Channel width of the stem and of each parallel path.
    :param kernel_sizes: Kernel size per parallel path.
    :param dropout: Dropout before the classification head.
    """

    def __init__(
        self,
        num_classes: int = 4,
        channels: int = 32,
        kernel_sizes: tuple = (3, 5, 7),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.paths = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = channels * len(kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """:param x: Image batch, ``(B, 3, H, W)``.
        :return: ``{"logits", "features"}``.
        """
        stem = self.stem(x)
        merged = torch.cat([torch.relu(path(stem)) for path in self.paths], dim=1)
        features = torch.flatten(self.pool(merged), 1)
        return {"logits": self.classifier(self.dropout(features)), "features": features}
