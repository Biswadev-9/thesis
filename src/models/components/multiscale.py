"""Adaptive multiscale kernel branch and its ablation arms (Step 11).

Step 11 asks for parallel convolutional paths with different receptive fields, combined by
a gate that outputs normalised weights, and requires an ablation against fixed kernels:

    "Ablation is essential: compare fixed 3x3 convolution, fixed 5x5 convolution, fixed
     dilated convolution, and the proposed adaptive multiscale module."

**This module follows the reference notebook rather than the specification's wording**, by
explicit instruction. Two consequences worth noting:

- The paths are 3x3, 5x5 and *dilated* 3x3 (dilation 3). The specification suggests
  "3x3, 5x5, and 7x7, or ... dilated convolutions"; the notebook mixed the two, taking two
  plain kernels and one dilated. The dilated path reaches a 7x7 receptive field with 3x3
  parameter cost, so the intent - fine, medium and broad context - is preserved.
- The gate is **per pixel**. It emits a softmax over the three paths at every spatial
  location, so the model can prefer fine detail in one region and broad context in
  another. This is the study's actual novelty, and the reason Arm 5 (global, SKNet-style
  gating) exists as its control: Arm 5 collapses the spatial dimension before gating, so
  the difference between them isolates *spatial* adaptivity from adaptivity as such.

The eight ablation arms, all built from :class:`MultiscaleClassifier`:

===== ==================== ==========================================================
Arm   variant              What it isolates
===== ==================== ==========================================================
1     ``fixed_3x3``        single fine kernel, no multiscale
2     ``fixed_5x5``        single medium kernel
3     ``fixed_dilated``    single broad kernel
4     ``concat_nogate``    all three paths, merged without any gating
5     ``global_gate``      one weight per path per image (SKNet-style)
6     ``spatial_gate``     one weight per path **per pixel** - proposed
7     ``spatial_gate`` +q  proposed branch feeding a fixed quantum circuit
8     ``global_gate`` +q   global gating feeding the same circuit
===== ==================== ==========================================================
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.base import FeatureNet

#: Branch variants. Arms 7 and 8 reuse ``spatial_gate``/``global_gate`` with a quantum head.
VARIANTS = ("fixed_3x3", "fixed_5x5", "fixed_dilated", "concat_nogate", "global_gate", "spatial_gate")

#: Arm name -> (variant, quantum). The single place the ablation's structure is defined.
ARMS: Dict[str, Tuple[str, bool]] = {
    "arm1_fixed_3x3": ("fixed_3x3", False),
    "arm2_fixed_5x5": ("fixed_5x5", False),
    "arm3_fixed_dilated": ("fixed_dilated", False),
    "arm4_concat_nogate": ("concat_nogate", False),
    "arm5_global_gate": ("global_gate", False),
    "arm6_spatial_gate": ("spatial_gate", False),
    "arm7_spatial_gate_quantum": ("spatial_gate", True),
    "arm8_global_gate_quantum": ("global_gate", True),
}

#: Human-readable path labels, used by the Step 11 morphology figures.
PATH_LABELS = ("3x3 (fine)", "5x5 (medium)", "dilated (broad)")


class ConvStem(nn.Module):
    """Shared convolutional stem, downsampling by 4.

    Every arm uses the identical stem so the ablation compares the *gating*, not the
    feature extractor in front of it. At the pipeline's 224px input this yields 56x56
    maps, which is coarse enough to be cheap and fine enough that a per-pixel gate still
    has meaningful spatial structure to learn.

    :param in_channels: Input channels.
    :param channels: Output channels, shared by every downstream path.
    """

    def __init__(self, in_channels: int = 3, channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Image batch, ``(B, C_in, H, W)``.
        :return: Feature maps, ``(B, channels, H/4, W/4)``.
        """
        return self.stem(x)


def _make_path(channels: int, kind: str) -> nn.Sequential:
    """Build one convolutional path.

    :param channels: Input and output channels.
    :param kind: ``fine`` (3x3), ``medium`` (5x5) or ``broad`` (dilated 3x3).
    :return: Conv-BN-ReLU block preserving spatial size.
    :raises ValueError: If ``kind`` is unknown.
    """
    if kind == "fine":
        conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    elif kind == "medium":
        conv = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
    elif kind == "broad":
        # Dilation 3 gives a 7x7 receptive field at 3x3 parameter cost.
        conv = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3)
    else:
        raise ValueError(f"Unknown path kind {kind!r}")

    return nn.Sequential(conv, nn.BatchNorm2d(channels), nn.ReLU(inplace=True))


class SpatialMultiScaleGate(nn.Module):
    """Three parallel paths fused by a **per-pixel** softmax gate.

    This is the proposed module. The gate head is deliberately small - two 1x1
    convolutions through a bottleneck - so the arm's advantage cannot be attributed to
    simply having added capacity.

    Weights sum to 1 across the three paths at every spatial location, which is what makes
    them interpretable as "which receptive field does the model trust *here*" and lets
    Step 11's morphology analysis correlate them against tumour extent.

    :param channels: Channels per path.
    :param bottleneck_divisor: Gate head bottleneck width, as a divisor of ``channels``.
    """

    def __init__(self, channels: int = 32, bottleneck_divisor: int = 2) -> None:
        super().__init__()
        self.channels = channels
        self.path_fine = _make_path(channels, "fine")
        self.path_medium = _make_path(channels, "medium")
        self.path_broad = _make_path(channels, "broad")

        hidden = max(1, channels // bottleneck_divisor)
        self.gate_head = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """:param x: Stem output, ``(B, C, H, W)``.
        :return: ``(fused, weight_maps)`` where weight_maps is ``(B, 3, H, W)`` and sums
            to 1 along dim 1.
        """
        paths = [self.path_fine(x), self.path_medium(x), self.path_broad(x)]
        scores = self.gate_head(torch.cat(paths, dim=1))
        weight_maps = F.softmax(scores, dim=1)

        fused = sum(weight_maps[:, i : i + 1] * path for i, path in enumerate(paths))
        return fused, weight_maps


class GlobalMultiScaleGate(nn.Module):
    """The same three paths fused by one weight per path **per image** (SKNet-style).

    Arm 5's control. It pools each path to a scalar summary before gating, so the model can
    still choose between receptive fields but cannot vary that choice across the image.
    Comparing it against :class:`SpatialMultiScaleGate` is what isolates the contribution
    of *spatial* adaptivity.

    :param channels: Channels per path.
    :param bottleneck_divisor: Gate bottleneck width, as a divisor of ``channels``.
    """

    def __init__(self, channels: int = 32, bottleneck_divisor: int = 2) -> None:
        super().__init__()
        self.channels = channels
        self.path_fine = _make_path(channels, "fine")
        self.path_medium = _make_path(channels, "medium")
        self.path_broad = _make_path(channels, "broad")

        hidden = max(1, channels // bottleneck_divisor)
        self.gate_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_fc = nn.Sequential(
            nn.Linear(3 * channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """:param x: Stem output, ``(B, C, H, W)``.
        :return: ``(fused, weights)`` where weights is ``(B, 3)`` - note the missing
            spatial dimensions relative to the spatial gate.
        """
        paths = [self.path_fine(x), self.path_medium(x), self.path_broad(x)]
        pooled = torch.cat([torch.flatten(self.gate_pool(p), 1) for p in paths], dim=1)
        weights = F.softmax(self.gate_fc(pooled), dim=1)

        fused = sum(weights[:, i].view(-1, 1, 1, 1) * path for i, path in enumerate(paths))
        return fused, weights


class MultiscaleBranch(nn.Module):
    """Stem plus one of the six path/gating variants, pooled to a feature vector.

    :param variant: One of :data:`VARIANTS`.
    :param channels: Channel width throughout.
    :raises ValueError: If ``variant`` is unknown.
    """

    def __init__(self, variant: str = "spatial_gate", channels: int = 32) -> None:
        super().__init__()

        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}. Available: {VARIANTS}")

        self.variant = variant
        self.channels = channels
        self.feature_dim = channels
        self.stem = ConvStem(channels=channels)

        if variant == "spatial_gate":
            self.gate = SpatialMultiScaleGate(channels)
        elif variant == "global_gate":
            self.gate = GlobalMultiScaleGate(channels)
        elif variant == "concat_nogate":
            self.path_fine = _make_path(channels, "fine")
            self.path_medium = _make_path(channels, "medium")
            self.path_broad = _make_path(channels, "broad")
            # 1x1 projection back to `channels`, so this arm's feature width - and its
            # head - match the gated arms exactly. Without it, Arm 4 would be compared
            # against the others with three times the features.
            self.project = nn.Conv2d(3 * channels, channels, kernel_size=1)
        else:
            kind = {"fixed_3x3": "fine", "fixed_5x5": "medium", "fixed_dilated": "broad"}[variant]
            self.path = _make_path(channels, kind)

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """:param x: Image batch, ``(B, 3, H, W)``.
        :return: ``(features, gate_weights)``. ``gate_weights`` is ``(B, 3, H/4, W/4)`` for
            the spatial gate, ``(B, 3)`` for the global gate, and ``None`` for the
            ungated arms.
        """
        stem = self.stem(x)
        weights: Optional[torch.Tensor] = None

        if self.variant in ("spatial_gate", "global_gate"):
            fused, weights = self.gate(stem)
        elif self.variant == "concat_nogate":
            paths = [self.path_fine(stem), self.path_medium(stem), self.path_broad(stem)]
            fused = self.project(torch.cat(paths, dim=1))
        else:
            fused = self.path(stem)

        return torch.flatten(self.pool(fused), 1), weights


class MultiscaleClassifier(FeatureNet):
    """A multiscale branch with a classification head - one of the eight ablation arms.

    :param variant: Branch variant; see :data:`VARIANTS`.
    :param num_classes: Number of output classes.
    :param channels: Channel width.
    :param dropout: Dropout before the head.
    :param quantum: Route features through a fixed quantum circuit before classifying
        (arms 7 and 8).
    :param n_qubits: Circuit width when ``quantum`` is set.
    :param circuit_name: Circuit variant when ``quantum`` is set.
    """

    def __init__(
        self,
        variant: str = "spatial_gate",
        num_classes: int = 4,
        channels: int = 32,
        dropout: float = 0.3,
        quantum: bool = False,
        n_qubits: int = 4,
        circuit_name: str = "fixed",
    ) -> None:
        super().__init__()

        self.branch = MultiscaleBranch(variant=variant, channels=channels)
        self.variant = variant
        self.quantum = quantum
        self.feature_dim = channels

        if quantum:
            # Imported lazily so the classical arms never pay PennyLane's import cost.
            from src.models.components.quantum import QuantumLayer

            self.reduce = nn.Linear(channels, n_qubits)
            self.quantum_layer = QuantumLayer(circuit_name=circuit_name, n_qubits=n_qubits)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(n_qubits, num_classes)
        else:
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(channels, num_classes)

    @classmethod
    def from_arm(cls, arm: str, **kwargs) -> "MultiscaleClassifier":
        """Build an arm by its ablation name.

        :param arm: Key of :data:`ARMS`, e.g. ``arm6_spatial_gate``.
        :param kwargs: Forwarded to the constructor.
        :return: The configured arm.
        :raises ValueError: If ``arm`` is unknown.
        """
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}. Available: {sorted(ARMS)}")
        variant, quantum = ARMS[arm]
        return cls(variant=variant, quantum=quantum, **kwargs)

    def extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """:param x: Image batch, ``(B, 3, H, W)``.
        :return: ``{"logits", "features", "gate_maps"}``. ``gate_maps`` is present only
            for the gated variants, and carries spatial dimensions only for
            ``spatial_gate``.
        """
        features, gate_weights = self.branch(x)

        outputs: Dict[str, torch.Tensor] = {"features": features}
        if gate_weights is not None:
            outputs["gate_maps"] = gate_weights

        if self.quantum:
            from src.models.components.quantum import scale_to_angles

            angles = scale_to_angles(self.reduce(features))
            quantum_features = self.quantum_layer(angles)
            outputs["quantum_features"] = quantum_features
            outputs["logits"] = self.classifier(self.dropout(quantum_features))
        else:
            outputs["logits"] = self.classifier(self.dropout(features))

        return outputs
