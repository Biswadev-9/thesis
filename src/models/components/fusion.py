"""Feature fusion strategies and the final classifier (Steps 13 and 14).

Step 13 prescribes the order in which fusion is tried:

    "Simple concatenation should be used as the first fusion baseline. Then add
     attention-based or gated fusion only if it improves validation performance."

so three strategies are implemented and compared rather than one being assumed:

:class:`ConcatFusion`
    Baseline 1 - project, concatenate, classify.
:class:`SEFusion`
    Baseline 2 - squeeze-and-excitation attention over the concatenated vector. Weights
    are per *channel*, so they say which feature dimensions matter, not which branch.
:class:`GatedFusion`
    Baseline 3 - one weight per **branch** per image. This is the variant that answers
    Step 13's "report ... learned fusion weights" directly, because its weights are
    interpretable as how much the model trusts each branch for a given image.

**Why every strategy projects first.** The three branches arrive at wildly different
widths: classical 1280, spatial-gate 32, quantum 4. Concatenating them raw would let the
classical branch dominate by sheer dimensionality - 97 % of the input - and any
"contribution" measured afterwards would mostly be measuring width. Projecting all three
to a shared width first makes the comparison about information rather than size.

Step 14 then puts :class:`FinalClassifier` on top of the fused vector: dense layers with
normalisation, dropout and a softmax output, deliberately small to avoid overfitting a
few thousand images.
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

#: The three branches, in the order every fusion module consumes them.
BRANCH_NAMES = ("classical", "spatial", "quantum")


class FusionNet(nn.Module):
    """Shared contract for fusion modules.

    Mirrors ``FeatureNet`` on the image-space side: ``forward`` returns logits, and
    ``extract`` additionally exposes the fused vector and any interpretable weights.
    """

    def extract(
        self, classical: torch.Tensor, spatial: torch.Tensor, quantum: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Fuse the branches and classify.

        :param classical: Classical branch features, ``(B, classical_dim)``.
        :param spatial: Spatial-gate branch features, ``(B, spatial_dim)``.
        :param quantum: Quantum branch features, ``(B, quantum_dim)``.
        :return: Dict containing at least ``logits`` and ``fused``.
        :raises NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement extract()")

    def forward(
        self, classical: torch.Tensor, spatial: torch.Tensor, quantum: torch.Tensor
    ) -> torch.Tensor:
        """:param classical: Classical branch features.
        :param spatial: Spatial-gate branch features.
        :param quantum: Quantum branch features.
        :return: Class logits.
        """
        return self.extract(classical, spatial, quantum)["logits"]


class BranchProjections(nn.Module):
    """Project each branch to a shared width.

    :param classical_dim: Classical branch width.
    :param spatial_dim: Spatial-gate branch width.
    :param quantum_dim: Quantum branch width.
    :param proj_dim: Shared width every branch is mapped to.
    """

    def __init__(
        self,
        classical_dim: int = 1280,
        spatial_dim: int = 32,
        quantum_dim: int = 4,
        proj_dim: int = 64,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.classical = nn.Sequential(nn.Linear(classical_dim, proj_dim), nn.ReLU(inplace=True))
        self.spatial = nn.Sequential(nn.Linear(spatial_dim, proj_dim), nn.ReLU(inplace=True))
        self.quantum = nn.Sequential(nn.Linear(quantum_dim, proj_dim), nn.ReLU(inplace=True))

    def forward(
        self, classical: torch.Tensor, spatial: torch.Tensor, quantum: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """:param classical: Classical branch features.
        :param spatial: Spatial-gate branch features.
        :param quantum: Quantum branch features.
        :return: The three projections, each ``(B, proj_dim)``.
        """
        return self.classical(classical), self.spatial(spatial), self.quantum(quantum)


def _mlp_head(input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
    """Build the small classifier head shared by the three fusion baselines.

    :param input_dim: Input width.
    :param hidden_dim: Hidden width.
    :param num_classes: Output classes.
    :param dropout: Dropout rate, applied before each linear layer.
    :return: The head.
    """
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


class ConcatFusion(FusionNet):
    """Fusion baseline 1: project, concatenate, classify.

    Step 13's mandated starting point. The attention and gated variants must beat this on
    validation to justify their extra machinery.

    :param classical_dim: Classical branch width.
    :param spatial_dim: Spatial-gate branch width.
    :param quantum_dim: Quantum branch width.
    :param proj_dim: Shared projection width.
    :param hidden_dim: Classifier hidden width.
    :param num_classes: Output classes.
    :param dropout: Dropout rate.
    """

    def __init__(
        self,
        classical_dim: int = 1280,
        spatial_dim: int = 32,
        quantum_dim: int = 4,
        proj_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.projections = BranchProjections(classical_dim, spatial_dim, quantum_dim, proj_dim)
        self.classifier = _mlp_head(proj_dim * 3, hidden_dim, num_classes, dropout)

    def extract(self, classical, spatial, quantum) -> Dict[str, torch.Tensor]:
        """:param classical: Classical branch features.
        :param spatial: Spatial-gate branch features.
        :param quantum: Quantum branch features.
        :return: ``{"logits", "fused"}``. No branch weights - this baseline learns none.
        """
        fused = torch.cat(self.projections(classical, spatial, quantum), dim=1)
        return {"logits": self.classifier(fused), "fused": fused}


class SEFusion(FusionNet):
    """Fusion baseline 2: squeeze-and-excitation attention over the concatenated vector.

    The gate is per **channel**, not per branch, so its weights say which feature
    dimensions matter rather than which branch does. That makes it less directly
    interpretable than :class:`GatedFusion` for Step 13's reporting requirement, which is
    why both are tried.

    :param classical_dim: Classical branch width.
    :param spatial_dim: Spatial-gate branch width.
    :param quantum_dim: Quantum branch width.
    :param proj_dim: Shared projection width.
    :param hidden_dim: Classifier hidden width.
    :param num_classes: Output classes.
    :param dropout: Dropout rate.
    :param reduction: Bottleneck divisor inside the excitation block.
    """

    def __init__(
        self,
        classical_dim: int = 1280,
        spatial_dim: int = 32,
        quantum_dim: int = 4,
        proj_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.3,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.projections = BranchProjections(classical_dim, spatial_dim, quantum_dim, proj_dim)

        concat_dim = proj_dim * 3
        self.excitation = nn.Sequential(
            nn.Linear(concat_dim, max(1, concat_dim // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, concat_dim // reduction), concat_dim),
            nn.Sigmoid(),
        )
        self.classifier = _mlp_head(concat_dim, hidden_dim, num_classes, dropout)

    def extract(self, classical, spatial, quantum) -> Dict[str, torch.Tensor]:
        """:param classical: Classical branch features.
        :param spatial: Spatial-gate branch features.
        :param quantum: Quantum branch features.
        :return: ``{"logits", "fused", "channel_weights"}``.
        """
        concatenated = torch.cat(self.projections(classical, spatial, quantum), dim=1)
        channel_weights = self.excitation(concatenated)
        fused = concatenated * channel_weights
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "channel_weights": channel_weights,
        }


class GatedFusion(FusionNet):
    """Fusion baseline 3: one softmax weight per **branch** per image.

    This is the variant that satisfies Step 13's requirement to "report branch
    contribution through ... learned fusion weights" directly: its three weights sum to 1
    and read as how much the model trusts the classical, spatial-gate and quantum branch
    for that particular image.

    Note it fuses by weighted **sum** rather than concatenation, so the classifier sees a
    vector of one branch's width rather than three. That makes it the smallest of the
    three heads, which is worth remembering when comparing their scores.

    :param classical_dim: Classical branch width.
    :param spatial_dim: Spatial-gate branch width.
    :param quantum_dim: Quantum branch width.
    :param proj_dim: Shared projection width.
    :param hidden_dim: Classifier hidden width.
    :param num_classes: Output classes.
    :param dropout: Dropout rate.
    :param gate_hidden: Hidden width of the gate network.
    """

    def __init__(
        self,
        classical_dim: int = 1280,
        spatial_dim: int = 32,
        quantum_dim: int = 4,
        proj_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.3,
        gate_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.projections = BranchProjections(classical_dim, spatial_dim, quantum_dim, proj_dim)
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 3, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, len(BRANCH_NAMES)),
        )
        self.classifier = _mlp_head(proj_dim, hidden_dim, num_classes, dropout)

    def extract(self, classical, spatial, quantum) -> Dict[str, torch.Tensor]:
        """:param classical: Classical branch features.
        :param spatial: Spatial-gate branch features.
        :param quantum: Quantum branch features.
        :return: ``{"logits", "fused", "branch_weights"}`` where ``branch_weights`` is
            ``(B, 3)`` summing to 1, ordered as :data:`BRANCH_NAMES`.
        """
        projections = self.projections(classical, spatial, quantum)
        branch_weights = F.softmax(self.gate(torch.cat(projections, dim=1)), dim=1)

        fused = sum(
            branch_weights[:, index : index + 1] * projection
            for index, projection in enumerate(projections)
        )
        return {"logits": self.classifier(fused), "fused": fused, "branch_weights": branch_weights}


class FinalClassifier(nn.Module):
    """Step 14's classifier head over the fused feature vector.

    Step 14 asks for "one or two dense layers with ReLU or GELU activation" plus dropout,
    weight decay and early stopping, and a four-class softmax output. It is deliberately
    small: the fused vector is only a few hundred dimensions and the dataset a few
    thousand images, so a large head would overfit before it helped.

    Raw logits are returned from ``forward``; softmax is applied explicitly by
    :meth:`predict_proba`. Keeping the loss on logits is numerically stabler than feeding
    it probabilities.

    :param input_dim: Fused vector width.
    :param hidden_dims: Hidden layer widths.
    :param num_classes: Output classes.
    :param dropout: Dropout after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 192,
        hidden_dims: Sequence[int] = (128, 64),
        num_classes: int = 4,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = width

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(previous, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Fused feature vector, ``(B, input_dim)``.
        :return: Raw logits, ``(B, num_classes)``.
        """
        return self.output(self.hidden(x))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Fused feature vector.
        :return: The four-class softmax probability vector Step 14 specifies.
        """
        return F.softmax(self.forward(x), dim=1)


class FusedFeatureClassifier(FusionNet):
    """The Step 14 deliverable: branch projections plus the final classifier.

    Structurally this is :class:`ConcatFusion` with Step 14's deeper, normalised head in
    place of the two-layer baseline head. Step 13 chooses the fusion strategy; Step 14
    replaces the head; Step 15 trains this under the fixed protocol across seeds.

    :param classical_dim: Classical branch width.
    :param spatial_dim: Spatial-gate branch width.
    :param quantum_dim: Quantum branch width.
    :param proj_dim: Shared projection width.
    :param hidden_dims: Classifier hidden widths.
    :param num_classes: Output classes.
    :param dropout: Dropout rate.
    """

    def __init__(
        self,
        classical_dim: int = 1280,
        spatial_dim: int = 32,
        quantum_dim: int = 4,
        proj_dim: int = 64,
        hidden_dims: Sequence[int] = (128, 64),
        num_classes: int = 4,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.projections = BranchProjections(classical_dim, spatial_dim, quantum_dim, proj_dim)
        self.final_classifier = FinalClassifier(
            input_dim=proj_dim * 3,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        )

    def extract(self, classical, spatial, quantum) -> Dict[str, torch.Tensor]:
        """:param classical: Classical branch features.
        :param spatial: Spatial-gate branch features.
        :param quantum: Quantum branch features.
        :return: ``{"logits", "fused"}``.
        """
        fused = torch.cat(self.projections(classical, spatial, quantum), dim=1)
        return {"logits": self.final_classifier(fused), "fused": fused}

    @torch.no_grad()
    def predict_proba(self, classical, spatial, quantum) -> torch.Tensor:
        """:param classical: Classical branch features.
        :param spatial: Spatial-gate branch features.
        :param quantum: Quantum branch features.
        :return: Four-class softmax probabilities.
        """
        return F.softmax(self.forward(classical, spatial, quantum), dim=1)


#: Fusion strategies compared in Step 13, in the order the specification prescribes.
FUSION_STRATEGIES = {
    "concat": ConcatFusion,
    "se": SEFusion,
    "gated": GatedFusion,
}
