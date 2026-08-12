"""Shared contract for every image-space network in the project.

The reference notebook grew five different forward signatures - ``x -> logits``,
``x -> (logits, features)``, ``x -> (logits, features, weight_maps)``, ``x -> dict`` and
``(c, s, q) -> (logits, weights)`` - which meant every model needed its own training
loop. Downstream analyses then had to know which shape each model returned.

One contract fixes that:

``forward(x)``
    Returns logits only. This is what the loss and the metrics consume, so a single
    ``training_step`` serves every model in the repository.

``extract(x)``
    Returns a dict that always contains ``logits`` and ``features``, plus whatever
    auxiliary tensors that architecture produces. Analyses request auxiliaries by key
    and simply skip models that do not expose them.

Auxiliary keys used downstream:

``features``
    Pre-classifier embedding. Step 10's separability analysis and the Step 13 fusion
    stage both read it.
``gate_maps``
    Per-pixel multiscale gate weights, ``(B, 3, H, W)``. Step 11's morphology analysis.
``branch_weights``
    Per-image, per-branch fusion weights, ``(B, 3)``. Step 13's contribution reporting.
``quantum_weights``
    Per-image quantum-expert mixture weights, ``(B, 5)``. Step 12's selection analysis.
"""

from typing import Dict

import torch
from torch import nn


class FeatureNet(nn.Module):
    """Base class implementing the shared ``forward``/``extract`` contract.

    Subclasses implement :meth:`extract` and inherit :meth:`forward`.
    """

    #: Dimensionality of the ``features`` entry returned by :meth:`extract`.
    feature_dim: int

    def extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the network and return logits plus any auxiliary tensors.

        :param x: Input batch of images, ``(B, C, H, W)``.
        :return: Dict containing at least ``logits`` and ``features``.
        :raises NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement extract()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits only.

        :param x: Input batch of images, ``(B, C, H, W)``.
        :return: Class logits, ``(B, num_classes)``.
        """
        return self.extract(x)["logits"]
