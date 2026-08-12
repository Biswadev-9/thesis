"""Loss functions for the class-imbalance ablation (Steps 8 and 14).

Every loss here exposes ``set_class_weights``, so the `LightningModule` can inject
weights derived from the training split without the config needing to know them ahead of
time. Losses that ignore weights implement it as a no-op, which keeps the config
interface uniform across the ablation.

**The focal loss is corrected relative to the reference notebook.** The specification
states the form explicitly:

    FL = -alpha_t (1 - p_t)^gamma log(p_t)

The notebook computed ``p_t = exp(-CE)`` where ``CE`` was *already class-weighted*. That
identity only holds when every class weight is 1: with weights ``w``, the weighted
cross-entropy for a sample is ``-w_t log(p_t)``, so ``exp(-CE) = p_t^{w_t}``, not
``p_t``. The modulating factor ``(1 - p_t)^gamma`` was therefore computed against a
distorted quantity whenever class weighting was active - which was exactly the
configuration under test. :class:`FocalLoss` implements the specification's form;
:class:`LegacyFocalLoss` preserves the notebook's behaviour so its published Step 8 and
Step 14 tables remain reproducible.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class WeightAwareLoss(nn.Module):
    """Base class for losses that can optionally consume training-split class weights.

    :param use_class_weights: Whether :meth:`set_class_weights` should take effect.
    """

    def __init__(self, use_class_weights: bool = False) -> None:
        super().__init__()
        self.use_class_weights = use_class_weights
        # Non-persistent: a buffer so `.to(device)` moves it with the module, but kept out
        # of the checkpoint because class weights are *derived* from the training split
        # rather than learned. Persisting them would also make checkpoints unloadable into
        # a freshly built module, whose buffer is still None and so has no matching key.
        # They are re-derived from the datamodule in `MRIClassificationModule.setup`.
        self.register_buffer("class_weights", None, persistent=False)

    def set_class_weights(self, weights: Optional[torch.Tensor]) -> None:
        """Install class weights derived from the training split.

        Called once by the `LightningModule` during ``setup``. Ignored when the loss was
        configured with ``use_class_weights=False``, so the same call site serves every
        candidate in the ablation.

        :param weights: Per-class weights, ``(num_classes,)``, or ``None``.
        """
        if not self.use_class_weights or weights is None:
            return
        self.class_weights = weights.detach().clone().to(dtype=torch.float32)

    def extra_repr(self) -> str:
        return f"use_class_weights={self.use_class_weights}"


class CrossEntropyLoss(WeightAwareLoss):
    """Cross-entropy, optionally class-weighted.

    Covers two of the specification's candidates from one class: plain cross-entropy
    (``use_class_weights=False``) and Step 8's class-weighted cross-entropy
    (``use_class_weights=True``).

    :param use_class_weights: Apply inverse-frequency class weights.
    :param label_smoothing: Optional label smoothing; 0 reproduces standard CE.
    """

    def __init__(self, use_class_weights: bool = False, label_smoothing: float = 0.0) -> None:
        super().__init__(use_class_weights=use_class_weights)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """:param logits: Raw scores, ``(B, K)``.
        :param targets: Integer labels, ``(B,)``.
        :return: Scalar loss.
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )


class FocalLoss(WeightAwareLoss):
    """Focal loss as specified: ``FL = -alpha_t (1 - p_t)^gamma log(p_t)``.

    ``p_t`` is read directly off the softmax for the true class, so the modulating factor
    is computed against a genuine probability regardless of whether class weighting is
    active. ``alpha_t`` is then applied as a separate multiplicative factor, exactly as
    the formula reads.

    :param gamma: Focusing parameter; larger values down-weight easy samples harder.
    :param use_class_weights: Use inverse-frequency weights as ``alpha``.
    :param reduction: ``mean``, ``sum`` or ``none``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        use_class_weights: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__(use_class_weights=use_class_weights)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction {reduction!r}")
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """:param logits: Raw scores, ``(B, K)``.
        :param targets: Integer labels, ``(B,)``.
        :return: Scalar loss, or per-sample losses when ``reduction="none"``.
        """
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        loss = -((1.0 - pt) ** self.gamma) * log_pt

        if self.class_weights is not None:
            loss = loss * self.class_weights.to(loss.device)[targets]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, {super().extra_repr()}"


class LegacyFocalLoss(WeightAwareLoss):
    """The reference notebook's focal loss, kept only for reproducing its results.

    Computes ``pt = exp(-CE_weighted)``, which equals the true class probability only
    when all class weights are 1. Retained so the notebook's Step 8 and Step 14 tables
    can be regenerated for comparison; :class:`FocalLoss` is what the pipeline uses.

    :param gamma: Focusing parameter.
    :param use_class_weights: Use inverse-frequency weights inside the cross-entropy.
    :param reduction: ``mean``, ``sum`` or ``none``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        use_class_weights: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__(use_class_weights=use_class_weights)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction {reduction!r}")
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """:param logits: Raw scores, ``(B, K)``.
        :param targets: Integer labels, ``(B,)``.
        :return: Scalar loss, or per-sample losses when ``reduction="none"``.
        """
        ce = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, {super().extra_repr()}"
