"""Tests for the Step 8 / Step 14 loss functions.

The focal-loss tests are the point of this file. The specification states the form
explicitly as ``FL = -alpha_t (1 - p_t)^gamma log(p_t)``, and the reference notebook did
not implement that when class weighting was active. These tests pin the corrected
behaviour and document the size of the discrepancy.
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.components.losses import (
    CrossEntropyLoss,
    FocalLoss,
    LegacyFocalLoss,
)


@pytest.fixture
def batch():
    """:return: A fixed ``(logits, targets)`` batch of 4-class predictions."""
    torch.manual_seed(0)
    return torch.randn(16, 4), torch.randint(0, 4, (16,))


@pytest.fixture
def weights():
    """:return: Deliberately non-uniform class weights, as inverse frequency produces."""
    return torch.tensor([0.5, 1.0, 1.5, 2.0])


def test_plain_ce_matches_torch(batch):
    """Unweighted cross-entropy must be exactly torch's."""
    logits, targets = batch
    loss = CrossEntropyLoss(use_class_weights=False)
    assert torch.allclose(loss(logits, targets), F.cross_entropy(logits, targets))


def test_weighted_ce_applies_weights_only_when_enabled(batch, weights):
    """`set_class_weights` must be a no-op unless the loss opted in."""
    logits, targets = batch

    unweighted = CrossEntropyLoss(use_class_weights=False)
    unweighted.set_class_weights(weights)
    assert unweighted.class_weights is None
    assert torch.allclose(unweighted(logits, targets), F.cross_entropy(logits, targets))

    weighted = CrossEntropyLoss(use_class_weights=True)
    weighted.set_class_weights(weights)
    assert torch.allclose(
        weighted(logits, targets), F.cross_entropy(logits, targets, weight=weights)
    )


def test_focal_with_gamma_zero_reduces_to_cross_entropy(batch):
    """gamma=0 removes the modulating factor, so focal must collapse onto CE.

    This is the identity the notebook's formulation fails once weights are non-uniform.
    """
    logits, targets = batch
    focal = FocalLoss(gamma=0.0, use_class_weights=False)
    assert torch.allclose(focal(logits, targets), F.cross_entropy(logits, targets), atol=1e-6)


def test_weighted_focal_with_gamma_zero_reduces_to_weighted_cross_entropy(batch, weights):
    """With weights applied as alpha_t, gamma=0 must give weighted CE up to normalisation.

    torch normalises weighted CE by the summed weights; the specification's formula does
    not, so compare against the unnormalised mean.
    """
    logits, targets = batch
    focal = FocalLoss(gamma=0.0, use_class_weights=True)
    focal.set_class_weights(weights)

    expected = (
        F.cross_entropy(logits, targets, weight=weights, reduction="none") * 1.0
    ).mean()
    assert torch.allclose(focal(logits, targets), expected, atol=1e-6)


def test_focal_uses_the_true_class_probability(batch):
    """The modulating factor must be built from softmax(logits)[target], not exp(-CE)."""
    logits, targets = batch
    gamma = 2.0

    probabilities = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
    expected = (-((1 - probabilities) ** gamma) * probabilities.log()).mean()

    focal = FocalLoss(gamma=gamma, use_class_weights=False)
    assert torch.allclose(focal(logits, targets), expected, atol=1e-6)


def test_corrected_and_legacy_focal_agree_without_class_weights(batch):
    """Unweighted, exp(-CE) does equal p_t, so both formulations coincide.

    This is why the notebook's bug stayed invisible: it only bites once weights differ
    from 1, which is exactly the configuration Step 8 puts under test.
    """
    logits, targets = batch
    corrected = FocalLoss(gamma=2.0, use_class_weights=False)
    legacy = LegacyFocalLoss(gamma=2.0, use_class_weights=False)
    assert torch.allclose(corrected(logits, targets), legacy(logits, targets), atol=1e-6)


def test_corrected_and_legacy_focal_diverge_with_class_weights(batch, weights):
    """With weights, the legacy form modulates against p_t^{w_t} instead of p_t."""
    logits, targets = batch

    corrected = FocalLoss(gamma=2.0, use_class_weights=True)
    corrected.set_class_weights(weights)
    legacy = LegacyFocalLoss(gamma=2.0, use_class_weights=True)
    legacy.set_class_weights(weights)

    assert not torch.allclose(corrected(logits, targets), legacy(logits, targets), atol=1e-3)


def test_legacy_focal_modulates_against_p_to_the_weight(batch, weights):
    """Pin the exact nature of the legacy defect, so the deviation register is verifiable.

    Weighted CE for a sample is ``-w_t log(p_t)``, so ``exp(-CE) = p_t ** w_t``.
    """
    logits, targets = batch
    gamma = 2.0

    probabilities = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
    distorted_pt = probabilities ** weights[targets]
    weighted_ce = -weights[targets] * probabilities.log()
    expected = ((1 - distorted_pt) ** gamma * weighted_ce).mean()

    legacy = LegacyFocalLoss(gamma=gamma, use_class_weights=True)
    legacy.set_class_weights(weights)
    assert torch.allclose(legacy(logits, targets), expected, atol=1e-5)


def test_focal_downweights_easy_samples(weights):
    """A confident correct prediction must contribute far less than an uncertain one."""
    confident = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    uncertain = torch.tensor([[0.3, 0.2, 0.25, 0.25]])
    target = torch.tensor([0])

    focal = FocalLoss(gamma=2.0, use_class_weights=False)
    assert focal(confident, target) < 0.01 * focal(uncertain, target)


@pytest.mark.parametrize("reduction,expected_shape", [("mean", ()), ("sum", ()), ("none", (16,))])
def test_focal_reductions(batch, reduction, expected_shape):
    """`none` is needed by analyses that inspect per-sample loss."""
    logits, targets = batch
    focal = FocalLoss(gamma=2.0, use_class_weights=False, reduction=reduction)
    assert focal(logits, targets).shape == expected_shape


def test_unknown_reduction_is_rejected():
    """Typos in a config should fail at construction, not silently pick a default."""
    with pytest.raises(ValueError, match="reduction"):
        FocalLoss(reduction="average")
